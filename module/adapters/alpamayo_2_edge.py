"""Adapter for Alpamayo 2 Edge, package ``alpamayo2_edge``.

Edge is a 2B release built on Cosmos3-Edge with its own input profile: six of the seven
PhysicalAI-AV cameras (no rear tele), each sampled at its own times (front wide and front
tele at -2/-1/-0.5/-0.1/0 s, the other four at -1/-0.1/0 s), 20 ego history poses, and a
navigation instruction in every driving prompt. VQA uses all six cameras at -0.3..0 s.
The package needs Transformers 5, so it runs in its own environment. There is no
classifier-free-guidance sampler, so non-1.0 navigation weights fall back to plain
conditioning, and neither 4-bit quantization nor OOM-free offloading is provided.
"""

from __future__ import annotations

import math

import torch

from .. import config as cfg
from ._rigs import FRONT_WIDE_SLOT, SIX_CAMERA_INDICES, SIX_CAMERA_RIG
from ._text import extract_text_field
from .base import AlpamayoAdapter

# Ticks (cfg.CONTROL_DT apart) before t0 that each camera index contributes.
DRIVING_TICKS = {
    camera: (20, 10, 5, 1, 0) if camera in (1, 6) else (10, 1, 0) for camera in SIX_CAMERA_INDICES
}
VQA_TICKS = {camera: (3, 2, 1, 0) for camera in SIX_CAMERA_INDICES}


class Alpamayo2EdgeAdapter(AlpamayoAdapter):
    version = "2-edge"
    model_id = "nvidia/Alpamayo2-Edge"
    #: Checkpoint revision documented by the pinned alpamayo2-edge submodule.
    model_revision = "f8951d095de3589b3face1dec3f4394c8ecedb38"
    display_name = "Alpamayo 2 Edge"

    source_camera_configs = SIX_CAMERA_RIG
    source_camera_indices = SIX_CAMERA_INDICES
    viz_camera_slot = FRONT_WIDE_SLOT
    # Union of the driving and VQA schedules; each task picks its frames from these.
    frame_ticks = (20, 10, 5, 3, 2, 1, 0)
    num_history_steps = 20

    supports_navigation = True
    supports_vqa = True
    supports_oom_free = False

    #: Edge requires a navigation instruction; this one is used when none is given.
    default_navigation_text = "Follow the road"

    def load_model(
        self, use_quantization=False, device_map="auto", oom_free=False, oom_kwargs=None,
        max_memory=None,
    ):
        if oom_free:
            raise ValueError("OOM-free demand layering is not supported for Alpamayo 2 Edge.")
        if use_quantization:
            raise ValueError("4-bit quantization is not supported for Alpamayo 2 Edge.")

        from alpamayo2_edge import helper
        from alpamayo2_edge.accel import AccelConfig, enable_acceleration
        from alpamayo2_edge.checkpoint import load_model

        # The 2B model fits on one GPU, so device_map / max_memory placement is not used.
        model, _loading_info = load_model(
            self.model_id, revision=self.model_revision, attn_implementation="flash_attention_2"
        )
        # NVIDIA's accelerated path (CLI --accel tuned): torch.compile + CUDA graphs for the
        # prefill, CoT decode and trajectory expert. The first inference of each input shape
        # compiles and captures its graphs, which takes tens of seconds.
        enable_acceleration(model, AccelConfig(mode="tuned"))
        processor = helper.get_processor(model.tokenizer, model.config)
        # prepare_model_inputs / prepare_vqa_inputs reload the processor from disk on every
        # call (~2 s, as long as the inference itself); reuse the one loaded here.
        helper.get_processor = lambda tokenizer, model_config: processor
        return model, processor

    def prepare_model_input(self, images_array, history_xyz, history_rot, t0_us):
        images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3)
        if images.shape[:2] != (self.num_cameras, len(self.frame_ticks)):
            raise ValueError(
                f"Alpamayo 2 Edge expects {self.num_cameras} cameras x "
                f"{len(self.frame_ticks)} frames, got {tuple(images.shape[:2])}"
            )
        return {
            "images": images,
            "ego_history_xyz": torch.from_numpy(history_xyz).float().unsqueeze(0),
            "ego_history_rot": torch.from_numpy(history_rot).float().unsqueeze(0),
            "t0_us": int(t0_us),
        }

    def _camera_inputs(self, data, ticks_by_camera):
        """Edge camera inputs for one task profile, with capture times in microseconds."""
        tick_us = round(cfg.CONTROL_DT * 1_000_000)
        frames, cameras, timestamps = [], [], []
        for slot, camera in enumerate(self.source_camera_indices):
            for tick in ticks_by_camera[camera]:
                frames.append(data["images"][slot, self.frame_ticks.index(tick)])
                cameras.append(camera)
                timestamps.append(data["t0_us"] - tick * tick_us)
        return {
            "image_frames": torch.stack(frames).unsqueeze(1).contiguous(),
            "camera_indices": torch.tensor(cameras, dtype=torch.int64),
            "absolute_timestamps": torch.tensor(timestamps, dtype=torch.int64),
        }

    def driving_sample(self, data, navigation_text=None):
        """The raw 22-image Edge driving sample for ``prepare_model_inputs``."""
        nav_text = navigation_text.strip() if isinstance(navigation_text, str) else ""
        return {
            **self._camera_inputs(data, DRIVING_TICKS),
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
            "nav_text": [nav_text or self.default_navigation_text],
        }

    def run_inference(
        self, model, processor, data,
        navigation_text=None, navigation_weight=1.0, vlm_generate_timing=None,
    ):
        from alpamayo2_edge.helper import prepare_model_inputs, to_device

        from module.vlm_generate_optimization import optimized_vlm_generate

        if not math.isfinite(float(navigation_weight)) or float(navigation_weight) < 0:
            raise ValueError("navigation_weight must be a non-negative finite number")
        if navigation_text and not math.isclose(float(navigation_weight), 1.0):
            print(
                "Alpamayo 2 Edge has no navigation CFG sampler; using plain nav conditioning "
                f"instead of weight {float(navigation_weight):.2f}."
            )
        sample = self.driving_sample(data, navigation_text)
        inputs = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), "cuda")
        with (
            optimized_vlm_generate(model, disable_output_logits=True, timing=vlm_generate_timing),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            pred_xyz, _pred_rot, _logprob, extra = model.sample_trajectories_from_data(
                inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=cfg.NUM_TRAJ_SAMPLES,
                max_generation_length=256,
                diffusion_kwargs={"inference_step": 10},
                return_extra=True,
            )
        return pred_xyz, extra

    def run_vqa(self, model, processor, data, question):
        from alpamayo2_edge.helper import prepare_vqa_inputs, to_device
        from alpamayo2_edge.text_tasks import generate_vqa_answer

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        inputs = prepare_vqa_inputs(
            self._camera_inputs(data, VQA_TICKS), model.config, model.tokenizer, question
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            answer = generate_vqa_answer(model, to_device(inputs, "cuda"), max_new_tokens=256)
        return {"answer": answer}

    def extract_cot_text(self, extra):
        return extract_text_field(extra, "cot")

    def extract_answer_text(self, extra):
        return extract_text_field(extra, "answer")
