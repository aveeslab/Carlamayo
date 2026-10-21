"""Adapter for Alpamayo 2 Super, package ``alpamayo2_super``.

Alpamayo 2 consumes the seven-camera PhysicalAI source ring and selects a fixed
six-camera profile per task via ``select_task_input``. It supports navigation text
conditioning (the released single-GPU path has no classifier-free-guidance sampler,
so non-1.0 weights fall back to plain conditioning) and VQA through the release
no-special-token text pipeline. OOM-free demand layering is 1.5-only and unavailable.
"""

from __future__ import annotations

import math

import torch

from ._rigs import FRONT_WIDE_SLOT, SEVEN_CAMERA_INDICES, SEVEN_CAMERA_RIG
from ._text import clean_generated_answer_text, extract_text_field
from .base import AlpamayoAdapter


class Alpamayo2Adapter(AlpamayoAdapter):
    version = "2"
    model_id = "nvidia/Alpamayo2-Super"
    display_name = "Alpamayo 2 Super"

    source_camera_configs = SEVEN_CAMERA_RIG
    source_camera_indices = SEVEN_CAMERA_INDICES
    viz_camera_slot = FRONT_WIDE_SLOT

    supports_navigation = True
    supports_vqa = True
    supports_oom_free = False

    def load_model(
        self, use_quantization=False, device_map="auto", oom_free=False, oom_kwargs=None,
        max_memory=None,
    ):
        if oom_free:
            raise ValueError("OOM-free demand layering is not supported for Alpamayo 2.")

        from alpamayo2_super import helper
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

        if use_quantization:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = Alpamayo2Super.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
            )
        elif device_map:
            model = Alpamayo2Super.from_pretrained(
                self.model_id, dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory
            )
        else:
            model = Alpamayo2Super.from_pretrained(self.model_id, dtype=torch.bfloat16).to("cuda")

        processor = helper.get_processor(model.tokenizer, model.config)
        return model, processor

    def prepare_model_input(self, images_array, history_xyz, history_rot, t0_us):
        from alpamayo2_super.input_profiles import CANONICAL_CAMERA_IDS, CANONICAL_CAMERA_NAMES

        from module.config import CONTROL_DT

        images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3).contiguous()
        num_cameras, num_frames = images.shape[0], images.shape[1]
        if num_cameras != len(CANONICAL_CAMERA_IDS):
            raise ValueError(
                f"Alpamayo 2 expects {len(CANONICAL_CAMERA_IDS)} source cameras, got {num_cameras}"
            )
        hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)

        frame_step_us = int(round(CONTROL_DT * 1_000_000))
        frame_offsets = torch.arange(-(num_frames - 1), 1, dtype=torch.int64) * frame_step_us
        absolute_timestamps = (int(t0_us) + frame_offsets).unsqueeze(0).repeat(num_cameras, 1)
        camera_tmin = int(absolute_timestamps.min().item())

        return {
            "image_frames": images,
            "camera_indices": torch.tensor(CANONICAL_CAMERA_IDS, dtype=torch.int64),
            "camera_names": list(CANONICAL_CAMERA_NAMES),
            "ego_history_xyz": hist_xyz,
            "ego_history_rot": hist_rot,
            "absolute_timestamps": absolute_timestamps,
            "relative_timestamps": (absolute_timestamps - camera_tmin).float() * 1e-6,
            "camera_tmin": camera_tmin,
            "ego_t0": torch.tensor([int(t0_us)], dtype=torch.int64),
            "ego_t0_frame_idx": torch.tensor([num_frames - 1], dtype=torch.int64),
        }

    def _tokenize_trajectory_prompt(self, processor, model_config, task_data, nav_text=None):
        from alpamayo2_super.chat_template.conversation import build_conversation

        data_for_prompt = dict(task_data)
        components_order = ["image", "traj_history", "prompt"]
        if nav_text:
            data_for_prompt["nav_text"] = [nav_text]
            components_order = ["image", "traj_history", "nav_instruction", "prompt"]

        messages = build_conversation(
            data=data_for_prompt,
            num_tokens_per_history_traj=model_config.tokens_per_history_traj,
            num_tokens_per_future_traj=model_config.tokens_per_future_traj,
            components_order=components_order,
            components_prompt=["cot", "traj_future"],
            generation_mode=True,
            include_camera_ids=model_config.include_camera_ids,
            camera_ids=data_for_prompt["camera_indices"],
            include_frame_nums=model_config.frame_label == "frame_num",
        )
        if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
            messages = messages[:-1]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
        )
        images = data_for_prompt["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tokenized_data = dict(
            processor(
                text=text, images=images, videos=None,
                padding=False, return_tensors="pt", do_rescale=False,
            )
        )
        if tokenized_data["input_ids"].shape[0] != 1:
            raise ValueError("trajectory prompt tokenization expects one sample at a time")
        return tokenized_data

    def run_inference(
        self, model, processor, data,
        navigation_text=None, navigation_weight=1.0, vlm_generate_timing=None,
    ):
        from alpamayo2_super import helper
        from alpamayo2_super.input_profiles import select_task_input

        from module.vlm_generate_optimization import optimized_vlm_generate
        from module.config import NUM_TRAJ_SAMPLES

        nav_text = navigation_text.strip() if isinstance(navigation_text, str) else ""
        if not math.isfinite(float(navigation_weight)) or float(navigation_weight) < 0:
            raise ValueError("navigation_weight must be a non-negative finite number")
        if nav_text and not math.isclose(float(navigation_weight), 1.0):
            print(
                "Navigation CFG is not available in the single-GPU Alpamayo 2 release path; "
                f"using plain nav conditioning instead of weight {float(navigation_weight):.2f}."
            )

        task_data = select_task_input(data, "trajectory")
        tokenized_data = self._tokenize_trajectory_prompt(
            processor, model.config, task_data, nav_text=nav_text or None
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized_data,
                "ego_history_xyz": task_data["ego_history_xyz"],
                "ego_history_rot": task_data["ego_history_rot"],
            },
            "cuda",
        )

        with (
            optimized_vlm_generate(model, disable_output_logits=True, timing=vlm_generate_timing),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            pred_xyz, _pred_rot, _logprob, extra = model.sample_trajectories_from_data(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=NUM_TRAJ_SAMPLES,
                diffusion_kwargs={"inference_step": 10},
                max_generation_length=256,
                return_extra=True,
            )
        return pred_xyz, extra

    def run_vqa(self, model, processor, data, question):
        from alpamayo2_super import helper
        from alpamayo2_super.input_profiles import select_task_input
        from alpamayo2_super.text_tasks import generate_text, prepare_vqa_inputs

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        vqa_data = select_task_input(data, "vqa")
        model_inputs = prepare_vqa_inputs(
            vqa_data, model.config, model.tokenizer, question=question
        )
        model_inputs = helper.to_device(model_inputs, "cuda")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return generate_text(
                model, model_inputs,
                top_p=0.98, temperature=0.6, num_samples=1, max_new_tokens=256,
            )

    def extract_cot_text(self, extra):
        return extract_text_field(extra, "cot")

    def extract_answer_text(self, extra):
        for key in ("answer", "cot", "raw_outputs"):
            answer = extract_text_field(extra, key)
            if answer:
                return clean_generated_answer_text(answer)
        return ""
