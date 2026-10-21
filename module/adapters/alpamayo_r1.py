"""Adapter for Alpamayo 1 (Alpamayo-R1), package ``alpamayo_r1``.

R1 is the trajectory + Chain-of-Thought release: four cameras, no navigation
conditioning, and no VQA. Its trajectory sampler and helpers match the 1.5 API,
but ``create_message`` takes only image frames.
"""

from __future__ import annotations

import torch

from ._rigs import FOUR_CAMERA_INDICES, FOUR_CAMERA_RIG, FRONT_WIDE_SLOT
from ._text import extract_text_field
from .base import AlpamayoAdapter


class AlpamayoR1Adapter(AlpamayoAdapter):
    version = "1"
    model_id = "nvidia/Alpamayo-R1-10B"
    display_name = "Alpamayo 1 (R1)"

    source_camera_configs = FOUR_CAMERA_RIG
    source_camera_indices = FOUR_CAMERA_INDICES
    viz_camera_slot = FRONT_WIDE_SLOT

    supports_navigation = False
    supports_vqa = False
    supports_oom_free = False

    def load_model(
        self, use_quantization=False, device_map="auto", oom_free=False, oom_kwargs=None,
        max_memory=None,
    ):
        if oom_free:
            raise ValueError("OOM-free demand layering is not supported for Alpamayo 1 (R1).")

        from alpamayo_r1 import helper
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        if use_quantization:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AlpamayoR1.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
            )
        elif device_map:
            model = AlpamayoR1.from_pretrained(
                self.model_id, dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory
            )
        else:
            model = AlpamayoR1.from_pretrained(self.model_id, dtype=torch.bfloat16).to("cuda")

        processor = helper.get_processor(model.tokenizer)
        return model, processor

    def prepare_model_input(self, images_array, history_xyz, history_rot, t0_us):
        images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3).contiguous()
        if images.shape[0] != self.num_cameras:
            raise ValueError(
                f"Alpamayo 1 expects {self.num_cameras} source cameras, got {images.shape[0]}"
            )
        hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)
        return {
            "image_frames": images,
            "camera_indices": torch.tensor(self.source_camera_indices, dtype=torch.int64),
            "ego_history_xyz": hist_xyz,
            "ego_history_rot": hist_rot,
        }

    def run_inference(
        self, model, processor, data,
        navigation_text=None, navigation_weight=1.0, vlm_generate_timing=None,
    ):
        if navigation_text:
            print("Alpamayo 1 (R1) has no navigation conditioning; ignoring navigation text.")

        from alpamayo_r1 import helper

        from module.vlm_generate_optimization import optimized_vlm_generate
        from module.config import NUM_TRAJ_SAMPLES

        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            "cuda",
        )

        with (
            optimized_vlm_generate(model, disable_output_logits=True, timing=vlm_generate_timing),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=NUM_TRAJ_SAMPLES,
                diffusion_kwargs={"inference_step": 10},
                max_generation_length=256,
                return_extra=True,
            )
        return pred_xyz, extra

    def extract_cot_text(self, extra):
        return extract_text_field(extra, "cot")
