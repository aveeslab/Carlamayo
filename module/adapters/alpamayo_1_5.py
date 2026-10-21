"""Adapter for Alpamayo 1.5, package ``alpamayo1_5``.

1.5 keeps R1's four-camera trajectory + CoT core and adds navigation conditioning
(with a classifier-free-guidance path), VQA text generation, and OOM-free CPU<->GPU
demand layering. This adapter ports the previously validated ``module.inference``
logic from the 1.5-only Carlamayo.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ._rigs import FOUR_CAMERA_INDICES, FOUR_CAMERA_RIG, FRONT_WIDE_SLOT
from ._text import extract_answer_from_decoded_text, extract_text_field
from .base import AlpamayoAdapter

#: Fixed Qwen-VL image-token budget used by the 1.5 CARLA integration.
VLM_IMAGE_PIXELS = 196608


class Alpamayo15Adapter(AlpamayoAdapter):
    version = "1.5"
    model_id = "nvidia/Alpamayo-1.5-10B"
    display_name = "Alpamayo 1.5"

    source_camera_configs = FOUR_CAMERA_RIG
    source_camera_indices = FOUR_CAMERA_INDICES
    viz_camera_slot = FRONT_WIDE_SLOT

    supports_navigation = True
    supports_vqa = True
    supports_oom_free = True

    def load_model(
        self, use_quantization=False, device_map="auto", oom_free=False, oom_kwargs=None,
        max_memory=None,
    ):
        from module.alpamayo_compat import patch_legacy_hydra_targets

        patch_legacy_hydra_targets()

        if oom_free:
            from module.oom_offload import load_offloaded_model

            return load_offloaded_model(**(oom_kwargs or {}))

        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        if use_quantization:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = Alpamayo1_5.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
            )
        elif device_map:
            model = Alpamayo1_5.from_pretrained(
                self.model_id, dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory
            )
        else:
            model = Alpamayo1_5.from_pretrained(self.model_id, dtype=torch.bfloat16).to("cuda")

        processor = helper.get_processor(model.tokenizer)
        return model, processor

    def prepare_model_input(self, images_array, history_xyz, history_rot, t0_us):
        images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3).contiguous()
        if images.shape[0] != self.num_cameras:
            raise ValueError(
                f"Alpamayo 1.5 expects {self.num_cameras} source cameras, got {images.shape[0]}"
            )
        hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)
        return {
            "image_frames": images,
            "ego_history_xyz": hist_xyz,
            "ego_history_rot": hist_rot,
        }

    @staticmethod
    def _prime_oom_pipeline(model):
        pipeline = getattr(model, "_oom_pipeline", None)
        if pipeline is not None:
            pipeline.start_iteration()

    def run_inference(
        self, model, processor, data,
        navigation_text=None, navigation_weight=1.0, vlm_generate_timing=None,
    ):
        from alpamayo1_5 import helper

        from module.vlm_generate_optimization import optimized_vlm_generate
        from module.config import NUM_TRAJ_SAMPLES

        nav_text = navigation_text.strip() if isinstance(navigation_text, str) else ""
        if not math.isfinite(float(navigation_weight)) or float(navigation_weight) < 0:
            raise ValueError("navigation_weight must be a non-negative finite number")

        messages = helper.create_message(
            data["image_frames"].flatten(0, 1),
            camera_indices=data.get("camera_indices"),
            nav_text=nav_text or None,
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            min_pixels=VLM_IMAGE_PIXELS,
            max_pixels=VLM_IMAGE_PIXELS,
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            "cuda",
        )

        diffusion_kwargs = {"inference_step": 10}
        inference_fn = model.sample_trajectories_from_data_with_vlm_rollout
        use_cfg_nav = False
        if nav_text and not math.isclose(float(navigation_weight), 1.0):
            use_cfg_nav = True
            inference_fn = model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav
            diffusion_kwargs = {
                **diffusion_kwargs,
                "use_classifier_free_guidance": True,
                "inference_guidance_weight": float(navigation_weight),
            }

        self._prime_oom_pipeline(model)
        with (
            optimized_vlm_generate(
                model, disable_output_logits=not use_cfg_nav, timing=vlm_generate_timing
            ),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            pred_xyz, _pred_rot, extra = inference_fn(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=NUM_TRAJ_SAMPLES,
                diffusion_kwargs=diffusion_kwargs,
                max_generation_length=256,
                return_extra=True,
            )
        return pred_xyz, extra

    def run_vqa(self, model, processor, data, question):
        from alpamayo1_5 import helper

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        messages = helper.create_vqa_message(
            data["image_frames"].flatten(0, 1),
            question=question,
            camera_indices=data.get("camera_indices"),
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device({"tokenized_data": inputs}, "cuda")

        self._prime_oom_pipeline(model)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if hasattr(model, "vlm") and hasattr(model, "tokenizer"):
                return self._generate_vqa_with_partial_answer_fallback(model, model_inputs)
            return model.generate_text(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_samples=1,
                max_generation_length=256,
            )

    @staticmethod
    def _generate_vqa_with_partial_answer_fallback(
        model, model_inputs, top_p=0.98, top_k=None, temperature=0.6,
        num_samples=1, max_generation_length=256,
    ):
        """Generate VQA text while preserving partial answers without ``answer_end``."""
        tokenized_data = dict(model_inputs["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")

        generation_config = copy.deepcopy(model.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = model.tokenizer.pad_token_id

        generated = model.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        sequences = generated["sequences"] if isinstance(generated, dict) else generated.sequences
        generated_tokens = sequences[:, input_ids.shape[1]:]
        decoded_batch = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        batch_size = int(input_ids.shape[0])
        raw = np.array(decoded_batch, dtype=object).reshape([batch_size, num_samples])
        answers = np.array(
            [extract_answer_from_decoded_text(text) for text in decoded_batch], dtype=object
        ).reshape([batch_size, num_samples])
        return {"answer": answers, "raw_answer": raw}

    def extract_cot_text(self, extra):
        return extract_text_field(extra, "cot")

    def extract_answer_text(self, extra):
        answer = extract_text_field(extra, "answer")
        if answer:
            return extract_answer_from_decoded_text(answer)
        for key in ("raw_answer", "raw_text", "decoded_answer", "decoded_text"):
            raw_answer = extract_text_field(extra, key)
            if raw_answer:
                return extract_answer_from_decoded_text(raw_answer)
        return ""
