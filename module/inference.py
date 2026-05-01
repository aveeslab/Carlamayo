"""Alpamayo inference utilities."""

import math

import torch
import numpy as np

from transformers import BitsAndBytesConfig

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper

from . import config as cfg
from .alpamayo_compat import patch_legacy_hydra_targets

patch_legacy_hydra_targets()

SUPPORTED_CUDA_LINALG_LIBRARIES = {"default", "cusolver", "magma"}


def configure_cuda_linalg_library(library: str | None):
    """Set PyTorch's preferred CUDA linalg backend when supported."""

    if library is None:
        return None

    normalized = library.strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized not in SUPPORTED_CUDA_LINALG_LIBRARIES:
        supported = ", ".join(sorted(SUPPORTED_CUDA_LINALG_LIBRARIES))
        raise ValueError(
            f"Unsupported CUDA linalg library '{library}'. Expected one of: {supported}."
        )
    if not torch.cuda.is_available():
        return None

    preferred_linalg_library = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    if preferred_linalg_library is None:
        return None
    return preferred_linalg_library(normalized)


def load_model(use_quantization: bool, device_map="auto"):
    """Load Alpamayo model and processor."""
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    else:
        if device_map:
            model = Alpamayo1_5.from_pretrained(
                "nvidia/Alpamayo-R1-10B",
                dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            model = Alpamayo1_5.from_pretrained(
                "nvidia/Alpamayo-R1-10B",
                dtype=torch.bfloat16,
            ).to("cuda")

    processor = helper.get_processor(model.tokenizer)
    return model, processor


def prepare_model_input(images_array, history_xyz, history_rot):
    """Convert CARLA data to model input format."""
    images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3).contiguous()
    hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)
    return {
        "image_frames": images,
        "ego_history_xyz": hist_xyz,
        "ego_history_rot": hist_rot,
    }


def run_inference(
    model,
    processor,
    data,
    navigation_text: str | None = None,
    navigation_weight: float = 1.0,
):
    """Run Alpamayo inference locally.

    ``navigation_text`` conditions the trajectory prompt. ``navigation_weight``
    uses Alpamayo's CFG navigation path when it differs from 1.0.
    """

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
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    diffusion_kwargs = {"inference_step": 10}
    inference_fn = model.sample_trajectories_from_data_with_vlm_rollout
    if nav_text and not math.isclose(float(navigation_weight), 1.0):
        inference_fn = model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav
        diffusion_kwargs = {
            **diffusion_kwargs,
            "use_classifier_free_guidance": True,
            "inference_guidance_weight": float(navigation_weight),
        }

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = inference_fn(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=cfg.NUM_TRAJ_SAMPLES,
            diffusion_kwargs=diffusion_kwargs,
            max_generation_length=256,
            return_extra=True,
        )

    return pred_xyz, extra


def _extract_text_field(extra, key):
    if not isinstance(extra, dict):
        return ""
    if key not in extra:
        return ""

    value = extra[key]
    if value is None:
        return ""

    while True:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return ""
            value = value[0]
            continue
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ""
            value = value.flat[0]
            continue
        if hasattr(value, "numel") and hasattr(value, "reshape"):
            if int(value.numel()) == 0:
                return ""
            value = value.reshape(-1)[0].item()
            continue
        return str(value)


def extract_cot_text(extra):
    return _extract_text_field(extra, "cot")


def extract_answer_text(extra):
    return _extract_text_field(extra, "answer")


def run_vqa(
    model,
    processor,
    data,
    question: str,
):
    """Run Alpamayo VQA text generation for a driving-relevant question."""

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

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.generate_text(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_samples=1,
            max_generation_length=256,
        )


def extract_trajectory_samples(pred_xyz):
    arr = pred_xyz.detach().cpu().numpy()
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected pred_xyz shape after squeeze: {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError(f"Trajectory last dim must be >= 3, got {arr.shape}")
    return arr[:, :, :3]


def select_trajectory_by_prev_similarity(traj_samples, prev_traj):
    num_samples = traj_samples.shape[0]
    if prev_traj is None:
        return 0, [None] * num_samples

    prev_xy = prev_traj[:, :2]
    scores = []
    for i in range(num_samples):
        curr_xy = traj_samples[i, :, :2]
        n = min(len(curr_xy), len(prev_xy))
        if n <= 0:
            score = float("inf")
        else:
            score = float(np.mean(np.linalg.norm(curr_xy[:n] - prev_xy[:n], axis=1)))
        scores.append(score)

    best_idx = int(np.argmin(scores))
    return best_idx, scores
