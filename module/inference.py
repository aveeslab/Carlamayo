"""Alpamayo 2 Super inference utilities."""

import math
import re

import torch
import numpy as np

from transformers import BitsAndBytesConfig

from alpamayo2_super import helper
from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.input_profiles import (
    CANONICAL_CAMERA_IDS,
    CANONICAL_CAMERA_NAMES,
    select_task_input,
)
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
from alpamayo2_super.text_tasks import generate_text, prepare_vqa_inputs

from . import config as cfg
from .vlm_generate_optimization import VlmGenerateTiming, optimized_vlm_generate

SUPPORTED_CUDA_LINALG_LIBRARIES = {"default", "cusolver", "magma"}
ALPAMAYO_MODEL_NAME = "nvidia/Alpamayo2-Super"
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>|</s>|<s>")


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
    """Load the Alpamayo 2 Super model and its checkpoint processor."""
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = Alpamayo2Super.from_pretrained(
            ALPAMAYO_MODEL_NAME,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    else:
        if device_map:
            model = Alpamayo2Super.from_pretrained(
                ALPAMAYO_MODEL_NAME,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
        else:
            model = Alpamayo2Super.from_pretrained(
                ALPAMAYO_MODEL_NAME,
                dtype=torch.bfloat16,
            ).to("cuda")

    processor = helper.get_processor(model.tokenizer, model.config)
    return model, processor


def prepare_model_input(images_array, history_xyz, history_rot, t0_us):
    """Convert CARLA data to the Alpamayo 2 Super seven-camera source format.

    ``images_array`` holds the canonical PhysicalAI camera ring in index order
    (``camera, frame, height, width, channel``). ``t0_us`` is the simulation
    timestamp of the most recent frame; earlier frames are spaced one control
    step apart, matching CARLA's fixed synchronous tick.
    """
    images = torch.from_numpy(images_array).permute(0, 1, 4, 2, 3).contiguous()
    num_cameras, num_frames = images.shape[0], images.shape[1]
    if num_cameras != len(CANONICAL_CAMERA_IDS):
        raise ValueError(
            f"Expected {len(CANONICAL_CAMERA_IDS)} source cameras, got {num_cameras}"
        )
    hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)

    frame_step_us = int(round(cfg.CONTROL_DT * 1_000_000))
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


def _tokenize_trajectory_prompt(processor, model_config, task_data, nav_text=None):
    """Tokenize one trajectory-task prompt, optionally nav-conditioned."""
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    images = data_for_prompt["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt",
            do_rescale=False,
        )
    )
    if tokenized_data["input_ids"].shape[0] != 1:
        raise ValueError("trajectory prompt tokenization expects one sample at a time")
    return tokenized_data


def run_inference(
    model,
    processor,
    data,
    navigation_text: str | None = None,
    navigation_weight: float = 1.0,
    vlm_generate_timing: VlmGenerateTiming | None = None,
):
    """Run Alpamayo 2 Super trajectory inference locally.

    ``navigation_text`` conditions the trajectory prompt through the model's
    ``nav_instruction`` component. The released single-GPU Alpamayo 2 Super
    inference path has no classifier-free-guidance sampler, so any
    ``navigation_weight`` other than 1.0 falls back to plain nav conditioning.
    """

    nav_text = navigation_text.strip() if isinstance(navigation_text, str) else ""
    if not math.isfinite(float(navigation_weight)) or float(navigation_weight) < 0:
        raise ValueError("navigation_weight must be a non-negative finite number")
    if nav_text and not math.isclose(float(navigation_weight), 1.0):
        print(
            "Navigation CFG is not available in the single-GPU Alpamayo 2 Super "
            f"release path; using plain nav conditioning instead of weight "
            f"{float(navigation_weight):.2f}."
        )

    task_data = select_task_input(data, "trajectory")
    tokenized_data = _tokenize_trajectory_prompt(
        processor,
        model.config,
        task_data,
        nav_text=nav_text or None,
    )

    model_inputs = {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": task_data["ego_history_xyz"],
        "ego_history_rot": task_data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    with (
        optimized_vlm_generate(
            model,
            disable_output_logits=True,
            timing=vlm_generate_timing,
        ),
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        pred_xyz, _pred_rot, _logprob, extra = model.sample_trajectories_from_data(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=cfg.NUM_TRAJ_SAMPLES,
            diffusion_kwargs={"inference_step": 10},
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
            return str(value).strip()
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
        return str(value).strip()


def _clean_generated_answer_text(text):
    text = SPECIAL_TOKEN_RE.sub("", str(text))
    return " ".join(text.split()).strip()


def extract_cot_text(extra):
    return _extract_text_field(extra, "cot")


def extract_answer_text(extra):
    for key in ("answer", "cot", "raw_outputs"):
        answer = _extract_text_field(extra, key)
        if answer:
            return _clean_generated_answer_text(answer)
    return ""


def run_vqa(
    model,
    processor,
    data,
    question: str,
):
    """Run Alpamayo 2 Super no-special VQA text generation."""

    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    vqa_data = select_task_input(data, "vqa")
    model_inputs = prepare_vqa_inputs(
        vqa_data,
        model.config,
        model.tokenizer,
        question=question,
    )
    model_inputs = helper.to_device(model_inputs, "cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return generate_text(
            model,
            model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_samples=1,
            max_new_tokens=256,
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
