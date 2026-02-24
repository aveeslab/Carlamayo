"""Alpamayo inference utilities."""

import torch
import numpy as np

from transformers import BitsAndBytesConfig

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

from . import config as cfg


def load_model(use_quantization: bool):
    """Load Alpamayo model and processor."""
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AlpamayoR1.from_pretrained(
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


def run_inference(model, processor, data):
    """Run Alpamayo inference locally."""
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

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

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=cfg.NUM_TRAJ_SAMPLES,
            diffusion_kwargs={"inference_step": 10},
            max_generation_length=256,
            return_extra=True,
        )

    return pred_xyz, extra


def extract_cot_text(extra):
    if not isinstance(extra, dict):
        return ""
    if "cot" not in extra:
        return ""

    cot = extra["cot"]
    if cot is None:
        return ""

    while True:
        if isinstance(cot, str):
            return cot
        if isinstance(cot, (list, tuple)):
            if len(cot) == 0:
                return ""
            cot = cot[0]
            continue
        if isinstance(cot, np.ndarray):
            if cot.size == 0:
                return ""
            cot = cot.flat[0]
            continue
        if hasattr(cot, "numel") and hasattr(cot, "reshape"):
            if int(cot.numel()) == 0:
                return ""
            cot = cot.reshape(-1)[0].item()
            continue
        return str(cot)


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
