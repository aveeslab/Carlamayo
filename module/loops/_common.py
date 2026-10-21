"""Shared helpers for the CARLA loop runners."""

import os

import numpy as np

from module import config as cfg


def collect_oom_kwargs(args):
    """Gather the optional OOM-free tuning knobs set on the parsed args."""
    oom_kwargs = {}
    if getattr(args, "oom_free_headroom_gb", None) is not None:
        oom_kwargs["headroom_gb"] = args.oom_free_headroom_gb
    if getattr(args, "oom_free_margin", None) is not None:
        oom_kwargs["margin"] = args.oom_free_margin
    if getattr(args, "oom_free_resident", None) is not None:
        oom_kwargs["resident_override"] = args.oom_free_resident
    return oom_kwargs


def model_max_memory(args):
    """``max_memory`` for ``--max-gpu-memory-gb``: cap each GPU, offload the rest to RAM."""
    if getattr(args, "max_gpu_memory_gb", None) is None:
        return None
    import torch

    max_memory = {i: f"{args.max_gpu_memory_gb}GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    return max_memory


def derive_pygame_ui_video_path(output_video_path):
    """Return the companion video path for recorded Pygame UI frames."""
    root, ext = os.path.splitext(output_video_path)
    return f"{root}_pygame_ui{ext or '.mp4'}"


def stack_frame_buffer(frame_buffer, num_cameras):
    """Stack a rolling camera frame buffer into a ``(cam, frame, H, W, C)`` array."""
    images_array = np.zeros(
        (num_cameras, cfg.NUM_FRAMES, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS),
        dtype=np.uint8,
    )
    for t, frame_images in enumerate(frame_buffer):
        for c in range(num_cameras):
            images_array[c, t] = frame_images[c]
    return images_array
