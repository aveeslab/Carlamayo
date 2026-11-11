"""Shared helpers for the CARLA loop runners."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# NumPy copies release the GIL; per-camera threads stack the frame buffer several times
# faster than one np.stack (Edge copies 6 cameras x 7 ticks of 1080p frames per tick).
_STACK_POOL = ThreadPoolExecutor(max_workers=8)


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


def stack_frame_buffer(frame_buffer, frame_ticks):
    """Stack the buffered ticks ``frame_ticks`` before the newest into ``(cam, frame, H, W, C)``."""
    frames = [frame_buffer[-1 - tick] for tick in frame_ticks]
    stacked = np.empty((len(frames[0]), len(frames), *frames[0].shape[1:]), frames[0].dtype)

    def fill(camera):
        for t, frame in enumerate(frames):
            stacked[camera, t] = frame[camera]

    list(_STACK_POOL.map(fill, range(len(frames[0]))))
    return stacked
