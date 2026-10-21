# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop inference over a recorded CARLA dataset for any Alpamayo version.

"""Open-loop runner: replay recorded CARLA frames through a selected Alpamayo model.

The vehicle never moves under model control here; each recorded frame is fed to the
model and its predicted trajectory and Chain-of-Thought are rendered to a summary
video. The recorded seven-camera superset is subset to the selected version's rig.
"""

import time

import numpy as np
import torch

from module import config as cfg
from module.inference import (
    configure_cuda_linalg_library,
    extract_trajectory_samples,
)
from module.loops._common import collect_oom_kwargs, model_max_memory
from module.open_loop_dataset import (
    load_front_camera_image,
    load_open_loop_arrays,
    load_trajectory_index,
)


def _load_model(adapter, args):
    if args.oom_free and not adapter.supports_oom_free:
        raise SystemExit(
            f"{adapter.display_name} does not support OOM-free demand layering "
            "(available for --version 1.5 only)."
        )
    configure_cuda_linalg_library(args.cuda_linalg_library)
    return adapter.load_model(
        use_quantization=args.quantization,
        device_map=args.device_map,
        oom_free=args.oom_free,
        oom_kwargs=collect_oom_kwargs(args),
        max_memory=model_max_memory(args),
    )


def run(adapter, args):
    """Run open-loop inference for ``adapter`` over the recorded dataset."""

    output_video = args.output_video or cfg.OPEN_LOOP_OUTPUT_VIDEO
    camera_order = list(adapter.source_camera_configs)
    front_camera_name = camera_order[adapter.viz_camera_slot]

    print("=" * 60)
    print(f"CARLA -> {adapter.display_name} Open-Loop Inference")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Quantization: {'ON (4-bit)' if args.quantization else 'OFF (full-precision)'}")
    if args.oom_free:
        print("Model loading: OOM-free CPU<->GPU demand layering")
    print(f"Cameras: {adapter.num_cameras} ({', '.join(camera_order)})")

    trajectory, frame_ids = load_trajectory_index(args.data_root)
    start_index = cfg.NUM_FRAMES - 1
    frames_to_process = max(0, len(frame_ids) - start_index)
    print(f"\nTotal recorded frames: {len(frame_ids)}")
    print(f"Frames selected for inference: {frames_to_process}")
    if frames_to_process == 0:
        print("Not enough frames to build the requested camera history.")
        return

    print("\nLoading model...")
    model, processor = _load_model(adapter, args)
    print("Model loaded!")

    predictions, cot_texts, inference_times, camera_images = [], [], [], []
    for frame_index in range(start_index, len(frame_ids)):
        display_index = frame_index - start_index + 1
        frame_id = frame_ids[frame_index]
        print(f"  Frame {display_index}/{frames_to_process} (CARLA frame={frame_id})")

        arrays = load_open_loop_arrays(
            args.data_root, trajectory, frame_ids, frame_index,
            num_history_steps=cfg.NUM_HISTORY, num_frames=cfg.NUM_FRAMES,
            camera_order=camera_order,
        )
        model_input = adapter.prepare_model_input(
            arrays["image_frames"], arrays["history_xyz"], arrays["history_rot"], arrays["t0_us"]
        )
        torch.cuda.manual_seed_all(42)
        inference_start = time.perf_counter()
        pred_xyz, extra = adapter.run_inference(model, processor, model_input)
        inference_time = time.perf_counter() - inference_start

        predictions.append(extract_trajectory_samples(pred_xyz))
        cot_text = adapter.extract_cot_text(extra)
        cot_texts.append(cot_text)
        inference_times.append(inference_time)
        camera_images.append(
            load_front_camera_image(args.data_root, frame_id, camera_name=front_camera_name)
        )
        print(f"    Inference: {inference_time:.2f}s | CoC: {cot_text[:80]}...")

    avg_time = float(np.mean(inference_times)) if inference_times else 0.0
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total frames processed: {len(predictions)}")
    print(f"Average inference time: {avg_time:.2f}s/frame")
    if torch.cuda.is_available():
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if predictions:
        from module.visualization import save_open_loop_video

        print("\nCreating video...")
        save_open_loop_video(
            predictions, camera_images, cot_texts, inference_times, output_video, fps=5
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
