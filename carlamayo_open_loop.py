# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop inference script for CARLA-collected data with Alpamayo 1.5.

"""Run open-loop Alpamayo inference on a recorded CARLA dataset."""

import argparse
import time

import numpy as np
import torch

from module import config as cfg
from module.open_loop_dataset import (
    load_front_camera_image,
    load_open_loop_arrays,
    load_trajectory_index,
)


DEFAULT_OUTPUT_VIDEO = "carla_alpamayo_open_loop_result.mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo open-loop inference on CARLA-collected data."
    )
    parser.add_argument(
        "--data-root",
        default="carla_data",
        help="Dataset root containing trajectory.json and camera folders.",
    )
    parser.add_argument(
        "--output-video",
        default=DEFAULT_OUTPUT_VIDEO,
        help=f"Output visualization video path. Default: {DEFAULT_OUTPUT_VIDEO}.",
    )
    parser.add_argument(
        "--quantization",
        dest="use_quantization",
        action="store_true",
        help="Use 4-bit quantized model instead of the default full-precision model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Model device_map passed to from_pretrained. Default: "auto".',
    )
    parser.add_argument(
        "--cuda-linalg-library",
        choices=("default", "cusolver", "magma"),
        default="magma",
        help=(
            "Preferred CUDA linalg backend for torch.linalg calls. "
            'Default: "magma" to match the closed-loop runner.'
        ),
    )
    return parser.parse_args()


def load_model_input(data_root, trajectory, frame_ids, frame_index):
    """Load one recorded CARLA frame and convert it to Alpamayo input tensors."""

    from module.inference import prepare_model_input

    arrays = load_open_loop_arrays(
        data_root,
        trajectory,
        frame_ids,
        frame_index,
        num_history_steps=cfg.NUM_HISTORY,
        num_frames=cfg.NUM_FRAMES,
    )
    return prepare_model_input(
        arrays["image_frames"],
        arrays["history_xyz"],
        arrays["history_rot"],
    )


def run_single_frame(model, processor, model_input):
    """Run Alpamayo once and return trajectory samples, CoT text, and latency."""

    from module.inference import extract_cot_text, extract_trajectory_samples, run_inference

    torch.cuda.manual_seed_all(42)
    inference_start = time.perf_counter()
    pred_xyz, extra = run_inference(
        model,
        processor,
        model_input,
        disable_unused_generate_logits=True,
        vlm_image_pixels=cfg.VLM_IMAGE_PIXELS,
    )
    inference_time = time.perf_counter() - inference_start
    return extract_trajectory_samples(pred_xyz), extract_cot_text(extra), inference_time


def load_open_loop_model(use_quantization, device_map, cuda_linalg_library):
    """Load the shared Alpamayo model stack used by the closed-loop runner."""

    from module.inference import configure_cuda_linalg_library, load_model

    configure_cuda_linalg_library(cuda_linalg_library)
    return load_model(use_quantization, device_map=device_map)


def main():
    args = parse_args()

    print("=" * 60)
    print("CARLA -> Alpamayo 1.5 Open-Loop Inference")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Quantization: {'ON (4-bit)' if args.use_quantization else 'OFF (full-precision)'}")
    print(f"Device map: {args.device_map}")
    print(f"CUDA linalg library: {args.cuda_linalg_library}")

    trajectory, frame_ids = load_trajectory_index(args.data_root)
    start_index = cfg.NUM_FRAMES - 1
    frames_to_process = max(0, len(frame_ids) - start_index)
    print(f"\nTotal recorded frames: {len(frame_ids)}")
    print(f"Frames selected for inference: {frames_to_process}")

    if frames_to_process == 0:
        print("Not enough frames to build the requested camera history.")
        return

    print("\nLoading model...")
    model, processor = load_open_loop_model(
        args.use_quantization,
        args.device_map,
        args.cuda_linalg_library,
    )
    print("Model loaded!")

    predictions = []
    cot_texts = []
    inference_times = []
    camera_images = []

    for frame_index in range(start_index, len(frame_ids)):
        display_index = frame_index - start_index + 1
        frame_id = frame_ids[frame_index]
        print(f"  Frame {display_index}/{frames_to_process} (CARLA frame={frame_id})")

        model_input = load_model_input(args.data_root, trajectory, frame_ids, frame_index)
        pred_xyz, cot_text, inference_time = run_single_frame(model, processor, model_input)

        predictions.append(pred_xyz)
        cot_texts.append(cot_text)
        inference_times.append(inference_time)
        camera_images.append(load_front_camera_image(args.data_root, frame_id))

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
            predictions,
            camera_images,
            cot_texts,
            inference_times,
            args.output_video,
            fps=5,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
