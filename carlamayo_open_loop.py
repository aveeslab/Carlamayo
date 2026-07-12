# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Open-loop inference script for CARLA-collected data with Alpamayo 1.5.

"""Run open-loop Alpamayo inference on a recorded CARLA dataset."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from module import config as cfg
from module.open_loop_dataset import (
    OPEN_LOOP_CAMERA_INDICES,
    load_front_camera_image,
    load_open_loop_arrays,
    load_trajectory_index,
)


DEFAULT_OUTPUT_VIDEO = "carla_alpamayo_open_loop_result.mp4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Alpamayo open-loop inference on CARLA-collected data."
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "nav", "navigation", "vqa"),
        default="normal",
        help="Open-loop task: trajectory, navigation-conditioned trajectory, or VQA.",
    )
    parser.add_argument(
        "--navigation-text",
        default="",
        help="Navigation instruction applied to every frame in navigation mode.",
    )
    parser.add_argument(
        "--navigation-weight",
        type=float,
        default=1.0,
        help="Navigation guidance weight. Default: 1.0.",
    )
    parser.add_argument(
        "--vqa-question",
        default="",
        help="Question applied to every frame in VQA mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Generation temperature. Default: 0.6 for trajectory modes and "
            "0 for deterministic VQA."
        ),
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
        "--output-json",
        default=None,
        help="VQA result JSON path. Default: output video path with a .json suffix.",
    )
    parser.add_argument(
        "--quantization",
        dest="use_quantization",
        action="store_true",
        help="Use 4-bit quantized model instead of the default full-precision model.",
    )
    parser.add_argument(
        "--oom-free",
        dest="oom_free",
        action="store_true",
        help=(
            "Use OOM-free CPU<->GPU demand layering (third_party/oom-free-alpamayo) "
            "instead of loading the whole model on the GPU. Streams full-precision "
            "VLM/ViT/Expert layers on demand to keep peak VRAM low. Mutually "
            "exclusive with --quantization."
        ),
    )
    parser.add_argument(
        "--oom-free-headroom-gb",
        type=float,
        default=None,
        help="OOM-free: VRAM (GB) reserved for activation spikes. Default ~3.5.",
    )
    parser.add_argument(
        "--oom-free-margin",
        type=int,
        default=None,
        help="OOM-free: safety margin subtracted from the max resident VLM layer count.",
    )
    parser.add_argument(
        "--oom-free-resident",
        type=int,
        default=None,
        help="OOM-free: force this many GPU-resident VLM layers (default: auto from free VRAM).",
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
    args = parser.parse_args(argv)
    args.mode = "navigation" if args.mode == "nav" else args.mode
    args.navigation_text = args.navigation_text.strip()
    args.vqa_question = args.vqa_question.strip()
    if args.temperature is None:
        args.temperature = 0.0 if args.mode == "vqa" else 0.6

    if args.mode == "navigation" and not args.navigation_text:
        parser.error("--navigation-text is required in navigation mode.")
    if args.mode == "vqa" and not args.vqa_question:
        parser.error("--vqa-question is required in VQA mode.")
    if not math.isfinite(args.navigation_weight) or args.navigation_weight < 0:
        parser.error("--navigation-weight must be a non-negative finite number.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be a non-negative finite number.")
    if args.mode != "vqa" and args.temperature == 0:
        parser.error(
            "--temperature must be greater than zero in trajectory modes; "
            "Alpamayo trajectory generation requires sampling."
        )
    if args.mode == "vqa":
        args.output_json = args.output_json or str(Path(args.output_video).with_suffix(".json"))
    elif args.output_json is not None:
        parser.error("--output-json is only valid in VQA mode.")
    _validate_output_paths(args, parser)
    if args.oom_free and args.use_quantization:
        parser.error("--oom-free and --quantization are mutually exclusive.")
    return args


def _validate_output_paths(args, parser):
    data_root = Path(args.data_root).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None
    protected_targets = set()
    protected_directories = []
    if data_root.is_dir():
        for source_path in data_root.iterdir():
            resolved_source = source_path.resolve()
            protected_targets.add(resolved_source)
            if source_path.is_dir():
                protected_directories.append(resolved_source)

    if output_json is not None and output_video == output_json:
        parser.error("--output-video and --output-json must be different paths.")

    for output_path in (output_video, output_json):
        if output_path is None:
            continue
        if output_path == data_root / "trajectory.json":
            parser.error("Output paths must not overwrite the source trajectory.json.")
        if output_path.is_relative_to(data_root):
            parser.error("Output paths must be outside the source dataset directory.")
        if output_path in protected_targets or any(
            output_path.is_relative_to(source_dir)
            for source_dir in protected_directories
        ):
            parser.error("Output paths must not overwrite resolved dataset sources.")


def get_start_index(mode):
    """Return the first usable frame while preserving trajectory history semantics."""

    return 0 if mode == "vqa" else cfg.NUM_FRAMES - 1


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
    model_input = prepare_model_input(
        arrays["image_frames"],
        arrays["history_xyz"],
        arrays["history_rot"],
    )
    model_input["camera_indices"] = torch.tensor(OPEN_LOOP_CAMERA_INDICES, dtype=torch.long)
    return model_input


def run_trajectory_frame(
    model,
    processor,
    model_input,
    navigation_text=None,
    navigation_weight=1.0,
    temperature=0.6,
):
    """Run Alpamayo once and return trajectory samples, CoT text, and latency."""

    from module.inference import extract_cot_text, extract_trajectory_samples, run_inference

    torch.cuda.manual_seed_all(42)
    inference_start = time.perf_counter()
    pred_xyz, extra = run_inference(
        model,
        processor,
        model_input,
        navigation_text=navigation_text,
        navigation_weight=navigation_weight,
        temperature=temperature,
        disable_unused_generate_logits=True,
        vlm_image_pixels=cfg.VLM_IMAGE_PIXELS,
    )
    inference_time = time.perf_counter() - inference_start
    return extract_trajectory_samples(pred_xyz), extract_cot_text(extra), inference_time


def run_vqa_frame(model, processor, model_input, question, temperature=0.0):
    """Run Alpamayo VQA once and return answer text and latency."""

    from module.inference import extract_answer_text, run_vqa

    torch.cuda.manual_seed_all(42)
    inference_start = time.perf_counter()
    extra = run_vqa(
        model,
        processor,
        model_input,
        question=question,
        temperature=temperature,
    )
    inference_time = time.perf_counter() - inference_start
    return extract_answer_text(extra), inference_time


def load_open_loop_model(args):
    """Load the shared Alpamayo model stack used by the closed-loop runner."""

    from module.inference import configure_cuda_linalg_library, load_model

    configure_cuda_linalg_library(args.cuda_linalg_library)

    if args.oom_free:
        from module.oom_offload import load_offloaded_model

        kwargs = {}
        if args.oom_free_headroom_gb is not None:
            kwargs["headroom_gb"] = args.oom_free_headroom_gb
        if args.oom_free_margin is not None:
            kwargs["margin"] = args.oom_free_margin
        if args.oom_free_resident is not None:
            kwargs["resident_override"] = args.oom_free_resident
        return load_offloaded_model(**kwargs)

    return load_model(args.use_quantization, device_map=args.device_map)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 60)
    print("CARLA -> Alpamayo 1.5 Open-Loop Inference")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Mode: {args.mode}")
    print(f"Temperature: {args.temperature}")
    if args.oom_free:
        print("Model loading: OOM-free CPU<->GPU demand layering (full-precision)")
    else:
        print(f"Quantization: {'ON (4-bit)' if args.use_quantization else 'OFF (full-precision)'}")
    print(f"Device map: {args.device_map}")
    print(f"CUDA linalg library: {args.cuda_linalg_library}")

    trajectory, frame_ids = load_trajectory_index(args.data_root)
    start_index = get_start_index(args.mode)
    frames_to_process = max(0, len(frame_ids) - start_index)
    print(f"\nTotal recorded frames: {len(frame_ids)}")
    print(f"Frames selected for inference: {frames_to_process}")

    if frames_to_process == 0:
        print("Not enough recorded frames for the selected mode.")
        return

    print("\nLoading model...")
    model, processor = load_open_loop_model(args)
    print("Model loaded!")

    predictions = []
    cot_texts = []
    answer_texts = []
    inference_times = []
    vqa_document = None
    if args.mode == "vqa":
        from module.open_loop_results import create_vqa_document

        vqa_document = create_vqa_document(
            args.data_root,
            args.vqa_question,
            args.temperature,
        )

    for display_index, frame_index in enumerate(
        range(start_index, len(frame_ids)), start=1
    ):
        frame_id = frame_ids[frame_index]
        print(f"  Frame {display_index}/{frames_to_process} (CARLA frame={frame_id})")

        model_input = load_model_input(args.data_root, trajectory, frame_ids, frame_index)
        if args.mode == "vqa":
            from module.open_loop_results import append_vqa_frame, write_json_atomic

            answer, inference_time = run_vqa_frame(
                model,
                processor,
                model_input,
                question=args.vqa_question,
                temperature=args.temperature,
            )
            answer_texts.append(answer)
            append_vqa_frame(
                vqa_document,
                display_index,
                frame_id,
                answer,
                inference_time,
            )
            write_json_atomic(vqa_document, args.output_json)
            print(f"    Inference: {inference_time:.2f}s | Answer: {answer[:80]}...")
        else:
            pred_xyz, cot_text, inference_time = run_trajectory_frame(
                model,
                processor,
                model_input,
                navigation_text=args.navigation_text if args.mode == "navigation" else None,
                navigation_weight=args.navigation_weight,
                temperature=args.temperature,
            )
            predictions.append(pred_xyz)
            cot_texts.append(cot_text)
            print(f"    Inference: {inference_time:.2f}s | CoC: {cot_text[:80]}...")

        inference_times.append(inference_time)

    avg_time = float(np.mean(inference_times)) if inference_times else 0.0
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total frames processed: {len(inference_times)}")
    print(f"Average inference time: {avg_time:.2f}s/frame")
    if torch.cuda.is_available():
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if args.mode == "vqa" and answer_texts:
        from module.visualization import save_open_loop_vqa_video

        print("\nCreating VQA video...")
        save_open_loop_vqa_video(
            (
                load_front_camera_image(args.data_root, frame_id)
                for frame_id in frame_ids[start_index:]
            ),
            answer_texts,
            inference_times,
            args.vqa_question,
            args.output_video,
            fps=5,
            total_frames=frames_to_process,
        )
    elif predictions:
        from module.visualization import save_open_loop_video

        print("\nCreating video...")
        save_open_loop_video(
            predictions,
            (
                load_front_camera_image(args.data_root, frame_id)
                for frame_id in frame_ids[start_index:]
            ),
            cot_texts,
            inference_times,
            args.output_video,
            fps=5,
            navigation_text=args.navigation_text if args.mode == "navigation" else "",
            navigation_weight=args.navigation_weight,
            total_frames=frames_to_process,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
