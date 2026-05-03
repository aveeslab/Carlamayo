"""Modular entrypoint for CARLA closed-loop control with Alpamayo."""

import argparse
import json
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np
import torch

from module import config as cfg
from module.latency_control import NormalModeLatencyStats, should_refresh_normal_inference
from module.navigation_control import NavigationControlState
from module.pid_controller import OfficialPIDFollower
from module.respawn_control import RespawnMonitor
from module.trajectory_cache import alpamayo_local_to_world, world_to_alpamayo_local
from module.vlm_generate_optimization import VlmGenerateTiming
from module.visualization import VideoRecorder, create_visualization_frame
from module.carla_interface import CARLAInterface
from module.inference import (
    configure_cuda_linalg_library,
    extract_answer_text,
    extract_cot_text,
    extract_trajectory_samples,
    load_model,
    prepare_model_input,
    run_inference,
    run_vqa,
    select_trajectory_by_prev_similarity,
)


def capture_initial_ui_frame(carla_if, frame_count):
    """Tick once so paused pygame starts with a real camera frame."""

    carla_if.tick()
    frame_count += 1
    state = carla_if.get_ego_state()
    carla_if.update_history(state)
    images = carla_if.get_camera_images()
    ui_frame = None
    if len(images) > 1:
        ui_frame = images[1]
    elif len(images) > 0:
        ui_frame = images[0]

    telemetry = {
        "frame": frame_count,
        "speed_kmh": state["speed"] * 3.6,
        "steering": 0.0,
        "inference_time": 0.0,
    }
    return frame_count, ui_frame, telemetry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CARLA closed-loop control with Alpamayo (modular)."
    )
    parser.add_argument(
        "--quantization",
        dest="quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantized model. This is the default for local closed-loop testing.",
    )
    parser.add_argument(
        "--no-quantization",
        dest="quantization",
        action="store_false",
        help="Use full-precision model instead of the default 4-bit quantized model.",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Run internal async inference mode (non-blocking world tick).",
    )
    parser.add_argument(
        "--pygame-ui",
        action="store_true",
        help="Show a pygame camera UI with prompt input and pause/resume controls.",
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "navigation", "vqa"),
        default="normal",
        help="Closed-loop inference mode. Default: normal.",
    )
    parser.add_argument(
        "--navigation-text",
        default="",
        help='Initial navigation instruction, e.g. "Turn right in 30m".',
    )
    parser.add_argument(
        "--navigation-weight",
        type=float,
        default=1.0,
        help="Navigation CFG weight. 1.0 uses normal nav conditioning; other values use CFG nav.",
    )
    parser.add_argument(
        "--vqa-question",
        default="",
        help='Initial VQA question for --mode vqa, e.g. "Describe the scene.".',
    )
    parser.add_argument(
        "--normal-inference-interval-frames",
        type=int,
        default=0,
        help=(
            "Minimum synchronous CARLA frames between normal-mode model refreshes. "
            "Default: 0 (refresh every ready frame). "
            "Set 0 to reproduce the per-ready-frame baseline."
        ),
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Start the pygame UI paused so navigation text can be entered before the first tick.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many CARLA frames. Default: 0 means run until interrupted.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable MP4 recording for latency benchmark runs.",
    )
    parser.add_argument(
        "--latency-stats-json",
        default="",
        help="Write normal-mode latency stats to this JSON file on shutdown.",
    )
    parser.add_argument(
        "--keep-generate-logits",
        dest="disable_unused_generate_logits",
        action="store_false",
        default=True,
        help=(
            "Keep Alpamayo VLM returned logits during trajectory generation. "
            "Default disables these unused returned logits to reduce single-call latency/memory."
        ),
    )
    parser.add_argument(
        "--vlm-image-pixels",
        type=int,
        default=65536,
        help=(
            "Per-image min/max pixel budget passed to the Qwen-VL processor. "
            "Default: 65536 for lower single-call VLM latency. Use 196608 for "
            "the original Alpamayo image-token budget."
        ),
    )
    parser.add_argument(
        "--carla-map",
        default=cfg.CARLA_MAP,
        help=f"CARLA map to load before spawning actors. Default: {cfg.CARLA_MAP}.",
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
            'Default: "magma" to avoid cuSOLVER cholesky handle failures.'
        ),
    )
    parser.add_argument(
        "--no-auto-respawn",
        dest="auto_respawn",
        action="store_false",
        default=True,
        help="Disable automatic ego respawn after collisions or repeated stuck frames.",
    )
    parser.add_argument(
        "--respawn-stuck-frames",
        type=int,
        default=cfg.RESPAWN_STUCK_FRAMES,
        help=(
            "Respawn after this many consecutive low-speed frames while throttle is commanded. "
            "Set 0 to disable stuck respawn."
        ),
    )
    parser.add_argument(
        "--respawn-stuck-speed-kmh",
        type=float,
        default=cfg.RESPAWN_STUCK_SPEED_KMH,
        help="Speed threshold for stuck respawn detection in km/h.",
    )
    parser.add_argument(
        "--respawn-collision-cooldown-frames",
        type=int,
        default=cfg.RESPAWN_COLLISION_COOLDOWN_FRAMES,
        help="Minimum frames between automatic respawns.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inference_interval_sec = 1.0
    if args.normal_inference_interval_frames < 0:
        raise ValueError("--normal-inference-interval-frames must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.vlm_image_pixels <= 0:
        raise ValueError("--vlm-image-pixels must be positive")
    if args.respawn_stuck_frames < 0:
        raise ValueError("--respawn-stuck-frames must be non-negative")
    if args.respawn_stuck_speed_kmh < 0:
        raise ValueError("--respawn-stuck-speed-kmh must be non-negative")
    if args.respawn_collision_cooldown_frames < 0:
        raise ValueError("--respawn-collision-cooldown-frames must be non-negative")

    print("=" * 60)
    print("CARLA Real-time Control with Alpamayo")
    print("=" * 60)
    print(f"Quantization: {'ON (4-bit)' if args.quantization else 'OFF (full-precision)'}")
    print(f"Execution: {'ASYNC' if args.async_mode else 'SYNC'}")
    print(f"Inference mode: {args.mode}")
    print(f"Pygame UI: {'ON' if args.pygame_ui else 'OFF'}")
    print(f"CARLA map: {args.carla_map}")
    print(f"Device map: {args.device_map}")
    print(f"CUDA linalg library: {args.cuda_linalg_library}")
    print(f"Video recording: {'OFF' if args.no_video else ('ON' if cfg.SAVE_VIDEO else 'OFF')}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    if args.mode == "normal":
        print(f"Normal model refresh interval: {args.normal_inference_interval_frames} frames")
    print(f"Auto respawn: {'ON' if args.auto_respawn else 'OFF'}")
    if args.auto_respawn:
        print(
            "  collision cooldown="
            f"{args.respawn_collision_cooldown_frames} frames, "
            f"stuck={args.respawn_stuck_frames} frames "
            f"@ <= {args.respawn_stuck_speed_kmh:.1f} km/h"
        )

    nav_state = NavigationControlState(
        args.navigation_text,
        args.navigation_weight,
        mode=args.mode,
        vqa_question=args.vqa_question,
    )
    nav_state.paused = bool(args.start_paused)
    if args.mode == "navigation" and nav_state.navigation_text:
        print(
            f"Initial navigation: {nav_state.navigation_text} "
            f"(weight={nav_state.navigation_weight:.2f})"
        )
    if args.mode == "vqa" and nav_state.vqa_question:
        print(f"Initial VQA question: {nav_state.vqa_question}")

    print("\nLoading model...")
    configure_cuda_linalg_library(args.cuda_linalg_library)
    model, processor = load_model(args.quantization, device_map=args.device_map)
    print("Model loaded!")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

    carla_if = CARLAInterface()
    save_video = cfg.SAVE_VIDEO and not args.no_video
    video_recorder = VideoRecorder(cfg.OUTPUT_VIDEO, fps=cfg.VIDEO_FPS) if save_video else None
    pygame_ui = None
    latest_ui_frame = None
    latest_telemetry = {}
    latency_stats = NormalModeLatencyStats()
    vlm_generate_timing = VlmGenerateTiming()
    respawn_monitor = RespawnMonitor(
        cooldown_frames=args.respawn_collision_cooldown_frames,
        stuck_frames=args.respawn_stuck_frames,
        stuck_speed_kmh=args.respawn_stuck_speed_kmh,
    )
    respawn_count = 0
    respawn_reasons = []
    inference_request_q = None
    inference_result_q = None
    inference_stop = None
    worker_thread = None

    if args.pygame_ui:
        from module.pygame_ui import ClosedLoopPygameUI

        pygame_ui = ClosedLoopPygameUI(
            width=cfg.PYGAME_WINDOW_WIDTH,
            height=cfg.PYGAME_WINDOW_HEIGHT,
            mode=args.mode,
        )

    try:
        carla_if.connect()
        carla_if.load_map(args.carla_map)
        carla_if.spawn_ego_vehicle()
        carla_if.enable_synchronous_mode()
        carla_if.spawn_npcs(num_vehicles=cfg.NPC_VEHICLE_COUNT, num_walkers=cfg.NPC_WALKER_COUNT)
        carla_if.setup_cameras()
        if args.auto_respawn:
            carla_if.setup_collision_sensor()
        time.sleep(1.0)

        pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)

        current_trajectory = None
        current_pred_xyz = None
        current_pred_world = None
        prev_selected_trajectory = None
        current_selected_traj_idx = 0
        current_cot = ""
        current_inference_time = 0.0
        frame_buffer = []
        current_trajectory_ts = None
        last_model_refresh_frame = None
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}

        pending_inference = False
        last_inference_submit_ts = 0.0
        respawn_revision = 0

        def _clear_async_queues():
            if inference_request_q is not None:
                while True:
                    try:
                        inference_request_q.get_nowait()
                    except queue.Empty:
                        break
            if inference_result_q is not None:
                while True:
                    try:
                        inference_result_q.get_nowait()
                    except queue.Empty:
                        break

        def _auto_respawn(reason):
            nonlocal current_trajectory, current_pred_xyz, current_pred_world
            nonlocal prev_selected_trajectory, current_selected_traj_idx, current_cot
            nonlocal current_inference_time, current_trajectory_ts, last_model_refresh_frame
            nonlocal prev_control, pending_inference, pid_follower, respawn_count
            nonlocal respawn_revision

            print(f"[Frame {frame_count}] Auto-respawn: {reason}")
            carla_if.respawn_ego_vehicle()
            pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)
            current_trajectory = None
            current_pred_xyz = None
            current_pred_world = None
            prev_selected_trajectory = None
            current_selected_traj_idx = 0
            current_cot = ""
            current_inference_time = 0.0
            current_trajectory_ts = None
            last_model_refresh_frame = None
            prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
            pending_inference = False
            respawn_revision += 1
            respawn_count += 1
            respawn_reasons.append({"frame": int(frame_count), "reason": reason})
            respawn_monitor.mark_respawn(
                frame_count=frame_count,
                collision_count=carla_if.get_collision_count(),
            )
            frame_buffer.clear()
            _clear_async_queues()

        def _run_inference_with_nav_fallback(model_data, navigation_text, navigation_weight):
            def _run_once(weight):
                return run_inference(
                    model,
                    processor,
                    model_data,
                    navigation_text=navigation_text,
                    navigation_weight=weight,
                    vlm_generate_timing=vlm_generate_timing,
                    disable_unused_generate_logits=args.disable_unused_generate_logits,
                    vlm_image_pixels=args.vlm_image_pixels,
                )

            try:
                return _run_once(navigation_weight)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if navigation_text and abs(float(navigation_weight) - 1.0) > 1e-6:
                    message = (
                        "Navigation CFG ran out of CUDA memory; "
                        "falling back to normal nav conditioning with weight 1.0."
                    )
                    print(message)
                    nav_state.set_error(message)
                    return _run_once(1.0)
                raise
            except RuntimeError as exc:
                if "CUSOLVER_STATUS_INTERNAL_ERROR" not in str(exc):
                    raise
                message = (
                    "cuSOLVER linalg backend failed; switching CUDA linalg backend "
                    "to MAGMA and retrying inference once."
                )
                print(message)
                nav_state.set_error(message)
                configure_cuda_linalg_library("magma")
                return _run_once(navigation_weight)

        def _run_vqa_with_linalg_fallback(model_data, question):
            try:
                return run_vqa(model, processor, model_data, question=question)
            except RuntimeError as exc:
                if "CUSOLVER_STATUS_INTERNAL_ERROR" not in str(exc):
                    raise
                message = (
                    "cuSOLVER linalg backend failed; switching CUDA linalg backend "
                    "to MAGMA and retrying VQA once."
                )
                print(message)
                nav_state.set_error(message)
                configure_cuda_linalg_library("magma")
                return run_vqa(model, processor, model_data, question=question)

        if args.async_mode:
            inference_request_q = queue.Queue(maxsize=1)
            inference_result_q = queue.Queue(maxsize=1)
            inference_stop = threading.Event()

            def _build_inference_request():
                images_array = np.zeros(
                    (
                        cfg.NUM_CAMERAS,
                        cfg.NUM_FRAMES,
                        cfg.IMG_HEIGHT,
                        cfg.IMG_WIDTH,
                        cfg.IMG_CHANNELS,
                    ),
                    dtype=np.uint8,
                )
                for t, frame_images in enumerate(frame_buffer):
                    for c in range(cfg.NUM_CAMERAS):
                        images_array[c, t] = frame_images[c]
                history_xyz, history_rot = carla_if.get_history_in_local_frame()
                return {
                    "mode": args.mode,
                    "images_array": images_array,
                    "history_xyz": history_xyz,
                    "history_rot": history_rot,
                    "ego_state": state.copy(),
                    "navigation_text": nav_state.navigation_text,
                    "navigation_weight": nav_state.navigation_weight,
                    "vqa_question": nav_state.vqa_question,
                    "prompt_revision": nav_state.revision,
                    "respawn_revision": respawn_revision,
                }

            def _inference_worker():
                while not inference_stop.is_set():
                    try:
                        req = inference_request_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if req is None:
                        break
                    req_frame = int(req["frame"])
                    try:
                        t0 = time.time()
                        model_data = prepare_model_input(
                            req["images_array"],
                            req["history_xyz"],
                            req["history_rot"],
                        )
                        if req["mode"] == "vqa":
                            extra = _run_vqa_with_linalg_fallback(
                                model_data,
                                question=req["vqa_question"],
                            )
                            result = {
                                "mode": "vqa",
                                "frame_submitted": req_frame,
                                "extra": extra,
                                "answer": extract_answer_text(extra),
                                "inference_time": time.time() - t0,
                                "result_ts": time.time(),
                                "vqa_question": req["vqa_question"],
                                "prompt_revision": req["prompt_revision"],
                                "respawn_revision": req["respawn_revision"],
                            }
                        else:
                            navigation_text = (
                                req["navigation_text"] if req["mode"] == "navigation" else ""
                            )
                            navigation_weight = (
                                req["navigation_weight"] if req["mode"] == "navigation" else 1.0
                            )
                            pred_xyz, extra = _run_inference_with_nav_fallback(
                                model_data,
                                navigation_text=navigation_text,
                                navigation_weight=navigation_weight,
                            )
                            result = {
                                "mode": req["mode"],
                                "frame_submitted": req_frame,
                                "pred_xyz": pred_xyz,
                                "extra": extra,
                                "inference_time": time.time() - t0,
                                "result_ts": time.time(),
                                "navigation_text": navigation_text,
                                "navigation_weight": navigation_weight,
                                "prompt_revision": req["prompt_revision"],
                                "respawn_revision": req["respawn_revision"],
                            }
                    except Exception as e:
                        result = {
                            "frame_submitted": req_frame,
                            "error": str(e),
                            "result_ts": time.time(),
                            "respawn_revision": req.get("respawn_revision"),
                        }

                    while True:
                        try:
                            inference_result_q.get_nowait()
                        except queue.Empty:
                            break
                    inference_result_q.put_nowait(result)

            worker_thread = threading.Thread(
                target=_inference_worker,
                name="alpamayo-inference-worker",
                daemon=True,
            )
            worker_thread.start()

        print("\nStarting control loop...")
        if save_video:
            print(f"Recording video to: {cfg.OUTPUT_VIDEO}")
        print("-" * 60)

        frame_count = 0
        last_seen_nav_revision = nav_state.revision
        last_vqa_submitted_revision = None
        last_vqa_completed_revision = None
        if pygame_ui is not None and nav_state.paused:
            frame_count, latest_ui_frame, latest_telemetry = capture_initial_ui_frame(
                carla_if,
                frame_count,
            )
            pygame_ui.draw(latest_ui_frame, nav_state, latest_telemetry)

        while True:
            if pygame_ui is not None:
                if not pygame_ui.process_events(nav_state):
                    print("\nPygame UI requested shutdown.")
                    break
                if nav_state.revision != last_seen_nav_revision:
                    if args.mode == "navigation":
                        print(
                            f"Navigation updated: {nav_state.navigation_text or '(none)'} "
                            f"(weight={nav_state.navigation_weight:.2f})"
                        )
                    elif args.mode == "vqa":
                        print(f"VQA question updated: {nav_state.vqa_question or '(none)'}")
                    prev_selected_trajectory = None
                    current_trajectory = None
                    current_pred_xyz = None
                    current_pred_world = None
                    current_trajectory_ts = None
                    pending_inference = False
                    last_seen_nav_revision = nav_state.revision
                if nav_state.paused:
                    carla_if.apply_control(0.0, 0.0, 1.0)
                    latest_telemetry = {
                        **latest_telemetry,
                        "frame": frame_count,
                        "inference_time": current_inference_time,
                    }
                    pygame_ui.draw(latest_ui_frame, nav_state, latest_telemetry)
                    continue

            carla_if.tick()
            frame_count += 1

            state = carla_if.get_ego_state()
            carla_if.update_history(state)

            images = carla_if.get_camera_images()
            if args.auto_respawn:
                collision_decision = respawn_monitor.check_collision(
                    frame_count=frame_count,
                    collision_count=carla_if.get_collision_count(),
                    last_collision_event=carla_if.get_last_collision_event(),
                )
                if collision_decision.should_respawn:
                    _auto_respawn(collision_decision.reason)
                    continue

            if len(images) > 1:
                latest_ui_frame = images[1]
            latest_telemetry = {
                "frame": frame_count,
                "speed_kmh": state["speed"] * 3.6,
                "steering": prev_control["steer"],
                "inference_time": current_inference_time,
            }
            frame_buffer.append(images)
            if len(frame_buffer) > cfg.NUM_FRAMES:
                frame_buffer.pop(0)
            if current_pred_world is not None:
                current_pred_xyz = world_to_alpamayo_local(current_pred_world, state)
                current_trajectory = current_pred_xyz[current_selected_traj_idx]

            if args.async_mode:
                now_ts = time.time()
                if args.mode == "vqa":
                    should_submit_inference = (
                        len(frame_buffer) >= cfg.NUM_FRAMES
                        and bool(nav_state.vqa_question)
                        and not pending_inference
                        and nav_state.revision != last_vqa_submitted_revision
                        and nav_state.revision != last_vqa_completed_revision
                    )
                elif args.mode == "normal":
                    frame_ready = len(frame_buffer) >= cfg.NUM_FRAMES
                    if frame_ready:
                        latency_stats.record_eligible_frame()
                    should_submit_inference = should_refresh_normal_inference(
                        frame_ready=frame_ready,
                        has_trajectory=current_trajectory is not None,
                        pending_inference=pending_inference,
                        frame_count=frame_count,
                        last_refresh_frame=last_model_refresh_frame,
                        min_interval_frames=args.normal_inference_interval_frames,
                    )
                    if (
                        frame_ready
                        and not should_submit_inference
                        and current_trajectory is not None
                    ):
                        latency_stats.record_reuse_frame()
                else:
                    should_submit_inference = (
                        len(frame_buffer) >= cfg.NUM_FRAMES
                        and not pending_inference
                        and (now_ts - last_inference_submit_ts) >= inference_interval_sec
                    )
                if should_submit_inference:
                    req = _build_inference_request()
                    req["frame"] = int(frame_count)
                    while True:
                        try:
                            inference_request_q.get_nowait()
                        except queue.Empty:
                            break
                    inference_request_q.put_nowait(req)
                    pending_inference = True
                    last_inference_submit_ts = now_ts
                    if args.mode == "vqa":
                        last_vqa_submitted_revision = nav_state.revision

                latest_result = None
                while True:
                    try:
                        latest_result = inference_result_q.get_nowait()
                    except queue.Empty:
                        break
                if latest_result is not None:
                    pending_inference = False
                    if (
                        latest_result.get("prompt_revision", nav_state.revision)
                        != nav_state.revision
                    ):
                        print(
                            f"[Frame {frame_count}] Discarded stale inference result for "
                            f"prompt revision {latest_result.get('prompt_revision')}"
                        )
                    elif (
                        latest_result.get("respawn_revision", respawn_revision)
                        != respawn_revision
                    ):
                        print(
                            f"[Frame {frame_count}] Discarded stale inference result from "
                            f"respawn revision {latest_result.get('respawn_revision')}"
                        )
                    elif "error" not in latest_result and latest_result.get("mode") == "vqa":
                        answer = latest_result.get("answer") or extract_answer_text(
                            latest_result.get("extra")
                        )
                        nav_state.set_vqa_answer(answer)
                        current_inference_time = float(latest_result["inference_time"])
                        last_vqa_completed_revision = latest_result.get("prompt_revision")
                        print(
                            f"[Frame {frame_count}] VQA done: {current_inference_time:.2f}s "
                            f"(submitted at frame {latest_result['frame_submitted']})"
                        )
                        print(f"    Q: {latest_result.get('vqa_question') or '(none)'}")
                        print(f"    A: {answer[:160]}...")
                    elif "error" not in latest_result:
                        pred_xyz = latest_result["pred_xyz"]
                        extra = latest_result["extra"]
                        inference_time = float(latest_result["inference_time"])
                        traj_samples = extract_trajectory_samples(pred_xyz)
                        selected_idx, _similarity_scores = select_trajectory_by_prev_similarity(
                            traj_samples,
                            prev_selected_trajectory,
                        )
                        current_selected_traj_idx = selected_idx
                        anchor_state = latest_result.get("ego_state", state)
                        current_pred_world = alpamayo_local_to_world(traj_samples, anchor_state)
                        current_pred_xyz = world_to_alpamayo_local(current_pred_world, state)
                        current_trajectory = current_pred_xyz[selected_idx]
                        prev_selected_trajectory = current_trajectory.copy()
                        current_cot = extract_cot_text(extra)
                        current_inference_time = inference_time
                        current_trajectory_ts = float(latest_result["result_ts"])
                        if latest_result.get("mode") == "normal":
                            last_model_refresh_frame = frame_count
                            latency_stats.record_model_refresh(inference_time)

                        print(
                            f"[Frame {frame_count}] Inference done: {inference_time:.2f}s "
                            f"(submitted at frame {latest_result['frame_submitted']})"
                        )
                        print(f"    CoT: {current_cot[:60]}...")
                        print(
                            f"    Nav: {latest_result.get('navigation_text') or '(none)'} "
                            f"(weight={latest_result.get('navigation_weight', 1.0):.2f})"
                        )
                        print(
                            f"    Selected traj sample: {current_selected_traj_idx}/"
                            f"{cfg.NUM_TRAJ_SAMPLES - 1}"
                        )
                        print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")
                    else:
                        print(f"[Frame {frame_count}] Inference error: {latest_result['error']}")
            else:
                if len(frame_buffer) >= cfg.NUM_FRAMES:
                    if args.mode == "vqa":
                        images_array = np.zeros(
                            (
                                cfg.NUM_CAMERAS,
                                cfg.NUM_FRAMES,
                                cfg.IMG_HEIGHT,
                                cfg.IMG_WIDTH,
                                cfg.IMG_CHANNELS,
                            ),
                            dtype=np.uint8,
                        )
                        for t, frame_images in enumerate(frame_buffer):
                            for c in range(cfg.NUM_CAMERAS):
                                images_array[c, t] = frame_images[c]
                        history_xyz, history_rot = carla_if.get_history_in_local_frame()
                        model_data = prepare_model_input(images_array, history_xyz, history_rot)
                        should_run_vqa = (
                            bool(nav_state.vqa_question)
                            and nav_state.revision != last_vqa_completed_revision
                        )
                        if should_run_vqa:
                            model_start_time = time.time()
                            extra = _run_vqa_with_linalg_fallback(
                                model_data,
                                question=nav_state.vqa_question,
                            )
                            model_inference_time = time.time() - model_start_time
                            answer = extract_answer_text(extra)
                            nav_state.set_vqa_answer(answer)
                            current_inference_time = model_inference_time
                            last_vqa_completed_revision = nav_state.revision
                            print(f"[Frame {frame_count}] VQA: {model_inference_time:.2f}s")
                            print(f"    Q: {nav_state.vqa_question}")
                            print(f"    A: {answer[:160]}...")
                    else:
                        should_run_inference = True
                        if args.mode == "normal":
                            latency_stats.record_eligible_frame()
                            should_run_inference = should_refresh_normal_inference(
                                frame_ready=True,
                                has_trajectory=current_trajectory is not None,
                                pending_inference=False,
                                frame_count=frame_count,
                                last_refresh_frame=last_model_refresh_frame,
                                min_interval_frames=args.normal_inference_interval_frames,
                            )
                            if not should_run_inference and current_trajectory is not None:
                                latency_stats.record_reuse_frame()
                        if not should_run_inference:
                            model_data = None
                        else:
                            images_array = np.zeros(
                                (
                                    cfg.NUM_CAMERAS,
                                    cfg.NUM_FRAMES,
                                    cfg.IMG_HEIGHT,
                                    cfg.IMG_WIDTH,
                                    cfg.IMG_CHANNELS,
                                ),
                                dtype=np.uint8,
                            )
                            for t, frame_images in enumerate(frame_buffer):
                                for c in range(cfg.NUM_CAMERAS):
                                    images_array[c, t] = frame_images[c]
                            history_xyz, history_rot = carla_if.get_history_in_local_frame()
                            model_data = prepare_model_input(images_array, history_xyz, history_rot)

                        navigation_text = (
                            nav_state.navigation_text if args.mode == "navigation" else ""
                        )
                        navigation_weight = (
                            nav_state.navigation_weight if args.mode == "navigation" else 1.0
                        )
                        if model_data is not None:
                            model_start_time = time.time()
                            pred_xyz, extra = _run_inference_with_nav_fallback(
                                model_data,
                                navigation_text=navigation_text,
                                navigation_weight=navigation_weight,
                            )
                            model_inference_time = time.time() - model_start_time

                            traj_samples = extract_trajectory_samples(pred_xyz)
                            selected_idx, _similarity_scores = select_trajectory_by_prev_similarity(
                                traj_samples,
                                prev_selected_trajectory,
                            )
                            current_selected_traj_idx = selected_idx
                            current_pred_world = alpamayo_local_to_world(traj_samples, state)
                            current_pred_xyz = world_to_alpamayo_local(current_pred_world, state)
                            current_trajectory = current_pred_xyz[selected_idx]
                            prev_selected_trajectory = current_trajectory.copy()
                            current_cot = extract_cot_text(extra)
                            current_inference_time = model_inference_time
                            current_trajectory_ts = time.time()
                            if args.mode == "normal":
                                last_model_refresh_frame = frame_count
                                latency_stats.record_model_refresh(model_inference_time)

                            print(f"[Frame {frame_count}] Inference: {model_inference_time:.2f}s")
                            print(f"    CoT: {current_cot[:60]}...")
                            if args.mode == "navigation":
                                print(
                                    f"    Nav: {nav_state.navigation_text or '(none)'} "
                                    f"(weight={nav_state.navigation_weight:.2f})"
                                )
                            print(
                                f"    Selected traj sample: {current_selected_traj_idx}/"
                                f"{cfg.NUM_TRAJ_SAMPLES - 1}"
                            )
                            print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")

            if current_trajectory is not None:
                vehicle_tf = carla_if.ego_vehicle.get_transform()
                steering_raw, throttle_raw, brake_raw, _ctrl_debug = pid_follower.compute_control(
                    vehicle_tf,
                    current_trajectory[:, :3],
                    float(state["speed"]),
                )

                alpha = cfg.CONTROL_SMOOTH_ALPHA
                steering = (1.0 - alpha) * prev_control["steer"] + alpha * steering_raw
                throttle = (1.0 - alpha) * prev_control["throttle"] + alpha * throttle_raw
                brake = (1.0 - alpha) * prev_control["brake"] + alpha * brake_raw

                if throttle >= brake:
                    brake = 0.0
                else:
                    throttle = 0.0

                prev_control = {"steer": steering, "throttle": throttle, "brake": brake}
                carla_if.apply_control(steering, throttle, brake)

                if current_pred_xyz is not None:
                    cam_img = images[1]
                    vis_frame = create_visualization_frame(
                        cam_img,
                        current_pred_xyz,
                        current_selected_traj_idx,
                        frame_count,
                        current_inference_time,
                        current_cot,
                        state["speed"] * 3.6,
                        steering,
                        navigation_text=nav_state.navigation_text,
                        navigation_weight=nav_state.navigation_weight,
                        paused=nav_state.paused,
                    )
                    latest_ui_frame = vis_frame
                    if save_video:
                        video_recorder.add_frame(vis_frame)

                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": steering,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    pygame_ui.draw(latest_ui_frame, nav_state, latest_telemetry)

                print(
                    f"[Frame {frame_count}] Speed: {state['speed']*3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}"
                )
                if current_trajectory_ts is not None and args.async_mode:
                    print(f"    Trajectory age: {time.time() - current_trajectory_ts:.2f}s")
                if args.auto_respawn:
                    stuck_decision = respawn_monitor.check_stuck(
                        frame_count=frame_count,
                        speed_kmh=state["speed"] * 3.6,
                        throttle=throttle,
                        brake=brake,
                        has_trajectory=True,
                    )
                    if stuck_decision.should_respawn:
                        _auto_respawn(stuck_decision.reason)
            else:
                if args.mode == "vqa":
                    carla_if.apply_control(0.0, 0.0, 1.0)
                else:
                    carla_if.apply_control(0.0, 0.3, 0.0)
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": 0.0,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    pygame_ui.draw(latest_ui_frame, nav_state, latest_telemetry)

            if args.max_frames and frame_count >= args.max_frames:
                print(f"\nReached --max-frames={args.max_frames}; stopping.")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if args.async_mode and inference_stop is not None:
            inference_stop.set()
            try:
                inference_request_q.put_nowait(None)
            except Exception:
                pass
            if worker_thread is not None:
                worker_thread.join(timeout=2.0)
        if save_video and video_recorder:
            video_recorder.save()
        if pygame_ui is not None:
            pygame_ui.close()
        carla_if.cleanup()
        if args.mode == "normal":
            stats_dict = latency_stats.to_dict(
                interval_frames=args.normal_inference_interval_frames,
                mode=args.mode,
            )
            stats_dict.update(vlm_generate_timing.to_dict())
            stats_dict["vlm_image_pixels"] = int(args.vlm_image_pixels)
            stats_dict["disable_unused_generate_logits"] = bool(
                args.disable_unused_generate_logits
            )
            stats_dict["respawn_count"] = int(respawn_count)
            stats_dict["respawn_reasons"] = respawn_reasons
            print(
                "Normal latency stats: "
                f"eligible_frames={stats_dict['eligible_frames']}, "
                f"model_refreshes={stats_dict['model_refreshes']}, "
                f"trajectory_reuse_frames={stats_dict['trajectory_reuse_frames']}, "
                f"vlm_call_reduction_vs_per_frame_baseline="
                f"{stats_dict['vlm_call_reduction_vs_per_frame_baseline'] * 100:.1f}%, "
                f"total_model_time={stats_dict['total_model_time_sec']:.2f}s, "
                f"avg_vlm_generate_time={stats_dict['avg_vlm_generate_time_sec']:.2f}s, "
                f"respawns={respawn_count}"
            )
            if args.latency_stats_json:
                stats_path = Path(args.latency_stats_json)
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")
                print(f"Wrote latency stats JSON: {stats_path}")

    print("\nStopped.")


if __name__ == "__main__":
    main()
