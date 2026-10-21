# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Closed-loop CARLA control: Alpamayo trajectories drive the ego via PID.

"""Closed-loop runner: the selected Alpamayo model controls the ego vehicle.

Every tick (or asynchronously) the model predicts a trajectory that a PID follower
converts to steering/throttle/brake. Supports ``normal`` / ``navigation`` / ``vqa``
modes; navigation and VQA require a version whose adapter advertises support.
"""

import queue
import threading
import time
import traceback

import torch

from module import config as cfg
from module.carla_interface import CARLAInterface
from module.data_collection import RunRecorder
from module.inference import (
    configure_cuda_linalg_library,
    extract_trajectory_samples,
    select_trajectory_by_prev_similarity,
)
from module.loops._common import (
    collect_oom_kwargs,
    derive_pygame_ui_video_path,
    stack_frame_buffer,
)
from module.navigation_control import NavigationControlState
from module.pid_controller import OfficialPIDFollower, trajectory_to_world
from module.respawn_control import RespawnMonitor
from module.visualization import VideoRecorder, create_visualization_frame
from module.vlm_generate_optimization import VlmGenerateTiming


def format_vqa_answer_preview(answer, limit=160):
    answer = str(answer or "").strip()
    if not answer:
        return "(empty answer)"
    suffix = "..." if len(answer) > limit else ""
    return f"{answer[:limit]}{suffix}"


def capture_initial_ui_frame(carla_if, frame_count, viz_slot):
    """Tick once so paused pygame starts with a real camera frame."""
    carla_if.tick()
    frame_count += 1
    state = carla_if.get_ego_state()
    carla_if.update_history(state)
    images = carla_if.get_camera_images()
    ui_frame = images[viz_slot] if len(images) > viz_slot else (images[0] if len(images) else None)
    telemetry = {
        "frame": frame_count,
        "speed_kmh": state["speed"] * 3.6,
        "steering": 0.0,
        "inference_time": 0.0,
    }
    return frame_count, ui_frame, telemetry


def _check_capabilities(adapter, args):
    if args.mode == "navigation" and not adapter.supports_navigation:
        raise SystemExit(
            f"{adapter.display_name} does not support navigation mode. "
            "Use --version 1.5 or 2, or --mode normal."
        )
    if args.mode == "vqa" and not adapter.supports_vqa:
        raise SystemExit(
            f"{adapter.display_name} does not support VQA mode. "
            "Use --version 1.5 or 2, or --mode normal."
        )
    if args.oom_free and not adapter.supports_oom_free:
        raise SystemExit(
            f"{adapter.display_name} does not support OOM-free demand layering "
            "(available for --version 1.5 only)."
        )
    if args.oom_free and args.quantization:
        raise SystemExit("--oom-free and --quantization are mutually exclusive.")


def run(adapter, args):
    _check_capabilities(adapter, args)
    inference_interval_sec = 1.0
    viz_slot = adapter.viz_camera_slot
    num_cameras = adapter.num_cameras
    output_video = args.output_video or cfg.OUTPUT_VIDEO
    start_paused = bool(args.pygame_ui)
    pygame_ui_video = derive_pygame_ui_video_path(output_video) if args.pygame_ui else None

    print("=" * 60)
    print(f"CARLA Real-time Closed-Loop Control with {adapter.display_name}")
    print("=" * 60)
    if args.oom_free:
        print("Model loading: OOM-free CPU<->GPU demand layering (full-precision)")
    else:
        print(f"Quantization: {'ON (4-bit)' if args.quantization else 'OFF (full-precision)'}")
    print(f"Execution: {'ASYNC' if args.async_mode else 'SYNC'}")
    print(f"Inference mode: {args.mode}")
    print(f"Pygame UI: {'ON' if args.pygame_ui else 'OFF'}")
    print(f"CARLA map: {cfg.CARLA_MAP}")
    print(f"Cameras: {num_cameras}")
    print("Auto respawn: ON after collisions")

    nav_state = NavigationControlState(
        args.navigation_text, args.navigation_weight, mode=args.mode, vqa_question=args.vqa_question
    )
    nav_state.paused = bool(start_paused)
    if args.mode == "navigation" and nav_state.navigation_text:
        print(
            f"Initial navigation: {nav_state.navigation_text} "
            f"(weight={nav_state.navigation_weight:.2f})"
        )
    if args.mode == "vqa" and nav_state.vqa_question:
        print(f"Initial VQA question: {nav_state.vqa_question}")

    print("\nLoading model...")
    configure_cuda_linalg_library(args.cuda_linalg_library)
    oom_kwargs = collect_oom_kwargs(args)
    model = processor = None
    if not args.oom_free:
        model, processor = adapter.load_model(
            use_quantization=args.quantization, device_map=args.device_map
        )
        print("Model loaded!")
        print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")
    else:
        # Defer OOM-free loading until CARLA has spawned so the residency plan
        # reflects the VRAM CARLA actually leaves free.
        print("OOM-free mode: Alpamayo loads after CARLA is fully spawned.")

    carla_if = CARLAInterface(camera_configs=adapter.source_camera_configs)
    video_recorder = VideoRecorder(output_video, fps=cfg.VIDEO_FPS) if cfg.SAVE_VIDEO else None
    pygame_ui = None
    pygame_ui_recorder = None
    run_recorder = None
    latest_ui_frame = None
    latest_telemetry = {}

    if args.pygame_ui:
        from module.pygame_ui import ClosedLoopPygameUI

        pygame_ui = ClosedLoopPygameUI(
            width=cfg.PYGAME_WINDOW_WIDTH, height=cfg.PYGAME_WINDOW_HEIGHT, mode=args.mode
        )
        pygame_ui_recorder = VideoRecorder(pygame_ui_video, fps=cfg.VIDEO_FPS)

    def draw_pygame_ui(frame_rgb, telemetry):
        if pygame_ui is None:
            return
        pygame_ui.draw(frame_rgb, nav_state, telemetry)
        if pygame_ui_recorder is not None:
            pygame_ui_recorder.add_frame(pygame_ui.capture_frame())

    try:
        carla_if.tm_port = args.tm_port
        carla_if.connect(host=args.carla_host, port=args.carla_port)
        carla_if.load_map(cfg.CARLA_MAP)
        carla_if.spawn_ego_vehicle()
        carla_if.enable_synchronous_mode()
        carla_if.spawn_npcs(num_vehicles=cfg.NPC_VEHICLE_COUNT, num_walkers=cfg.NPC_WALKER_COUNT)
        carla_if.setup_cameras()
        carla_if.setup_collision_sensor()
        if args.record_dir:
            run_recorder = RunRecorder(args.record_dir, carla_if.camera_order)
            print(f"Recording debug run to: {args.record_dir}")
        time.sleep(1.0)

        if args.oom_free:
            print("Loading model (OOM-free CPU<->GPU demand layering)...")
            model, processor = adapter.load_model(
                use_quantization=False, device_map=args.device_map,
                oom_free=True, oom_kwargs=oom_kwargs,
            )
            print("Model loaded!")
            print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

        pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)

        current_trajectory = None
        current_trajectory_world = None
        current_pred_xyz = None
        prev_selected_trajectory = None
        current_selected_traj_idx = 0
        current_cot = ""
        current_inference_time = 0.0
        vlm_generate_timing = VlmGenerateTiming()
        respawn_monitor = RespawnMonitor(cooldown_frames=cfg.RESPAWN_COLLISION_COOLDOWN_FRAMES)
        frame_buffer = []
        current_trajectory_frame = None
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}

        pending_inference = False
        last_inference_submit_ts = 0.0
        respawn_revision = 0
        inference_request_q = None
        inference_result_q = None
        inference_stop = None
        worker_thread = None

        def _clear_async_queues():
            for maybe_queue in (inference_request_q, inference_result_q):
                if maybe_queue is None:
                    continue
                while True:
                    try:
                        maybe_queue.get_nowait()
                    except queue.Empty:
                        break

        def _auto_respawn(reason):
            nonlocal current_trajectory, current_pred_xyz, prev_selected_trajectory
            nonlocal current_selected_traj_idx, current_cot, current_inference_time
            nonlocal current_trajectory_frame, prev_control, pending_inference, pid_follower
            nonlocal respawn_revision, last_vqa_submitted_revision, last_vqa_completed_revision

            print(f"[Frame {frame_count}] Auto-respawn: {reason}")
            carla_if.respawn_ego_vehicle()
            pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)
            current_trajectory = None
            current_pred_xyz = None
            prev_selected_trajectory = None
            current_selected_traj_idx = 0
            current_cot = ""
            current_inference_time = 0.0
            current_trajectory_frame = None
            prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
            pending_inference = False
            respawn_revision += 1
            last_vqa_submitted_revision = None
            last_vqa_completed_revision = None
            respawn_monitor.mark_respawn(
                frame_count=frame_count, collision_count=carla_if.get_collision_count()
            )
            frame_buffer.clear()
            _clear_async_queues()

        def _run_inference_with_nav_fallback(model_data, navigation_text, navigation_weight):
            def _run_once():
                return adapter.run_inference(
                    model, processor, model_data,
                    navigation_text=navigation_text,
                    navigation_weight=navigation_weight,
                    vlm_generate_timing=vlm_generate_timing,
                )

            try:
                return _run_once()
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
                return _run_once()

        def _run_vqa_with_linalg_fallback(model_data, question):
            try:
                return adapter.run_vqa(model, processor, model_data, question=question)
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
                return adapter.run_vqa(model, processor, model_data, question=question)

        if args.async_mode:
            inference_request_q = queue.Queue(maxsize=1)
            inference_result_q = queue.Queue(maxsize=1)
            inference_stop = threading.Event()

            def _build_inference_request():
                images_array = stack_frame_buffer(frame_buffer, num_cameras)
                history_xyz, history_rot = carla_if.get_history_in_local_frame()
                return {
                    "mode": args.mode,
                    "images_array": images_array,
                    "history_xyz": history_xyz,
                    "history_rot": history_rot,
                    "t0_us": int(frame_count * cfg.CONTROL_DT * 1_000_000),
                    "navigation_text": nav_state.navigation_text,
                    "navigation_weight": nav_state.navigation_weight,
                    "vqa_question": nav_state.vqa_question,
                    "prompt_revision": nav_state.revision,
                    "respawn_revision": respawn_revision,
                    "ego_tf": carla_if.ego_vehicle.get_transform(),
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
                        model_data = adapter.prepare_model_input(
                            req["images_array"], req["history_xyz"], req["history_rot"],
                            req["t0_us"],
                        )
                        if req["mode"] == "vqa":
                            extra = _run_vqa_with_linalg_fallback(
                                model_data, question=req["vqa_question"]
                            )
                            result = {
                                "mode": "vqa",
                                "frame_submitted": req_frame,
                                "extra": extra,
                                "answer": adapter.extract_answer_text(extra),
                                "inference_time": time.time() - t0,
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
                                "navigation_text": navigation_text,
                                "navigation_weight": navigation_weight,
                                "history_xyz": req["history_xyz"],
                                "history_rot": req["history_rot"],
                                "ego_tf": req["ego_tf"],
                                "prompt_revision": req["prompt_revision"],
                                "respawn_revision": req["respawn_revision"],
                            }
                    except Exception as e:
                        result = {
                            "frame_submitted": req_frame,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "respawn_revision": req.get("respawn_revision"),
                        }

                    while True:
                        try:
                            inference_result_q.get_nowait()
                        except queue.Empty:
                            break
                    inference_result_q.put_nowait(result)

            worker_thread = threading.Thread(
                target=_inference_worker, name="alpamayo-inference-worker", daemon=True
            )
            worker_thread.start()

        print("\nStarting control loop...")
        if cfg.SAVE_VIDEO:
            print(f"Recording video to: {output_video}")
        if pygame_ui_recorder is not None:
            print(f"Recording Pygame UI to: {pygame_ui_video}")
        print("-" * 60)

        frame_count = 0
        last_seen_nav_revision = nav_state.revision
        last_vqa_submitted_revision = None
        last_vqa_completed_revision = None
        if pygame_ui is not None and nav_state.paused:
            frame_count, latest_ui_frame, latest_telemetry = capture_initial_ui_frame(
                carla_if, frame_count, viz_slot
            )
            draw_pygame_ui(latest_ui_frame, latest_telemetry)

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
                    current_trajectory_frame = None
                    pending_inference = False
                    last_seen_nav_revision = nav_state.revision
                if nav_state.paused:
                    carla_if.apply_control(0.0, 0.0, 1.0)
                    latest_telemetry = {
                        **latest_telemetry,
                        "frame": frame_count,
                        "inference_time": current_inference_time,
                    }
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)
                    continue

            carla_if.tick()
            frame_count += 1

            state = carla_if.get_ego_state()
            carla_if.update_history(state)

            collision_decision = respawn_monitor.check_collision(
                frame_count=frame_count,
                collision_count=carla_if.get_collision_count(),
                last_collision_event=carla_if.get_last_collision_event(),
            )
            if collision_decision.should_respawn:
                _auto_respawn(collision_decision.reason)
                continue

            try:
                images = carla_if.get_camera_images()
            except TimeoutError as exc:
                print(f"[Frame {frame_count}] Warning: {exc}; braking and skipping this tick.")
                carla_if.apply_control(0.0, 0.0, 1.0)
                prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": 0.0,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)
                continue
            if len(images) > viz_slot:
                latest_ui_frame = images[viz_slot]
            latest_telemetry = {
                "frame": frame_count,
                "speed_kmh": state["speed"] * 3.6,
                "steering": prev_control["steer"],
                "inference_time": current_inference_time,
            }
            frame_buffer.append(images)
            if len(frame_buffer) > cfg.NUM_FRAMES:
                frame_buffer.pop(0)
            if run_recorder is not None:
                run_recorder.add_frame(carla_if.frame_id, images, state)
            new_inference = None

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
                    prompt_rev = latest_result.get("prompt_revision", nav_state.revision)
                    if prompt_rev != nav_state.revision:
                        print(
                            f"[Frame {frame_count}] Discarded stale inference result for "
                            f"prompt revision {latest_result.get('prompt_revision')}"
                        )
                    elif (
                        latest_result.get("respawn_revision", respawn_revision) != respawn_revision
                    ):
                        print(
                            f"[Frame {frame_count}] Discarded stale inference result from "
                            f"respawn revision {latest_result.get('respawn_revision')}"
                        )
                    elif "error" not in latest_result and latest_result.get("mode") == "vqa":
                        answer = latest_result.get("answer") or adapter.extract_answer_text(
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
                        print(f"    A: {format_vqa_answer_preview(answer)}")
                    elif "error" not in latest_result:
                        pred_xyz = latest_result["pred_xyz"]
                        extra = latest_result["extra"]
                        inference_time = float(latest_result["inference_time"])
                        traj_samples = extract_trajectory_samples(pred_xyz)
                        selected_idx, _scores = select_trajectory_by_prev_similarity(
                            traj_samples, prev_selected_trajectory
                        )
                        current_selected_traj_idx = selected_idx
                        current_trajectory = traj_samples[selected_idx]
                        # Anchor the plan at the pose it was predicted from, not the
                        # pose the ego has reached by the time the result arrives.
                        current_trajectory_world = trajectory_to_world(
                            latest_result["ego_tf"], current_trajectory
                        )
                        prev_selected_trajectory = current_trajectory.copy()
                        current_pred_xyz = traj_samples
                        current_cot = adapter.extract_cot_text(extra)
                        current_inference_time = inference_time
                        current_trajectory_frame = latest_result["frame_submitted"]
                        new_inference = {
                            key: latest_result[key]
                            for key in ("frame_submitted", "history_xyz", "history_rot")
                        }
                        print(
                            f"[Frame {frame_count}] Inference done: {inference_time:.2f}s "
                            f"(submitted at frame {latest_result['frame_submitted']})"
                        )
                        print(f"    CoT: {current_cot[:60]}...")
                        print(
                            f"    Selected traj sample: {current_selected_traj_idx}/"
                            f"{cfg.NUM_TRAJ_SAMPLES - 1}"
                        )
                    else:
                        print(f"[Frame {frame_count}] Inference error: {latest_result['error']}")
                        if args.debug_worker_traceback and latest_result.get("traceback"):
                            print(latest_result["traceback"].rstrip())
            else:
                if len(frame_buffer) >= cfg.NUM_FRAMES:
                    images_array = stack_frame_buffer(frame_buffer, num_cameras)
                    history_xyz, history_rot = carla_if.get_history_in_local_frame()
                    model_data = adapter.prepare_model_input(
                        images_array, history_xyz, history_rot,
                        int(frame_count * cfg.CONTROL_DT * 1_000_000),
                    )
                    if args.mode == "vqa":
                        should_run_vqa = (
                            bool(nav_state.vqa_question)
                            and nav_state.revision != last_vqa_completed_revision
                        )
                        if should_run_vqa:
                            model_start_time = time.time()
                            extra = _run_vqa_with_linalg_fallback(
                                model_data, question=nav_state.vqa_question
                            )
                            model_inference_time = time.time() - model_start_time
                            answer = adapter.extract_answer_text(extra)
                            nav_state.set_vqa_answer(answer)
                            current_inference_time = model_inference_time
                            last_vqa_completed_revision = nav_state.revision
                            print(f"[Frame {frame_count}] VQA: {model_inference_time:.2f}s")
                            print(f"    Q: {nav_state.vqa_question}")
                            print(f"    A: {format_vqa_answer_preview(answer)}")
                    else:
                        navigation_text = (
                            nav_state.navigation_text if args.mode == "navigation" else ""
                        )
                        navigation_weight = (
                            nav_state.navigation_weight if args.mode == "navigation" else 1.0
                        )
                        model_start_time = time.time()
                        pred_xyz, extra = _run_inference_with_nav_fallback(
                            model_data,
                            navigation_text=navigation_text,
                            navigation_weight=navigation_weight,
                        )
                        model_inference_time = time.time() - model_start_time
                        traj_samples = extract_trajectory_samples(pred_xyz)
                        selected_idx, _scores = select_trajectory_by_prev_similarity(
                            traj_samples, prev_selected_trajectory
                        )
                        current_selected_traj_idx = selected_idx
                        current_trajectory = traj_samples[selected_idx]
                        current_trajectory_world = trajectory_to_world(
                            carla_if.ego_vehicle.get_transform(), current_trajectory
                        )
                        prev_selected_trajectory = current_trajectory.copy()
                        current_pred_xyz = traj_samples
                        current_cot = adapter.extract_cot_text(extra)
                        current_inference_time = model_inference_time
                        current_trajectory_frame = frame_count
                        new_inference = {
                            "frame_submitted": frame_count,
                            "history_xyz": history_xyz,
                            "history_rot": history_rot,
                        }
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

            trajectory_age = (
                None if current_trajectory_frame is None
                else (frame_count - current_trajectory_frame) * cfg.CONTROL_DT
            )
            if current_trajectory is not None and trajectory_age <= cfg.TRAJECTORY_MAX_AGE_SEC:
                vehicle_tf = carla_if.ego_vehicle.get_transform()
                steering_raw, throttle_raw, brake_raw, ctrl_debug = pid_follower.compute_control(
                    vehicle_tf, current_trajectory_world, float(state["speed"])
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
                control_record = {
                    "target_speed_kmh": ctrl_debug.get("target_speed_kmh"),
                    "control_raw": [steering_raw, throttle_raw, brake_raw],
                    "control_applied": [steering, throttle, brake],
                }

                if current_pred_xyz is not None:
                    cam_img = images[viz_slot]
                    vis_frame = create_visualization_frame(
                        cam_img, current_pred_xyz, current_selected_traj_idx, frame_count,
                        current_inference_time, current_cot, state["speed"] * 3.6, steering,
                        navigation_text=nav_state.navigation_text,
                        navigation_weight=nav_state.navigation_weight,
                        paused=nav_state.paused,
                    )
                    latest_ui_frame = vis_frame
                    if cfg.SAVE_VIDEO:
                        video_recorder.add_frame(vis_frame)

                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": steering,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

                print(
                    f"[Frame {frame_count}] Speed: {state['speed']*3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}"
                )
                if args.async_mode:
                    print(f"    Trajectory age: {trajectory_age:.1f}s (sim)")
            else:
                if current_trajectory is not None:
                    print(
                        f"[Frame {frame_count}] Trajectory is {trajectory_age:.1f}s old (sim); "
                        "braking until a newer prediction arrives."
                    )
                carla_if.apply_control(0.0, 0.0, 1.0)
                control_record = {"control_applied": [0.0, 0.0, 1.0]}
                latest_telemetry = {
                    "frame": frame_count,
                    "speed_kmh": state["speed"] * 3.6,
                    "steering": 0.0,
                    "inference_time": current_inference_time,
                }
                if pygame_ui is not None:
                    draw_pygame_ui(latest_ui_frame, latest_telemetry)

            if run_recorder is not None:
                if new_inference is not None:
                    new_inference.update(
                        cot=current_cot,
                        trajectory=current_trajectory,
                        inference_time=current_inference_time,
                    )
                run_recorder.add_record({
                    "frame_id": carla_if.frame_id,
                    "frame_count": frame_count,
                    "ego": state,
                    "trajectory_age_s": trajectory_age,
                    "inference": new_inference,
                    **control_record,
                    "nearby": carla_if.get_nearby_actors(),
                })

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
        if cfg.SAVE_VIDEO and video_recorder:
            video_recorder.save()
        if pygame_ui_recorder is not None:
            pygame_ui_recorder.save()
        if run_recorder is not None:
            run_recorder.close()
        if pygame_ui is not None:
            pygame_ui.close()
        carla_if.cleanup()

    print("\nStopped.")
