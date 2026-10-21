# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Live open-loop: CARLA autopilot drives; Alpamayo observes and predicts.

"""Live open-loop runner: real-time CARLA with an autopilot-driven ego.

Unlike open-loop (offline replay) and closed-loop (model drives via PID), this loop
lets the CARLA Traffic Manager autopilot drive the ego vehicle while the selected
Alpamayo model runs live on the same camera stream. The model's predicted trajectory
and Chain-of-Thought are overlaid against the autopilot's actual control, so you can
watch the model open-loop in a live simulation without handing it the wheel.
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
from module.loops._common import collect_oom_kwargs, model_max_memory, stack_frame_buffer
from module.visualization import VideoRecorder, create_live_open_loop_visualization_frame
from module.vlm_generate_optimization import VlmGenerateTiming


def _check_capabilities(adapter, args):
    if args.oom_free and not adapter.supports_oom_free:
        raise SystemExit(
            f"{adapter.display_name} does not support OOM-free demand layering "
            "(available for --version 1.5 only)."
        )
    if args.oom_free and args.quantization:
        raise SystemExit("--oom-free and --quantization are mutually exclusive.")
    if args.navigation_text and not adapter.supports_navigation:
        print(
            f"{adapter.display_name} has no navigation conditioning; "
            "ignoring --navigation-text."
        )


def run(adapter, args):
    _check_capabilities(adapter, args)
    viz_slot = adapter.viz_camera_slot
    num_cameras = adapter.num_cameras
    output_video = args.output_video or cfg.LIVE_OPEN_LOOP_OUTPUT_VIDEO
    inference_interval_sec = 1.0
    nav_text = args.navigation_text if adapter.supports_navigation else ""

    print("=" * 60)
    print(f"CARLA Live Open-Loop with {adapter.display_name} (autopilot driving)")
    print("=" * 60)
    if args.oom_free:
        print("Model loading: OOM-free CPU<->GPU demand layering (full-precision)")
    else:
        print(f"Quantization: {'ON (4-bit)' if args.quantization else 'OFF (full-precision)'}")
    print(f"Execution: {'ASYNC' if args.async_mode else 'SYNC'}")
    print("Control: CARLA Traffic Manager autopilot (model does NOT drive)")
    print(f"CARLA map: {cfg.CARLA_MAP}")
    print(f"Cameras: {num_cameras}")
    if nav_text:
        print(f"Navigation prompt (observation only): {nav_text}")

    print("\nLoading model...")
    configure_cuda_linalg_library(args.cuda_linalg_library)
    oom_kwargs = collect_oom_kwargs(args)
    model = processor = None
    if not args.oom_free:
        model, processor = adapter.load_model(
            use_quantization=args.quantization, device_map=args.device_map,
            max_memory=model_max_memory(args),
        )
        print("Model loaded!")
        print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")
    else:
        print("OOM-free mode: Alpamayo loads after CARLA is fully spawned.")

    carla_if = CARLAInterface(camera_configs=adapter.source_camera_configs)
    video_recorder = VideoRecorder(output_video, fps=cfg.VIDEO_FPS) if cfg.SAVE_VIDEO else None
    vlm_generate_timing = VlmGenerateTiming()
    run_recorder = None

    current_pred_xyz = None
    current_selected_traj_idx = 0
    prev_selected_trajectory = None
    current_cot = ""
    current_inference_time = 0.0
    new_inference = None
    frame_buffer = []

    pending_inference = False
    last_inference_submit_ts = 0.0
    inference_request_q = inference_result_q = inference_stop = worker_thread = None

    def _run_inference_with_linalg_fallback(model_data):
        def _run_once():
            torch.manual_seed(cfg.INFERENCE_SEED)
            torch.cuda.manual_seed_all(cfg.INFERENCE_SEED)
            return adapter.run_inference(
                model, processor, model_data,
                navigation_text=nav_text or None, navigation_weight=1.0,
                vlm_generate_timing=vlm_generate_timing,
            )

        try:
            return _run_once()
        except RuntimeError as exc:
            if "CUSOLVER_STATUS_INTERNAL_ERROR" not in str(exc):
                raise
            print("cuSOLVER linalg backend failed; switching to MAGMA and retrying once.")
            configure_cuda_linalg_library("magma")
            return _run_once()

    def _ingest_result(pred_xyz, extra, inference_time, submitted_frame, history_xyz, history_rot):
        nonlocal current_pred_xyz, current_selected_traj_idx, prev_selected_trajectory
        nonlocal current_cot, current_inference_time, new_inference
        traj_samples = extract_trajectory_samples(pred_xyz)
        selected_idx, _scores = select_trajectory_by_prev_similarity(
            traj_samples, prev_selected_trajectory
        )
        current_selected_traj_idx = selected_idx
        prev_selected_trajectory = traj_samples[selected_idx].copy()
        current_pred_xyz = traj_samples
        current_cot = adapter.extract_cot_text(extra)
        current_inference_time = inference_time
        new_inference = {
            "frame_submitted": submitted_frame,
            "history_xyz": history_xyz,
            "history_rot": history_rot,
            "cot": current_cot,
            "trajectory": prev_selected_trajectory,
            "inference_time": inference_time,
        }
        print(f"[Frame {frame_count}] Inference: {inference_time:.2f}s "
              f"(submitted at frame {submitted_frame})")
        print(f"    CoT: {current_cot[:70]}")

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

        # Hand the wheel to CARLA's Traffic Manager autopilot.
        carla_if.set_ego_autopilot(True)
        print("Ego autopilot: ON (Traffic Manager driving)")

        if args.async_mode:
            inference_request_q = queue.Queue(maxsize=1)
            inference_result_q = queue.Queue(maxsize=1)
            inference_stop = threading.Event()

            def _inference_worker():
                while not inference_stop.is_set():
                    try:
                        req = inference_request_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if req is None:
                        break
                    try:
                        t0 = time.time()
                        model_data = adapter.prepare_model_input(
                            req["images_array"], req["history_xyz"], req["history_rot"],
                            req["t0_us"],
                        )
                        pred_xyz, extra = _run_inference_with_linalg_fallback(model_data)
                        result = {
                            "pred_xyz": pred_xyz, "extra": extra,
                            "inference_time": time.time() - t0,
                            "frame_submitted": int(req["frame"]),
                            "history_xyz": req["history_xyz"],
                            "history_rot": req["history_rot"],
                        }
                    except Exception as e:
                        result = {"error": str(e), "traceback": traceback.format_exc()}
                    while True:
                        try:
                            inference_result_q.get_nowait()
                        except queue.Empty:
                            break
                    inference_result_q.put_nowait(result)

            worker_thread = threading.Thread(
                target=_inference_worker, name="alpamayo-live-worker", daemon=True
            )
            worker_thread.start()

        print("\nStarting live open-loop...")
        if cfg.SAVE_VIDEO:
            print(f"Recording video to: {output_video}")
        print("-" * 60)

        frame_count = 0
        while True:
            carla_if.tick()
            frame_count += 1

            state = carla_if.get_ego_state()
            carla_if.update_history(state)
            control = carla_if.get_ego_control()

            if carla_if.get_collision_count() > 0:
                event = carla_if.get_last_collision_event()
                if event is not None:
                    print(f"[Frame {frame_count}] Collision logged: actor="
                          f"{event['other_actor']} impulse={event['intensity']:.1f}")
                carla_if.reset_collision_history()

            try:
                images = carla_if.get_camera_images()
            except TimeoutError as exc:
                print(f"[Frame {frame_count}] Warning: {exc}; skipping this tick.")
                continue

            frame_buffer.append(images)
            if len(frame_buffer) > cfg.NUM_FRAMES:
                frame_buffer.pop(0)
            if run_recorder is not None:
                run_recorder.add_frame(carla_if.frame_id, images, state)
            new_inference = None

            if len(frame_buffer) >= cfg.NUM_FRAMES:
                if args.async_mode:
                    now_ts = time.time()
                    due = (now_ts - last_inference_submit_ts) >= inference_interval_sec
                    if not pending_inference and due:
                        history_xyz, history_rot = carla_if.get_history_in_local_frame()
                        req = {
                            "images_array": stack_frame_buffer(frame_buffer, num_cameras),
                            "history_xyz": history_xyz,
                            "history_rot": history_rot,
                            "t0_us": int(frame_count * cfg.CONTROL_DT * 1_000_000),
                            "frame": int(frame_count),
                        }
                        while True:
                            try:
                                inference_request_q.get_nowait()
                            except queue.Empty:
                                break
                        inference_request_q.put_nowait(req)
                        pending_inference = True
                        last_inference_submit_ts = now_ts

                    latest_result = None
                    while True:
                        try:
                            latest_result = inference_result_q.get_nowait()
                        except queue.Empty:
                            break
                    if latest_result is not None:
                        pending_inference = False
                        if "error" in latest_result:
                            print(f"[Frame {frame_count}] Inference error: "
                                  f"{latest_result['error']}")
                            if args.debug_worker_traceback and latest_result.get("traceback"):
                                print(latest_result["traceback"].rstrip())
                        else:
                            _ingest_result(
                                latest_result["pred_xyz"], latest_result["extra"],
                                float(latest_result["inference_time"]),
                                latest_result["frame_submitted"],
                                latest_result["history_xyz"],
                                latest_result["history_rot"],
                            )
                else:
                    images_array = stack_frame_buffer(frame_buffer, num_cameras)
                    history_xyz, history_rot = carla_if.get_history_in_local_frame()
                    model_data = adapter.prepare_model_input(
                        images_array, history_xyz, history_rot,
                        int(frame_count * cfg.CONTROL_DT * 1_000_000),
                    )
                    model_start = time.time()
                    pred_xyz, extra = _run_inference_with_linalg_fallback(model_data)
                    _ingest_result(
                        pred_xyz, extra, time.time() - model_start, frame_count,
                        history_xyz, history_rot,
                    )

            if current_pred_xyz is not None and cfg.SAVE_VIDEO:
                vis_frame = create_live_open_loop_visualization_frame(
                    images[viz_slot], current_pred_xyz, current_selected_traj_idx,
                    frame_count, current_inference_time, current_cot,
                    state["speed"] * 3.6, control,
                )
                video_recorder.add_frame(vis_frame)

            print(
                f"[Frame {frame_count}] Autopilot -> Speed: {state['speed']*3.6:.1f} km/h, "
                f"Steer: {control['steer']:.4f}, Throttle: {control['throttle']:.3f}, "
                f"Brake: {control['brake']:.3f}"
            )
            if run_recorder is not None:
                run_recorder.add_record({
                    "frame_id": carla_if.frame_id,
                    "frame_count": frame_count,
                    "ego": state,
                    "inference": new_inference,
                    "autopilot_control": control,
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
        if carla_if.ego_vehicle is not None:
            try:
                carla_if.set_ego_autopilot(False)
            except Exception:
                pass
        if cfg.SAVE_VIDEO and video_recorder:
            video_recorder.save()
        if run_recorder is not None:
            run_recorder.close()
        carla_if.cleanup()

    print("\nStopped.")
