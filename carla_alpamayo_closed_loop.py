"""Modular entrypoint for CARLA closed-loop control with Alpamayo."""

import argparse
import queue
import threading
import time
import traceback

import numpy as np
import torch

from module import config as cfg
from module.pid_controller import OfficialPIDFollower
from module.visualization import VideoRecorder, create_visualization_frame
from module.carla_interface import CARLAInterface
from module.inference import (
    extract_cot_text,
    extract_trajectory_samples,
    load_model,
    prepare_model_input,
    run_inference,
    select_trajectory_by_prev_similarity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CARLA closed-loop control with Alpamayo (modular).")
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Use 4-bit quantized model. Default is full-precision.",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Run internal async inference mode (non-blocking world tick).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inference_interval_sec = 1.0

    print("=" * 60)
    print("CARLA Real-time Control with Alpamayo")
    print("=" * 60)
    print(f"Quantization: {'ON (4-bit)' if args.quantization else 'OFF (full-precision)'}")
    print(f"Mode: {'ASYNC' if args.async_mode else 'SYNC'}")

    print("\nLoading model...")
    model, processor = load_model(args.quantization)
    print("Model loaded!")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

    carla_if = CARLAInterface()
    video_recorder = VideoRecorder(cfg.OUTPUT_VIDEO, fps=cfg.VIDEO_FPS) if cfg.SAVE_VIDEO else None

    try:
        carla_if.connect()
        carla_if.load_map(cfg.CARLA_MAP)
        carla_if.spawn_ego_vehicle()
        carla_if.enable_synchronous_mode()
        carla_if.spawn_npcs(num_vehicles=cfg.NPC_VEHICLE_COUNT, num_walkers=cfg.NPC_WALKER_COUNT)
        carla_if.setup_cameras()
        time.sleep(1.0)

        pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)

        current_trajectory = None
        current_pred_xyz = None
        prev_selected_trajectory = None
        current_selected_traj_idx = 0
        current_cot = ""
        current_inference_time = 0.0
        frame_buffer = []
        current_trajectory_ts = None
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}

        pending_inference = False
        last_inference_submit_ts = 0.0
        inference_request_q = None
        inference_result_q = None
        inference_stop = None
        worker_thread = None

        if args.async_mode:
            inference_request_q = queue.Queue(maxsize=1)
            inference_result_q = queue.Queue(maxsize=1)
            inference_stop = threading.Event()

            def _build_inference_request():
                images_array = np.zeros(
                    (cfg.NUM_CAMERAS, cfg.NUM_FRAMES, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS),
                    dtype=np.uint8,
                )
                for t, frame_images in enumerate(frame_buffer):
                    for c in range(cfg.NUM_CAMERAS):
                        images_array[c, t] = frame_images[c]
                history_xyz, history_rot = carla_if.get_history_in_local_frame()
                return {"images_array": images_array, "history_xyz": history_xyz, "history_rot": history_rot}

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
                        model_data = prepare_model_input(req["images_array"], req["history_xyz"], req["history_rot"])
                        pred_xyz, extra = run_inference(model, processor, model_data)
                        result = {
                            "frame_submitted": req_frame,
                            "pred_xyz": pred_xyz,
                            "extra": extra,
                            "inference_time": time.time() - t0,
                            "result_ts": time.time(),
                        }
                    except Exception as e:
                        result = {"frame_submitted": req_frame, "error": str(e), "result_ts": time.time()}

                    while True:
                        try:
                            inference_result_q.get_nowait()
                        except queue.Empty:
                            break
                    inference_result_q.put_nowait(result)

            worker_thread = threading.Thread(target=_inference_worker, name="alpamayo-inference-worker", daemon=True)
            worker_thread.start()

        print("\nStarting control loop...")
        if cfg.SAVE_VIDEO:
            print(f"Recording video to: {cfg.OUTPUT_VIDEO}")
        print("-" * 60)

        frame_count = 0

        while True:
            carla_if.tick()
            frame_count += 1

            state = carla_if.get_ego_state()
            carla_if.update_history(state)

            images = carla_if.get_camera_images()
            frame_buffer.append(images)
            if len(frame_buffer) > cfg.NUM_FRAMES:
                frame_buffer.pop(0)

            if args.async_mode:
                now_ts = time.time()
                if (
                    len(frame_buffer) >= cfg.NUM_FRAMES
                    and not pending_inference
                    and (now_ts - last_inference_submit_ts) >= inference_interval_sec
                ):
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

                latest_result = None
                while True:
                    try:
                        latest_result = inference_result_q.get_nowait()
                    except queue.Empty:
                        break
                if latest_result is not None:
                    pending_inference = False
                    if "error" not in latest_result:
                        pred_xyz = latest_result["pred_xyz"]
                        extra = latest_result["extra"]
                        inference_time = float(latest_result["inference_time"])
                        traj_samples = extract_trajectory_samples(pred_xyz)
                        selected_idx, _similarity_scores = select_trajectory_by_prev_similarity(
                            traj_samples,
                            prev_selected_trajectory,
                        )
                        current_selected_traj_idx = selected_idx
                        current_trajectory = traj_samples[selected_idx]
                        prev_selected_trajectory = current_trajectory.copy()
                        current_pred_xyz = traj_samples
                        current_cot = extract_cot_text(extra)
                        current_inference_time = inference_time
                        current_trajectory_ts = float(latest_result["result_ts"])

                        print(
                            f"[Frame {frame_count}] Inference done: {inference_time:.2f}s "
                            f"(submitted at frame {latest_result['frame_submitted']})"
                        )
                        print(f"    CoT: {current_cot[:60]}...")
                        print(f"    Selected traj sample: {current_selected_traj_idx}/{cfg.NUM_TRAJ_SAMPLES - 1}")
                        print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")
            else:
                if len(frame_buffer) >= cfg.NUM_FRAMES:
                    images_array = np.zeros(
                        (cfg.NUM_CAMERAS, cfg.NUM_FRAMES, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS),
                        dtype=np.uint8,
                    )
                    for t, frame_images in enumerate(frame_buffer):
                        for c in range(cfg.NUM_CAMERAS):
                            images_array[c, t] = frame_images[c]

                    history_xyz, history_rot = carla_if.get_history_in_local_frame()

                    model_data = prepare_model_input(images_array, history_xyz, history_rot)
                    model_start_time = time.time()
                    pred_xyz, extra = run_inference(model, processor, model_data)
                    model_inference_time = time.time() - model_start_time

                    traj_samples = extract_trajectory_samples(pred_xyz)
                    selected_idx, _similarity_scores = select_trajectory_by_prev_similarity(
                        traj_samples,
                        prev_selected_trajectory,
                    )
                    current_selected_traj_idx = selected_idx
                    current_trajectory = traj_samples[selected_idx]
                    prev_selected_trajectory = current_trajectory.copy()
                    current_pred_xyz = traj_samples
                    current_cot = extract_cot_text(extra)
                    current_inference_time = model_inference_time
                    current_trajectory_ts = time.time()

                    print(f"[Frame {frame_count}] Inference: {model_inference_time:.2f}s")
                    print(f"    CoT: {current_cot[:60]}...")
                    print(f"    Selected traj sample: {current_selected_traj_idx}/{cfg.NUM_TRAJ_SAMPLES - 1}")
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

                if cfg.SAVE_VIDEO and current_pred_xyz is not None:
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
                    )
                    video_recorder.add_frame(vis_frame)

                print(
                    f"[Frame {frame_count}] Speed: {state['speed']*3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}"
                )
                if current_trajectory_ts is not None and args.async_mode:
                    print(f"    Trajectory age: {time.time() - current_trajectory_ts:.2f}s")
            else:
                carla_if.apply_control(0.0, 0.3, 0.0)

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
        carla_if.cleanup()

    print("\nStopped.")


if __name__ == "__main__":
    main()
