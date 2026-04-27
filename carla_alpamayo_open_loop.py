# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: MIT
#
# Open-loop inference script for CARLA-collected data with Alpamayo 1.5.

import os
import shutil
import json
import copy
import time
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
from einops import rearrange
from transformers import BitsAndBytesConfig

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper

from module.alpamayo_compat import patch_legacy_hydra_targets
from module.visualization import transcode_video_for_browser_compat

patch_legacy_hydra_targets()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo open-loop inference on CARLA collected data."
    )
    parser.add_argument(
        "--data-root",
        default="carla_data",
        help="Path to CARLA dataset root containing trajectory.json and camera folders.",
    )
    parser.add_argument(
        "--quantization",
        dest="use_quantization",
        action="store_true",
        help="Use 4-bit quantized model (default is full-precision).",
    )
    parser.set_defaults(use_quantization=False)
    return parser.parse_args()


def project_trajectory_to_camera_plt(cam_img, pred_xyz):
    """Project trajectory onto camera using matplotlib (test_inference_gif.py style)."""
    img_height, img_width = cam_img.shape[:2]

    # Camera parameters (CARLA cam_front_wide: 120 FOV)
    camera_height = 2.4
    fov = 120
    focal_length_px = img_width / (2 * np.tan(np.radians(fov / 2)))

    def project_to_image(xyz_points):
        """Perspective projection from ego frame to image coordinates."""
        x, y, z = xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]
        z_cam = z + camera_height
        valid = x > 0.5

        with np.errstate(divide='ignore', invalid='ignore'):
            u = img_width / 2 - (y / x) * focal_length_px
            v_temp = img_height / 2 - (z_cam / x) * focal_length_px
            v = img_height - v_temp

        u = np.clip(u, 0, img_width - 1)
        v = np.clip(v, 0, img_height - 1)

        return u, v, valid

    # Use all trajectory samples
    num_samples = pred_xyz.shape[2]
    all_points = []

    for i in range(num_samples):
        pred_points = pred_xyz.cpu()[0, 0, i, :, :3].numpy()
        u, v, valid = project_to_image(pred_points)
        points = np.column_stack([u[valid], v[valid]])
        if len(points) > 1:
            all_points.append(points)

    # Draw on image using matplotlib for consistent style
    fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
    ax.imshow(cam_img)
    ax.axis('off')

    # Draw trajectories (red like test_inference_gif.py)
    for points in all_points:
        ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=6, alpha=0.9)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    # Resize back to original dimensions
    img_array = cv2.resize(img_array, (img_width, img_height))

    return img_array


def add_text_overlay_gif_style(img, frame_idx, total_frames, inference_time, cot_text):
    """Add text overlay in test_inference_gif.py style using matplotlib."""
    import textwrap

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img)
    ax.axis('off')

    # Top center: Frame info and inference time
    title_str = f'Frame: {frame_idx+1}/{total_frames} | Inference: {inference_time:.2f}s'
    ax.text(0.5, 0.98, title_str, transform=ax.transAxes,
           fontsize=13, color='white', weight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
           ha='center', va='top')

    # Bottom left: Chain-of-Causation
    wrapped_cot = '\n'.join(textwrap.wrap(cot_text, width=80))
    ax.text(0.02, 0.02, f'Chain-of-Causation:\n{wrapped_cot}',
           transform=ax.transAxes,
           fontsize=12, color='white', weight='normal',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
           ha='left', va='bottom',
           family='monospace')

    plt.tight_layout()
    fig.canvas.draw()

    # Convert to image
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img_array


def run_inference_single(model, processor, data):
    """Run single inference and return results."""
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

    torch.cuda.manual_seed_all(42)
    inference_start = time.time()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1, # 예측 경로 개수
            diffusion_kwargs={"inference_step": 10}, # diffusion step 수
            max_generation_length=256,
            return_extra=True,
        )

    inference_time = time.time() - inference_start
    cot_text = extra["cot"][0][0][0]

    return pred_xyz, cot_text, inference_time


def main():
    args = parse_args()
    data_root = args.data_root
    use_quantization = args.use_quantization
    num_frames_for_history = 4

    print("=" * 60)
    print("CARLA -> Alpamayo-R1 Inference")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Quantization: {'ON (4-bit)' if use_quantization else 'OFF (full-precision)'}")

    # Load model
    print("\nLoading model...")
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16
        ).to("cuda")

    print("Model loaded!")
    processor = helper.get_processor(model.tokenizer)

    # Load trajectory
    json_path = os.path.join(data_root, "trajectory.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Trajectory file not found: {json_path}")

    with open(json_path, 'r') as f:
        traj_data = json.load(f)
    sorted_frame_ids = sorted([int(k) for k in traj_data.keys()])
    total_frames = len(sorted_frame_ids)

    print(f"\nTotal frames: {total_frames}")
    print(f"Skipping first {num_frames_for_history} frames for history")

    # Results
    all_predictions = []
    all_cots = []
    all_inference_times = []
    all_camera_images = []

    start_idx = num_frames_for_history - 1

    for t0_idx in range(start_idx, total_frames):
        frame_num = t0_idx - start_idx + 1
        print(f"  Frame {frame_num}/{total_frames - start_idx} (idx={t0_idx})")

        # Load data for this frame
        data = load_carla_dataset_at_index(data_root, traj_data, sorted_frame_ids, t0_idx)

        # Load camera image for visualization
        fid = sorted_frame_ids[t0_idx]
        cam_path = os.path.join(data_root, "cam_front_wide", f"{fid:06d}.jpg")
        cam_img = np.array(Image.open(cam_path).convert('RGB'))
        all_camera_images.append(cam_img)

        # Run inference
        pred_xyz, cot_text, inference_time = run_inference_single(model, processor, data)

        all_predictions.append(pred_xyz)
        all_cots.append(cot_text)
        all_inference_times.append(inference_time)

        print(f"    Inference: {inference_time:.2f}s | CoC: {cot_text[:50]}...")

    # Summary
    avg_time = np.mean(all_inference_times) if all_inference_times else 0
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total frames processed: {len(all_predictions)}")
    print(f"Average inference time: {avg_time:.2f}s/frame")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Create video
    if all_predictions:
        output_video = "carla_alpamayo_open_loop_result.mp4"
        create_video_full(all_predictions, all_camera_images, all_cots,
                         all_inference_times, output_video)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def load_carla_dataset_at_index(data_root, traj_data, sorted_frame_ids, t0_idx,
                                 num_history_steps=16, num_frames=4):
    """Load CARLA data at a specific frame index."""
    total_frames = len(sorted_frame_ids)

    # Get t0 pose for reference
    t0_fid = str(sorted_frame_ids[t0_idx])
    t0_pose = traj_data[t0_fid]
    t0_pos = np.array([t0_pose['x'], t0_pose['y'], t0_pose['z']])
    t0_rot = R.from_euler('xyz', [t0_pose['roll'], t0_pose['pitch'], -t0_pose['yaw']], degrees=True)
    t0_rot_inv = t0_rot.inv()

    # Build history trajectory
    history_xyz = []
    history_rot = []
    for i in range(num_history_steps):
        hist_idx = t0_idx - (num_history_steps - 1 - i)
        hist_idx = max(0, hist_idx)
        fid = str(sorted_frame_ids[hist_idx])
        pose = traj_data[fid]
        pos = np.array([pose['x'], pose['y'], pose['z']])
        rot = R.from_euler('xyz', [pose['roll'], pose['pitch'], -pose['yaw']], degrees=True)
        rel_pos = t0_rot_inv.apply(pos - t0_pos)
        rel_rot = (t0_rot_inv * rot).as_matrix()
        history_xyz.append(rel_pos)
        history_rot.append(rel_rot)

    # Load camera images
    camera_order = ["cam_front_left", "cam_front_wide", "cam_front_right", "cam_front_tele"]
    image_frames_list = []

    for cam_name in camera_order:
        cam_dir = os.path.join(data_root, cam_name)
        frames_for_cam = []
        for i in range(num_frames):
            frame_idx = t0_idx - (num_frames - 1 - i)
            frame_idx = max(0, frame_idx)
            fid = sorted_frame_ids[frame_idx]
            img_path = os.path.join(cam_dir, f"{fid:06d}.jpg")
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            frames_for_cam.append(img_array)
        cam_frames = np.stack(frames_for_cam, axis=0)
        cam_tensor = torch.from_numpy(cam_frames)
        cam_tensor = rearrange(cam_tensor, "t h w c -> t c h w")
        image_frames_list.append(cam_tensor)

    image_frames = torch.stack(image_frames_list, dim=0)
    ego_history_xyz = torch.tensor(np.array(history_xyz), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.tensor(np.array(history_rot), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return {
        "image_frames": image_frames,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def create_video_full(all_predictions, camera_images, all_cots, all_inference_times, output_path):
    """Create video from all inference results."""
    print("\nCreating video...")

    total_frames = len(all_predictions)
    video_frames = []

    for frame_idx in range(total_frames):
        pred_xyz = all_predictions[frame_idx]
        cam_img = camera_images[frame_idx].copy()
        cot_text = all_cots[frame_idx]
        inference_time = all_inference_times[frame_idx]

        # Project trajectory onto camera image
        cam_img_with_traj = project_trajectory_to_camera_plt(cam_img, pred_xyz)

        # Add text overlay
        final_img = add_text_overlay_gif_style(
            cam_img_with_traj, frame_idx, total_frames, inference_time, cot_text
        )

        video_frames.append(final_img)

        if frame_idx % 10 == 0:
            print(f"  Rendering frame {frame_idx+1}/{total_frames}...")

    # Save an OpenCV MP4 first, then transcode to H.264/yuv420p when ffmpeg is available.
    video_fps = 5
    h, w = video_frames[0].shape[:2]
    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    temp_path = os.path.join(output_dir, f".{os.path.basename(output_path)}.opencv-tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(temp_path, fourcc, video_fps, (w, h))

    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()

    transcoded, transcode_msg = transcode_video_for_browser_compat(temp_path, output_path)
    if not transcoded:
        shutil.move(temp_path, output_path)
        print(
            f"Warning: H.264 transcode skipped ({transcode_msg}); "
            "saved OpenCV mp4v output."
        )
    else:
        os.remove(temp_path)

    print(f"Video saved: {output_path}")
    print(
        f"  Codec: {transcode_msg if transcoded else 'mp4v'}, "
        f"Resolution: {w}x{h}, FPS: {video_fps}, Frames: {len(video_frames)}"
    )
    print(f"  Total frames: {total_frames}")

    # GIF 생성 (주석처리)
    # gif_path = output_path.replace('.mp4', '.gif')
    # gif_frames = [Image.fromarray(f) for f in video_frames]
    # gif_frames[0].save(
    #     gif_path,
    #     save_all=True,
    #     append_images=gif_frames[1:],
    #     duration=200,
    #     loop=0
    # )
    # print(f"GIF saved: {gif_path}")


if __name__ == "__main__":
    main()
