"""Visualization and video recording helpers."""

import os
import shutil
import subprocess
import textwrap

import cv2
import numpy as np
import torch


def _project_one_trajectory(
    result,
    points_3d,
    img_width,
    img_height,
    focal_length_px,
    camera_height,
    line_color,
    point_color,
    line_thickness,
):
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_cam = z + camera_height
    valid = x > 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        u = img_width / 2 - (y / x) * focal_length_px
        v_temp = img_height / 2 - (z_cam / x) * focal_length_px
        v = img_height - v_temp

    u = np.clip(u, 0, img_width - 1).astype(np.int32)
    v = np.clip(v, 0, img_height - 1).astype(np.int32)
    points_2d = np.column_stack([u[valid], v[valid]])
    if len(points_2d) <= 1:
        return

    for i in range(len(points_2d) - 1):
        cv2.line(
            result,
            tuple(points_2d[i]),
            tuple(points_2d[i + 1]),
            line_color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    for pt in points_2d:
        cv2.circle(result, tuple(pt), max(4, line_thickness), point_color, -1, cv2.LINE_AA)


def project_trajectory_to_image(cam_img, pred_xyz, selected_idx=0, camera_height=2.4, fov=120):
    """Project one or multiple trajectories onto image."""
    img_height, img_width = cam_img.shape[:2]
    focal_length_px = img_width / (2 * np.tan(np.radians(fov / 2)))

    result = cam_img.copy()
    if isinstance(pred_xyz, torch.Tensor):
        arr = pred_xyz.detach().cpu().numpy()
    else:
        arr = np.asarray(pred_xyz)

    if arr.ndim == 2:
        traj_samples = arr[None, :, :3]
    elif arr.ndim == 3:
        traj_samples = arr[:, :, :3]
    else:
        raise ValueError(f"Expected trajectory with ndim 2 or 3, got shape {arr.shape}")

    num_samples = traj_samples.shape[0]
    selected_idx = int(np.clip(selected_idx, 0, max(0, num_samples - 1)))

    for i in range(num_samples):
        if i == selected_idx:
            continue
        _project_one_trajectory(
            result=result,
            points_3d=traj_samples[i],
            img_width=img_width,
            img_height=img_height,
            focal_length_px=focal_length_px,
            camera_height=camera_height,
            line_color=(255, 255, 255),
            point_color=(255, 255, 255),
            line_thickness=4,
        )

    _project_one_trajectory(
        result=result,
        points_3d=traj_samples[selected_idx],
        img_width=img_width,
        img_height=img_height,
        focal_length_px=focal_length_px,
        camera_height=camera_height,
        line_color=(255, 0, 0),
        point_color=(255, 100, 100),
        line_thickness=8,
    )

    return result


def create_visualization_frame(
    cam_img,
    pred_xyz,
    selected_idx,
    frame_count,
    inference_time,
    cot_text,
    speed_kmh,
    steering,
    navigation_text="",
    navigation_weight=1.0,
    paused=False,
):
    """Create a single visualization frame with all overlays."""
    vis_img = project_trajectory_to_image(cam_img, pred_xyz, selected_idx=selected_idx)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    h, w = vis_img.shape[:2]

    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, h - 190), (w - 10, h - 10), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    info_text = (
        f"Frame: {frame_count} | Inference: {inference_time:.2f}s | "
        f"Speed: {speed_kmh:.1f} km/h | Steer: {steering:.2f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
    pad_x = 14
    pad_y = 14
    box_x1, box_y1 = 10, 10
    box_x2 = min(w - 10, box_x1 + tw + pad_x * 2)
    box_y2 = box_y1 + th + pad_y * 2
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    cv2.putText(
        vis_img,
        info_text,
        (20, 50),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    status = "PAUSED" if paused else "RUNNING"
    nav_display = navigation_text or "(no navigation text)"
    nav_display = nav_display[:160] + "..." if len(nav_display) > 160 else nav_display
    cv2.putText(
        vis_img,
        f"{status} | Nav: {nav_display} | Weight: {navigation_weight:.2f}",
        (20, h - 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cot_display = cot_text[:200] + "..." if len(cot_text) > 200 else cot_text
    lines = textwrap.wrap(f"CoT: {cot_display}", width=120)
    y_offset = h - 115
    for line in lines[:3]:
        cv2.putText(
            vis_img,
            line,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


def create_open_loop_visualization_frame(
    cam_img,
    pred_xyz,
    frame_count,
    total_frames,
    inference_time,
    cot_text,
):
    """Create one open-loop visualization frame with trajectory and text overlays."""

    vis_img = project_trajectory_to_image(cam_img, pred_xyz, selected_idx=0)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    height, width = vis_img.shape[:2]

    header = f"Frame: {frame_count}/{total_frames} | Inference: {inference_time:.2f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(header, font, font_scale, thickness)
    pad_x = 14
    pad_y = 14
    box_x1 = 10
    box_y1 = 10
    box_x2 = min(width - 10, box_x1 + text_width + pad_x * 2)
    box_y2 = box_y1 + text_height + pad_y * 2
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.65, vis_img, 0.35, 0)
    cv2.putText(
        vis_img,
        header,
        (box_x1 + pad_x, box_y1 + pad_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, height - 170), (width - 10, height - 10), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    cot_display = str(cot_text or "").strip()
    cot_display = cot_display[:240] + "..." if len(cot_display) > 240 else cot_display
    lines = textwrap.wrap(f"Chain-of-Causation: {cot_display}", width=120)
    y_offset = height - 125
    for line in lines[:4]:
        cv2.putText(
            vis_img,
            line,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 30

    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


def save_open_loop_video(
    predictions,
    camera_images,
    cot_texts,
    inference_times,
    output_path,
    fps=5,
):
    """Render and save the open-loop inference summary video."""

    total_frames = len(predictions)
    recorder = VideoRecorder(output_path, fps=fps)
    for frame_index, (pred_xyz, cam_img, cot_text, inference_time) in enumerate(
        zip(predictions, camera_images, cot_texts, inference_times, strict=True),
        start=1,
    ):
        recorder.add_frame(
            create_open_loop_visualization_frame(
                cam_img=cam_img,
                pred_xyz=pred_xyz,
                frame_count=frame_index,
                total_frames=total_frames,
                inference_time=inference_time,
                cot_text=cot_text,
            )
        )
        if frame_index == 1 or frame_index % 10 == 0 or frame_index == total_frames:
            print(f"  Rendering frame {frame_index}/{total_frames}...")

    recorder.save()


def transcode_video_for_browser_compat(source_path, output_path):
    """Transcode OpenCV output to H.264/yuv420p for VS Code/browser players."""
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg not found"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "fast",
        "-crf",
        "18",
        output_path,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False, result.stderr.strip() or "ffmpeg failed"
    return True, "H.264/yuv420p"


class VideoRecorder:
    """Records frames and saves to video file."""

    def __init__(self, output_path, fps=10):
        self.output_path = output_path
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def _create_writer(self, width, height, output_path):
        for codec in ("mp4v", "avc1", "H264"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, None

    def save(self):
        if not self.frames:
            print("No frames to save.")
            return

        print(f"\nSaving video with {len(self.frames)} frames...")
        h, w = self.frames[0].shape[:2]
        output_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(
            output_dir,
            f".{os.path.basename(self.output_path)}.opencv-tmp.mp4",
        )

        writer, selected_codec = self._create_writer(w, h, temp_path)
        if writer is None:
            print("Failed to initialize video writer.")
            return
        if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

        transcoded, transcode_msg = transcode_video_for_browser_compat(temp_path, self.output_path)
        if not transcoded:
            shutil.move(temp_path, self.output_path)
            print(
                f"Warning: H.264 transcode skipped ({transcode_msg}); "
                f"saved OpenCV {selected_codec} output."
            )
        else:
            os.remove(temp_path)

        print(f"Video saved: {self.output_path}")
        print(
            f"  Codec: {transcode_msg if transcoded else selected_codec}, "
            f"Resolution: {w}x{h}, FPS: {self.fps}, Frames: {len(self.frames)}"
        )
