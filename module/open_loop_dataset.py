"""Dataset loading helpers for open-loop CARLA inference."""

import json
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from . import config as cfg

OPEN_LOOP_CAMERA_ORDER = (
    "cam_front_left",
    "cam_front_wide",
    "cam_front_right",
    "cam_front_tele",
)
FRONT_CAMERA_NAME = "cam_front_wide"


def load_trajectory_index(data_root):
    """Load ``trajectory.json`` and return its sorted integer frame IDs."""

    json_path = os.path.join(data_root, "trajectory.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Trajectory file not found: {json_path}")

    with open(json_path) as f:
        trajectory = json.load(f)

    frame_ids = sorted(int(frame_id) for frame_id in trajectory)
    return trajectory, frame_ids


def _pose_to_position_and_rotation(pose):
    position = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
    rotation = R.from_euler(
        "xyz",
        [pose["roll"], pose["pitch"], -pose["yaw"]],
        degrees=True,
    )
    return position, rotation


def build_ego_history(trajectory, frame_ids, t0_index, num_history_steps=cfg.NUM_HISTORY):
    """Build ego pose history in the current frame's local coordinate system."""

    t0_frame_id = str(frame_ids[t0_index])
    t0_position, t0_rotation = _pose_to_position_and_rotation(trajectory[t0_frame_id])
    t0_rotation_inv = t0_rotation.inv()

    history_xyz = []
    history_rot = []
    for offset in range(num_history_steps):
        history_index = max(0, t0_index - (num_history_steps - 1 - offset))
        frame_id = str(frame_ids[history_index])
        position, rotation = _pose_to_position_and_rotation(trajectory[frame_id])
        history_xyz.append(t0_rotation_inv.apply(position - t0_position))
        history_rot.append((t0_rotation_inv * rotation).as_matrix())

    return (
        np.asarray(history_xyz, dtype=np.float32),
        np.asarray(history_rot, dtype=np.float32),
    )


def load_camera_history(
    data_root,
    frame_ids,
    t0_index,
    camera_order=OPEN_LOOP_CAMERA_ORDER,
    num_frames=cfg.NUM_FRAMES,
):
    """Load camera history as ``camera, time, height, width, channel`` arrays."""

    camera_frames = []
    for camera_name in camera_order:
        frames = []
        camera_dir = os.path.join(data_root, camera_name)
        for offset in range(num_frames):
            frame_index = max(0, t0_index - (num_frames - 1 - offset))
            frame_id = frame_ids[frame_index]
            image_path = os.path.join(camera_dir, f"{frame_id:06d}.jpg")
            image = Image.open(image_path).convert("RGB")
            frames.append(np.asarray(image, dtype=np.uint8))
        camera_frames.append(np.stack(frames, axis=0))

    return np.stack(camera_frames, axis=0)


def load_front_camera_image(data_root, frame_id, camera_name=FRONT_CAMERA_NAME):
    """Load the front camera image used for open-loop visualization."""

    image_path = os.path.join(data_root, camera_name, f"{frame_id:06d}.jpg")
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def load_open_loop_arrays(
    data_root,
    trajectory,
    frame_ids,
    t0_index,
    num_history_steps=cfg.NUM_HISTORY,
    num_frames=cfg.NUM_FRAMES,
):
    """Load images and ego history arrays for one open-loop inference step."""

    image_frames = load_camera_history(
        data_root,
        frame_ids,
        t0_index,
        num_frames=num_frames,
    )
    history_xyz, history_rot = build_ego_history(
        trajectory,
        frame_ids,
        t0_index,
        num_history_steps=num_history_steps,
    )
    return {
        "image_frames": image_frames,
        "history_xyz": history_xyz,
        "history_rot": history_rot,
    }
