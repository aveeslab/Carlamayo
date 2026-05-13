import json

import numpy as np
from PIL import Image

from module.open_loop_dataset import (
    OPEN_LOOP_CAMERA_ORDER,
    build_ego_history,
    load_camera_history,
    load_open_loop_arrays,
    load_trajectory_index,
)


def _write_rgb_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_load_trajectory_index_sorts_numeric_frame_ids(tmp_path):
    trajectory = {
        "10": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        "2": {"x": 1, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))

    loaded_trajectory, frame_ids = load_trajectory_index(tmp_path)

    assert loaded_trajectory == trajectory
    assert frame_ids == [2, 10]


def test_build_ego_history_clamps_early_frames_and_ends_at_origin():
    trajectory = {
        "1": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        "2": {"x": 10, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
    }

    history_xyz, history_rot = build_ego_history(
        trajectory,
        [1, 2],
        t0_index=1,
        num_history_steps=3,
    )

    np.testing.assert_allclose(history_xyz[:, 0], [-10.0, -10.0, 0.0])
    np.testing.assert_allclose(history_xyz[-1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(history_rot[-1], np.eye(3), atol=1e-6)


def test_load_open_loop_arrays_preserves_camera_and_frame_order(tmp_path):
    frame_ids = [1, 2]
    trajectory = {
        "1": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
        "2": {"x": 1, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
    }
    for camera_index, camera_name in enumerate(OPEN_LOOP_CAMERA_ORDER):
        for frame_id in frame_ids:
            value = camera_index * 10 + frame_id
            _write_rgb_image(tmp_path / camera_name / f"{frame_id:06d}.jpg", value)

    camera_history = load_camera_history(tmp_path, frame_ids, t0_index=1, num_frames=2)
    sample = load_open_loop_arrays(
        tmp_path,
        trajectory,
        frame_ids,
        t0_index=1,
        num_history_steps=2,
        num_frames=2,
    )

    assert camera_history.shape == (4, 2, 2, 3, 3)
    assert camera_history[0, 0, 0, 0, 0] == 1
    assert camera_history[0, 1, 0, 0, 0] == 2
    assert camera_history[3, 1, 0, 0, 0] == 32
    assert sample["image_frames"].shape == (4, 2, 2, 3, 3)
    assert sample["history_xyz"].shape == (2, 3)
    assert sample["history_rot"].shape == (2, 3, 3)
