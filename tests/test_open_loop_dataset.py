import json

import numpy as np
from PIL import Image

from module.open_loop_dataset import (
    OPEN_LOOP_CAMERA_ORDER,
    _pose_to_position_and_rotation,
    build_ego_history,
    load_camera_history,
    load_open_loop_arrays,
    load_trajectory_index,
)


def _pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    return {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw}


def _carla_rotation_matrix(roll, pitch, yaw):
    """Rotation block of CARLA's Transform::GetMatrix (left-handed, degrees)."""
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])


def test_pose_conversion_is_carla_pose_mirrored_into_flu():
    mirror = np.diag([1.0, -1.0, 1.0])
    for roll, pitch, yaw in [(5.0, 10.0, 30.0), (-20.0, 7.0, -135.0), (3.0, -15.0, 90.0)]:
        position, rotation = _pose_to_position_and_rotation(
            _pose(1.0, 2.0, 3.0, roll, pitch, yaw)
        )
        expected = mirror @ _carla_rotation_matrix(roll, pitch, yaw) @ mirror
        np.testing.assert_allclose(rotation.as_matrix(), expected, atol=1e-6)
        np.testing.assert_allclose(position, [1.0, -2.0, 3.0])


def test_build_ego_history_is_in_alpamayo_flu_frame():
    # Heading CARLA yaw=90 after driving 10 m: the previous pose is 10 m behind.
    history_xyz, _ = build_ego_history(
        {"1": _pose(y=-10.0, yaw=90.0), "2": _pose(yaw=90.0)}, [1, 2], 1, num_history_steps=2
    )
    np.testing.assert_allclose(history_xyz[0], [-10.0, 0.0, 0.0], atol=1e-5)

    # A pose on CARLA's left (y=-3) that was heading 10 deg further left lies at
    # Alpamayo +y with a positive relative yaw.
    history_xyz, history_rot = build_ego_history(
        {"1": _pose(y=-3.0, yaw=-10.0), "2": _pose()}, [1, 2], 1, num_history_steps=2
    )
    np.testing.assert_allclose(history_xyz[0], [0.0, 3.0, 0.0], atol=1e-5)
    rel_yaw = np.degrees(np.arctan2(history_rot[0][1, 0], history_rot[0][0, 0]))
    np.testing.assert_allclose(rel_yaw, 10.0, atol=1e-4)


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
    # The newest frame's clip-relative timestamp comes from the fixed control tick.
    assert sample["t0_us"] == 100_000
