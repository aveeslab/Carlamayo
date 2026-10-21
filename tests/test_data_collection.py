import json
import queue

import numpy as np

from module.data_collection import (
    RunRecorder,
    collect_synchronous_sensor_frame,
    frame_file_path,
    frame_is_complete,
)
from module.open_loop_dataset import load_front_camera_image, load_trajectory_index


def test_frame_is_complete_requires_all_sensor_outputs(tmp_path):
    (tmp_path / "cam_front_wide").mkdir()
    (tmp_path / "lidar_top").mkdir()
    (tmp_path / "cam_front_wide" / "000007.jpg").write_bytes(b"image")

    assert not frame_is_complete(tmp_path, ["cam_front_wide", "lidar_top"], 7)

    (tmp_path / "lidar_top" / "000007.ply").write_bytes(b"lidar")

    assert frame_is_complete(tmp_path, ["cam_front_wide", "lidar_top"], 7)
    assert frame_file_path(tmp_path, "lidar_top", 7).endswith("lidar_top/000007.ply")


def test_collect_synchronous_sensor_frame_keeps_only_exact_tick():
    sensor_queue = queue.Queue()
    sensor_queue.put((9, "camera", "old-camera"))
    sensor_queue.put((10, "camera", "current-camera"))
    sensor_queue.put((11, "lidar", "future-lidar"))
    sensor_queue.put((10, "lidar", "current-lidar"))

    frame = collect_synchronous_sensor_frame(
        sensor_queue,
        ["camera", "lidar"],
        frame_id=10,
        timeout=0.01,
    )

    assert frame == {"camera": "current-camera", "lidar": "current-lidar"}


def test_run_recorder_writes_an_open_loop_replayable_run(tmp_path):
    recorder = RunRecorder(tmp_path, ["cam_a"])
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255
    pose = {"x": 1, "y": 2, "z": 0, "roll": 0, "pitch": 0, "yaw": 90, "speed": 3}
    recorder.add_frame(12, [red], pose)
    recorder.add_record({"frame_id": 12, "trajectory": np.zeros((2, 3), dtype=np.float32)})
    recorder.close()

    trajectory, frame_ids = load_trajectory_index(tmp_path)
    assert frame_ids == [12] and trajectory["12"]["yaw"] == 90
    image = load_front_camera_image(tmp_path, 12, camera_name="cam_a")
    assert image[0, 0, 0] > 200 and image[0, 0, 2] < 50  # saved as RGB, not swapped
    record = json.loads((tmp_path / "records.jsonl").read_text())
    assert record["trajectory"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_run_recorder_keeps_a_replayable_trajectory_before_close(tmp_path, monkeypatch):
    monkeypatch.setattr(RunRecorder, "TRAJECTORY_SAVE_EVERY", 2)
    recorder = RunRecorder(tmp_path, ["cam_a"])
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    pose = {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}
    recorder.add_frame(1, [image], pose)
    recorder.add_frame(2, [image], pose)
    recorder.add_record({"frame_id": 2})

    # No close(): an aborted run still leaves the trajectory and the records on disk.
    _, frame_ids = load_trajectory_index(tmp_path)
    assert frame_ids == [1, 2]
    assert json.loads((tmp_path / "records.jsonl").read_text()) == {"frame_id": 2}
