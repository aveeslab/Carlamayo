import queue

from module.data_collection import (
    collect_synchronous_sensor_frame,
    frame_file_path,
    frame_is_complete,
)


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
