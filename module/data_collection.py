"""Reusable helpers for CARLA data collection."""

import os
import queue
import time


def frame_file_path(output_dir, sensor_name, frame_id):
    """Return the on-disk path for one saved sensor frame."""

    ext = "ply" if "lidar" in sensor_name else "jpg"
    return os.path.join(output_dir, sensor_name, f"{frame_id:06d}.{ext}")


def frame_is_complete(output_dir, sensor_names, frame_id):
    """Return True when every expected sensor file exists for a frame."""

    return all(
        os.path.exists(frame_file_path(output_dir, sensor_name, frame_id))
        for sensor_name in sensor_names
    )


def collect_synchronous_sensor_frame(
    sensor_queue,
    expected_sensor_names,
    frame_id,
    timeout=5.0,
):
    """Collect one complete synchronous sensor packet for a world tick frame.

    CARLA sensors can leave older or newer frame messages in the shared queue,
    especially after map reloads or when image encoding is slower than the
    simulation tick. Filter by the exact frame returned from ``world.tick()``
    so trajectory poses and sensor files stay aligned.
    """

    deadline = time.time() + timeout
    frame_data = {}
    expected = set(expected_sensor_names)

    while time.time() < deadline and set(frame_data) != expected:
        remaining = max(0.1, deadline - time.time())
        try:
            sensor_frame, name, data = sensor_queue.get(True, remaining)
        except queue.Empty:
            break

        if sensor_frame != frame_id:
            continue
        if name in expected:
            frame_data[name] = data

    return frame_data
