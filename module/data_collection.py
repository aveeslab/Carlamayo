"""Reusable helpers for CARLA data collection."""

import json
import os
import queue
import time

import cv2


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


class RunRecorder:
    """Record a live CARLA run for offline debugging (``--record-dir``).

    Camera JPEGs and ``trajectory.json`` use the ``data_collect.py`` layout, so a
    recording replays directly with ``--loop open --data-root <dir>``. Per-tick
    model and control state goes to ``records.jsonl``.
    """

    def __init__(self, output_dir, camera_names):
        self.output_dir = output_dir
        self.camera_names = list(camera_names)
        for name in self.camera_names:
            os.makedirs(os.path.join(output_dir, name), exist_ok=True)
        self.poses = {}
        self._records = open(os.path.join(output_dir, "records.jsonl"), "w")

    def add_frame(self, frame_id, images_rgb, pose):
        for name, image in zip(self.camera_names, images_rgb):
            cv2.imwrite(
                frame_file_path(self.output_dir, name, frame_id),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
        self.poses[int(frame_id)] = {
            key: round(float(pose[key]), 6) for key in ("x", "y", "z", "pitch", "yaw", "roll")
        }

    def add_record(self, record):
        self._records.write(json.dumps(record, default=_to_jsonable) + "\n")

    def close(self):
        self._records.close()
        with open(os.path.join(self.output_dir, "trajectory.json"), "w") as f:
            json.dump(self.poses, f, indent=4)


def _to_jsonable(value):
    """JSON fallback for numpy/torch arrays and scalars."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
