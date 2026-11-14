import math
import types

import numpy as np
import pytest

from module import config as cfg
from module.trajectory_follower import TrajectoryTracker, pose_in_plan_frame


def _straight_path(speeds_mps):
    """CARLA-local plan along +x whose 0.1 s steps run at ``speeds_mps``."""
    x = np.cumsum(np.asarray(speeds_mps, dtype=float) * 0.1)
    return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


def test_speed_target_follows_trajectory_time_and_clamps_at_the_end():
    path = _straight_path(np.arange(1, 65) * 0.1 + 1.0)
    tracker = TrajectoryTracker()
    for index, expected in [(0, 1.1), (9, 2.0), (500, 7.4)]:
        command = tracker.command(path, [0, 0, 0], 0.0, 0.0, trajectory_time_index=index)
        assert command.target_speed_mps == pytest.approx(expected)
        assert command.speed_index == min(index, 63)


def test_planned_stop_brakes_while_moving_and_holds_once_stopped():
    path = _straight_path([0.1] * 64)
    tracker = TrajectoryTracker()
    moving = tracker.command(path, [0, 0, 0], 0.0, 0.5)
    assert moving.throttle == 0.0 and moving.brake >= cfg.TRACKER_MIN_STOP_BRAKE
    assert not moving.hold_active
    stopped = tracker.command(path, [0, 0, 0], 0.0, 0.1)
    assert stopped.hold_active and stopped.brake == cfg.TRACKER_HOLD_BRAKE


def test_throttle_feedforward_holds_the_planned_speed():
    tracker = TrajectoryTracker()
    command = tracker.command(_straight_path([2.0] * 64), [0, 0, 0], 0.0, 2.0)
    feedforward = cfg.TRACKER_FF_THROTTLE_AT_ZERO + 2.0 * cfg.TRACKER_FF_THROTTLE_PER_MPS
    assert command.throttle == pytest.approx(feedforward, abs=0.01)


def test_pid_does_not_wind_up_into_a_saturated_pedal():
    tracker = TrajectoryTracker()
    path = _straight_path([20.0] * 64)
    tracker.command(path, [0, 0, 0], 0.0, 0.0)
    before = tracker._speed_integral
    command = tracker.command(path, [0, 0, 0], 0.0, 0.0)
    assert command.throttle == 1.0 and tracker._speed_integral == before


def test_steering_is_pure_pursuit_from_the_nearest_point_and_rate_limited():
    # A plan that starts behind the ego and veers right (CARLA +y) ahead of it.
    x = np.arange(-5.0, 20.0, 0.5)
    path = np.column_stack([x, np.where(x > 0, 0.05 * x**2, 0.0), np.zeros_like(x)])
    tracker = TrajectoryTracker(max_steer_rate_per_s=100.0)
    command = tracker.command(path, [0, 0, 0], 0.0, 3.0)
    target = path[command.target_index]
    assert target[0] > 0 and math.hypot(*target[:2]) >= cfg.TRACKER_LOOKAHEAD_M
    curvature = 2.0 * target[1] / (target[:2] @ target[:2])
    expected = math.atan(cfg.EGO_WHEELBASE_M * curvature) / math.radians(70.0)
    assert command.steer == pytest.approx(expected) and command.steer > 0

    limited = TrajectoryTracker()
    assert limited.command(path, [0, 0, 0], 0.0, 3.0).steer == pytest.approx(0.1)


def _yaw_transform(x, y, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    matrix = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])
    return types.SimpleNamespace(
        get_matrix=lambda: matrix, get_inverse_matrix=lambda: np.linalg.inv(matrix)
    )


def test_pose_in_plan_frame_is_rear_axle_relative():
    plan = _yaw_transform(10.0, 5.0, 90.0)
    # 4 m further along the plan's heading and turned 30 degrees right.
    position, yaw = pose_in_plan_frame(plan, _yaw_transform(10.0, 9.0, 120.0))
    rear_axle_shift = cfg.EGO_REAR_AXLE_X * (1 - math.cos(math.radians(30)))
    assert position[0] == pytest.approx(4.0 - rear_axle_shift)
    assert yaw == pytest.approx(math.radians(30))
