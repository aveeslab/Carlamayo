import math
import types

import numpy as np
import pytest

from module import config as cfg
from module import trajectory_follower as tf


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class FakeTransform:
    """Identity-rotation transform at ``location`` (world == local + location)."""

    def __init__(self, location=None, yaw=0.0):
        self.location = location or FakeLocation()
        self.rotation = types.SimpleNamespace(yaw=yaw)

    def transform(self, location):
        return FakeLocation(
            self.location.x + location.x, self.location.y + location.y, self.location.z + location.z
        )


def fake_vehicle(max_steer_deg=70.0):
    wheels = [types.SimpleNamespace(max_steer_angle=max_steer_deg)] * 2 + [
        types.SimpleNamespace(max_steer_angle=0.0)
    ] * 2
    return types.SimpleNamespace(
        get_physics_control=lambda: types.SimpleNamespace(wheels=wheels)
    )


@pytest.fixture(autouse=True)
def fake_carla(monkeypatch):
    monkeypatch.setattr(tf, "carla", types.SimpleNamespace(Location=FakeLocation))
    monkeypatch.setattr(cfg, "FOLLOW_TARGET_MEDIAN_TICKS", 1)  # no filtering unless a test sets it


def rear_axle_at(x, y=0.0, yaw=0.0):
    """Transform whose rear axle sits at (x, y)."""
    return FakeTransform(FakeLocation(x - cfg.EGO_REAR_AXLE_X, y), yaw=yaw)


def straight(n=64, step=1.0):
    return np.array([[step * (i + 1), 0.0, 0.0] for i in range(n)])


def test_planned_speeds_come_from_the_waypoint_timing():
    wp_ego = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    np.testing.assert_allclose(tf.planned_speeds(wp_ego), [10.0, 10.0, 20.0])


def test_trajectory_to_world_mirrors_y_and_starts_at_the_rear_axle():
    wp_world = tf.trajectory_to_world(FakeTransform(FakeLocation(10.0, 20.0)), [[5.0, 2.0, 0.0]])
    np.testing.assert_allclose(wp_world[0], [15.0 + cfg.EGO_REAR_AXLE_X, 18.0, 0.0])


def test_pure_pursuit_steers_toward_the_target_in_carla_signs():
    max_rad = math.radians(70.0)
    assert tf.pure_pursuit_steer((0, 0), 0.0, (10, 0), 2.86, max_rad) == 0.0
    assert tf.pure_pursuit_steer((0, 0), 0.0, (8, 3), 2.86, max_rad) > 0.0  # right in CARLA
    assert tf.pure_pursuit_steer((0, 0), 0.0, (8, -3), 2.86, max_rad) < 0.0
    # Heading 90 deg (CARLA +y): a target further along +y is straight ahead.
    assert abs(tf.pure_pursuit_steer((0, 0), math.pi / 2, (0, 10), 2.86, max_rad)) < 1e-9
    # Known geometry: 5 m ahead, 1 m right -> curvature 2*sin(a)/d, wheel angle atan(L*k).
    bearing, d = math.atan2(1, 5), math.hypot(5, 1)
    expected = math.atan(2.86 * 2 * math.sin(bearing) / d) / max_rad
    assert tf.pure_pursuit_steer((0, 0), 0.0, (5, 1), 2.86, max_rad) == pytest.approx(expected)


def test_steering_is_rate_limited_per_tick(monkeypatch):
    monkeypatch.setattr(cfg, "FOLLOW_STEER_RATE_PER_SEC", 1.0)
    follower = tf.TrajectoryFollower(fake_vehicle())
    hard_right = np.array([[3.0, 3.0, 0.0], [4.0, 6.0, 0.0], [5.0, 9.0, 0.0]])
    steer, _, _, _ = follower.compute_control(rear_axle_at(0.0), hard_right, 5.0, np.full(3, 5.0))
    assert steer == pytest.approx(1.0 * cfg.CONTROL_DT)


def test_speed_control_uses_feedforward_plus_pi(monkeypatch):
    monkeypatch.setattr(cfg, "FOLLOW_FF_THROTTLE_AT_ZERO", 0.1)
    monkeypatch.setattr(cfg, "FOLLOW_FF_THROTTLE_PER_MPS", 0.04)
    monkeypatch.setattr(cfg, "FOLLOW_KP", 0.35)
    monkeypatch.setattr(cfg, "FOLLOW_KI", 0.0)
    monkeypatch.setattr(cfg, "FOLLOW_KD", 0.0)
    monkeypatch.setattr(cfg, "FOLLOW_BRAKE_SPEED_RATIO", 0.6)
    monkeypatch.setattr(cfg, "FOLLOW_BRAKE_RELEASE_RATIO", 0.8)
    monkeypatch.setattr(cfg, "FOLLOW_BRAKE_ON", 0.3)
    follower = tf.TrajectoryFollower(fake_vehicle())
    plan = np.full(64, 5.0)
    # On target: throttle is the feedforward for 5 m/s, no brake.
    _, throttle, brake, debug = follower.compute_control(rear_axle_at(0.0), straight(), 5.0, plan)
    assert (throttle, brake) == (pytest.approx(0.1 + 0.2), 0.0)
    assert debug["target_speed_kmh"] == pytest.approx(18.0)
    # Too fast by 2 m/s (target is 71% of speed): coast, both pedals released.
    _, throttle, brake, _ = follower.compute_control(rear_axle_at(0.0), straight(), 7.0, plan)
    assert (throttle, brake) == (0.0, 0.0)
    # Target below 60% of the speed: the on/off brake engages ...
    _, throttle, brake, _ = follower.compute_control(rear_axle_at(0.0), straight(), 9.0, plan)
    assert (throttle, brake) == (0.0, 0.3)
    # ... and stays on until the target is back above 80% of the speed.
    assert follower.compute_control(rear_axle_at(0.0), straight(), 7.0, plan)[2] == 0.3
    assert follower.compute_control(rear_axle_at(0.0), straight(), 6.0, plan)[2] == 0.0


def test_integrator_builds_holding_throttle_but_not_while_saturated(monkeypatch):
    monkeypatch.setattr(cfg, "FOLLOW_FF_THROTTLE_AT_ZERO", 0.0)
    monkeypatch.setattr(cfg, "FOLLOW_FF_THROTTLE_PER_MPS", 0.0)
    monkeypatch.setattr(cfg, "FOLLOW_KP", 0.1)
    monkeypatch.setattr(cfg, "FOLLOW_KI", 0.5)
    monkeypatch.setattr(cfg, "FOLLOW_KD", 0.0)
    monkeypatch.setattr(cfg, "FOLLOW_KI_EFFORT_LIMIT", 0.5)
    follower = tf.TrajectoryFollower(fake_vehicle())
    plan = np.full(64, 5.0)
    # 1 m/s below target, unsaturated: the integral term grows every tick.
    throttles = [
        follower.compute_control(rear_axle_at(0.0), straight(), 4.0, plan)[1] for _ in range(3)
    ]
    assert throttles[0] < throttles[1] < throttles[2]
    # Far below a fast target: effort saturates at THROTTLE_MAX, the integral must not wind up.
    before = follower._integral
    fast = np.full(64, 9.0)
    _, throttle, _, _ = follower.compute_control(rear_axle_at(0.0), straight(), 0.0, fast)
    assert throttle == cfg.THROTTLE_MAX and follower._integral == before


def test_one_tick_dip_in_the_plan_is_filtered_out(monkeypatch):
    monkeypatch.setattr(cfg, "FOLLOW_TARGET_MEDIAN_TICKS", 3)
    monkeypatch.setattr(cfg, "FOLLOW_KI", 0.0)
    follower = tf.TrajectoryFollower(fake_vehicle())
    steady, dip, slow = np.full(64, 8.0), np.full(64, 0.5), np.full(64, 4.0)
    for _ in range(3):
        follower.compute_control(rear_axle_at(0.0), straight(), 8.0, steady)
    _, throttle, brake, debug = follower.compute_control(rear_axle_at(0.0), straight(), 8.0, dip)
    # The median ignores the single dip: target unchanged, no brake.
    assert debug["target_speed_filtered_kmh"] == pytest.approx(8.0 * 3.6)
    assert brake == 0.0
    # A sustained change passes after two ticks.
    follower.compute_control(rear_axle_at(0.0), straight(), 8.0, slow)
    _, _, brake, debug = follower.compute_control(rear_axle_at(0.0), straight(), 8.0, slow)
    assert debug["target_speed_filtered_kmh"] == pytest.approx(4.0 * 3.6) and brake > 0.0
    # A planned stop (<= PID_STOP_SPEED_MPS) bypasses the filter and brakes at once.
    stop = np.zeros(64)
    _, throttle, brake, debug = follower.compute_control(rear_axle_at(0.0), straight(), 8.0, stop)
    assert debug["planned_stop"] and (throttle, brake) == (0.0, cfg.FOLLOW_BRAKE_ON)


def test_planned_stop_holds_and_launch_from_rest_overrides_a_slow_start(monkeypatch):
    follower = tf.TrajectoryFollower(fake_vehicle())
    # At rest with a stop plan: full brake, no launch.
    stop = np.zeros(64)
    _, throttle, brake, debug = follower.compute_control(rear_axle_at(0.0), straight(), 0.0, stop)
    assert (throttle, brake, debug["launch"]) == (0.0, 1.0, False)
    # At rest with a slow-start plan that still covers several metres: launch.
    slow_start = np.array([0.0] * 10 + [0.2] * 20 + [1.0] * 34)
    _, throttle, brake, debug = follower.compute_control(
        rear_axle_at(0.0), straight(), 0.0, slow_start
    )
    assert debug["launch"] and throttle > 0.0 and brake == 0.0
    assert debug["target_speed_kmh"] == pytest.approx(cfg.PID_LAUNCH_SPEED_MPS * 3.6)


def test_tracking_starts_at_the_nearest_waypoint_of_an_older_plan():
    follower = tf.TrajectoryFollower(fake_vehicle())
    # World-frame plan along +x; the rear axle has advanced to x=4.2, past the first points.
    _, _, _, debug = follower.compute_control(rear_axle_at(4.2), straight(), 0.0, np.full(64, 5.0))
    # Lookahead (4 m at standstill) is measured from the ego, landing near x=8.2.
    assert debug["target_wp_xy"][0] >= 8.0


def test_launch_judges_an_older_plan_from_now_not_from_the_lead():
    follower = tf.TrajectoryFollower(fake_vehicle())
    # A 1.6 m crawl plan that ends after 2.5 s: at rest in --async with 1.2 s elapsed the
    # window after the lead is empty, but the plan still intends to move from now.
    crawl = np.array([0.0] * 10 + [1.0] * 15 + [0.0] * 39)
    follower._launching = True
    _, throttle, _, debug = follower.compute_control(rear_axle_at(0.0), straight(), 0.0, crawl, 1.2)
    assert debug["launch"] and throttle > 0.0


def test_launch_floor_holds_until_the_launch_speed_is_reached():
    follower = tf.TrajectoryFollower(fake_vehicle())
    # "Wait, then go": ~0.2 m/s for the next 1.5 s, accelerating later (13 m in total).
    wait_then_go = np.array([0.2] * 15 + [4.0] * 49) * (13.0 / (0.2 * 1.5 + 4.0 * 4.9))
    _, _, _, debug = follower.compute_control(rear_axle_at(0.0), straight(), 0.0, wait_then_go)
    assert debug["launch"]
    # Moving at 1 m/s (above rest, below the launch speed): still launching, no stop.
    _, throttle, brake, debug = follower.compute_control(
        rear_axle_at(0.0), straight(), 1.0, wait_then_go
    )
    assert debug["launch"] and not debug["planned_stop"] and brake == 0.0 and throttle > 0.0
    # Once at the launch speed the plan's own window applies again.
    _, _, _, debug = follower.compute_control(rear_axle_at(0.0), straight(), 2.5, wait_then_go)
    assert not debug["launch"]


def test_speed_is_capped_by_the_path_curvature_ahead(monkeypatch):
    monkeypatch.setattr(cfg, "FOLLOW_LAT_ACCEL_MAX", 3.0)
    follower = tf.TrajectoryFollower(fake_vehicle())
    # Rear axle at the origin on a 10 m radius circle (curvature 0.1): v_max = sqrt(3/0.1).
    angles = np.linspace(0.1, 1.5, 15)
    circle = np.column_stack([10 * np.sin(angles), 10 - 10 * np.cos(angles), np.zeros(15)])
    assert tf.max_path_curvature(circle[:, :2]) == pytest.approx(0.1, rel=1e-3)
    _, _, _, debug = follower.compute_control(rear_axle_at(0.0), circle, 8.0, np.full(64, 10.0))
    assert debug["curve_limit_kmh"] == pytest.approx(math.sqrt(30.0) * 3.6, rel=1e-3)
    assert debug["target_speed_kmh"] == pytest.approx(math.sqrt(30.0) * 3.6, rel=1e-3)
    # A straight plan is not capped.
    _, _, _, debug = follower.compute_control(rear_axle_at(0.0), straight(), 8.0, np.full(64, 9.0))
    assert debug["target_speed_kmh"] == pytest.approx(9.0 * 3.6)
