import types

import numpy as np
import pytest

from module import config as cfg
from module import pid_controller


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        return FakeLocation(self.x + other.x, self.y + other.y, self.z + other.z)


class FakeRotation:
    pass


class FakeTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or FakeLocation()
        self.rotation = rotation or FakeRotation()

    def transform(self, location):
        return FakeLocation(
            self.location.x + location.x,
            self.location.y + location.y,
            self.location.z + location.z,
        )

    def get_right_vector(self):
        return types.SimpleNamespace(x=0.0, y=1.0, z=0.0)


class RecordingVehiclePIDController:
    def __init__(self, *_args, **_kwargs):
        self.last_waypoint = None

    def run_step(self, _target_speed, waypoint):
        self.last_waypoint = waypoint
        return types.SimpleNamespace(steer=0.1, throttle=0.2, brake=0.0)


class RaisingMap:
    def __init__(self):
        self.called = False

    def get_waypoint(self, *_args, **_kwargs):
        self.called = True
        raise AssertionError(
            "PID should not project raw Alpamayo targets to CARLA map waypoints"
        )


class FakeVehicle:
    def get_world(self):
        return object()

    def get_control(self):
        return types.SimpleNamespace(steer=0.0)


class FakeWorld:
    def __init__(self, fake_map):
        self.fake_map = fake_map

    def get_map(self):
        return self.fake_map


def test_pid_follower_uses_raw_target_without_carla_waypoint_projection(monkeypatch):
    monkeypatch.setattr(cfg, "PID_SPEED_LEAD_SEC", 0.0)
    fake_map = RaisingMap()
    fake_carla = types.SimpleNamespace(
        LaneType=types.SimpleNamespace(Driving=object()),
        Location=FakeLocation,
        Rotation=FakeRotation,
        Transform=FakeTransform,
    )
    monkeypatch.setattr(pid_controller, "carla", fake_carla)
    monkeypatch.setattr(
        pid_controller,
        "_resolve_vehicle_pid_controller",
        lambda: RecordingVehiclePIDController,
    )

    follower = pid_controller.OfficialPIDFollower(FakeWorld(fake_map), FakeVehicle())
    vehicle_tf = FakeTransform(FakeLocation(10.0, 20.0, 0.0))
    wp_ego = np.array(
        [
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    wp_world = pid_controller.trajectory_to_world(vehicle_tf, wp_ego)
    steer, throttle, brake, debug = follower.compute_control(
        vehicle_tf,
        wp_world,
        0.0,
        pid_controller.planned_speeds(wp_ego),
    )

    assert fake_map.called is False
    assert (steer, throttle, brake) == pytest.approx((0.1, 0.2, 0.0))
    target_loc = follower.pid.last_waypoint.transform.location
    # Waypoints are rear-axle relative; the actor origin sits EGO_REAR_AXLE_X ahead.
    assert target_loc.x == pytest.approx(15.0 + cfg.EGO_REAR_AXLE_X)
    assert target_loc.y == pytest.approx(20.0)
    assert debug["target_idx"] == 1
    assert debug["target_raw_xy"] == pytest.approx([15.0 + cfg.EGO_REAR_AXLE_X, 20.0])
    assert debug["target_projected_to_road"] is False


def test_pid_follower_skips_waypoints_the_ego_already_passed(monkeypatch):
    monkeypatch.setattr(pid_controller, "carla", types.SimpleNamespace(
        Location=FakeLocation, Rotation=FakeRotation, Transform=FakeTransform,
    ))
    monkeypatch.setattr(
        pid_controller, "_resolve_vehicle_pid_controller", lambda: RecordingVehiclePIDController
    )
    follower = pid_controller.OfficialPIDFollower(FakeWorld(RaisingMap()), FakeVehicle())
    # World-frame plan from an older (async) prediction along +x.
    wp_world = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    # The ego's rear axle has since reached x=4.2, past the first waypoint.
    vehicle_tf = FakeTransform(FakeLocation(4.2 - cfg.EGO_REAR_AXLE_X, 0.0, 0.0))

    follower.compute_control(vehicle_tf, wp_world, 0.0, np.full(4, 5.0))

    # Lookahead (4 m at standstill) is measured from the nearest waypoint (x=4), not x=0.
    assert follower.pid.last_waypoint.transform.location.x == pytest.approx(8.0)


def _follower(monkeypatch, brake=0.0):
    class FixedPID(RecordingVehiclePIDController):
        def run_step(self, target_speed, waypoint):
            self.last_target_speed = target_speed
            return types.SimpleNamespace(steer=0.0, throttle=0.0 if brake else 0.5, brake=brake)

    monkeypatch.setattr(pid_controller, "carla", types.SimpleNamespace(
        Location=FakeLocation, Rotation=FakeRotation, Transform=FakeTransform,
    ))
    monkeypatch.setattr(pid_controller, "_resolve_vehicle_pid_controller", lambda: FixedPID)
    return pid_controller.OfficialPIDFollower(FakeWorld(RaisingMap()), FakeVehicle())


def test_planned_speeds_come_from_the_waypoint_timing():
    wp_ego = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    np.testing.assert_allclose(pid_controller.planned_speeds(wp_ego), [10.0, 10.0, 20.0])


def test_pid_target_speed_follows_the_plan_at_the_elapsed_time(monkeypatch):
    monkeypatch.setattr(cfg, "PID_SPEED_PREVIEW_SEC", 0.5)
    monkeypatch.setattr(cfg, "PID_SPEED_LEAD_SEC", 0.0)
    follower = _follower(monkeypatch)
    wp_world = np.array([[float(x), 0.0, 0.0] for x in range(1, 11)])
    speeds = np.array([2.0] * 5 + [8.0] * 5)
    vehicle_tf = FakeTransform(FakeLocation(-cfg.EGO_REAR_AXLE_X, 0.0, 0.0))

    _, _, _, debug = follower.compute_control(vehicle_tf, wp_world, 2.0, speeds)
    assert debug["target_speed_kmh"] == pytest.approx(2.0 * 3.6)
    # Half a second after the prediction the plan has sped up.
    _, _, _, debug = follower.compute_control(vehicle_tf, wp_world, 2.0, speeds, 0.5)
    assert debug["target_speed_kmh"] == pytest.approx(8.0 * 3.6)
    # The lead looks the same half second further into the plan from now.
    monkeypatch.setattr(cfg, "PID_SPEED_LEAD_SEC", 0.5)
    _, _, _, debug = follower.compute_control(vehicle_tf, wp_world, 2.0, speeds)
    assert debug["target_speed_kmh"] == pytest.approx(8.0 * 3.6)


def test_planned_stop_holds_the_brake_instead_of_creeping(monkeypatch):
    follower = _follower(monkeypatch)
    wp_world = np.zeros((4, 3))
    vehicle_tf = FakeTransform(FakeLocation(-cfg.EGO_REAR_AXLE_X, 0.0, 0.0))
    stop = np.zeros(4)

    _, throttle, brake, debug = follower.compute_control(vehicle_tf, wp_world, 0.0, stop)
    assert (throttle, brake, debug["planned_stop"]) == (0.0, 1.0, True)
    _, throttle, brake, _ = follower.compute_control(vehicle_tf, wp_world, 5.0, stop)
    assert (throttle, brake) == (0.0, 0.25)


def test_launch_from_rest_when_the_plan_still_moves(monkeypatch):
    follower = _follower(monkeypatch)
    wp_world = np.array([[float(x), 0.0, 0.0] for x in range(1, 11)])
    vehicle_tf = FakeTransform(FakeLocation(-cfg.EGO_REAR_AXLE_X, 0.0, 0.0))
    # Typical plan from standstill: ~0 for the first second, then a slow ramp (3.8 m in total).
    slow_start = np.array([0.0] * 10 + [0.2] * 20 + [1.0] * 34)

    _, throttle, _, debug = follower.compute_control(vehicle_tf, wp_world, 0.0, slow_start)
    assert debug["launch"] and debug["target_speed_kmh"] == pytest.approx(
        cfg.PID_LAUNCH_SPEED_MPS * 3.6
    )
    assert throttle > 0.0
    # Already moving: the plan's own (slow) speed is followed, no launch boost.
    _, _, _, debug = follower.compute_control(vehicle_tf, wp_world, 3.0, slow_start)
    assert not debug["launch"]
    # A plan that barely moves keeps holding the brake at rest.
    _, throttle, brake, debug = follower.compute_control(
        vehicle_tf, wp_world, 0.0, np.full(64, 0.02)
    )
    assert (throttle, brake, debug["launch"]) == (0.0, 1.0, False)


def test_launch_holds_while_the_plan_hovers_below_the_start_threshold(monkeypatch):
    follower = _follower(monkeypatch)
    wp_world = np.array([[float(x), 0.0, 0.0] for x in range(1, 11)])
    vehicle_tf = FakeTransform(FakeLocation(-cfg.EGO_REAR_AXLE_X, 0.0, 0.0))

    def plan(total_m):
        return np.full(64, total_m / 6.4)

    assert follower.compute_control(vehicle_tf, wp_world, 0.0, plan(4.0))[3]["launch"]
    # Still at rest; the next plan shrinks to 1.5 m but keeps the launch going.
    assert follower.compute_control(vehicle_tf, wp_world, 0.0, plan(1.5))[3]["launch"]
    # A plan that collapses to a stop ends it, and 1.5 m alone does not restart it.
    assert not follower.compute_control(vehicle_tf, wp_world, 0.0, plan(0.5))[3]["launch"]
    assert not follower.compute_control(vehicle_tf, wp_world, 0.0, plan(1.5))[3]["launch"]
