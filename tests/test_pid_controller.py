import math
import types

import numpy as np
import pytest

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

    def transform(self, loc):
        return FakeLocation(
            self.location.x + loc.x,
            self.location.y + loc.y,
            self.location.z + loc.z,
        )

    def get_forward_vector(self):
        return types.SimpleNamespace(x=1.0, y=0.0, z=0.0)

    def get_right_vector(self):
        return types.SimpleNamespace(x=0.0, y=1.0, z=0.0)


class RecordingVehiclePIDController:
    def __init__(self, *args, **kwargs):
        self.last_speed = None
        self.last_waypoint = None

    def run_step(self, target_speed, waypoint):
        self.last_speed = target_speed
        self.last_waypoint = waypoint
        return types.SimpleNamespace(steer=-0.7, throttle=0.2, brake=0.0)


class RaisingMap:
    def __init__(self):
        self.called = False

    def get_waypoint(self, *_args, **_kwargs):
        self.called = True
        raise AssertionError("controller should not project raw Alpamayo targets to map waypoints")


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


def _install_fakes(monkeypatch):
    fake_carla = types.SimpleNamespace(
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


def test_pid_follower_uses_official_pid_steer_with_8m_low_speed_target(monkeypatch):
    _install_fakes(monkeypatch)
    fake_map = RaisingMap()
    follower = pid_controller.OfficialPIDFollower(FakeWorld(fake_map), FakeVehicle())
    vehicle_tf = FakeTransform(FakeLocation(10.0, 20.0, 0.0))
    wp_ego = np.array(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, -2.0, 0.0],
            [9.0, -2.0, 0.0],
        ],
        dtype=np.float64,
    )

    steer, throttle, brake, debug = follower.compute_control(
        vehicle_tf,
        wp_ego,
        speed_mps=0.0,
    )

    expected_local = np.array([10.0 - math.sqrt(8.0), 2.0])

    assert fake_map.called is False
    assert (steer, throttle, brake) == pytest.approx((-0.7, 0.2, 0.0))
    assert debug["mode"] == "official_pid"
    assert debug["lookahead_m"] == pytest.approx(8.0)
    assert debug["target_idx"] == 3
    assert debug["target_local_xy"] == pytest.approx(expected_local.tolist())
    assert debug["lookahead_path_m"] == pytest.approx(8.0)
    target_loc = follower.pid.last_waypoint.transform.location
    assert target_loc.x == pytest.approx(10.0 + expected_local[0])
    assert target_loc.y == pytest.approx(20.0 + expected_local[1])
    assert debug["target_projected_to_road"] is False


def test_pid_follower_returns_neutral_control_without_target(monkeypatch):
    _install_fakes(monkeypatch)
    follower = pid_controller.OfficialPIDFollower(FakeWorld(RaisingMap()), FakeVehicle())
    vehicle_tf = FakeTransform(FakeLocation(10.0, 20.0, 0.0))

    steer, throttle, brake, debug = follower.compute_control(
        vehicle_tf,
        np.empty((0, 3), dtype=np.float64),
        speed_mps=0.0,
    )

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.0, 0.0))
    assert debug["mode"] == "official_pid_no_target"
