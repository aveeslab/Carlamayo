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

    steer, throttle, brake, debug = follower.compute_control(
        vehicle_tf,
        wp_ego,
        speed_mps=0.0,
    )

    assert fake_map.called is False
    assert (steer, throttle, brake) == pytest.approx((0.1, 0.2, 0.0))
    target_loc = follower.pid.last_waypoint.transform.location
    assert target_loc.x == pytest.approx(15.0)
    assert target_loc.y == pytest.approx(20.0)
    assert debug["target_idx"] == 1
    assert debug["target_raw_xy"] == pytest.approx([15.0, 20.0])
    assert debug["target_projected_to_road"] is False
