import math
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
        # The follower should replace this lateral output with trajectory pure pursuit.
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


def _expected_pure_pursuit_steer(local_xy):
    local_xy = np.asarray(local_xy, dtype=np.float64)
    distance_m = float(np.linalg.norm(local_xy[:2]))
    response_distance_m = min(distance_m, cfg.PID_STEER_RESPONSE_MAX_LOOKAHEAD_M)
    response_distance_sq = max(response_distance_m * response_distance_m, 1e-6)
    curvature = 2.0 * float(local_xy[1]) / response_distance_sq
    angle = math.atan(cfg.PID_WHEELBASE_M * curvature)
    steer = np.clip(angle / cfg.PID_STEER_NORMALIZATION_RAD, -cfg.PID_MAX_STEER, cfg.PID_MAX_STEER)
    return steer, curvature, angle, response_distance_m


def test_pid_follower_uses_interpolated_trajectory_lookahead_without_map_projection(monkeypatch):
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

    expected_local = np.array([3.0 + 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
    expected_steer, _curvature, _angle, response_distance_m = _expected_pure_pursuit_steer(
        expected_local
    )

    assert fake_map.called is False
    assert (throttle, brake) == pytest.approx((0.2, 0.0))
    assert steer == pytest.approx(expected_steer)
    assert steer > 0.0
    assert debug["mode"] == "trajectory_pure_pursuit"
    assert debug["target_idx"] == 2
    assert debug["target_local_xy"] == pytest.approx(expected_local.tolist())
    assert debug["lookahead_path_m"] == pytest.approx(4.0)
    target_loc = follower.pid.last_waypoint.transform.location
    assert target_loc.x == pytest.approx(10.0 + expected_local[0])
    assert target_loc.y == pytest.approx(20.0 + expected_local[1])
    assert debug["target_projected_to_road"] is False


def test_pid_follower_uses_zero_steer_for_straight_trajectory(monkeypatch):
    _install_fakes(monkeypatch)
    follower = pid_controller.OfficialPIDFollower(FakeWorld(RaisingMap()), FakeVehicle())
    vehicle_tf = FakeTransform(FakeLocation(10.0, 20.0, 0.0))
    wp_ego = np.array(
        [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    steer, throttle, brake, debug = follower.compute_control(vehicle_tf, wp_ego, speed_mps=0.0)

    assert (steer, throttle, brake) == pytest.approx((0.0, 0.2, 0.0))
    assert debug["target_local_xy"] == pytest.approx([4.0, 0.0])


def test_pid_follower_uses_more_responsive_steer_normalization(monkeypatch):
    _install_fakes(monkeypatch)
    follower = pid_controller.OfficialPIDFollower(FakeWorld(RaisingMap()), FakeVehicle())

    target_local = np.array([1.5, 0.1, 0.0], dtype=np.float64)
    steer, _curvature, _angle, _response_distance_m = follower._pure_pursuit_steer(target_local)

    distance_sq = float(np.dot(target_local[:2], target_local[:2]))
    expected_curvature = 2.0 * target_local[1] / distance_sq
    expected_angle = math.atan(cfg.PID_WHEELBASE_M * expected_curvature)
    old_normalization_steer = expected_angle / 0.7
    responsive_steer = expected_angle / 0.45

    assert cfg.PID_STEER_NORMALIZATION_RAD == pytest.approx(0.45)
    assert steer == pytest.approx(responsive_steer)
    assert abs(steer) > abs(old_normalization_steer) * 1.5


def test_pid_follower_boosts_small_lateral_offsets_at_long_lookahead(monkeypatch):
    _install_fakes(monkeypatch)
    follower = pid_controller.OfficialPIDFollower(FakeWorld(RaisingMap()), FakeVehicle())

    target_local = np.array([4.0, 0.05, 0.0], dtype=np.float64)
    steer, curvature, _angle, response_distance_m = follower._pure_pursuit_steer(target_local)

    old_curvature = 2.0 * target_local[1] / float(np.dot(target_local[:2], target_local[:2]))
    old_angle = math.atan(cfg.PID_WHEELBASE_M * old_curvature)
    old_steer = old_angle / cfg.PID_STEER_NORMALIZATION_RAD

    expected_steer, expected_curvature, _expected_angle, expected_response_m = (
        _expected_pure_pursuit_steer(target_local[:2])
    )

    assert cfg.PID_STEER_RESPONSE_MAX_LOOKAHEAD_M == pytest.approx(2.5)
    assert response_distance_m == pytest.approx(expected_response_m)
    assert curvature == pytest.approx(expected_curvature)
    assert steer == pytest.approx(expected_steer)
    assert abs(steer) > abs(old_steer) * 2.3
    assert abs(steer) < abs(old_steer) * 2.8
