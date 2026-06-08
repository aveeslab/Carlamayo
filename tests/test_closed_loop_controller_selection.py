import queue
import types

import numpy as np
import pytest

import carlamayo_closed_loop


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeRotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)


class FakeTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.location = FakeLocation(x, y, z)
        self.rotation = FakeRotation(yaw)


def test_parse_args_defaults_to_pid_controller():
    args = carlamayo_closed_loop.parse_args([])

    assert args.controller == "pid"
    assert args.inference_interval_frames > 0


def test_parse_args_accepts_mpc_controller():
    args = carlamayo_closed_loop.parse_args(["--controller", "mpc"])

    assert args.controller == "mpc"


def test_create_controller_builds_pid_controller(monkeypatch):
    class FakePIDFollower:
        def __init__(self, world, vehicle):
            self.world = world
            self.vehicle = vehicle

    world = object()
    vehicle = object()
    monkeypatch.setattr(carlamayo_closed_loop, "OfficialPIDFollower", FakePIDFollower)

    controller = carlamayo_closed_loop.create_controller("pid", world, vehicle)

    assert isinstance(controller, FakePIDFollower)
    assert controller.world is world
    assert controller.vehicle is vehicle


def test_create_controller_builds_mpc_follower(monkeypatch):
    class FakeMPCFollower:
        pass

    fake_module = types.SimpleNamespace(MPCFollower=FakeMPCFollower)
    monkeypatch.setattr(carlamayo_closed_loop.importlib, "import_module", lambda name: fake_module)

    controller = carlamayo_closed_loop.create_controller("mpc", object(), object())

    assert isinstance(controller, FakeMPCFollower)


def test_compute_controller_control_passes_latency_only_when_supported():
    class LatencyAwareController:
        def compute_control(self, vehicle_tf, trajectory_xyz, speed_mps, *, latency_ms=None):
            return vehicle_tf, trajectory_xyz, speed_mps, latency_ms

    result = carlamayo_closed_loop.compute_controller_control(
        LatencyAwareController(),
        "tf",
        "traj",
        3.0,
        latency_ms=42.0,
    )

    assert result == ("tf", "traj", 3.0, 42.0)


def test_compute_controller_control_preserves_pid_call_shape():
    class LegacyController:
        def compute_control(self, vehicle_tf, trajectory_xyz, speed_mps):
            return vehicle_tf, trajectory_xyz, speed_mps

    result = carlamayo_closed_loop.compute_controller_control(
        LegacyController(),
        "tf",
        "traj",
        3.0,
        latency_ms=42.0,
    )

    assert result == ("tf", "traj", 3.0)


def test_record_applied_controller_control_uses_optional_hook():
    class Controller:
        def __init__(self):
            self.applied = None

        def record_applied_control(self, steer, throttle, brake):
            self.applied = (steer, throttle, brake)

    controller = Controller()

    carlamayo_closed_loop.record_applied_controller_control(
        controller,
        steer=0.1,
        throttle=0.2,
        brake=0.0,
    )

    assert controller.applied == (0.1, 0.2, 0.0)


def test_record_applied_controller_control_ignores_legacy_controllers():
    controller = object()

    carlamayo_closed_loop.record_applied_controller_control(
        controller,
        steer=0.1,
        throttle=0.2,
        brake=0.0,
    )


def test_enqueue_inference_stop_replaces_stale_request_with_stop_sentinel():
    request_q = queue.Queue(maxsize=1)
    stale_request = object()
    request_q.put_nowait(stale_request)

    carlamayo_closed_loop.enqueue_inference_stop(request_q)

    assert request_q.get_nowait() is None
    assert request_q.empty()


def test_sync_trajectory_latency_ignores_wall_clock_inference_time():
    assert (
        carlamayo_closed_loop.compute_trajectory_latency_ms(
            async_mode=False,
            inference_time_s=3.0,
            trajectory_ts=10.0,
            now_ts=14.0,
        )
        == 0.0
    )


def test_async_trajectory_latency_includes_inference_and_age():
    assert carlamayo_closed_loop.compute_trajectory_latency_ms(
        async_mode=True,
        inference_time_s=3.0,
        trajectory_ts=10.0,
        now_ts=14.0,
    ) == pytest.approx(7000.0)


def test_local_trajectory_world_path_round_trip():
    origin = FakeTransform(x=10.0, y=20.0, yaw=90.0)
    trajectory = np.array([[2.0, 1.0, 0.0]], dtype=np.float64)

    world_path = carlamayo_closed_loop.local_trajectory_to_world_path(origin, trajectory)
    local = carlamayo_closed_loop.world_path_to_local_trajectory(origin, world_path)

    assert local == pytest.approx(trajectory)


def test_mpc_world_reference_tracks_remaining_path_from_vehicle_pose():
    path = carlamayo_closed_loop.local_trajectory_to_world_path(
        FakeTransform(),
        np.column_stack([np.linspace(0.0, 10.0, 11), np.zeros(11), np.zeros(11)]),
    )

    reference, progress_m, cte_m = carlamayo_closed_loop.build_mpc_reference_from_world_path(
        path,
        FakeTransform(x=5.0, y=0.0, yaw=0.0),
    )

    assert progress_m == pytest.approx(5.0)
    assert cte_m == pytest.approx(0.0)
    assert reference[0, 0] == pytest.approx(0.0)
    assert reference[-1, 0] > reference[0, 0]


def test_mpc_world_reference_progress_does_not_move_backward():
    path = carlamayo_closed_loop.local_trajectory_to_world_path(
        FakeTransform(),
        np.column_stack([np.linspace(0.0, 10.0, 11), np.zeros(11), np.zeros(11)]),
    )

    _reference, progress_m, _cte_m = carlamayo_closed_loop.build_mpc_reference_from_world_path(
        path,
        FakeTransform(x=5.0, y=0.0, yaw=0.0),
        previous_progress_m=6.0,
    )

    assert progress_m == pytest.approx(6.0)
