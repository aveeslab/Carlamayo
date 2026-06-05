import types

import carlamayo_closed_loop


def test_parse_args_defaults_to_pid_controller():
    args = carlamayo_closed_loop.parse_args([])

    assert args.controller == "pid"


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
