import importlib
import importlib.machinery
import importlib.util
import queue
import sys
import types

import pytest


fake_carla = types.SimpleNamespace(
    command=types.SimpleNamespace(DestroyActor=lambda actor_id: ("destroy", actor_id)),
    VehicleControl=lambda: types.SimpleNamespace(steer=0.0, throttle=0.0, brake=0.0),
    Vector3D=lambda: types.SimpleNamespace(),
)
sys.modules.setdefault("carla", fake_carla)
if importlib.util.find_spec("cv2") is None:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    fake_cv2.COLOR_BGR2RGB = 0
    fake_cv2.cvtColor = lambda image, _code: image
    sys.modules.setdefault("cv2", fake_cv2)

CARLAInterface = importlib.import_module("module.carla_interface").CARLAInterface


class EmptyCameraQueue:
    def get(self, timeout):
        raise queue.Empty


def test_get_camera_images_raises_when_camera_frame_missing():
    carla_if = CARLAInterface()
    carla_if.camera_order = ["cam_front_wide"]
    carla_if.sensor_queues = {"cam_front_wide": EmptyCameraQueue()}

    with pytest.raises(TimeoutError, match=r"Missing camera frames: \['cam_front_wide'\]"):
        carla_if.get_camera_images()


def test_cleanup_reports_recoverable_teardown_failures(capsys):
    class FailingTrafficManager:
        def set_synchronous_mode(self, _enabled):
            raise RuntimeError("traffic manager failed")

    class FailingClient:
        def get_trafficmanager(self, _port):
            return FailingTrafficManager()

        def apply_batch(self, _commands):
            raise RuntimeError("destroy batch failed")

    class FailingController:
        id = 10

        def stop(self):
            raise RuntimeError("controller stop failed")

    class FailingSettings:
        synchronous_mode = True
        fixed_delta_seconds = 0.1

    class FailingWorld:
        def get_actors(self, _actor_ids):
            return [FailingController()]

        def get_settings(self):
            return FailingSettings()

        def apply_settings(self, _settings):
            raise RuntimeError("world settings failed")

    class FailingSensor:
        id = 20

        def stop(self):
            raise RuntimeError("sensor stop failed")

        def destroy(self):
            raise RuntimeError("sensor destroy failed")

    class FailingEgoVehicle:
        def destroy(self):
            raise RuntimeError("ego destroy failed")

    carla_if = CARLAInterface()
    carla_if.client = FailingClient()
    carla_if.world = FailingWorld()
    carla_if.npc_walker_controller_ids = [10]
    carla_if.npc_walker_ids = [11]
    carla_if.npc_vehicle_ids = [12]
    carla_if.sensors = {"cam_front_wide": FailingSensor()}
    carla_if.ego_vehicle = FailingEgoVehicle()

    carla_if.cleanup()

    output = capsys.readouterr().out
    assert "Warning: failed to disable TrafficManager synchronous mode" in output
    assert "Warning: failed to stop walker controller 10" in output
    assert "Warning: failed to destroy walker controllers" in output
    assert "Warning: failed to destroy NPC walkers" in output
    assert "Warning: failed to destroy NPC vehicles" in output
    assert "Warning: failed to stop sensor cam_front_wide" in output
    assert "Warning: failed to destroy sensor cam_front_wide" in output
    assert "Warning: failed to destroy ego vehicle" in output
    assert "Warning: failed to restore world asynchronous mode" in output


def test_select_vehicle_blueprint_reports_preferred_id_miss_before_compatible_choice(capsys):
    carla_interface = importlib.import_module("module.carla_interface")

    class FakeBlueprint:
        def __init__(self, blueprint_id):
            self.id = blueprint_id
            self.attrs = {"number_of_wheels": "4"}

        def has_attribute(self, name):
            return name in self.attrs or name == "role_name"

        def get_attribute(self, name):
            return self.attrs[name]

        def set_attribute(self, name, value):
            self.attrs[name] = value

    class FakeBlueprintLibrary:
        def find(self, blueprint_id):
            raise RuntimeError(f"missing {blueprint_id}")

        def filter(self, pattern):
            assert pattern == "vehicle.*"
            return [FakeBlueprint("vehicle.lincoln.mkz"), FakeBlueprint("vehicle.firetruck")]

    selected = carla_interface.select_vehicle_blueprint(
        FakeBlueprintLibrary(),
        preferred_id="vehicle.tesla.model3",
        role_name="hero",
    )

    assert selected.id == "vehicle.lincoln.mkz"
    assert selected.attrs["role_name"] == "hero"
    output = capsys.readouterr().out
    assert "Warning: preferred CARLA vehicle blueprint vehicle.tesla.model3 unavailable" in output
