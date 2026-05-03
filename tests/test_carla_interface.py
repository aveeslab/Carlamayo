import pytest

from module.carla_interface import is_allowed_npc_vehicle_blueprint


class FakeAttribute:
    def __init__(self, value):
        self.value = str(value)

    def __int__(self):
        return int(self.value)

    def __str__(self):
        return self.value


class FakeBlueprint:
    def __init__(self, blueprint_id, number_of_wheels=4):
        self.id = blueprint_id
        self._attrs = {"number_of_wheels": FakeAttribute(number_of_wheels)}

    def has_attribute(self, name):
        return name in self._attrs

    def get_attribute(self, name):
        return self._attrs[name]


def test_allowed_npc_vehicle_blueprint_accepts_regular_four_wheel_car():
    assert is_allowed_npc_vehicle_blueprint(FakeBlueprint("vehicle.audi.tt")) is True


def test_allowed_npc_vehicle_blueprint_rejects_large_vehicle_ids():
    large_ids = [
        "vehicle.carlamotors.carlacola",
        "vehicle.carlamotors.firetruck",
        "vehicle.ford.ambulance",
        "vehicle.mercedes.sprinter",
        "vehicle.mitsubishi.fusorosa",
        "vehicle.tesla.cybertruck",
    ]

    for blueprint_id in large_ids:
        assert is_allowed_npc_vehicle_blueprint(FakeBlueprint(blueprint_id)) is False


def test_allowed_npc_vehicle_blueprint_rejects_non_four_wheel_actors():
    assert is_allowed_npc_vehicle_blueprint(FakeBlueprint("vehicle.yamaha.yzf", 2)) is False


def test_apply_control_sends_manual_control_flags_and_returns_command():
    from module.carla_interface import CARLAInterface

    class FakeEgoVehicle:
        def __init__(self):
            self.control = None

        def apply_control(self, control):
            self.control = control

        def get_control(self):
            return self.control

    carla_if = CARLAInterface()
    carla_if.ego_vehicle = FakeEgoVehicle()

    commanded = carla_if.apply_control(steering=0.25, throttle=0.4, brake=0.1)

    assert commanded["steer"] == pytest.approx(0.25)
    assert commanded["throttle"] == pytest.approx(0.4)
    assert commanded["brake"] == pytest.approx(0.1)
    assert commanded["hand_brake"] is False
    assert commanded["reverse"] is False
    assert commanded["manual_gear_shift"] is False
    assert carla_if.ego_vehicle.control.hand_brake is False
    assert carla_if.ego_vehicle.control.reverse is False
    assert carla_if.ego_vehicle.control.manual_gear_shift is False


def test_get_applied_control_reads_actor_control():
    from module.carla_interface import CARLAInterface

    class FakeControl:
        steer = -0.2
        throttle = 0.6
        brake = 0.0
        hand_brake = False
        reverse = False
        manual_gear_shift = False

    class FakeEgoVehicle:
        def get_control(self):
            return FakeControl()

    carla_if = CARLAInterface()
    carla_if.ego_vehicle = FakeEgoVehicle()

    applied = carla_if.get_applied_control()

    assert applied["steer"] == pytest.approx(-0.2)
    assert applied["throttle"] == pytest.approx(0.6)
    assert applied["brake"] == pytest.approx(0.0)
    assert applied["hand_brake"] is False
    assert applied["reverse"] is False
    assert applied["manual_gear_shift"] is False
