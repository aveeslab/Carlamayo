from types import SimpleNamespace

from module.carla_interface import CARLAInterface, is_allowed_npc_vehicle_blueprint


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


def test_collision_callback_records_count_and_last_event():
    interface = CARLAInterface()
    event = SimpleNamespace(
        frame=7,
        normal_impulse=SimpleNamespace(x=3.0, y=4.0, z=12.0),
        other_actor=SimpleNamespace(id=99, type_id="vehicle.audi.tt"),
    )

    interface._collision_callback(event)

    assert interface.get_collision_count() == 1
    last_event = interface.get_last_collision_event()
    assert last_event["frame"] == 7
    assert last_event["intensity"] == 13.0
    assert last_event["other_actor"] == "vehicle.audi.tt"
    assert last_event["other_actor_id"] == 99


def test_respawn_ego_vehicle_clears_runtime_state(monkeypatch):
    interface = CARLAInterface()
    spawn_point = SimpleNamespace(location=SimpleNamespace(x=1.0, y=2.0, z=3.0))
    controls = []

    class FakeVehicle:
        def __init__(self):
            self.transforms = []

        def apply_control(self, control):
            controls.append(control)

        def set_target_velocity(self, velocity):
            self.velocity = velocity

        def set_target_angular_velocity(self, velocity):
            self.angular_velocity = velocity

        def set_transform(self, transform):
            self.transforms.append(transform)

    interface.ego_vehicle = FakeVehicle()
    interface.history_buffer = [{"x": 1.0}]
    interface.collision_events = [{"frame": 1}]
    interface.sensor_queues = {}
    monkeypatch.setattr(interface, "_select_ego_spawn_point", lambda: spawn_point)

    returned_spawn = interface.respawn_ego_vehicle()

    assert returned_spawn is spawn_point
    assert interface.ego_vehicle.transforms == [spawn_point]
    assert interface.history_buffer == []
    assert interface.collision_events == []
    assert controls[-1].brake == 1.0
