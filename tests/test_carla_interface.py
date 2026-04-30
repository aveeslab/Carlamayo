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
