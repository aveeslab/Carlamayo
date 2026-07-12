import importlib
import sys
import types

import pytest


fake_carla = types.ModuleType("carla")
fake_carla.WeatherParameters = types.SimpleNamespace(HardRainNoon=object())
sys.modules.setdefault("carla", fake_carla)
data_collect = importlib.import_module("data_collect")


def test_data_collection_defaults_preserve_existing_behavior():
    args = data_collect.parse_args([])

    assert args.map_name == "Town02"
    assert args.weather is None
    assert args.frames is None
    assert args.output_dir == "carla_data"
    assert args.host == "localhost"
    assert args.port == 2000
    assert args.traffic_manager_port == 8000


def test_data_collection_accepts_finite_flood_capture():
    args = data_collect.parse_args(
        [
            "--map",
            "Town10HD_Opt_Flood",
            "--weather",
            "HardRainNoon",
            "--frames",
            "1000",
            "--output-dir",
            "/tmp/town10-flood",
            "--host",
            "127.0.0.1",
            "--port",
            "2020",
            "--traffic-manager-port",
            "8020",
        ]
    )

    assert args.map_name == "Town10HD_Opt_Flood"
    assert args.weather == "HardRainNoon"
    assert args.frames == 1000
    assert args.output_dir == "/tmp/town10-flood"
    assert args.host == "127.0.0.1"
    assert args.port == 2020
    assert args.traffic_manager_port == 8020


@pytest.mark.parametrize("value", ["0", "-1"])
def test_data_collection_rejects_nonpositive_frame_limit(value):
    with pytest.raises(SystemExit):
        data_collect.parse_args(["--frames", value])


def test_resolve_weather_returns_exact_carla_preset():
    assert data_collect.resolve_weather("HardRainNoon", fake_carla) is fake_carla.WeatherParameters.HardRainNoon


def test_unknown_weather_is_rejected():
    with pytest.raises(ValueError, match="Unknown CARLA weather preset"):
        data_collect.resolve_weather("FloodMagic", fake_carla)


def test_apply_weather_waits_until_all_preset_values_match():
    preset = types.SimpleNamespace(
        cloudiness=100.0,
        precipitation=100.0,
        wetness=100.0,
        fog_falloff=0.1,
        mie_scattering_scale=0.03,
        rayleigh_scattering_scale=0.0331,
    )

    class World:
        def __init__(self):
            self.states = iter(
                [
                    types.SimpleNamespace(
                        cloudiness=100.0,
                        precipitation=100.0,
                        wetness=100.0,
                        fog_falloff=0.0,
                        mie_scattering_scale=0.0,
                        rayleigh_scattering_scale=0.0,
                    ),
                    types.SimpleNamespace(
                        cloudiness=100.0,
                        precipitation=100.0,
                        wetness=100.0,
                        fog_falloff=0.1,
                        mie_scattering_scale=0.03,
                        rayleigh_scattering_scale=0.0331,
                    ),
                ]
            )
            self.ticks = 0
            self.applied = None

        def set_weather(self, weather):
            self.applied = weather

        def tick(self):
            self.ticks += 1

        def get_weather(self):
            return next(self.states)

    world = World()

    actual = data_collect.apply_weather_until_ready(world, preset, max_ticks=3)

    assert world.applied is preset
    assert world.ticks == 2
    assert actual.wetness == 100.0


def test_apply_weather_raises_when_preset_never_becomes_active():
    preset = types.SimpleNamespace(cloudiness=100.0, precipitation=100.0, wetness=100.0)

    class World:
        def set_weather(self, weather):
            pass

        def tick(self):
            pass

        def get_weather(self):
            return types.SimpleNamespace(cloudiness=100.0, precipitation=100.0, wetness=0.0)

    with pytest.raises(RuntimeError, match="did not become active"):
        data_collect.apply_weather_until_ready(World(), preset, max_ticks=2)


def test_weather_match_requires_every_expected_field():
    expected = types.SimpleNamespace(cloudiness=100.0, precipitation=100.0)
    actual = types.SimpleNamespace(cloudiness=100.0)

    assert not data_collect._weather_matches(expected, actual)


@pytest.mark.parametrize(
    ("recorded", "limit", "expected"),
    [
        (999, 1000, False),
        (1000, 1000, True),
        (1001, 1000, True),
        (1000, None, False),
    ],
)
def test_capture_limit_counts_only_complete_frames(recorded, limit, expected):
    assert data_collect.capture_complete(recorded, limit) is expected
