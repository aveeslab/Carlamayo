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
