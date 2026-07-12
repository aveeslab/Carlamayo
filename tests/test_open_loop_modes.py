import pytest

import carlamayo_open_loop as open_loop


def test_open_loop_defaults_to_normal_mode():
    args = open_loop.parse_args([])

    assert args.mode == "normal"
    assert args.temperature == 0.0
    assert args.output_json is None


def test_nav_alias_normalizes_to_navigation():
    args = open_loop.parse_args(
        [
            "--mode",
            "nav",
            "--navigation-text",
            "Keep right",
            "--navigation-weight",
            "1.25",
        ]
    )

    assert args.mode == "navigation"
    assert args.navigation_text == "Keep right"
    assert args.navigation_weight == 1.25


@pytest.mark.parametrize("mode", ["nav", "navigation"])
def test_navigation_requires_nonempty_text(mode):
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--mode", mode, "--navigation-text", "  "])


def test_vqa_requires_nonempty_question():
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--mode", "vqa", "--vqa-question", "  "])


def test_vqa_derives_json_path_from_output_video():
    args = open_loop.parse_args(
        [
            "--mode",
            "vqa",
            "--vqa-question",
            "Is the road flooded?",
            "--output-video",
            "run.mp4",
        ]
    )

    assert args.output_json == "run.json"


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_temperature_must_be_nonnegative_and_finite(value):
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--temperature", value])


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_navigation_weight_must_be_nonnegative_and_finite(value):
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--navigation-weight", value])


def test_output_json_is_rejected_outside_vqa():
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--output-json", "answers.json"])
