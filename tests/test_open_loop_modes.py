import pytest
import numpy as np
import torch

import carlamayo_open_loop as open_loop
from module import open_loop_dataset


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


def test_open_loop_camera_indices_match_training_camera_labels():
    assert open_loop_dataset.OPEN_LOOP_CAMERA_INDICES == (0, 1, 2, 6)


def test_load_model_input_attaches_camera_indices(tmp_path, monkeypatch):
    arrays = {
        "image_frames": np.zeros((4, 4, 2, 3, 3), dtype=np.uint8),
        "history_xyz": np.zeros((16, 3), dtype=np.float32),
        "history_rot": np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
    }
    monkeypatch.setattr(open_loop, "load_open_loop_arrays", lambda *args, **kwargs: arrays)

    model_input = open_loop.load_model_input(tmp_path, {}, [1], 0)

    assert torch.equal(model_input["camera_indices"], torch.tensor([0, 1, 2, 6]))


def test_navigation_frame_forwards_prompt_weight_and_temperature(monkeypatch):
    calls = []

    def fake_run_inference(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros((1, 1, 1, 2, 3)), {"cot": ["safe"]}

    monkeypatch.setattr("module.inference.run_inference", fake_run_inference)

    open_loop.run_trajectory_frame(
        object(),
        object(),
        {},
        navigation_text="Turn left",
        navigation_weight=1.4,
        temperature=0.0,
    )

    assert calls[0]["navigation_text"] == "Turn left"
    assert calls[0]["navigation_weight"] == 1.4
    assert calls[0]["temperature"] == 0.0


def test_vqa_frame_returns_only_extracted_answer_and_latency(monkeypatch):
    monkeypatch.setattr(
        "module.inference.run_vqa",
        lambda *args, **kwargs: {"answer": ["<|answer_start|>Yes.<|answer_end|>"]},
    )

    answer, inference_time = open_loop.run_vqa_frame(
        object(), object(), {}, question="Is it flooded?", temperature=0.0
    )

    assert answer == "Yes."
    assert inference_time >= 0
