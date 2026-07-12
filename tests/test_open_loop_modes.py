import importlib
import json

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


def test_vqa_document_checkpoints_full_frame_identity(tmp_path):
    results = importlib.import_module("module.open_loop_results")
    document = results.create_vqa_document("dataset", "Is it flooded?", temperature=0.0)
    results.append_vqa_frame(document, 1, 77, "Yes, standing water is visible.", 1.25)
    output = tmp_path / "answers.json"

    results.write_json_atomic(document, output)

    saved = json.loads(output.read_text())
    assert saved["question"] == "Is it flooded?"
    assert saved["frames"] == [
        {
            "frame_index": 1,
            "carla_frame_id": 77,
            "answer": "Yes, standing water is visible.",
            "inference_time_seconds": 1.25,
        }
    ]
    assert not (tmp_path / "answers.json.tmp").exists()


def test_vqa_main_processes_every_frame_without_trajectory_inference(tmp_path, monkeypatch):
    frame_ids = [101, 102]
    answers = iter([("No standing water.", 0.4), ("Yes, flooding is visible.", 0.5)])
    captured_video = {}

    monkeypatch.setattr(open_loop, "load_trajectory_index", lambda data_root: ({}, frame_ids))
    monkeypatch.setattr(open_loop, "load_open_loop_model", lambda args: (object(), object()))
    monkeypatch.setattr(open_loop, "load_model_input", lambda *args, **kwargs: {})
    monkeypatch.setattr(open_loop, "run_vqa_frame", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(
        open_loop,
        "run_trajectory_frame",
        lambda *args, **kwargs: pytest.fail("trajectory inference ran in VQA mode"),
    )
    monkeypatch.setattr(
        open_loop,
        "load_front_camera_image",
        lambda *args, **kwargs: np.zeros((80, 120, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "module.visualization.save_open_loop_vqa_video",
        lambda camera_images, answer_texts, inference_times, question, output_path, fps=5: (
            captured_video.update(
                images=len(camera_images), answers=list(answer_texts), question=question
            )
        ),
        raising=False,
    )
    output_json = tmp_path / "answers.json"

    open_loop.main(
        [
            "--mode",
            "vqa",
            "--vqa-question",
            "Is the road flooded?",
            "--data-root",
            str(tmp_path),
            "--output-video",
            str(tmp_path / "answers.mp4"),
            "--output-json",
            str(output_json),
        ]
    )

    saved = json.loads(output_json.read_text())
    assert [item["carla_frame_id"] for item in saved["frames"]] == frame_ids
    assert captured_video == {
        "images": 2,
        "answers": ["No standing water.", "Yes, flooding is visible."],
        "question": "Is the road flooded?",
    }
