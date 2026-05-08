import sys

import pytest

import carla_alpamayo_closed_loop as closed_loop


def test_closed_loop_does_not_expose_carla_map_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert not hasattr(args, "carla_map")


def test_closed_loop_rejects_carla_map_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--carla-map", "Town05"],
    )

    with pytest.raises(SystemExit):
        closed_loop.parse_args()


def test_closed_loop_defaults_to_auto_device_map(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.device_map == "auto"


def test_closed_loop_defaults_to_magma_cuda_linalg(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.cuda_linalg_library == "magma"


def test_closed_loop_defaults_to_normal_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.mode == "normal"


def test_closed_loop_does_not_expose_auto_respawn_toggle(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert not hasattr(args, "auto_respawn")
    assert not hasattr(args, "respawn_collision_cooldown_frames")


def test_closed_loop_rejects_auto_respawn_disable_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--no-auto-respawn"],
    )

    with pytest.raises(SystemExit):
        closed_loop.parse_args()


def test_closed_loop_defaults_to_full_precision_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.quantization is False


def test_closed_loop_quantization_flag_enables_4bit_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--quantization"],
    )

    args = closed_loop.parse_args()

    assert args.quantization is True


def test_closed_loop_rejects_no_quantization_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--no-quantization"],
    )

    with pytest.raises(SystemExit):
        closed_loop.parse_args()


def test_closed_loop_disables_unused_generate_logits_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.disable_unused_generate_logits is True


def test_closed_loop_can_keep_generate_logits_for_baseline(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--keep-generate-logits"],
    )

    args = closed_loop.parse_args()

    assert args.disable_unused_generate_logits is False


def test_closed_loop_does_not_expose_vlm_image_pixels_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert not hasattr(args, "vlm_image_pixels")


def test_closed_loop_rejects_vlm_image_pixels_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--vlm-image-pixels", "65536"],
    )

    with pytest.raises(SystemExit):
        closed_loop.parse_args()


def test_closed_loop_pygame_ui_starts_paused(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--pygame-ui"],
    )

    args = closed_loop.parse_args()

    assert args.pygame_ui is True
    assert args.start_paused is True
    assert args.pygame_ui_video == "carla_alpamayo_closed_loop_result_pygame_ui.mp4"


def test_closed_loop_without_pygame_ui_does_not_start_paused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.pygame_ui is False
    assert args.start_paused is False
    assert args.pygame_ui_video is None


def test_pygame_ui_video_path_is_derived_from_output_video():
    assert (
        closed_loop.derive_pygame_ui_video_path("results/closed-loop.mp4")
        == "results/closed-loop_pygame_ui.mp4"
    )


def test_vqa_answer_preview_does_not_turn_empty_answer_into_ellipsis():
    assert closed_loop.format_vqa_answer_preview("") == "(empty answer)"
    assert closed_loop.format_vqa_answer_preview("clear", limit=10) == "clear"
    assert closed_loop.format_vqa_answer_preview("a" * 12, limit=10) == "aaaaaaaaaa..."


def test_closed_loop_rejects_start_paused_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--start-paused"],
    )

    with pytest.raises(SystemExit):
        closed_loop.parse_args()


def test_closed_loop_accepts_vqa_mode_and_initial_question(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "carla_alpamayo_closed_loop.py",
            "--mode",
            "vqa",
            "--vqa-question",
            "Describe the scene.",
        ],
    )

    args = closed_loop.parse_args()

    assert args.mode == "vqa"
    assert args.vqa_question == "Describe the scene."


def test_capture_initial_ui_frame_ticks_and_returns_front_wide_camera():
    import numpy as np

    frame0 = np.zeros((2, 2, 3), dtype=np.uint8)
    frame1 = np.ones((2, 2, 3), dtype=np.uint8)

    class FakeCarlaInterface:
        def __init__(self):
            self.tick_count = 0
            self.history = []

        def tick(self):
            self.tick_count += 1

        def get_ego_state(self):
            return {"speed": 2.0}

        def update_history(self, state):
            self.history.append(state)

        def get_camera_images(self):
            return np.array([frame0, frame1])

    fake = FakeCarlaInterface()

    frame_count, ui_frame, telemetry = closed_loop.capture_initial_ui_frame(fake, frame_count=4)

    assert frame_count == 5
    assert fake.tick_count == 1
    assert fake.history == [{"speed": 2.0}]
    assert np.array_equal(ui_frame, frame1)
    assert telemetry["frame"] == 5
    assert telemetry["speed_kmh"] == 7.2
