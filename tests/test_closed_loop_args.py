import sys

import carla_alpamayo_closed_loop as closed_loop


def test_closed_loop_defaults_to_town03(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.carla_map == "Town03"


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


def test_closed_loop_defaults_normal_inference_interval_to_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.normal_inference_interval_frames == 0


def test_closed_loop_accepts_baseline_normal_inference_interval(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--normal-inference-interval-frames", "0"],
    )

    args = closed_loop.parse_args()

    assert args.normal_inference_interval_frames == 0


def test_closed_loop_accepts_latency_benchmark_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "carla_alpamayo_closed_loop.py",
            "--max-frames",
            "120",
            "--no-video",
            "--latency-stats-json",
            "run.json",
        ],
    )

    args = closed_loop.parse_args()

    assert args.max_frames == 120
    assert args.no_video is True
    assert args.latency_stats_json == "run.json"


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


def test_closed_loop_defaults_vlm_image_pixels_to_fast_balanced_cap(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.vlm_image_pixels == 65536


def test_closed_loop_accepts_full_vlm_image_pixel_baseline(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["carla_alpamayo_closed_loop.py", "--vlm-image-pixels", "196608"],
    )

    args = closed_loop.parse_args()

    assert args.vlm_image_pixels == 196608


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
