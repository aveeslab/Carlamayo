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


def test_closed_loop_vqa_defaults_reduce_generation_memory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["carla_alpamayo_closed_loop.py"])

    args = closed_loop.parse_args()

    assert args.vqa_camera_index == 1
    assert args.vqa_num_frames == 1
    assert args.vqa_max_generation_length == 96


def test_closed_loop_accepts_vqa_memory_knobs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "carla_alpamayo_closed_loop.py",
            "--mode",
            "vqa",
            "--vqa-camera-index",
            "2",
            "--vqa-num-frames",
            "2",
            "--vqa-max-generation-length",
            "64",
        ],
    )

    args = closed_loop.parse_args()

    assert args.vqa_camera_index == 2
    assert args.vqa_num_frames == 2
    assert args.vqa_max_generation_length == 64


def test_build_vqa_images_array_selects_recent_frames_from_one_camera():
    import numpy as np

    frame0 = np.array(
        [
            np.full((2, 2, 3), 0, dtype=np.uint8),
            np.full((2, 2, 3), 10, dtype=np.uint8),
        ]
    )
    frame1 = np.array(
        [
            np.full((2, 2, 3), 1, dtype=np.uint8),
            np.full((2, 2, 3), 11, dtype=np.uint8),
        ]
    )
    frame2 = np.array(
        [
            np.full((2, 2, 3), 2, dtype=np.uint8),
            np.full((2, 2, 3), 12, dtype=np.uint8),
        ]
    )

    images_array = closed_loop.build_vqa_images_array(
        [frame0, frame1, frame2],
        camera_index=1,
        num_frames=2,
    )

    assert images_array.shape == (1, 2, 2, 2, 3)
    assert int(images_array[0, 0, 0, 0, 0]) == 11
    assert int(images_array[0, 1, 0, 0, 0]) == 12


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
