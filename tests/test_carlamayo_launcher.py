import pytest

import carlamayo


def test_version_and_loop_are_required():
    parser = carlamayo.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--version", "2"])  # missing --loop
    with pytest.raises(SystemExit):
        parser.parse_args(["--loop", "open"])  # missing --version


def test_invalid_version_and_loop_rejected():
    parser = carlamayo.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version", "3", "--loop", "open"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--version", "2", "--loop", "sideways"])


def test_preset_loop_drops_loop_flag():
    parser = carlamayo.build_parser(preset_loop="closed")
    args = parser.parse_args(["--version", "1.5"])
    assert args.version == "1.5"
    assert not hasattr(args, "loop")


def test_parsed_defaults_and_overrides():
    parser = carlamayo.build_parser()
    args = parser.parse_args(
        ["--version", "2", "--loop", "live-open", "--async", "--navigation-text", "Turn right"]
    )
    assert args.version == "2"
    assert args.loop == "live-open"
    assert args.async_mode is True
    assert args.navigation_text == "Turn right"
    assert args.mode == "normal"
    assert args.device_map == "auto"
    assert args.output_video is None


def test_max_gpu_memory_caps_each_gpu_and_offloads_the_rest_to_cpu(monkeypatch):
    import torch

    from module.loops._common import model_max_memory

    parser = carlamayo.build_parser()
    args = parser.parse_args(["--version", "2", "--loop", "open", "--max-gpu-memory-gb", "38"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    max_memory = model_max_memory(args)

    assert max_memory[0] == "38.0GiB"
    assert max_memory["cpu"] > 0
    assert model_max_memory(parser.parse_args(["--version", "2", "--loop", "open"])) is None


def test_stack_frame_buffer_picks_ticks_before_the_newest_frame():
    import numpy as np

    from module.loops._common import stack_frame_buffer

    # Five ticks of a two-camera rig; pixel value = tick index, camera 1 offset by 100.
    buffer = [np.array([np.full((2, 2, 3), t), np.full((2, 2, 3), 100 + t)]) for t in range(5)]

    images = stack_frame_buffer(buffer, frame_ticks=(4, 1, 0))

    assert images.shape == (2, 3, 2, 2, 3)
    assert images[:, :, 0, 0, 0].tolist() == [[0, 3, 4], [100, 103, 104]]
