from pathlib import Path

import numpy as np
import pytest
import torch

from module.visualization import (
    VideoRecorder,
    create_open_loop_visualization_frame,
    create_visualization_frame,
    project_trajectory_to_image,
    save_open_loop_video,
)


def test_project_trajectory_to_image_draws_selected_path_in_red():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    trajectory = np.array([[[2.0, 0.0, 0.0], [5.0, 0.1, 0.0], [8.0, 0.2, 0.0]]])

    rendered = project_trajectory_to_image(image, trajectory, selected_idx=0)

    assert rendered.shape == image.shape
    assert rendered[..., 0].max() == 255
    assert rendered.sum() > 0


def test_project_trajectory_to_image_accepts_torch_tensor():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    trajectory = torch.tensor([[[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]], dtype=torch.float32)

    rendered = project_trajectory_to_image(image, trajectory)

    assert rendered.shape == image.shape
    assert rendered.sum() > 0


def test_project_trajectory_to_image_rejects_invalid_rank():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    trajectory = np.zeros((2, 3, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Expected trajectory with ndim 2 or 3"):
        project_trajectory_to_image(image, trajectory)


def test_create_visualization_frame_preserves_rgb_shape_and_adds_overlay():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    trajectory = np.array([[[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]]])

    frame = create_visualization_frame(
        image,
        trajectory,
        selected_idx=0,
        frame_count=3,
        inference_time=0.25,
        cot_text="Stop because the light is red.",
        speed_kmh=12.0,
        steering=0.1,
        navigation_text="Stop at the light",
        navigation_weight=1.0,
        paused=True,
    )

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8
    assert frame.sum() > 0


def test_create_open_loop_visualization_frame_adds_header_and_cot_overlay():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    trajectory = np.array([[[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]]])

    frame = create_open_loop_visualization_frame(
        image,
        trajectory,
        frame_count=1,
        total_frames=2,
        inference_time=0.12,
        cot_text="Follow the lane.",
    )

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8
    assert frame.sum() > 0


def test_video_recorder_no_frames_does_not_create_output(tmp_path):
    output_path = tmp_path / "empty.mp4"

    VideoRecorder(output_path).save()

    assert not output_path.exists()


def test_save_open_loop_video_writes_nonempty_video(tmp_path):
    output_path = tmp_path / "open_loop.mp4"
    predictions = [np.array([[[2.0, 0.0, 0.0], [5.0, 0.2, 0.0], [8.0, 0.4, 0.0]]])]
    camera_images = [np.zeros((80, 120, 3), dtype=np.uint8)]

    save_open_loop_video(
        predictions,
        camera_images,
        ["Follow the lane."],
        [0.1],
        output_path,
        fps=5,
    )

    assert Path(output_path).exists()
    assert output_path.stat().st_size > 0
