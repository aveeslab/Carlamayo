import numpy as np
import pytest
import torch

from module import inference


def test_model_name_uses_alpamayo_2_super_weights():
    assert inference.ALPAMAYO_MODEL_NAME == "nvidia/Alpamayo2-Super"


def test_prepare_model_input_builds_expected_source_ring():
    images = np.zeros((7, 4, 8, 12, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 16, axis=0)

    model_input = inference.prepare_model_input(images, history_xyz, history_rot, t0_us=5_000_000)

    assert model_input["image_frames"].shape == (7, 4, 3, 8, 12)
    assert model_input["image_frames"].dtype == torch.uint8
    assert model_input["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert model_input["ego_history_xyz"].dtype == torch.float32
    assert model_input["ego_history_rot"].shape == (1, 1, 16, 3, 3)
    assert model_input["camera_indices"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert model_input["camera_names"][1] == "camera_front_wide_120fov"
    assert model_input["absolute_timestamps"].shape == (7, 4)
    assert model_input["absolute_timestamps"][0].tolist() == [
        4_700_000,
        4_800_000,
        4_900_000,
        5_000_000,
    ]
    assert model_input["ego_t0"].tolist() == [5_000_000]
    assert model_input["ego_t0_frame_idx"].tolist() == [3]
    np.testing.assert_allclose(
        model_input["relative_timestamps"][0].numpy(),
        [0.0, 0.1, 0.2, 0.3],
        atol=1e-6,
    )


def test_prepare_model_input_rejects_partial_camera_rings():
    images = np.zeros((4, 4, 8, 12, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 16, axis=0)

    with pytest.raises(ValueError, match="Expected 7 source cameras"):
        inference.prepare_model_input(images, history_xyz, history_rot, t0_us=5_000_000)


def test_prepare_model_input_feeds_task_profile_selection():
    from alpamayo2_super.input_profiles import select_task_input

    images = np.zeros((7, 4, 8, 12, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 16, axis=0)

    model_input = inference.prepare_model_input(images, history_xyz, history_rot, t0_us=5_000_000)
    trajectory_input = select_task_input(model_input, "trajectory")
    vqa_input = select_task_input(model_input, "vqa")

    assert trajectory_input["camera_indices"].tolist() == [0, 1, 2, 3, 5, 6]
    assert trajectory_input["image_frames"].shape == (6, 4, 3, 8, 12)
    assert vqa_input["camera_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert vqa_input["image_frames"].shape == (6, 4, 3, 8, 12)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"cot": [[[["  reason about lanes  "]]]]}, "reason about lanes"),
        ({"cot": np.array(["stop for light"], dtype=object)}, "stop for light"),
        ({"cot": torch.tensor([7])}, "7"),
        ({}, ""),
    ],
)
def test_extract_cot_text_handles_nested_common_return_shapes(extra, expected):
    assert inference.extract_cot_text(extra) == expected


def test_extract_answer_text_prefers_answer_and_strips_special_tokens():
    extra = {
        "answer": np.array(["The traffic light is red.<|im_end|>"], dtype=object),
        "cot": np.array(["unused fallback"], dtype=object),
    }

    assert inference.extract_answer_text(extra) == "The traffic light is red."


def test_extract_answer_text_falls_back_to_cot_then_raw_outputs():
    assert inference.extract_answer_text({"cot": ["slow down"]}) == "slow down"
    assert inference.extract_answer_text({"raw_outputs": ["raw text"]}) == "raw text"
    assert inference.extract_answer_text({}) == ""


def test_extract_trajectory_samples_squeezes_batch_axes_and_keeps_xyz_only():
    pred_xyz = torch.arange(1 * 1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)

    samples = inference.extract_trajectory_samples(pred_xyz)

    assert samples.shape == (2, 3, 3)
    np.testing.assert_allclose(samples, pred_xyz.numpy()[0, 0, :, :, :3])


def test_select_trajectory_by_prev_similarity_prefers_closest_xy_path():
    previous = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    candidates = np.array(
        [
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
            [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
        ],
        dtype=np.float32,
    )

    best_idx, scores = inference.select_trajectory_by_prev_similarity(candidates, previous)

    assert best_idx == 1
    assert scores[1] < scores[0]
