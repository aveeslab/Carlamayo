import numpy as np
import pytest
import torch

from module import inference


def test_model_name_uses_alpamayo_15_weights():
    assert inference.ALPAMAYO_MODEL_NAME == "nvidia/Alpamayo-1.5-10B"


def test_prepare_model_input_builds_expected_tensor_shapes_and_types():
    images = np.zeros((4, 4, 8, 12, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 16, axis=0)

    model_input = inference.prepare_model_input(images, history_xyz, history_rot)

    assert model_input["image_frames"].shape == (4, 4, 3, 8, 12)
    assert model_input["image_frames"].dtype == torch.uint8
    assert model_input["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert model_input["ego_history_xyz"].dtype == torch.float32
    assert model_input["ego_history_rot"].shape == (1, 1, 16, 3, 3)




def test_prepare_model_input_adds_carla_010_front_camera_indices():
    images = np.zeros((4, 4, 2, 3, 3), dtype=np.uint8)
    history_xyz = np.zeros((16, 3), dtype=np.float32)
    history_rot = np.zeros((16, 3, 3), dtype=np.float32)

    model_input = inference.prepare_model_input(images, history_xyz, history_rot)

    assert model_input["camera_indices"].tolist() == [0, 1, 2, 6]

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


def test_extract_answer_text_removes_special_tokens_and_terminators():
    extra = {
        "answer": np.array(
            ["<|answer_start|>The traffic light is red.<|answer_end|><|im_end|>"],
            dtype=object,
        )
    }

    assert inference.extract_answer_text(extra) == "The traffic light is red."


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
