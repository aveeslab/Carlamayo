import copy
from contextlib import nullcontext

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


def test_run_inference_forwards_temperature(monkeypatch):
    captured = {}

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    class Model:
        def sample_trajectories_from_data_with_vlm_rollout(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 1, 1, 2, 3)), None, {}

    monkeypatch.setattr(inference, "optimized_vlm_generate", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(inference.torch, "autocast", lambda *args, **kwargs: nullcontext())

    inference.run_inference(Model(), Processor(), _minimal_model_input(), temperature=0.0)

    assert captured["temperature"] == 0.0


@pytest.mark.parametrize(
    ("temperature", "expected_sampling", "expected_temperature"),
    [(0.0, False, None), (0.4, True, 0.4)],
)
def test_vqa_generation_selects_decoding_mode(
    temperature, expected_sampling, expected_temperature
):
    class GenerationConfig:
        top_p = None
        temperature = None
        do_sample = None
        num_return_sequences = None
        max_new_tokens = None
        output_logits = None
        return_dict_in_generate = None
        top_k = None
        pad_token_id = None

        def __deepcopy__(self, memo):
            duplicate = GenerationConfig()
            duplicate.__dict__.update(copy.deepcopy(self.__dict__, memo))
            return duplicate

    class VLM:
        generation_config = GenerationConfig()
        seen_config = None

        def generate(self, input_ids, generation_config, **kwargs):
            self.seen_config = generation_config
            return {"sequences": torch.tensor([[1, 2, 3]])}

    class Tokenizer:
        pad_token_id = 0

        def batch_decode(self, generated_tokens, skip_special_tokens=False):
            return ["<|answer_start|>Yes.<|answer_end|>"]

    class Model:
        vlm = VLM()
        tokenizer = Tokenizer()

    model = Model()
    result = inference._generate_vqa_text_with_partial_answer_fallback(
        model,
        {"tokenized_data": {"input_ids": torch.tensor([[1, 2]])}},
        temperature=temperature,
    )

    assert result["answer"].tolist() == [["Yes."]]
    assert model.vlm.seen_config.do_sample is expected_sampling
    assert model.vlm.seen_config.temperature == expected_temperature


def _minimal_model_input():
    return {
        "image_frames": torch.zeros((4, 4, 3, 2, 2), dtype=torch.uint8),
        "ego_history_xyz": torch.zeros((1, 1, 16, 3)),
        "ego_history_rot": torch.zeros((1, 1, 16, 3, 3)),
    }
