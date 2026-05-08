from types import SimpleNamespace

import numpy as np
import torch

from module import inference


class FakeFrames:
    def flatten(self, start_dim, end_dim):
        assert (start_dim, end_dim) == (0, 1)
        return "flattened-frames"


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": "fake-ids", "messages": messages, "kwargs": kwargs}


class FakeModel:
    def __init__(self):
        self.normal_kwargs = None
        self.cfg_kwargs = None
        self.vqa_kwargs = None

    def sample_trajectories_from_data_with_vlm_rollout(self, **kwargs):
        self.normal_kwargs = kwargs
        return "pred-normal", "rot-normal", {"cot": ["normal"]}

    def sample_trajectories_from_data_with_vlm_rollout_cfg_nav(self, **kwargs):
        self.cfg_kwargs = kwargs
        return "pred-cfg", "rot-cfg", {"cot": ["cfg"]}

    def generate_text(self, **kwargs):
        self.vqa_kwargs = kwargs
        return {"answer": ["There is a traffic light ahead."]}


def _patch_helper(monkeypatch):
    seen = {}

    def fake_create_message(
        frames,
        camera_indices=None,
        num_frames_per_camera=4,
        nav_text=None,
        use_nav_prompt=False,
    ):
        seen["frames"] = frames
        seen["camera_indices"] = camera_indices
        seen["nav_text"] = nav_text
        seen["use_nav_prompt"] = use_nav_prompt
        return [{"role": "user", "content": []}]

    def fake_to_device(model_inputs, device):
        seen["device"] = device
        return model_inputs

    monkeypatch.setattr(inference.helper, "create_message", fake_create_message)
    monkeypatch.setattr(inference.helper, "to_device", fake_to_device)
    return seen


def test_run_inference_passes_navigation_text_to_chat_prompt(monkeypatch):
    seen = _patch_helper(monkeypatch)
    model = FakeModel()
    data = {
        "image_frames": FakeFrames(),
        "ego_history_xyz": "history-xyz",
        "ego_history_rot": "history-rot",
    }

    pred, extra = inference.run_inference(
        model,
        FakeProcessor(),
        data,
        navigation_text="Turn right in 30m",
        navigation_weight=1.0,
    )

    assert pred == "pred-normal"
    assert extra == {"cot": ["normal"]}
    assert seen["nav_text"] == "Turn right in 30m"
    assert seen["device"] == "cuda"
    assert model.normal_kwargs["diffusion_kwargs"] == {"inference_step": 10}
    assert model.cfg_kwargs is None


def test_run_inference_uses_cfg_nav_when_navigation_weight_is_not_one(monkeypatch):
    _patch_helper(monkeypatch)
    model = FakeModel()
    data = {
        "image_frames": FakeFrames(),
        "ego_history_xyz": "history-xyz",
        "ego_history_rot": "history-rot",
    }

    pred, extra = inference.run_inference(
        model,
        FakeProcessor(),
        data,
        navigation_text="Turn left in 20m",
        navigation_weight=1.5,
    )

    assert pred == "pred-cfg"
    assert extra == {"cot": ["cfg"]}
    assert model.normal_kwargs is None
    assert model.cfg_kwargs["diffusion_kwargs"] == {
        "inference_step": 10,
        "use_classifier_free_guidance": True,
        "inference_guidance_weight": 1.5,
    }


def test_run_inference_passes_vlm_image_pixel_cap_to_processor(monkeypatch):
    _patch_helper(monkeypatch)
    model = FakeModel()
    processor = FakeProcessor()
    data = {
        "image_frames": FakeFrames(),
        "ego_history_xyz": "history-xyz",
        "ego_history_rot": "history-rot",
    }

    inference.run_inference(
        model,
        processor,
        data,
        vlm_image_pixels=196608,
    )

    tokenized_data = model.normal_kwargs["data"]["tokenized_data"]
    assert tokenized_data["kwargs"]["min_pixels"] == 196608
    assert tokenized_data["kwargs"]["max_pixels"] == 196608


def test_run_vqa_passes_question_to_vqa_prompt(monkeypatch):
    seen = {}

    def fake_create_vqa_message(
        frames,
        question,
        camera_indices=None,
        num_frames_per_camera=4,
    ):
        seen["frames"] = frames
        seen["question"] = question
        seen["camera_indices"] = camera_indices
        return [{"role": "user", "content": []}]

    def fake_to_device(model_inputs, device):
        seen["device"] = device
        return model_inputs

    monkeypatch.setattr(inference.helper, "create_vqa_message", fake_create_vqa_message)
    monkeypatch.setattr(inference.helper, "to_device", fake_to_device)

    model = FakeModel()
    data = {
        "image_frames": FakeFrames(),
        "camera_indices": "camera-indices",
    }

    extra = inference.run_vqa(
        model,
        FakeProcessor(),
        data,
        question="What is visible?",
    )

    assert extra == {"answer": ["There is a traffic light ahead."]}
    assert seen["question"] == "What is visible?"
    assert seen["camera_indices"] == "camera-indices"
    assert seen["device"] == "cuda"
    assert model.vqa_kwargs["max_generation_length"] == 256


def test_run_vqa_preserves_notebook_vqa_message_template(monkeypatch):
    seen = {}

    def fake_create_vqa_message(
        frames,
        question,
        camera_indices=None,
        num_frames_per_camera=4,
    ):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a driving assistant that generates safe actions.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "frame"},
                    {"type": "text", "text": f"<|question_start|>{question}<|question_end|>"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<|answer_start|>"}],
            },
        ]

    class CapturingProcessor(FakeProcessor):
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(inference.helper, "create_vqa_message", fake_create_vqa_message)
    monkeypatch.setattr(inference.helper, "to_device", lambda model_inputs, device: model_inputs)

    inference.run_vqa(
        FakeModel(),
        CapturingProcessor(),
        {
            "image_frames": FakeFrames(),
            "camera_indices": "camera-indices",
        },
        question="what is the color of traffic light?",
    )

    system_text = seen["messages"][0]["content"][0]["text"]
    user_text = seen["messages"][1]["content"][1]["text"]

    assert system_text == "You are a driving assistant that generates safe actions."
    assert "<|question_start|>what is the color of traffic light?<|question_end|>" in user_text


def test_extract_answer_text_handles_nested_answer():
    assert (
        inference.extract_answer_text({"answer": [["The lane is clear."]]})
        == "The lane is clear."
    )


def test_extract_answer_text_falls_back_to_raw_answer_when_structured_answer_is_empty():
    extra = {
        "answer": np.array([[""]]),
        "raw_answer": np.array([["There is a red traffic light ahead.<|im_end|>"]]),
    }

    assert inference.extract_answer_text(extra) == "There is a red traffic light ahead."


def test_run_vqa_keeps_partial_answer_when_answer_end_token_is_missing(monkeypatch):
    seen = {}

    def fake_create_vqa_message(
        frames,
        question,
        camera_indices=None,
        num_frames_per_camera=4,
    ):
        seen["frames"] = frames
        seen["question"] = question
        seen["camera_indices"] = camera_indices
        return [{"role": "user", "content": []}]

    def fake_to_device(model_inputs, device):
        seen["device"] = device
        return model_inputs

    class FakeProcessorWithTensorIds:
        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": torch.tensor([[11, 12, 13]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "messages": messages,
                "kwargs": kwargs,
            }

    class FakeTokenizer:
        pad_token_id = 0

        def batch_decode(self, output_tokens, skip_special_tokens=False):
            seen["decoded_tokens_shape"] = tuple(output_tokens.shape)
            seen["skip_special_tokens"] = skip_special_tokens
            return ["There is a red traffic light ahead.<|im_end|>"]

    class FakeVlm:
        def __init__(self):
            self.generation_config = SimpleNamespace()
            self.seen_generation_config = None

        def generate(self, input_ids, generation_config, **tokenized_data):
            seen["input_ids_shape"] = tuple(input_ids.shape)
            seen["attention_mask_shape"] = tuple(tokenized_data["attention_mask"].shape)
            self.seen_generation_config = generation_config
            generated_suffix = torch.tensor([[101, 102, 103]])
            return SimpleNamespace(sequences=torch.cat([input_ids, generated_suffix], dim=1))

    class FakeVqaModel:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.vlm = FakeVlm()

    monkeypatch.setattr(inference.helper, "create_vqa_message", fake_create_vqa_message)
    monkeypatch.setattr(inference.helper, "to_device", fake_to_device)

    model = FakeVqaModel()
    extra = inference.run_vqa(
        model,
        FakeProcessorWithTensorIds(),
        {
            "image_frames": FakeFrames(),
            "camera_indices": "camera-indices",
        },
        question="What is visible?",
    )

    assert inference.extract_answer_text(extra) == "There is a red traffic light ahead."
    assert seen["question"] == "What is visible?"
    assert seen["camera_indices"] == "camera-indices"
    assert seen["device"] == "cuda"
    assert seen["decoded_tokens_shape"] == (1, 3)
    assert seen["skip_special_tokens"] is False
    assert model.vlm.seen_generation_config.max_new_tokens == 256
