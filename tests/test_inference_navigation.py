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
        vlm_image_pixels=65536,
    )

    tokenized_data = model.normal_kwargs["data"]["tokenized_data"]
    assert tokenized_data["kwargs"]["min_pixels"] == 65536
    assert tokenized_data["kwargs"]["max_pixels"] == 65536


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


def test_extract_answer_text_handles_nested_answer():
    assert inference.extract_answer_text({"answer": [["The lane is clear."]]}) == "The lane is clear."
