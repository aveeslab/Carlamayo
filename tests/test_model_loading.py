from module import inference


class FakeModel:
    tokenizer = object()

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_quantized_load_model_uses_auto_device_map_by_default(monkeypatch):
    calls = []
    fake_model = FakeModel()

    def fake_from_pretrained(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return fake_model

    monkeypatch.setattr(inference.Alpamayo1_5, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(inference.helper, "get_processor", lambda tokenizer: "processor")

    model, processor = inference.load_model(use_quantization=True)

    assert model is fake_model
    assert processor == "processor"
    assert calls[0][0] == "nvidia/Alpamayo-1.5-10B"
    assert calls[0][1]["device_map"] == "auto"
    assert fake_model.moved_to is None


def test_full_precision_load_model_uses_device_map_when_requested(monkeypatch):
    calls = []
    fake_model = FakeModel()

    def fake_from_pretrained(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return fake_model

    monkeypatch.setattr(inference.Alpamayo1_5, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(inference.helper, "get_processor", lambda tokenizer: "processor")

    model, _processor = inference.load_model(use_quantization=False, device_map="auto")

    assert model is fake_model
    assert calls[0][0] == "nvidia/Alpamayo-1.5-10B"
    assert calls[0][1]["device_map"] == "auto"
    assert fake_model.moved_to is None


def test_configure_cuda_linalg_library_sets_preferred_backend(monkeypatch):
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    def fake_preferred_linalg_library(library=None):
        calls.append(library)
        return "old-backend"

    monkeypatch.setattr(inference.torch, "cuda", FakeCuda())
    monkeypatch.setattr(
        inference.torch.backends.cuda,
        "preferred_linalg_library",
        fake_preferred_linalg_library,
    )

    previous = inference.configure_cuda_linalg_library("magma")

    assert previous == "old-backend"
    assert calls == ["magma"]


def test_configure_cuda_linalg_library_rejects_unknown_backend():
    import pytest

    with pytest.raises(ValueError, match="Unsupported CUDA linalg"):
        inference.configure_cuda_linalg_library("cublas")
