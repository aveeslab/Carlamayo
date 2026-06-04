"""Tests for CARLA 0.10 model-loading quantization policy."""

import importlib
import sys
import types

from module.model_quantization import resolve_effective_quantization


def _quantization_arg_kwargs(script_path):
    import ast
    from pathlib import Path

    tree = ast.parse(Path(script_path).read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue

        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--quantization":
            return {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }

    raise AssertionError(f"{script_path} does not define --quantization")


def test_effective_quantization_is_forced_even_when_cli_default_is_false():
    decision = resolve_effective_quantization(requested=False)

    assert decision.requested is False
    assert decision.effective is True
    assert decision.forced is True


def test_effective_quantization_preserves_explicit_true_request():
    decision = resolve_effective_quantization(requested=True)

    assert decision.requested is True
    assert decision.effective is True
    assert decision.forced is False


def test_shared_load_model_uses_effective_quantization_policy():
    from pathlib import Path

    source = Path("module/inference.py").read_text()

    assert "resolve_effective_quantization(use_quantization)" in source
    assert "if quantization.effective:" in source


def test_cli_quantization_request_defaults_remain_false():
    for script_path in ("carlamayo_open_loop.py", "carlamayo_closed_loop.py"):
        kwargs = _quantization_arg_kwargs(script_path)

        assert kwargs["action"] == "store_true"
        assert kwargs["default"] is False


def test_load_model_forces_quantized_pretrained_call_when_request_default_is_false(monkeypatch):
    calls = []

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAlpamayo:
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return cls()

    fake_helper = types.SimpleNamespace(get_processor=lambda tokenizer: ("processor", tokenizer))
    fake_pkg = types.ModuleType("alpamayo1_5")
    fake_pkg.__path__ = []
    fake_pkg.helper = fake_helper
    fake_config_module = types.ModuleType("alpamayo1_5.config")
    fake_config_module.Alpamayo1_5Config = type(
        "Alpamayo1_5Config",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    fake_models_pkg = types.ModuleType("alpamayo1_5.models")
    fake_model_module = types.ModuleType("alpamayo1_5.models.alpamayo1_5")
    fake_model_module.Alpamayo1_5 = FakeAlpamayo

    monkeypatch.setitem(sys.modules, "alpamayo1_5", fake_pkg)
    monkeypatch.setitem(sys.modules, "alpamayo1_5.config", fake_config_module)
    monkeypatch.setitem(sys.modules, "alpamayo1_5.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "alpamayo1_5.models.alpamayo1_5", fake_model_module)
    monkeypatch.setattr("transformers.BitsAndBytesConfig", FakeBitsAndBytesConfig)
    sys.modules.pop("module.inference", None)

    inference = importlib.import_module("module.inference")
    model, processor = inference.load_model(False, device_map="auto")

    assert isinstance(model, FakeAlpamayo)
    assert processor == ("processor", FakeAlpamayo.tokenizer)
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert isinstance(kwargs["quantization_config"], FakeBitsAndBytesConfig)
    assert kwargs["quantization_config"].kwargs["load_in_4bit"] is True
    assert kwargs["device_map"] == "auto"
