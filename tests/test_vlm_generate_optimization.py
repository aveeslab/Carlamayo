from types import SimpleNamespace

from module.vlm_generate_optimization import (
    VlmGenerateTiming,
    optimized_vlm_generate,
)


class FakeVlm:
    def __init__(self):
        self.observed_output_logits = None

    def generate(self, *args, generation_config=None, **kwargs):
        self.observed_output_logits = generation_config.output_logits
        return SimpleNamespace(sequences=[1, 2, 3])


def test_optimized_vlm_generate_disables_unused_logits_during_call_and_restores_config():
    model = SimpleNamespace(vlm=FakeVlm())
    generation_config = SimpleNamespace(output_logits=True)

    with optimized_vlm_generate(model, disable_output_logits=True):
        result = model.vlm.generate(generation_config=generation_config)

    assert result.sequences == [1, 2, 3]
    assert model.vlm.observed_output_logits is False
    assert generation_config.output_logits is True


def test_optimized_vlm_generate_keeps_logits_when_disabled_flag_is_false():
    model = SimpleNamespace(vlm=FakeVlm())
    generation_config = SimpleNamespace(output_logits=True)

    with optimized_vlm_generate(model, disable_output_logits=False):
        model.vlm.generate(generation_config=generation_config)

    assert model.vlm.observed_output_logits is True
    assert generation_config.output_logits is True


def test_optimized_vlm_generate_records_call_timing():
    model = SimpleNamespace(vlm=FakeVlm())
    timing = VlmGenerateTiming()

    with optimized_vlm_generate(model, disable_output_logits=True, timing=timing):
        model.vlm.generate(generation_config=SimpleNamespace(output_logits=True))
        model.vlm.generate(generation_config=SimpleNamespace(output_logits=True))

    assert timing.calls == 2
    assert timing.total_time_sec >= 0.0
    assert timing.last_time_sec >= 0.0
    assert timing.avg_time_sec >= 0.0
