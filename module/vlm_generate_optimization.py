"""Runtime helpers for Alpamayo VLM generation memory profiling and safe optimization."""

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class VlmGenerateTiming:
    """Accumulate wall-time measurements for ``model.vlm.generate`` calls."""

    calls: int = 0
    total_time_sec: float = 0.0
    last_time_sec: float = 0.0

    def record(self, elapsed_sec: float) -> None:
        self.calls += 1
        self.last_time_sec = float(elapsed_sec)
        self.total_time_sec += float(elapsed_sec)

    @property
    def avg_time_sec(self) -> float:
        if self.calls <= 0:
            return 0.0
        return self.total_time_sec / self.calls

    def to_dict(self) -> dict[str, float | int]:
        return {
            "vlm_generate_calls": int(self.calls),
            "total_vlm_generate_time_sec": float(self.total_time_sec),
            "avg_vlm_generate_time_sec": float(self.avg_time_sec),
            "last_vlm_generate_time_sec": float(self.last_time_sec),
        }


@contextmanager
def optimized_vlm_generate(
    model,
    *,
    disable_output_logits: bool,
    timing: VlmGenerateTiming | None = None,
):
    """Patch ``model.vlm.generate`` during one inference call.

    Alpamayo's non-CFG trajectory path does not consume returned generation
    logits. Disabling those returned logits avoids a large per-token allocation
    while preserving sampling, generated sequences, KV cache behavior, diffusion
    settings, and the original image-token budget.
    """

    vlm = getattr(model, "vlm", None)
    if vlm is None or not hasattr(vlm, "generate"):
        yield
        return

    original_generate = vlm.generate

    def wrapped_generate(*args, **kwargs):
        generation_config = kwargs.get("generation_config")
        restore_output_logits = None
        if (
            disable_output_logits
            and generation_config is not None
            and hasattr(generation_config, "output_logits")
        ):
            restore_output_logits = generation_config.output_logits
            generation_config.output_logits = False

        start = perf_counter()
        try:
            return original_generate(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            if timing is not None:
                timing.record(elapsed)
            if restore_output_logits is not None:
                generation_config.output_logits = restore_output_logits

    vlm.generate = wrapped_generate
    try:
        yield
    finally:
        vlm.generate = original_generate
