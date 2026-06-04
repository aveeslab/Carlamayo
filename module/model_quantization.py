"""Model-loading quantization policy for the CARLA 0.10 branch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationDecision:
    """Requested CLI value and effective model-loading value."""

    requested: bool
    effective: bool
    forced: bool


def resolve_effective_quantization(requested: bool) -> QuantizationDecision:
    """Force 4-bit loading while preserving the CLI default as a requested value."""

    requested_bool = bool(requested)
    return QuantizationDecision(
        requested=requested_bool,
        effective=True,
        forced=not requested_bool,
    )
