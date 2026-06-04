"""Model-loading quantization policy shared by Alpamayo entrypoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationDecision:
    """Requested CLI value and effective model-loading value."""

    requested: bool
    effective: bool
    forced: bool


def resolve_effective_quantization(requested: bool) -> QuantizationDecision:
    """Resolve the effective model-loading quantization from the CLI request."""

    requested_bool = bool(requested)
    return QuantizationDecision(
        requested=requested_bool,
        effective=requested_bool,
        forced=False,
    )
