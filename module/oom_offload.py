# SPDX-FileCopyrightText: Copyright (c) 2026 AVEES Lab
# SPDX-License-Identifier: Apache-2.0
#
# Optional OOM-free Alpamayo loading via CPU-GPU layer streaming.

"""OOM-free Alpamayo loading via CPU<->GPU demand layering.

This is the integration glue between Carlamayo and ``oom-free-alpamayo``
(``third_party/oom-free-alpamayo``). When the entry scripts are run with
``--oom-free``, Alpamayo's VLM / ViT / Expert transformer layers are kept in
pinned host memory and streamed to the GPU on demand (``alpamayo_memopt``'s
``TriHookPipeline``), while the always-resident modules and a planned subset of
VLM layers stay on the GPU. This keeps Alpamayo's peak VRAM low enough to run
**alongside a live CARLA server** without quantization or accuracy loss.

The residency plan (how many VLM layers stay GPU-resident) is computed at load
time from the *currently free* VRAM, so it adapts automatically to however much
the running CARLA server is already using — no separate profiling step or
``config.json`` is required.

Design notes
------------
* The returned model is a normal :class:`alpamayo1_5.models.alpamayo1_5.Alpamayo1_5`
  with a streaming pipeline attached as ``model._oom_pipeline``. ``module.inference``
  calls ``model._oom_pipeline.start_iteration()`` before every inference, so the
  rest of the open/closed-loop code paths are unchanged.
* Offloading streams full-precision (bf16) layers, so it is mutually exclusive
  with ``--quantization``.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from . import config as cfg

# Default activation/transient headroom (GB) reserved on top of essentials +
# resident layers + double-flat-buffer, to absorb the ViT/diffusion activation
# spikes. Higher = safer (more streaming, slightly slower).
DEFAULT_HEADROOM_GB = 3.0
DEFAULT_MARGIN = 2

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OOM_FREE_PKG = _REPO_ROOT / "third_party" / "oom-free-alpamayo"


def _ensure_memopt_importable():
    """Make ``alpamayo_memopt`` importable even if it was not pip-installed."""
    try:
        import alpamayo_memopt  # noqa: F401
        return
    except ImportError:
        pass
    if _OOM_FREE_PKG.is_dir() and str(_OOM_FREE_PKG) not in sys.path:
        sys.path.insert(0, str(_OOM_FREE_PKG))
    import alpamayo_memopt  # noqa: F401  (raise a clear ImportError if still missing)


def _free_vram_gb(device) -> float:
    free_b, _total_b = torch.cuda.mem_get_info(device)
    return free_b / (1024 ** 3)


def plan_resident_layers(
    num_layers: int,
    layer_size_mb: float,
    free_after_essentials_gb: float,
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    margin: int = DEFAULT_MARGIN,
):
    """Pick how many / which VLM layers stay GPU-resident given free VRAM.

    Returns ``(resident_indices, num_resident, max_possible)``. The double-flat
    buffer (~2 layers) and ``headroom_gb`` of activation space are subtracted
    from the free VRAM before filling the rest with resident layers.
    """
    from alpamayo_memopt.profiler import (
        apply_conservative_margin,
        interleaved_placement,
    )

    dfb_gb = 2.0 * layer_size_mb / 1024.0
    available_gb = free_after_essentials_gb - headroom_gb - dfb_gb
    if available_gb <= 0:
        max_possible = 0
    else:
        max_possible = int(available_gb * 1024.0 / layer_size_mb)
    max_possible = min(max_possible, num_layers)
    num_resident = apply_conservative_margin(max_possible, margin=margin, minimum=0)
    num_resident = min(num_resident, num_layers)
    resident = interleaved_placement(num_resident, num_layers) if num_resident > 0 else []
    return resident, len(resident), max_possible


def load_offloaded_model(
    device: str = "cuda",
    headroom_gb: float = DEFAULT_HEADROOM_GB,
    margin: int = DEFAULT_MARGIN,
    resident_override: int | None = None,
    attn_implementation: str | None = None,
    verbose: bool = True,
):
    """Load Alpamayo 1.5 with CPU<->GPU demand layering (oom-free-alpamayo).

    Mirrors ``module.inference.load_model`` (returns ``(model, processor)``) but
    the model's VLM/ViT/Expert layers stream from pinned host memory and a
    ``TriHookPipeline`` is attached as ``model._oom_pipeline``.
    """
    _ensure_memopt_importable()

    from alpamayo_memopt.models import TriHookPipeline, get_adapter
    from alpamayo_memopt.profiler import (
        get_vlm_layer_size_mb,
        verify_cpu_can_hold_weights,
    )
    from alpamayo1_5 import helper

    if not torch.cuda.is_available():
        raise RuntimeError("OOM-free offloading requires a CUDA device.")

    device = str(device)
    adapter = get_adapter("r15")

    # Minimal argparse-like namespace the r15 adapter's loader reads. ``None``
    # lets each knob resolve from env/config/fallback (HF cache by default).
    args = SimpleNamespace(
        alpamayo_src=None,
        model_id=None,
        model_cache_dir=None,
        model_revision=None,
        local_files_only=None,
        attn_implementation=attn_implementation,
        clip_id=None,
        t0_us=None,
        dataset_revisions=None,
    )

    if verbose:
        print("[oom-free] Loading Alpamayo 1.5 onto CPU (no quantization)...")
    loaded = adapter.load(args, config=None)

    # Fail fast (and clearly) if host RAM cannot hold the weights.
    verify_cpu_can_hold_weights(loaded.model)

    if verbose:
        print("[oom-free] Moving always-resident modules to GPU...")
    adapter.setup_essentials(loaded, device)
    torch.cuda.synchronize(device)

    essentials_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    free_after_essentials_gb = _free_vram_gb(device)
    layer_size_mb = get_vlm_layer_size_mb(loaded.vlm_layers)
    num_layers = len(loaded.vlm_layers)

    if resident_override is not None:
        from alpamayo_memopt.profiler import interleaved_placement

        n = max(0, min(int(resident_override), num_layers))
        resident = interleaved_placement(n, num_layers) if n > 0 else []
        max_possible = n
    else:
        resident, _n, max_possible = plan_resident_layers(
            num_layers,
            layer_size_mb,
            free_after_essentials_gb,
            headroom_gb=headroom_gb,
            margin=margin,
        )

    if verbose:
        print(
            f"[oom-free] VLM layers: {num_layers} x {layer_size_mb:.0f} MB | "
            f"essentials {essentials_gb:.2f} GB | free {free_after_essentials_gb:.2f} GB"
        )
        print(
            f"[oom-free] Resident VLM layers: {len(resident)}/{num_layers} "
            f"(max fit {max_possible}, margin {margin}, headroom {headroom_gb:.1f} GB)"
        )
        print(f"[oom-free] Streaming {num_layers - len(resident)} VLM + all ViT + all Expert layers")

    for i in resident:
        loaded.vlm_layers[i].to(device)

    pipeline = TriHookPipeline(
        loaded.vlm_layers,
        loaded.vit_blocks,
        loaded.expert_layers,
        vlm_resident=resident,
        device=device,
    )

    model = loaded.model
    processor = helper.get_processor(model.tokenizer)

    # Attached so module.inference.{run_inference,run_vqa} prime the pipeline
    # (start_iteration) before each inference without further script changes.
    model._oom_pipeline = pipeline

    torch.cuda.synchronize(device)
    if verbose:
        used_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"[oom-free] Alpamayo VRAM after setup: {used_gb:.2f} GB allocated")
    return model, processor
