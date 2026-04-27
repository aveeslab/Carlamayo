"""Compatibility helpers for loading released Alpamayo configs.

The Hugging Face Alpamayo-R1/1.5 config can contain Hydra ``_target_``
strings that reference the historical ``alpamayo_r1`` package name.  This
repository installs NVIDIA's current ``alpamayo1_5`` package from the submodule,
so we normalize those target strings at config-construction time without editing
or vendoring the third-party source tree.
"""

from __future__ import annotations

import copy
from typing import Any


def _rewrite_legacy_hydra_targets(value: Any) -> Any:
    """Return a copy with legacy Hydra targets rewritten to ``alpamayo1_5``."""
    if isinstance(value, dict):
        rewritten: dict[Any, Any] = {}
        for key, item in value.items():
            if key == "_target_" and isinstance(item, str):
                rewritten[key] = item.replace("alpamayo_r1.", "alpamayo1_5.")
            else:
                rewritten[key] = _rewrite_legacy_hydra_targets(item)
        return rewritten

    if isinstance(value, list):
        return [_rewrite_legacy_hydra_targets(item) for item in value]

    return value


def patch_legacy_hydra_targets() -> None:
    """Patch Alpamayo config loading once for legacy Hugging Face targets."""
    from alpamayo1_5.config import Alpamayo1_5Config

    if getattr(Alpamayo1_5Config, "_carlamayo_legacy_target_patch", False):
        return

    original_init = Alpamayo1_5Config.__init__

    def patched_init(
        self,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
        **kwargs: Any,
    ) -> None:
        original_init(
            self,
            diffusion_cfg=_rewrite_legacy_hydra_targets(copy.deepcopy(diffusion_cfg)),
            action_space_cfg=_rewrite_legacy_hydra_targets(copy.deepcopy(action_space_cfg)),
            action_in_proj_cfg=_rewrite_legacy_hydra_targets(copy.deepcopy(action_in_proj_cfg)),
            action_out_proj_cfg=_rewrite_legacy_hydra_targets(copy.deepcopy(action_out_proj_cfg)),
            expert_cfg=_rewrite_legacy_hydra_targets(copy.deepcopy(expert_cfg)),
            keep_same_dtype=keep_same_dtype,
            expert_non_causal_attention=expert_non_causal_attention,
            include_camera_ids=include_camera_ids,
            include_frame_nums=include_frame_nums,
            **_rewrite_legacy_hydra_targets(copy.deepcopy(kwargs)),
        )

    Alpamayo1_5Config.__init__ = patched_init
    Alpamayo1_5Config._carlamayo_legacy_target_patch = True
