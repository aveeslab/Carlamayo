"""Alpamayo version adapters and the ``--version`` dispatcher."""

from __future__ import annotations

from .base import AlpamayoAdapter

SUPPORTED_VERSIONS = ("1", "1.5", "2", "2-edge")


def get_adapter(version: str) -> AlpamayoAdapter:
    """Return the adapter for one supported Alpamayo ``--version`` token.

    Adapters are imported lazily so selecting one version never imports another
    version's model package.
    """
    normalized = str(version).strip().lower()
    aliases = {
        "1": "1", "r1": "1", "1.0": "1", "alpamayo1": "1", "alpamayo-r1": "1",
        "1.5": "1.5", "15": "1.5", "alpamayo1.5": "1.5",
        "2": "2", "2.0": "2", "super": "2", "alpamayo2": "2",
        "2-edge": "2-edge", "2edge": "2-edge", "edge": "2-edge", "alpamayo2-edge": "2-edge",
    }
    resolved = aliases.get(normalized)
    if resolved == "1":
        from .alpamayo_r1 import AlpamayoR1Adapter

        return AlpamayoR1Adapter()
    if resolved == "1.5":
        from .alpamayo_1_5 import Alpamayo15Adapter

        return Alpamayo15Adapter()
    if resolved == "2":
        from .alpamayo_2 import Alpamayo2Adapter

        return Alpamayo2Adapter()
    if resolved == "2-edge":
        from .alpamayo_2_edge import Alpamayo2EdgeAdapter

        return Alpamayo2EdgeAdapter()
    raise ValueError(
        f"Unsupported Alpamayo version {version!r}. Expected one of: "
        f"{', '.join(SUPPORTED_VERSIONS)}."
    )


__all__ = ["AlpamayoAdapter", "SUPPORTED_VERSIONS", "get_adapter"]
