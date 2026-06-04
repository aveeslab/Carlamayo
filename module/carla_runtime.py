"""CARLA 0.10 runtime command helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from . import config as cfg


RENDER_OFFSCREEN_ARG = "-RenderOffScreen"


def resolve_carla_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured CARLA root directory."""

    if root is not None:
        return Path(root).expanduser()

    env_value = os.environ.get("CARLA_010_ROOT")
    if env_value:
        return Path(env_value).expanduser()

    return Path(cfg.CARLA_AGENT_ROOT).expanduser()


def find_carla_server_executable(root: str | os.PathLike[str] | None = None) -> Path:
    """Find the CARLA 0.10 server launcher, preferring the UE5 name."""

    carla_root = resolve_carla_root(root)
    candidate = carla_root / "CarlaUnreal.sh"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"CARLA 0.10 server launcher not found; expected {candidate}")


def build_carla_server_command(
    root: str | os.PathLike[str] | None = None,
    extra_args: Iterable[str] | None = None,
) -> list[str]:
    """Build the CARLA launch command with offscreen rendering and no quality override."""

    args = [str(arg) for arg in (extra_args or ())]
    quality_args = [arg for arg in args if "quality-level" in arg.lower()]
    if quality_args:
        raise ValueError("Do not pass quality-level arguments for this CARLA 0.10 setup.")

    command = [str(find_carla_server_executable(root)), RENDER_OFFSCREEN_ARG]
    command.extend(args)
    return command


def main() -> None:
    """Print a shell-ready CARLA 0.10 launch command."""

    print(" ".join(build_carla_server_command()))


if __name__ == "__main__":
    main()
