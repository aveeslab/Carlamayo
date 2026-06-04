"""Tests for CARLA 0.10 runtime command helpers."""

import pytest

from module.carla_runtime import (
    build_carla_server_command,
    find_carla_server_executable,
    resolve_carla_root,
)


def test_build_carla_server_command_uses_unreal_010_executable_and_offscreen_only(tmp_path):
    root = tmp_path / "Carla-0.10.0"
    exe = root / "CarlaUnreal.sh"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")

    command = build_carla_server_command(root)

    assert command == [str(exe), "-RenderOffScreen"]
    assert not any("quality-level" in arg.lower() for arg in command)


def test_build_carla_server_command_rejects_quality_level_args(tmp_path):
    root = tmp_path / "Carla-0.10.0"
    exe = root / "CarlaUnreal.sh"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")

    with pytest.raises(ValueError, match="quality-level"):
        build_carla_server_command(root, extra_args=["-quality-level=Low"])


def test_find_carla_server_executable_uses_010_name(tmp_path):
    root = tmp_path / "Carla-0.10.0"
    unreal = root / "CarlaUnreal.sh"
    root.mkdir()
    unreal.write_text("#!/bin/sh\n")

    assert find_carla_server_executable(root) == unreal


def test_resolve_carla_root_ignores_stale_generic_carla_root(monkeypatch):
    monkeypatch.setenv("CARLA_ROOT", "/old/carla")
    monkeypatch.delenv("CARLA_010_ROOT", raising=False)

    assert str(resolve_carla_root()).endswith("Carla-0.10.0")


def test_resolve_carla_root_allows_dedicated_010_override(monkeypatch, tmp_path):
    override = tmp_path / "Carla-0.10.0-custom"
    monkeypatch.setenv("CARLA_010_ROOT", str(override))

    assert resolve_carla_root() == override
