# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Reply with Korean.
When you edit code, must do git commit.
Do not git push without user's permission.

## What this is

CarlaMayo integrates **NVIDIA Alpamayo 1.5** (a 10B vision-language driving model) with the **CARLA 0.9.16** simulator. It provides three workflows: collect synchronized CARLA sensor data, run open-loop inference on recorded data, and run closed-loop control where Alpamayo drives the ego vehicle in real time.

Alpamayo itself is **not vendored** — it is a git submodule at `third_party/alpamayo1.5` (NVlabs/alpamayo1.5), installed `--no-deps -e`. Model weights are gated on Hugging Face (`nvidia/Alpamayo-1.5-10B`) and are downloaded at runtime by the submodule. Read the [HF model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) before working on inference.

## Environments

Three separate Python environments are used because CARLA and Alpamayo pin incompatible Python/torch stacks. See `docs/environment-setup.md` for full setup.

| Env | Python | Purpose | Install |
|-----|--------|---------|---------|
| `venv-carla` | 3.10 | data collection only | `pip install -r requirements-carla.txt` |
| `a1_5_venv` | 3.12 (uv) | open-loop inference | `uv sync --active` + `pip install --no-deps -e third_party/alpamayo1.5` + `pip install -r requirements-alpamayo.txt` |
| `a1_5_carla_venv` | 3.12 (uv) | closed-loop (needs both stacks) | same as `a1_5_venv` plus `-r requirements-carla.txt` |

The submodule must be present: `git submodule update --init --recursive`.

## Commands

```bash
# Lint / format (Ruff, line-length 100; config in pyproject.toml)
ruff check .
ruff format .

# Run the test suite (simulator-free unit tests only)
python -m pytest -q tests

# Run a single test
python -m pytest tests/test_config.py::test_runtime_dimensions_match_alpamayo_camera_history_contract

# CI invocation (matches .github/workflows/ci.yml — disables plugin autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

### Running the pipeline (requires a running CARLA server: `~/carla/CarlaUE4.sh -RenderOffScreen`)

```bash
python data_collect.py                                   # -> carla_data/{trajectory.json, cam_*/, lidar_top/}
python carlamayo_open_loop.py [--quantization]           # -> carla_alpamayo_open_loop_result.mp4
python carlamayo_closed_loop.py --mode {normal|navigation|vqa} [--pygame-ui] [--async] [--quantization]
```

CARLA PythonAPI path: set `CARLA_ROOT`/`CARLA_HOME` env var, or edit `CARLA_AGENT_ROOT` in `module/config.py` (used to locate `agents.navigation.controller`). Detailed runtime docs: `docs/inference-workflows.md`, `docs/navigation-mode.md`, `docs/vqa-mode.md`.

## Architecture

Three top-level entry scripts orchestrate the shared `module/` package:

- **`data_collect.py`** (CARLA env) — autopilot ego in CARLA, records 8 cameras + LiDAR + ego poses. Has its **own independent config block** (top of file, e.g. hardcoded `Town02`, 8 sensors) — it does **not** read `module/config.py`. Uses `module/data_collection.py` to keep only sensor packets whose frame matches the exact `world.tick()` frame, so camera/LiDAR files and `trajectory.json` poses stay aligned.
- **`carlamayo_open_loop.py`** (Alpamayo env) — replays recorded `carla_data/` through Alpamayo and renders a video. Loads frames via `module/open_loop_dataset.py`, shares the inference stack in `module/inference.py`.
- **`carlamayo_closed_loop.py`** (combined env) — the main real-time loop: tick CARLA → gather camera history → Alpamayo predicts a trajectory (or answers VQA) → PID follower converts trajectory to vehicle control. This is the largest file and ties together every module.

### `module/` package

- `config.py` — **central tunables** for the open/closed-loop runners (camera/history dims, control gains, PID, map, VRAM-related knobs). Most behavior changes happen here, not via CLI. Note the closed-loop map is config-only (`CARLA_MAP`, no CLI flag).
- `carla_interface.py` — `CARLAInterface`: connection, map load, ego/NPC spawning, camera+collision sensors, synchronous-mode lifecycle, ego-state history, control application, cleanup.
- `inference.py` — Alpamayo load/run. `load_model` (with optional 4-bit BitsAndBytes), `prepare_model_input`, `run_inference` (trajectory), `run_vqa`, and trajectory/CoT/answer extraction helpers. Imports `Alpamayo1_5` from the submodule. `run_inference`/`run_vqa` call `_prime_oom_pipeline(model)` so a model loaded in OOM-free mode is re-primed each inference.
- `oom_offload.py` — optional `--oom-free` loader. Wraps `third_party/oom-free-alpamayo` (`alpamayo_memopt`): loads Alpamayo on CPU, moves always-resident modules to GPU, auto-plans GPU-resident VLM layers from the *currently free* VRAM, and attaches a `TriHookPipeline` as `model._oom_pipeline` that streams the rest on demand. Lets full-precision Alpamayo coexist with a live CARLA server on a 16 GB GPU. Mutually exclusive with `--quantization`. See `docs/oom-free-mode.md`.
- `pid_controller.py` — `OfficialPIDFollower` wraps CARLA's official `VehiclePIDController` and tracks a speed-dependent lookahead point on the predicted trajectory.
- `navigation_control.py` — `NavigationControlState`: mutable, UI-shared prompt + pause + VQA state, with `revision` counter; parses `"text | weight"` commands.
- `respawn_control.py` — `RespawnMonitor`: decides when a new collision should trigger an ego respawn (with frame cooldown).
- `vlm_generate_optimization.py` — context manager that patches `model.vlm.generate` to drop unused returned logits (memory) and time calls.
- `alpamayo_compat.py` — monkey-patches `Alpamayo1_5Config.__init__` to rewrite legacy Hydra `_target_` strings (`alpamayo_r1.` → `alpamayo1_5.`) so released HF configs load against the current submodule package. Called once at import of `inference.py`.
- `visualization.py` — trajectory→image projection, visualization frame composition, `VideoRecorder`, and ffmpeg transcoding to browser-compatible H.264.
- `pygame_ui.py` — `ClosedLoopPygameUI` interactive panel (camera view, text input, pause/resume).
- `data_collection.py` / `open_loop_dataset.py` — frame-sync collection helpers and dataset loaders, respectively.

### Cross-cutting concepts (require reading multiple files)

- **Coordinate frames.** Alpamayo uses ego-local with **y = left**; CARLA uses **y = right**. `pid_controller.alpamayo_to_carla_local` negates y before world projection. Ego history is built in the *current frame's* local coordinates using **negated yaw** (CARLA's left-handed convention) — consistently in `carla_interface.get_history_in_local_frame` (closed-loop) and `open_loop_dataset.build_ego_history` (open-loop, via scipy `Rotation`). When touching trajectory/pose math, keep these two paths in sync.
- **Camera contract.** Alpamayo consumes **4 cameras × 4 history frames**, order `[front_left, front_wide, front_right, front_tele]`. Index **1 (front_wide)** is used for UI/visualization. Data collection records 8 cameras + LiDAR, but only these 4 front cameras feed inference.
- **Sync loop.** Closed-loop runs CARLA in synchronous mode at `fixed_delta_seconds = 0.1`. Camera frames are pulled from per-sensor queues with a timeout; a missing frame raises `TimeoutError` → brake and skip the tick.
- **Async mode (`--async`).** An inference worker thread runs Alpamayo off the tick loop. Results carry `prompt_revision` and `respawn_revision` tags; stale results (prompt changed or ego respawned) are discarded.
- **Control smoothing.** PID outputs are EMA-smoothed (`CONTROL_SMOOTH_ALPHA`) and throttle/brake are made mutually exclusive each tick.
- **Three modes.** `normal` (no nav text), `navigation` (`"text | weight"`; weight ≠ 1.0 switches to Alpamayo's classifier-free-guidance nav path), `vqa` (text answer only; ego is held braked).
- **VRAM / robustness fallbacks** (mostly in `carlamayo_closed_loop.py`): CUDA OOM during CFG-nav falls back to weight 1.0; cuSOLVER linalg errors switch the backend to MAGMA and retry (MAGMA is also the default `--cuda-linalg-library`); `--quantization` enables 4-bit; `--oom-free` streams full-precision layers from host memory (see `oom_offload.py` / `docs/oom-free-mode.md`) so Alpamayo fits next to a running CARLA on 16 GB where `--quantization` still OOMs — in closed-loop the offloaded model is loaded **after** CARLA spawns so the residency plan sees the real free VRAM; `VLM_IMAGE_PIXELS` pins the Qwen-VL image-token budget; collisions trigger auto-respawn.

## Testing notes

Tests are **simulator-free** and target lightweight helpers. Heavy dependencies are stubbed so they run on CPU/CI:

- `tests/conftest.py` adds the repo root to `sys.path`.
- Tests that touch CARLA code stub the `carla` module via `sys.modules.setdefault("carla", fake_carla)` (see `test_carla_interface.py`, `test_pid_controller.py`) — do the same when adding tests for modules that `import carla`.
- `test_inference_utils.py` imports `module.inference`, which requires `torch`, `transformers`, and the installed `alpamayo1_5` submodule. CI installs CPU torch + the submodule (`--no-deps`) for this.
- When adding helpers, prefer putting simulator-independent logic in small functions/modules so they remain unit-testable without CARLA or a GPU (the existing split between e.g. `data_collection.py` helpers and `data_collect.py` follows this pattern).

## Licensing constraint

This repo is Apache-2.0, but Alpamayo **model weights are not** — they have their own (possibly non-commercial) license per the HF model card. Do not add code or docs that imply the weights are redistributable under this repo's license.
