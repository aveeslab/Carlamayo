# Repository Guidelines

## Project Structure & Module Organization

This repository integrates NVIDIA Alpamayo 1.5 with the CARLA simulator. Root scripts are runnable entry points: `data_collect.py`, `carla_alpamayo_open_loop.py`, and `carla_alpamayo_closed_loop.py`. Shared integration code lives in `module/` (`config.py`, CARLA adapters, inference helpers, PID control, and visualization). User documentation is in `docs/`; demo media is in `assets/`. `third_party/alpamayo1.5` is a git submodule and should remain upstream-clean. Generated datasets and videos (`carla_data*/`, `carla_alpamayo_*.mp4`) are local artifacts, not source files.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` — fetch the Alpamayo submodule after cloning.
- `uv venv a1_5_venv --python 3.12 && source a1_5_venv/bin/activate && uv sync --active` — create the Alpamayo inference environment.
- `python -m pip install --no-deps -e third_party/alpamayo1.5` — expose the submodule package for imports.
- `python -m pip install -r requirements-alpamayo.txt` — install inference extras such as OpenCV and bitsandbytes.
- `python3.10 -m venv venv-carla && source venv-carla/bin/activate && pip install -r requirements-carla.txt` — create the CARLA data-collection environment; install `carla==0.9.16` separately to match the server.
- `python -m py_compile data_collect.py carla_alpamayo_open_loop.py carla_alpamayo_closed_loop.py module/*.py` — quick syntax check.
- `pytest` — run tests when Python test files are present.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep lines within the configured Ruff limit of 100 characters. Prefer `snake_case` for functions and variables, `PascalCase` for classes, and uppercase for constants such as configuration paths. Keep hardware, CARLA, and model settings centralized in `module/config.py`; avoid duplicating paths in scripts.

## Testing Guidelines

Add regression tests under `tests/` as `test_*.py` files and run them with `pytest`. Favor small unit tests for helpers in `module/`; mark or document tests that require CARLA, GPU, Hugging Face credentials, or model weights. For inference or video changes, include a smoke run and verify produced MP4s can be decoded.

## Commit & Pull Request Guidelines

Recent history uses short intent-focused commit subjects plus explanatory bodies and git trailers such as `Constraint:`, `Rejected:`, `Tested:`, and `Not-tested:`. Follow that style, especially for hardware- or license-sensitive changes. Pull requests should describe the workflow changed, list verification commands, note required CARLA/GPU/model access, and include screenshots or video snippets when visual output changes.

## Security & Configuration Tips

Do not commit Hugging Face tokens, local virtual environments, generated CARLA data, or model weights. Review Alpamayo model-card restrictions before downloading or sharing outputs.
