# Repository Guidelines

## Project Structure & Module Organization
- Root scripts are the main workflows: `data_collect.py` records CARLA sensor/trajectory data, `carlamayo_open_loop.py` runs Alpamayo inference on recorded data, and `carlamayo_closed_loop.py` drives simulator control modes.
- Shared implementation lives in `module/` for CARLA interfaces, inference helpers, configuration, PID/control logic, UI, and visualization.
- Tests live in `tests/` and mirror functional areas, for example `test_open_loop_dataset.py`, `test_carla_interface.py`, and `test_pid_controller.py`.
- Documentation is in `docs/`; demo/media assets are in `assets/`. NVIDIA Alpamayo source is a git submodule under `third_party/alpamayo1.5/`.
- Keep generated outputs such as `carla_data/` and `carla_alpamayo_*.mp4` out of git.

## Build, Test, and Development Commands
- `uv sync --group dev` installs the Python 3.12 Alpamayo/dev environment from `pyproject.toml` and `uv.lock`.
- `python -m pytest -q tests` runs the local lightweight test suite.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests` matches the CI test invocation.
- `uvx ruff check .` runs Ruff linting using the repository line-length setting.
- Runtime entry points are `python data_collect.py`, `python carlamayo_open_loop.py --help`, and `python carlamayo_closed_loop.py --mode normal|navigation|vqa ...`. CARLA 0.9.16 runtime dependencies are tracked separately in `requirements-carla.txt` and may require a Python/CARLA-specific wheel.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and a 100-character line limit (`pyproject.toml`). Prefer clear snake_case for functions, variables, and module files; use PascalCase for classes. Keep simulator-specific constants centralized in `module/config.py` where practical. Avoid broad refactors when changing control or inference behavior.

## Testing Guidelines
Use pytest. Name tests `test_*.py` and test functions `test_*`. Add focused regression tests in `tests/` when changing dataset loading, control logic, inference utilities, or CARLA interface behavior. Prefer lightweight fakes/mocks over requiring a live CARLA simulator or GPU in CI.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Align with CI and CARLA setup` and `Expose simulator runtime failures`. Keep commits focused, explain hardware/runtime assumptions when relevant, and do not commit generated data. Pull requests should include a concise summary, test results, linked issue if applicable, and screenshots or videos for UI/visualization changes.

## Security & Configuration Tips
Review third-party and model licenses before redistributing outputs or weights. Do not store credentials, Hugging Face tokens, local CARLA paths, or large model/data artifacts in the repository.
