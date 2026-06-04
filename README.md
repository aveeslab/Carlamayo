# CarlaMayo

NVIDIA Alpamayo 1.5 integrated with CARLA 0.10.0 for data collection, open-loop inference, and closed-loop driving.

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

[![CI](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml/badge.svg)](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml)

Before downloading weights or datasets, review the [Hugging Face model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for architecture, inputs/outputs, hardware requirements, and license terms.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.12.x for Alpamayo and combined CARLA runs |
| CARLA | 0.10.0 Linux package with `CarlaUnreal.sh` |
| GPU | NVIDIA GPU; full precision needs high VRAM, `--quantization` enables 4-bit loading |
| OS | Linux tested |

CARLA 0.10.0 is expected at `~/Carla-0.10.0` unless `CARLA_010_ROOT` points to another root. The root must contain `CarlaUnreal.sh` and `PythonAPI/`.

## Quick Start

```bash
git clone --recurse-submodules https://github.com/aveeslab/Carlamayo.git
cd Carlamayo
uv sync --group dev
export CARLA_010_ROOT=~/Carla-0.10.0
uv pip install --no-deps "$CARLA_010_ROOT/PythonAPI/carla/dist/carla-0.10.0-cp312-cp312-linux_x86_64.whl"
```

Authenticate to Hugging Face before model inference:

```bash
uv pip install huggingface_hub
huggingface-cli login
```

Start CARLA 0.10.0 from its install directory:

```bash
cd ${CARLA_010_ROOT:-$HOME/Carla-0.10.0}
./CarlaUnreal.sh -RenderOffScreen
```

Do not pass `-quality-level=Low`; low-quality rendering can degrade camera inputs.

## Common Workflows

Collect synchronized CARLA data:

```bash
.venv/bin/python data_collect.py
```

Run open-loop Alpamayo inference on `carla_data/`:

```bash
.venv/bin/python carlamayo_open_loop.py
```

Run closed-loop CARLA control:

```bash
.venv/bin/python carlamayo_closed_loop.py --device-map cuda:0
```

Default model loading is full precision. Use 4-bit quantization only when VRAM is limited:

```bash
.venv/bin/python carlamayo_closed_loop.py --quantization --device-map cuda:0
```

Closed-loop visualization videos include the projected predicted trajectory/path overlay and are written to `carla_alpamayo_closed_loop_result.mp4` when `SAVE_VIDEO=True` in `module/config.py`.

## Project Layout

```text
.
├── data_collect.py                 # CARLA camera/LiDAR/trajectory recording
├── carlamayo_open_loop.py          # Alpamayo inference on recorded data
├── carlamayo_closed_loop.py        # CARLA closed-loop control modes
├── module/                         # CARLA, inference, control, UI, visualization helpers
├── tests/                          # Lightweight pytest suite
├── docs/                           # Setup and workflow guides
├── assets/                         # Demo/media assets
└── third_party/alpamayo1.5/         # NVIDIA Alpamayo 1.5 submodule
```

Generated outputs such as `carla_data/` and `carla_alpamayo_*.mp4` are ignored by git.

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
uvx ruff check .
```

See `docs/environment-setup.md` for detailed environment setup and `docs/inference-workflows.md` for workflow-specific commands.

## Troubleshooting

- **CARLA launcher not found:** set `CARLA_010_ROOT` to the directory containing `CarlaUnreal.sh`.
- **CARLA Python import fails:** install the Python 3.12 CARLA 0.10.0 wheel from `$CARLA_010_ROOT/PythonAPI/carla/dist/`.
- **CUDA out of memory:** retry with `--quantization`, reduce concurrent GPU load, or use a larger GPU.
- **Browser/VS Code cannot play MP4:** install `ffmpeg`; videos are transcoded to H.264/yuv420p when available.

## License

This repository is Apache License 2.0. NVIDIA Alpamayo 1.5 source is included as a submodule under `third_party/alpamayo1.5`; model weights are not redistributed. Review the upstream model and dataset license terms before use.
