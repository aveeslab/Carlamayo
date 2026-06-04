# Environment Setup

This guide sets up CARLA 0.10.0 and Alpamayo 1.5 for data collection, open-loop inference, and closed-loop CARLA execution.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.12.x for this repository environment |
| CARLA | 0.10.0 Linux package |
| GPU | NVIDIA GPU; full precision needs high VRAM, `--quantization` enables 4-bit loading |
| Video tools | `ffmpeg` recommended for H.264/yuv420p MP4 output |

## Clone with Alpamayo Submodule

```bash
git clone --recurse-submodules https://github.com/aveeslab/Carlamayo.git
cd Carlamayo
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Alpamayo source lives in `third_party/alpamayo1.5/`; this repository does not vendor a copied source tree.

## Install CARLA 0.10.0

Unpack CARLA so the root contains `CarlaUnreal.sh` and `PythonAPI/`. The default expected root is:

```bash
~/Carla-0.10.0
```

If CARLA is elsewhere, export the root before starting CARLA or running Python scripts:

```bash
export CARLA_010_ROOT=/path/to/Carla-0.10.0
```

Start CARLA 0.10.0 from its install directory:

```bash
cd ${CARLA_010_ROOT:-$HOME/Carla-0.10.0}
./CarlaUnreal.sh -RenderOffScreen
```

Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

## Create the Python Environment

Install the project and development dependencies with `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --group dev
```

Install the CARLA 0.10.0 Python wheel that matches Python 3.12:

```bash
export CARLA_010_ROOT=${CARLA_010_ROOT:-$HOME/Carla-0.10.0}
uv pip install --no-deps "$CARLA_010_ROOT/PythonAPI/carla/dist/carla-0.10.0-cp312-cp312-linux_x86_64.whl"
```

If the wheel filename differs, choose the `cp312` wheel from `$CARLA_010_ROOT/PythonAPI/carla/dist/`.

## Hugging Face Access

Request access to:

- [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Authenticate before inference:

```bash
uv pip install huggingface_hub
huggingface-cli login
```

Alternatively, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the shell that runs inference.

## Optional Video Tooling

Install `ffmpeg` so OpenCV output is transcoded to browser-friendly H.264/yuv420p MP4:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Smoke Checks

```bash
.venv/bin/python - <<'PY'
import carla, torch
print("carla", carla.__file__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
```
