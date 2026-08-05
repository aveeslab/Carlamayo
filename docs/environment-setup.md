# Environment Setup

This guide covers the required environments for CARLA data collection, Alpamayo inference, and closed-loop CARLA execution.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | NVIDIA GPU with ≥80 GB VRAM for full-precision Alpamayo 2 Super (34B, bf16); 4-bit quantization is experimental and can reduce memory usage |
| **OS** | Linux tested |
| **CARLA** | 0.9.16 |
| **CUDA** | CUDA Toolkit 12.x with `nvcc`; `flash-attn` builds during installation |
| **ffmpeg** | Recommended for VS Code/browser-compatible H.264 MP4 video output |

> Alpamayo 2 Super is a 34B-parameter model. Full-precision bf16 weights alone occupy roughly 68 GB of VRAM, so plan for a ≥80 GB GPU (or run CARLA on a second GPU). The 4-bit quantization option is untested territory for this checkpoint and may not preserve output quality.

## 0. Clone the Repository and Submodules

Clone with the NVIDIA Alpamayo 2 submodule:

```bash
git clone --recurse-submodules https://github.com/aveeslab/Carlamayo.git
cd Carlamayo
```

If you already cloned the repository without submodules, initialize them from the repository root:

```bash
git submodule update --init --recursive
```

The Alpamayo 2 source remains in `third_party/alpamayo2` as a submodule. This repository does not vendor a copied `src/alpamayo2_super` tree.

## 1. CARLA Environment Setup

Use this environment for CARLA and data collection.

### 1.1 Install and run CARLA 0.9.16

```bash
mkdir -p ~/carla && cd ~/carla
wget https://tiny.carla.org/carla-0-9-16-linux
tar -xvzf carla-0-9-16-linux
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

> Do not add `-quality-level=Low`; low-quality rendering can make error.

If your CARLA archive extracts into a nested package directory, move or symlink the CARLA root so that `~/carla` contains `CarlaUE4.sh` and `PythonAPI/`.

### 1.2 Create a CARLA Python environment

From the repository root, install `requirements-carla.txt`; it pins `carla==0.9.16` to match the CARLA server.

```bash
python3.10 -m venv venv-carla
source venv-carla/bin/activate
pip install -r requirements-carla.txt
```


### 1.3 Install ffmpeg for compatible MP4 output

The inference scripts can write an OpenCV fallback video directly, but VS Code and browser-based players usually require H.264/yuv420p MP4. Install `ffmpeg` so videos are automatically transcoded to that compatible format:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 2. Alpamayo Environment Setup

Use this environment for model inference.

### 2.1 Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2.2 Set up the environment

From the repository root:

```bash
uv venv a2_venv --python 3.12
source a2_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install --no-deps -e third_party/alpamayo2
python -m pip install -r requirements-alpamayo.txt
```

`uv sync` installs the pinned `torch==2.8.0`, `transformers==4.57.1`, and builds `flash-attn`. If the `flash-attn` build fails, check that the CUDA Toolkit, `nvcc`, and a compatible compiler are visible in the same shell; set `MAX_JOBS` lower if the build exhausts host memory.

### 2.3 Authenticate with Hugging Face

The model requires access to gated resources. Request access first:

- [Alpamayo 2 Super Model Weights](https://huggingface.co/nvidia/Alpamayo2-Super)

Then authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
```

Create or copy your token from: <https://huggingface.co/settings/tokens>

## 3. Combined Closed-Loop Environment

Closed-loop execution needs Alpamayo and CARLA Python packages in the same environment.

From the repository root, install `requirements-carla.txt`; it pins `carla==0.9.16` to match the CARLA server.

```bash
uv venv a2_carla_venv --python 3.12
source a2_carla_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install --no-deps -e third_party/alpamayo2
python -m pip install -r requirements-alpamayo.txt -r requirements-carla.txt
```

If `agents.navigation.controller` is not found, set `CARLA_ROOT` to the directory that contains `PythonAPI/carla`:

```bash
export CARLA_ROOT=~/carla
```

Alternatively, edit `CARLA_AGENT_ROOT` in `module/config.py`.
