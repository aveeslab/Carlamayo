# Environment Setup

This guide covers the required environments for CARLA data collection, Alpamayo inference, and closed-loop CARLA execution.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | ≥24 GB VRAM for Alpamayo 1 / 1.5 (10B); ≥80 GB for Alpamayo 2 (34B); 4-bit `--quantization` or `--oom-free` (1.5) reduce VRAM |
| **OS** | Linux tested |
| **CARLA** | 0.9.16 |
| **CUDA** | CUDA Toolkit 12.x with `nvcc`; `flash-attn` builds during install |
| **ffmpeg** | Recommended for VS Code/browser-compatible H.264 MP4 video output |

## 0. Clone the Repository and Submodules

Clone with all Alpamayo submodules (1 / 1.5 / 2 and OOM-free):

```bash
git clone --recurse-submodules https://github.com/aveeslab/Carlamayo.git
cd Carlamayo
```

If you already cloned the repository without submodules, initialize them from the repository root:

```bash
git submodule update --init --recursive
```

Each Alpamayo release stays as a submodule and is not vendored:
`third_party/alpamayo1` (R1, `--version 1`), `third_party/alpamayo1.5`
(`--version 1.5`), `third_party/alpamayo2` (`--version 2`), and
`third_party/oom-free-alpamayo` (used by `--oom-free`). You only need to install
the package(s) for the version(s) you plan to run.

## 1. CARLA Environment Setup

Use this environment for CARLA and data collection.

### 1.1 Install and run CARLA 0.9.16

```bash
mkdir -p ~/carla && cd ~/carla
wget https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz
tar -xvzf carla-0-9-16-linux
./CarlaUE4.sh -RenderOffScreen
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

From the repository root. Install the package(s) for the version(s) you run;
all three share the same pinned dependencies, so they can coexist in one env:

```bash
uv venv a_venv --python 3.12
source a_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
# Install one or more Alpamayo versions (--no-deps; shared torch/transformers pins):
python -m pip install --no-deps -e third_party/alpamayo1      # --version 1 (R1)
python -m pip install --no-deps -e third_party/alpamayo1.5    # --version 1.5
python -m pip install --no-deps -e third_party/alpamayo2      # --version 2
python -m pip install --no-deps -e third_party/oom-free-alpamayo  # optional: --oom-free (1.5)
python -m pip install -r requirements-alpamayo.txt
```

### 2.3 Authenticate with Hugging Face

The models require access to gated resources. Request access for the version(s)
you use, then authenticate:

- [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
- [Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super)

```bash
pip install huggingface_hub
huggingface-cli login
```

Create or copy your token from: <https://huggingface.co/settings/tokens>

## 3. Combined Closed-Loop Environment

Closed-loop and live-open-loop execution need Alpamayo and CARLA Python packages
in the same environment. `requirements-carla.txt` pins `carla==0.9.16`:

```bash
uv venv a_carla_venv --python 3.12
source a_carla_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install --no-deps -e third_party/alpamayo1 -e third_party/alpamayo1.5 -e third_party/alpamayo2
python -m pip install --no-deps -e third_party/oom-free-alpamayo  # optional: --oom-free (1.5)
python -m pip install -r requirements-alpamayo.txt -r requirements-carla.txt
```

If `agents.navigation.controller` is not found, set `CARLA_ROOT` to the directory that contains `PythonAPI/carla`:

```bash
export CARLA_ROOT=~/carla
```

Alternatively, edit `CARLA_AGENT_ROOT` in `module/config.py`.
