# Environment Setup

This guide covers the required environments for CARLA data collection, Alpamayo inference, and closed-loop CARLA execution.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | NVIDIA GPU with ≥24 GB VRAM recommended for Alpamayo; 4-bit quantization can reduce memory usage |
| **OS** | Linux tested |
| **CARLA** | 0.9.16 |

> GPUs with less than 24 GB VRAM may encounter CUDA out-of-memory errors. The 4-bit quantization option can run with lower VRAM, depending on the full workload.

## 0. Clone the Repository and Submodules

Clone with the NVIDIA Alpamayo 1.5 submodule:

```bash
git clone --recurse-submodules https://github.com/aveeslab/Alpamayo-CARLA.git
cd Alpamayo-CARLA
```

If you already cloned the repository without submodules, initialize them from the repository root:

```bash
git submodule update --init --recursive
```

The Alpamayo source remains in `third_party/alpamayo1.5` as a submodule. This repository does not vendor a copied `src/alpamayo1_5` tree.

## 1. CARLA Environment Setup

Use this environment for CARLA and data collection.

### 1.1 Install and run CARLA 0.9.16

```bash
mkdir -p ~/carla && cd ~/carla
wget https://tiny.carla.org/carla-0-9-16-linux
tar -xvzf carla-0-9-16-linux
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

If your CARLA archive extracts into a nested package directory, move or symlink the CARLA root so that `~/carla` contains `CarlaUE4.sh` and `PythonAPI/`.

### 1.2 Create a CARLA Python environment

From the repository root:

```bash
python3.10 -m venv venv-carla
source venv-carla/bin/activate
pip install -r requirements-carla.txt
pip install carla==0.9.16
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
uv venv a1_5_venv --python 3.12
source a1_5_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install --no-deps -e third_party/alpamayo1.5
python -m pip install -r requirements-alpamayo.txt
```

`uv sync --active` installs the core `pyproject.toml` dependencies for this integration project. The editable install step exposes the `alpamayo1_5` Python package from the submodule. The extra `requirements-alpamayo.txt` step is still required for the CARLA inference scripts because they import OpenCV, SciPy, and optional 4-bit quantization support (`bitsandbytes`).

### 2.3 Authenticate with Hugging Face

The model requires access to gated resources. Request access first:

- [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Then authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
```

Create or copy your token from: <https://huggingface.co/settings/tokens>

## 3. Combined Closed-Loop Environment

Closed-loop execution needs Alpamayo and CARLA Python packages in the same environment.

From the repository root:

```bash
uv venv a1_5_carla_venv --python 3.12
source a1_5_carla_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install --no-deps -e third_party/alpamayo1.5
python -m pip install carla==0.9.16
python -m pip install -r requirements-alpamayo.txt -r requirements-carla.txt
```

If `agents.navigation.controller` is not found, set `CARLA_ROOT` to the directory that contains `PythonAPI/carla`:

```bash
export CARLA_ROOT=~/carla
```

Alternatively, edit `CARLA_AGENT_ROOT` in `module/config.py`.

## 4. License Boundaries

- This repository's CARLA integration code is MIT licensed. See `LICENSE`.
- NVIDIA Alpamayo 1.5 source code is a submodule licensed under Apache License 2.0. See `third_party/alpamayo1.5/LICENSE`.
- NVIDIA Alpamayo 1.5 model weights are not redistributed here and are governed by the Hugging Face model card/license terms, including non-commercial restrictions where applicable.
