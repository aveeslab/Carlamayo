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

## 1. CARLA Environment Setup

Use this environment for CARLA and data collection.

### 1.1 Install and run CARLA 0.9.16

```bash
mkdir -p ~/carla && cd ~/carla
wget https://tiny.carla.org/carla-0-9-16-linux
tar -xvzf carla-0-9-16-linux
./CarlaUE4.sh
```

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
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 2.3 Authenticate with Hugging Face

The model requires access to gated resources. Request access first:

- [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

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
uv venv ar1_carla_venv
source ar1_carla_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install carla==0.9.16
python -m pip install -r requirements-alpamayo.txt -r requirements-carla.txt
```

If `agents.navigation.controller` is not found, set `CARLA_ROOT` to the directory that contains `PythonAPI/carla`:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
```

Alternatively, edit `CARLA_AGENT_ROOT` in `module/config.py`.

## Troubleshooting

### Flash Attention issues

If Flash Attention 2 causes compatibility issues, use PyTorch scaled dot-product attention in your local model configuration:

```python
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Try the `--quantization` option.
2. Confirm available VRAM.
3. Reduce `num_traj_samples` when generating multiple trajectories.
4. Close other GPU-intensive processes.
