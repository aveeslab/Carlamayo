<div align="center">

# CarlaMayo

### Nvidia Alpamayo-R1 + CARLA Simulator

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

</div>

_Note: Following the release of [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) at CES 2026, Alpamayo-R1 has been renamed to Alpamayo 1._

> **📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) first!**
> The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x (see `pyproject.toml`), 3.10.x (for CARLA) |
| **GPU** | NVIDIA GPU with ≥24 GB VRAM for Alpamayo,  ≥6GB VRAM for CARLA|
| **OS** | Linux (tested); other platforms unverified |

> ⚠️ **Note**: GPUs with less than 24 GB VRAM will likely encounter CUDA out-of-memory errors.
> Demo) 4-bit Quantization Model requires 12GB VRAM.

## Repository Scope

Tracked files in this repo:
- `data_collect.py`
- `carla_alpamayo_open_loop.py`
- `carla_alpamayo_closed_loop.py`
- `requirements-carla.txt`
- `requirements-alpamayo.txt`
- `README.md`

## Installation

## 1) CARLA Environment Setup (Data Collection for Open-loop test)

### 1-1. Install and run CARLA 0.9.16 (Quick Install)

```bash
mkdir -p ~/carla && cd ~/carla
wget https://tiny.carla.org/carla-0-9-16-linux
tar -xvzf carla-0-9-16-linux
./CarlaUE4.sh
```

### 1-2. Create CARLA Python environment

```bash
python3.10 -m venv venv-carla
source venv-carla/bin/activate
pip install -r requirements-carla.txt
```

Install the CARLA Python API matching your CARLA server version:

```bash
pip install carla==0.9.16
```

## 2) Alpamayo Environment Setup

### 2-1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2-2. Set up the environment

```bash
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 2-3. Authenticate with HuggingFace

The model requires access to gated resources. Request access here:
- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

Then authenticate using the HuggingFace CLI:

```bash
# Install huggingface-cli if not already installed (included in transformers)
pip install huggingface_hub

# Login with your token
huggingface-cli login
```

Get your access token at: https://huggingface.co/settings/tokens

> 💡 **Tip**: For more details on HuggingFace authentication, see the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).

## 3) CarlaMayo Environment Setup

### 2-3. Closed-loop environment (Alpamayo + CARLA in one env)

```bash
cd ~/carlamayo
uv venv ar1_carla_venv
source ar1_carla_venv/bin/activate
uv sync --active
python -m ensurepip --upgrade
python -m pip install carla==0.9.16
python -m pip install -r ../requirements-alpamayo.txt -r ../requirements-carla.txt
```

If `agents.navigation.controller` is not found, set:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
```

## Running Inference

## 4) Open-Loop Inference


Run Data collection:

```bash
cd ~/carlamayo
source venv-carla/bin/activate
python data_collect.py
```

Outputs:
- `carla_data/trajectory.json`
- `carla_data/cam_*/<frame>.jpg`
- `carla_data/lidar_top/<frame>.ply`

Run Open-loop Test:

```bash
cd ~/carlamayo
source ar1_venv/bin/activate
python carla_alpamayo_open_loop.py

# Optional: quantized 4-bit mode
python carla_alpamayo_open_loop.py --quantization
```

Output:
- `carla_alpamayo_open_loop_result.mp4`

## 4) Closed-Loop Inference

Before running, set your local CARLA PythonAPI path in `module/config.py`:

```python
# User Config (top of module/config.py)
CARLA_AGENT_ROOT = "carla/CARLA_0.9.16"
```

Use the path that contains `PythonAPI/carla` on your machine.

Run:

```bash
cd ~/carla
/.CarlaUe4.sh

cd ~/carlamayo
source alpamayo/ar1_carla_venv/bin/activate
python carla_alpamayo_closed_loop.py

# Optional: quantized 4-bit mode
python carla_alpamayo_closed_loop.py --quantization
```

Output:
- `carla_alpamayo_closed_loop_result.mp4`

### Nvidia's Original Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo_r1/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to change
the `num_traj_samples=1` argument to a higher number (Line 60).


## Project Structure

```
~/carla/
├── Agents/
├── PythonAPI/
...
└── CarlaUE4.sh

~/<repo-root>/
├── data_collect.py
├── carla_alpamayo_open_loop.py
├── carla_alpamayo_closed_loop.py
├── module/
│   ├── config.py
│   ├── pid_controller.py
│   ├── visualization.py
│   ├── carla_interface.py
│   └── inference.py
├── requirements-carla.txt
├── requirements-alpamayo.txt
├── README.md
├── carla_data/                 # generated by data_collect.py
├── venv-carla/                 # CARLA env
└── alpamayo/                   # cloned by Nvidia Alpamayo-R1 github
    └── ar1_venv/               # Alpamayo env (created by uv)
    └── ar1_carla_venv/         # Alpamayo + CARLA env
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues:

```python
# Use PyTorch's scaled dot-product attention instead
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:
1. Try Quantization option
2. Ensure you have a GPU with at least 12 GB VRAM
3. Reduce `num_traj_samples` if generating multiple trajectories
4. Close other GPU-intensive applications


## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: Non-commercial license - see [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.

## Disclaimer

Alpamayo 1 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
