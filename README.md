<div align="center">

# CarlaMayo

### NVIDIA Alpamayo 1.5 + CARLA Simulator

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

</div>

> **📖 Please read the [Hugging Face Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) first.**
> The model card contains model architecture, inputs/outputs, licensing, and tested hardware details. This repository focuses on CARLA setup, data collection, and open/closed-loop inference scripts.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | NVIDIA GPU with ≥24 GB VRAM for Alpamayo, ≥6 GB VRAM for CARLA |
| **OS** | Linux tested; other platforms unverified |
| **CARLA** | 0.9.16 |

> ⚠️ GPUs with less than 24 GB VRAM will likely encounter CUDA out-of-memory errors for full-precision Alpamayo inference. The 4-bit quantization path can reduce memory usage.

## Installation

Clone with submodules, or initialize submodules after cloning:

```bash
git clone --recurse-submodules https://github.com/aveeslab/Alpamayo-CARLA.git
cd Alpamayo-CARLA
# If the repo was cloned without --recurse-submodules:
git submodule update --init --recursive
```

Environment setup has been moved to a separate document:

- [Environment Setup](docs/environment-setup.md)

## Running Inference

Data collection, open-loop inference, and closed-loop inference instructions have been moved to a separate document:

- [Data Collection and Inference](docs/inference-workflows.md)

### Closed-Loop UI Modes

The closed-loop runner supports `normal`, `navigation`, and `vqa` modes through
`--mode`. It defaults to 4-bit quantization, `Town03`, `device_map=auto`, and
the MAGMA CUDA linalg backend for local GPU stability.

```bash
source a1_5_carla_venv/bin/activate
export CARLA_ROOT=~/carla
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --start-paused
```

In `navigation` mode, type `Turn right in 30m | 1.0`, then press `Enter`. In
`vqa` mode, run `python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui
--start-paused`, type a driving-scene question, and press `Enter` to generate
an answer using Alpamayo VQA. VQA defaults to the latest front-camera frame and
`--vqa-max-generation-length 96` to fit 16 GB GPUs; lower it to `64` if needed.
Use `Ctrl+P` to pause/resume and `Esc` to quit. Plain spaces and `p` characters
are accepted in the text input.

## Project Structure

```
~/carla/
├── Agents/
├── PythonAPI/
...
└── CarlaUE4.sh

<repo-root>/
├── data_collect.py
├── carla_alpamayo_open_loop.py
├── carla_alpamayo_closed_loop.py
├── module/
│   ├── config.py
│   ├── pid_controller.py
│   ├── navigation_control.py
│   ├── pygame_ui.py
│   ├── visualization.py
│   ├── carla_interface.py
│   └── inference.py
├── third_party/
│   └── alpamayo1.5/            # NVIDIA Alpamayo 1.5 git submodule
├── docs/
│   ├── environment-setup.md
│   └── inference-workflows.md
├── requirements-carla.txt
├── requirements-alpamayo.txt
├── LICENSE
├── NOTICE
└── README.md
```

Generated data and videos such as `carla_data/` and `carla_alpamayo_*.mp4` are ignored by git.

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues, use PyTorch's scaled dot-product attention instead in the Alpamayo config:

```python
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Try 4-bit quantization.
2. Ensure you have a GPU with enough VRAM for the selected precision and trajectory count.
3. Keep `num_traj_samples` low on smaller GPUs.
4. Close other GPU-intensive applications.

## License and Third-Party Notices

This repository's CARLA integration code and documentation are licensed under the MIT License. See [LICENSE](LICENSE).

This repository does not vendor NVIDIA Alpamayo 1.5 source code directly. Alpamayo is linked as a git submodule under `third_party/alpamayo1.5` and is licensed separately under Apache License 2.0. See `third_party/alpamayo1.5/LICENSE` and [NOTICE](NOTICE).

NVIDIA Alpamayo 1.5 model weights are not redistributed by this repository and are not covered by this repository's MIT License. Review the [Hugging Face model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for the model license and usage restrictions, including non-commercial restrictions where applicable.
