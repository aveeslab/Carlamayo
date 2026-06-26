<div align="center">

# CarlaMayo

### NVIDIA Alpamayo 1.5 + CARLA Simulator

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

[![CI](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml/badge.svg)](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml)

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

Environment setup by following document:

- [Environment Setup](docs/environment-setup.md)

## Running Inference

Data collection, open-loop inference, and closed-loop inference by following document:

- [Data Collection and Inference](docs/inference-workflows.md)

### Closed-Loop UI Modes

The closed-loop runner supports `normal`, `navigation`, and `vqa` modes through
`--mode`. See the mode-specific usage guides:

- [Navigation Mode](docs/navigation-mode.md)
- [VQA Mode](docs/vqa-mode.md)

## Project Structure

```
<repo-root>/
├── data_collect.py              # Collect synchronized CARLA camera/LiDAR/trajectory data.
├── carlamayo_open_loop.py       # Run Alpamayo 1.5 inference on recorded CARLA data.
├── carlamayo_closed_loop.py     # Run closed-loop CARLA control modes.
├── module/                      # Shared CARLA, inference, control, UI, and visualization helpers.
├── tests/                       # Simulator-free unit tests for lightweight helpers.
├── docs/                        # Environment setup and workflow guides.
├── assets/                      # README images and demo media.
├── third_party/alpamayo1.5/     # NVIDIA Alpamayo 1.5 git submodule.
├── .github/workflows/ci.yml     # Lightweight GitHub Actions test workflow.
├── pyproject.toml               # Python project metadata and Ruff configuration.
├── uv.lock                      # Locked uv dependency graph for reproducible installs.
├── requirements-alpamayo.txt    # Additional Alpamayo runtime packages.
└── requirements-carla.txt       # CARLA 0.9.16 data-collection/runtime packages.
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

1. Use **OOM-free mode** (`--oom-free`) to stream Alpamayo's layers from host memory on demand, so the full-precision model fits alongside a running CARLA server without quantization or accuracy loss — this is the recommended fix when CARLA and Alpamayo must share a ≤16 GB GPU. See [OOM-Free Mode](docs/oom-free-mode.md).
2. Try 4-bit quantization with `--quantization`.
3. Ensure you have a GPU with enough VRAM for the selected precision and trajectory count.
4. Keep `num_traj_samples` low on smaller GPUs.
5. Close other GPU-intensive applications.

## License and Third-Party Licenses

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This repository does not vendor NVIDIA Alpamayo 1.5 source code directly. Alpamayo is linked as a git submodule under `third_party/alpamayo1.5` and is licensed separately under Apache License 2.0. See `third_party/alpamayo1.5/LICENSE`.

NVIDIA Alpamayo 1.5 model weights are not redistributed by this repository and are not covered by this repository's Apache License 2.0. Review the [Hugging Face model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for the model license and usage restrictions, including non-commercial restrictions where applicable.
