<div align="center">

# CarlaMayo

### NVIDIA Alpamayo 2 Super + CARLA Simulator

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

[![CI](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml/badge.svg)](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml)

</div>

> **📖 Please read the [Hugging Face Model Card](https://huggingface.co/nvidia/Alpamayo2-Super) first.**
> The model card contains model architecture, inputs/outputs, licensing, and tested hardware details. This repository focuses on CARLA setup, data collection, and open/closed-loop inference scripts.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | NVIDIA GPU with ≥80 GB VRAM for full-precision Alpamayo 2 Super, ≥6 GB VRAM for CARLA |
| **OS** | Linux tested; other platforms unverified |
| **CARLA** | 0.9.16 |

> ⚠️ Alpamayo 2 Super is a 34B-parameter model (32B VLM backbone + 2B diffusion expert); bf16 weights alone occupy roughly 68 GB of VRAM. Run the CARLA server on a second GPU when one GPU cannot host both. The 4-bit quantization path is experimental for this checkpoint.

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

### Camera Ring and Task Profiles

CARLA spawns the seven-camera PhysicalAI source ring that Alpamayo 2 Super was
trained on (cross left/right 120°, front wide 120°, front telephoto 30°, rear
left/right 70°, rear telephoto 30°). Each inference task then selects its fixed
six-camera profile from that ring exactly as in the release package: trajectory,
meta-action, and grounding use cameras `[0, 1, 2, 3, 5, 6]`, while VQA uses
cameras `[0, 1, 2, 3, 4, 5]`.

## Project Structure

```
<repo-root>/
├── data_collect.py              # Collect synchronized CARLA camera/LiDAR/trajectory data.
├── carlamayo_open_loop.py       # Run Alpamayo 2 Super inference on recorded CARLA data.
├── carlamayo_closed_loop.py     # Run closed-loop CARLA control modes.
├── module/                      # Shared CARLA, inference, control, UI, and visualization helpers.
├── tests/                       # Simulator-free unit tests for lightweight helpers.
├── docs/                        # Environment setup and workflow guides.
├── assets/                      # README images and demo media.
├── third_party/alpamayo2/       # NVIDIA Alpamayo 2 Super git submodule.
├── third_party/alpamayo1.5/     # NVIDIA Alpamayo 1.5 git submodule (legacy branches).
├── .github/workflows/ci.yml     # Lightweight GitHub Actions test workflow.
├── pyproject.toml               # Python project metadata and Ruff configuration.
├── uv.lock                      # Locked uv dependency graph for reproducible installs.
├── requirements-alpamayo.txt    # Additional Alpamayo runtime packages.
└── requirements-carla.txt       # CARLA 0.9.16 data-collection/runtime packages.
```

Generated data and videos such as `carla_data/` and `carla_alpamayo_*.mp4` are ignored by git.

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default, and `flash-attn` is compiled during
environment setup. If the build fails, check that CUDA Toolkit 12.x with `nvcc`
and a compatible compiler are visible in the same shell; set `MAX_JOBS` lower if
the build exhausts host memory.

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Run the CARLA server on a different GPU than the model (`-graphicsadapter=<n>`).
2. Try 4-bit quantization with `--quantization` (experimental).
3. Ensure you have a GPU with enough VRAM for the selected precision and trajectory count.
4. Keep `num_traj_samples` low on smaller GPUs.
5. Close other GPU-intensive applications.

## License and Third-Party Licenses

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This repository does not vendor NVIDIA Alpamayo 2 source code directly. Alpamayo 2 is linked as a git submodule under `third_party/alpamayo2` and is licensed separately under Apache License 2.0. See `third_party/alpamayo2/LICENSE`.

NVIDIA Alpamayo 2 Super model weights are not redistributed by this repository and are not covered by this repository's Apache License 2.0. The weights on Hugging Face [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super) are released under the [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license; review the model card before downloading or using them.
