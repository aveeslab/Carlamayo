<div align="center">

# CarlaMayo

### NVIDIA Alpamayo (1 / 1.5 / 2) + CARLA Simulator

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

[![CI](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml/badge.svg)](https://github.com/aveeslab/Carlamayo/actions/workflows/ci.yml)

</div>

> **📖 Please read the Hugging Face model card for the version you run first**
> ([Alpamayo-R1](https://huggingface.co/nvidia/Alpamayo-R1-10B),
> [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B),
> [Alpamayo 2 Super](https://huggingface.co/nvidia/Alpamayo2-Super)).
> Each card covers architecture, inputs/outputs, licensing, and tested hardware. This
> repository focuses on CARLA setup, data collection, and open/closed/live-open inference.

## Alpamayo Versions

Pick the model with `--version`. All three are tracked as git submodules under
`third_party/` and imported at runtime by a per-version adapter.

| `--version` | Model | Package | Cameras | Navigation | VQA | OOM-free |
|-------------|-------|---------|---------|------------|-----|----------|
| `1`   | [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)   | `alpamayo_r1`    | 4 | – | – | – |
| `1.5` | [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | `alpamayo1_5`    | 4 | ✅ (+CFG) | ✅ | ✅ |
| `2`   | [Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super)   | `alpamayo2_super`| 7-cam ring → 6 per task | ✅ | ✅ | – |

## Loop Modes

The unified `carlamayo.py` launcher selects a loop with `--loop`:

| `--loop` | Who drives | Description |
|----------|-----------|-------------|
| `open`      | nobody (offline) | Replay a recorded dataset through the model; render prediction video. |
| `closed`    | the model | Model trajectories drive the ego via a PID follower. |
| `live-open` | CARLA autopilot | Autopilot drives live while the model runs open-loop and is overlaid. |

```bash
# Alpamayo 2, closed-loop control:
python carlamayo.py --loop closed --version 2

# Alpamayo 1.5, live-open-loop (autopilot drives, model observes):
python carlamayo.py --loop live-open --version 1.5 --async

# Alpamayo 1 (R1), open-loop replay of recorded data:
python carlamayo.py --loop open --version 1 --data-root carla_data
```

`--version` is required. The legacy `carlamayo_open_loop.py` and
`carlamayo_closed_loop.py` scripts still work as thin wrappers (they preset
`--loop`) and also require `--version`.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x for Alpamayo, 3.10.x for CARLA |
| **GPU** | ≥24 GB VRAM for Alpamayo 1 / 1.5 (10B); ≥80 GB for Alpamayo 2 (34B); ≥6 GB for CARLA |
| **OS** | Linux tested; other platforms unverified |
| **CARLA** | 0.9.16 |

> ⚠️ Alpamayo 2 is a 34B model (~70 GB VRAM in bf16). Run the CARLA server on a
> separate GPU when one GPU cannot host both. For the 10B versions the 4-bit
> `--quantization` path or `--oom-free` (1.5 only) reduces VRAM.

## Installation

Environment setup by following document:

- [Environment Setup](docs/environment-setup.md)

## Running Inference

Data collection, open-loop, closed-loop, and live-open-loop inference:

- [Data Collection and Inference](docs/inference-workflows.md)

### Closed-Loop UI Modes

The closed-loop runner supports `normal`, `navigation`, and `vqa` modes through
`--mode` (navigation/VQA require `--version 1.5` or `2`). See the mode guides:

- [Navigation Mode](docs/navigation-mode.md)
- [VQA Mode](docs/vqa-mode.md)

## Project Structure

```
<repo-root>/
├── carlamayo.py                 # Unified launcher: --loop {open,closed,live-open} --version {1,1.5,2}.
├── carlamayo_open_loop.py       # Thin wrapper: carlamayo.py --loop open.
├── carlamayo_closed_loop.py     # Thin wrapper: carlamayo.py --loop closed.
├── data_collect.py              # Collect the seven-camera superset + LiDAR + trajectory.
├── module/
│   ├── adapters/                # Per-version model adapters + the --version dispatcher.
│   ├── loops/                   # open / closed / live-open loop runners.
│   └── ...                      # Shared CARLA, control, UI, and visualization helpers.
├── tests/                       # Simulator-free unit tests.
├── docs/                        # Environment setup and workflow guides.
├── third_party/alpamayo1/       # NVIDIA Alpamayo-R1 git submodule (--version 1).
├── third_party/alpamayo1.5/     # NVIDIA Alpamayo 1.5 git submodule (--version 1.5).
├── third_party/alpamayo2/       # NVIDIA Alpamayo 2 Super git submodule (--version 2).
├── third_party/oom-free-alpamayo/ # OOM-free demand-layering submodule (1.5 --oom-free).
├── .github/workflows/ci.yml     # Lightweight GitHub Actions test workflow.
├── pyproject.toml               # Python project metadata and Ruff configuration.
└── requirements-*.txt           # Alpamayo runtime and CARLA 0.9.16 packages.
```

Generated data and videos such as `carla_data/` and `carla_alpamayo_*.mp4` are ignored by git.

## Troubleshooting

### Flash Attention issues

The models use Flash Attention 2 by default. If you encounter compatibility issues, use PyTorch's scaled dot-product attention instead in the Alpamayo config:

```python
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:

1. Use **OOM-free mode** (`--oom-free`, Alpamayo 1.5 only). See [OOM-Free Mode](docs/oom-free-mode.md).
2. Try 4-bit quantization with `--quantization`, or keep full precision and offload the
   layers that do not fit to CPU RAM with `--max-gpu-memory-gb <GB>` (much slower).
3. Run the CARLA server on a different GPU than the model.
4. Ensure enough VRAM for the selected version and precision.
5. Close other GPU-intensive applications.

## License and Third-Party Licenses

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This repository does not vendor NVIDIA Alpamayo source code directly. Each Alpamayo
release is linked as a git submodule under `third_party/` and is licensed separately
under Apache License 2.0. See each submodule's `LICENSE`.

NVIDIA Alpamayo model weights are not redistributed by this repository and are not
covered by this repository's Apache License 2.0. Review each version's Hugging Face
model card for its model license and usage restrictions, including non-commercial
restrictions where applicable.
