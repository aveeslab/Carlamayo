# OOM-Free Mode (CPU↔GPU Demand Layering)

`--oom-free` runs Alpamayo 1.5 with **CPU↔GPU demand layering** from
[`third_party/oom-free-alpamayo`](../third_party/oom-free-alpamayo) instead of
loading the whole model onto the GPU. Alpamayo's VLM / ViT / Expert transformer
layers are kept in pinned host memory and streamed to the GPU on demand, while
the always-resident modules and a planned subset of VLM layers stay on the GPU.

This keeps Alpamayo's **peak VRAM low enough to run alongside a live CARLA
server** on a 16 GB GPU — the case where plain `--quantization` still runs out
of memory because CARLA (~6 GB) and the 4-bit model (~8 GB) together exceed the
card. Unlike quantization, demand layering keeps the model in **full precision**
(no accuracy loss).

> `--oom-free` and `--quantization` are mutually exclusive: demand layering
> streams full-precision (bf16) layers.

## Requirements

The `alpamayo_memopt` package must be importable. Install it once into the
inference environment (editable, no extra deps):

```bash
pip install --no-deps -e third_party/oom-free-alpamayo
```

It also needs enough host RAM to hold the full weights (**≥ 22 GB DRAM**) and a
CUDA GPU. No separate profiling step or `config.json` is required — the residency
plan is computed at load time (see below).

## Usage

Open-loop (no CARLA needed):

```bash
python carlamayo_open_loop.py --oom-free
```

Closed-loop (keep CARLA running the whole time):

```bash
# CARLA stays up; only Alpamayo is offloaded.
python carlamayo_closed_loop.py --mode normal --oom-free
# Recommended for closed-loop: run inference off the tick loop.
python carlamayo_closed_loop.py --mode normal --oom-free --async
```

In closed-loop, the model is loaded **after** CARLA has spawned its map, NPCs and
cameras, so the residency plan is computed against the VRAM CARLA actually
leaves free.

## How residency is planned

At load time the loader:

1. loads the model onto CPU and moves the always-resident modules to the GPU,
2. measures the **currently free** VRAM (so it adapts to however much the running
   CARLA server is using),
3. fills the free VRAM (minus a `headroom` reserve for activation spikes and the
   double-flat-buffer) with GPU-resident VLM layers using interleaved placement,
4. streams every remaining VLM layer plus all ViT and Expert layers on demand
   via `alpamayo_memopt.TriHookPipeline`.

More resident layers → faster but more VRAM; fewer → slower but smaller peak.

## Tuning knobs

| Flag | Default | Meaning |
|---|---|---|
| `--oom-free-headroom-gb` | `3.0` | VRAM (GB) reserved for activation spikes. Lower → more resident layers (faster), higher → safer. |
| `--oom-free-margin` | `2` | Safety margin subtracted from the max resident VLM-layer count. |
| `--oom-free-resident` | auto | Force an exact number of GPU-resident VLM layers (overrides the auto plan). |

## Performance note

Demand layering trades latency for VRAM: each autoregressive decode token streams
the offloaded VLM layers over PCIe, so inference is **several times slower** than
a fully-resident model — the more layers that must be streamed (i.e. the less
free VRAM), the slower it gets. PCIe link width matters a lot: a GPU in an x8
slot streams at half the bandwidth of x16. For closed-loop driving, `--async`
keeps the simulation ticking while inference runs in the background.
