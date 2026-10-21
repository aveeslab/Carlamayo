# Data Collection and Inference Workflows

This guide covers CARLA data collection and the three loop modes (open, closed,
live-open) across Alpamayo versions. Every command selects the model with
`--version {1,1.5,2}` (required) and, on the unified launcher, the loop with
`--loop {open,closed,live-open}`.

## 1. Data Collection

Start CARLA first:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

> Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

Then run data collection from the repository root:

```bash
source venv-carla/bin/activate
python data_collect.py
```

Outputs:

- `carla_data/trajectory.json`
- `carla_data/camera_*/<frame>.jpg`
- `carla_data/lidar_top/<frame>.ply`

`data_collect.py` records the seven-camera Alpamayo 2 source ring as a superset, so
one recording replays for any version (the four-camera Alpamayo 1 / 1.5 rig is a
subset selected by folder name at replay time). It records only complete synchronous
frames, matched to the exact frame returned by `world.tick()`.

## 2. Open-Loop Inference

Replay recorded data through the model (no driving):

```bash
source a2_venv/bin/activate
python carlamayo.py --loop open --version 2
# equivalently: python carlamayo_open_loop.py --version 2
```

Options: `--quantization` (4-bit), `--oom-free` (Alpamayo 1.5 only; see
[OOM-Free Mode](oom-free-mode.md)), `--data-root <dir>`, `--output-video <path>`.

Output: `carla_alpamayo_open_loop_result.mp4`

## 3. Closed-Loop Inference

The model drives the ego vehicle via a PID follower. Start CARLA and set the
PythonAPI path if needed:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
export CARLA_ROOT=~/carla
```

The CARLA town/map is set in `module/config.py` (`CARLA_MAP`). Run:

```bash
source a2_carla_venv/bin/activate
python carlamayo.py --loop closed --version 2
# equivalently: python carlamayo_closed_loop.py --version 2
```

> Full-precision Alpamayo 2 needs ~70 GB VRAM. Run the CARLA server on a second
> GPU (`-graphicsadapter=1`) when one GPU cannot host both.
> On a single smaller GPU, `--max-gpu-memory-gb <GB>` keeps that much of the model on
> each GPU and offloads the rest to CPU RAM; inference becomes much slower.

Optional pygame UI (starts paused so you can enter navigation/VQA text first):

```bash
python carlamayo.py --loop closed --version 2 --mode normal --pygame-ui
```

Mode-specific guides (navigation/VQA require `--version 1.5` or `2`):

- [Navigation Mode](navigation-mode.md)
- [VQA Mode](vqa-mode.md)

Other options: `--quantization`, `--async` (non-blocking inference worker),
`--oom-free` (Alpamayo 1.5). Example with OOM-free + async on a small GPU:

```bash
python carlamayo.py --loop closed --version 1.5 --mode normal --oom-free --async
```

Output: `carla_alpamayo_closed_loop_result.mp4`

## 4. Live-Open-Loop Inference

The CARLA Traffic Manager autopilot drives the ego vehicle while the model runs
live and is observed open-loop: its predicted trajectory and Chain-of-Thought are
overlaid against the autopilot's actual control, without the model touching the
wheel. This is useful for comparing model predictions to a real driving policy in a
live simulation.

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
export CARLA_ROOT=~/carla

source a2_carla_venv/bin/activate
# --async keeps the sim moving in real time while inference runs in the background:
python carlamayo.py --loop live-open --version 2 --async
```

Options: `--async`, `--quantization`, `--oom-free` (1.5), `--navigation-text`
(observation-only, for 1.5 / 2), `--output-video <path>`. The model never controls
the vehicle in this mode.

Output: `carla_alpamayo_live_open_loop_result.mp4`

## 5. Debug Recording

Add `--record-dir <dir>` to a closed-loop or live-open-loop run to save what the
model saw and what the controller did:

- `<dir>/camera_*/<frame>.jpg` and `<dir>/trajectory.json` in the data collection
  layout, so the run replays offline with `--loop open --data-root <dir>`.
- `<dir>/records.jsonl`, one line per tick: ego pose and speed, the ego history,
  CoT, and trajectory of each new inference, trajectory age, raw and applied
  controls, and ground truth for nearby actors (Alpamayo-frame position plus
  OpenDRIVE road/lane ids and a same-lane flag).

```bash
python carlamayo.py --loop closed --version 1.5 --record-dir runs/closed_001
python carlamayo.py --loop open --version 1.5 --data-root runs/closed_001
```

Recording writes one JPEG per camera per tick, which slows the loop.

## 6. NVIDIA Original Test Scripts

Each Alpamayo submodule ships its own smoke script and downloads its own gated
weights (large; see each model card for the license before use):

```bash
python third_party/alpamayo1/src/alpamayo_r1/test_inference.py         # Alpamayo 1 (R1)
python third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py       # Alpamayo 1.5
python -m alpamayo2_super.inference_smoke --help                        # Alpamayo 2
```
