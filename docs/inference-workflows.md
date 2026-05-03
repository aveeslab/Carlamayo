# Data Collection and Inference Workflows

This guide covers CARLA data collection, open-loop inference, and closed-loop inference.

## 1. Data Collection

Start CARLA first:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

Then run data collection from the repository root:

```bash
source venv-carla/bin/activate
python data_collect.py
```

Outputs:

- `carla_data/trajectory.json`
- `carla_data/cam_*/<frame>.jpg`
- `carla_data/lidar_top/<frame>.ply`

`data_collect.py` records only complete synchronous frames. Sensor messages are matched to the exact frame returned by `world.tick()` so camera/LiDAR files and trajectory poses remain aligned under slower Epic rendering or after map reloads.

## 2. Open-Loop Inference

Run open-loop inference on collected CARLA data:

```bash
source a1_5_venv/bin/activate
python carla_alpamayo_open_loop.py
```

Optional 4-bit quantized mode:

```bash
python carla_alpamayo_open_loop.py --quantization
```

Output:

- `carla_alpamayo_open_loop_result.mp4`

Smoke validation used a 4-frame subset and `--quantization` on an RTX 4080 SUPER 16 GB. Full-precision mode may require the larger VRAM budget described in the README.

## 3. Closed-Loop Inference

Before running, make sure CARLA is running:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

Set the CARLA PythonAPI path if needed:

```bash
export CARLA_ROOT=~/carla
```

or edit `module/config.py`:

```python
CARLA_AGENT_ROOT = "~/carla"
```

Run closed-loop inference from the repository root:

```bash
source a1_5_carla_venv/bin/activate
python carla_alpamayo_closed_loop.py
```

Closed-loop now defaults to 4-bit quantized mode for local testing. Use
`--no-quantization` only when you have enough GPU memory for full precision.
Model loading also defaults to `--device-map auto` to let Accelerate place
weights across available devices instead of forcing all weights onto `cuda:0`.
The closed-loop script also defaults to `--cuda-linalg-library magma`; this
avoids a cuSOLVER `torch.linalg.cholesky` initialization failure observed in
the Alpamayo action-space conversion path.

Normal mode latency optimization:

- Default normal mode keeps the original per-ready-frame refresh behavior and
  optimizes the single-call `vlm.generate()` hot path by disabling returned VLM
  logits that the trajectory rollout does not consume, plus a smaller Qwen-VL
  image-token budget (`--vlm-image-pixels 65536`). Token sampling, generated
  sequence handling, KV cache use, and diffusion parameters are unchanged.
- Baseline run: add `--keep-generate-logits --vlm-image-pixels 196608` to
  preserve Alpamayo's original returned-logits behavior and image-token budget.
- On shutdown, compare `avg_vlm_generate_time_sec` in the printed/written normal
  latency stats; the target optimization gate is `>=30%`.
- For repeatable benchmark runs, add `--max-frames N --no-video` to stop
  automatically and remove MP4 encoding overhead. Add `--latency-stats-json`
  to write machine-readable stats, then compare runs with
  `python tools/compare_latency_runs.py baseline.json optimized.json`.
- Automatic ego respawn is on by default. A collision sensor triggers immediate
  respawn, and repeated low-speed throttle deadlocks trigger after
  `--respawn-stuck-frames 40`. Use `--no-auto-respawn` to keep the previous
  behavior.

Optional pygame UI modes:

```bash
# Normal closed-loop trajectory control with camera UI.
python carla_alpamayo_closed_loop.py --mode normal --pygame-ui

# Baseline for latency comparison.
python carla_alpamayo_closed_loop.py --mode normal --pygame-ui \
  --keep-generate-logits --vlm-image-pixels 196608 --max-frames 100 --no-video \
  --latency-stats-json baseline.json

# Optimized run and 30% gate check.
python carla_alpamayo_closed_loop.py --mode normal --pygame-ui \
  --max-frames 100 --no-video --latency-stats-json optimized.json
python tools/compare_latency_runs.py baseline.json optimized.json \
  --metric vlm-generate --min-reduction 0.30

# Navigation-controlled trajectory generation.
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --start-paused

# VQA over the current camera frames; ego vehicle is held braked.
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --start-paused
```

Controls:

- `Ctrl+P`: pause or resume the synchronous CARLA loop.
- `Enter`: apply the text in the input box.
- `Esc`: quit.
- Navigation input format: `Turn right in 30m | 1.0`. The text before `|` becomes
  `navigation_text`; the number after `|` becomes the navigation guidance weight.
  Weight `1.0` uses normal nav conditioning, while other values use Alpamayo's
  classifier-free guidance navigation path.
- VQA input format: plain driving-scene question, for example
  `What traffic elements are visible and how should they influence driving?`.
  The answer is shown in the pygame panel.

Optional async inference mode:

```bash
python carla_alpamayo_closed_loop.py --async
```

Output:

- `carla_alpamayo_closed_loop_result.mp4`

For lower VRAM machines, the validated command was:

```bash
source a1_5_carla_venv/bin/activate
export CARLA_ROOT=~/carla
python carla_alpamayo_closed_loop.py --async
```

## 4. NVIDIA Original Test Script

The original Alpamayo test script is provided by the submodule. It downloads example data and model weights. The model weights are large and may take time depending on network speed.

```bash
source a1_5_venv/bin/activate
python third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py
```

To generate more trajectories and reasoning traces, increase `num_traj_samples` in that submodule script. Review the model card/license terms before downloading or using the model weights.
