# Data Collection and Inference Workflows

This guide covers CARLA data collection, open-loop inference, and closed-loop inference.

## 1. Data Collection

Start CARLA first:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen
```

> Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

Then run data collection from the repository root:

```bash
source venv-carla/bin/activate
python data_collect.py
```

For a finite capture on a specific map and weather preset:

```bash
python data_collect.py \
  --map Town10HD_Opt_Flood \
  --weather HardRainNoon \
  --frames 1000 \
  --port 2020 \
  --traffic-manager-port 8020 \
  --output-dir carla_data/town10hd_flood_hardrain_1000
```

`--frames` counts only complete synchronized records. Missing sensor packets do
not consume the requested count. RGB cameras are recorded with CARLA
post-processing effects enabled.

Outputs:

- `carla_data/trajectory.json`
- `carla_data/cam_*/<frame>.jpg`
- `carla_data/lidar_top/<frame>.ply`

`data_collect.py` records only complete synchronous frames. Sensor messages are matched to the exact frame returned by `world.tick()` so camera/LiDAR files and trajectory poses remain aligned under slower Epic rendering or after map reloads.

## 2. Open-Loop Inference

Run open-loop inference on collected CARLA data:

```bash
source a1_5_venv/bin/activate
python carlamayo_open_loop.py
```

The open-loop runner supports three modes through one entry point. `nav` is an
alias for `navigation`.

Normal trajectory and Chain-of-Causation inference:

```bash
python carlamayo_open_loop.py \
  --mode normal \
  --data-root carla_data/town10hd_flood_hardrain_1000 \
  --output-video normal.mp4
```

Navigation-conditioned trajectory inference applies one instruction to every
selected frame after the initial four-frame history is available:

```bash
python carlamayo_open_loop.py \
  --mode navigation \
  --navigation-text "Proceed cautiously through standing water" \
  --navigation-weight 1.0 \
  --data-root carla_data/town10hd_flood_hardrain_1000 \
  --output-video navigation.mp4
```

VQA applies one question to every recorded frame. It does not run trajectory
inference; the video contains only the question and answer, and the JSON keeps
the full answer, CARLA frame ID, and inference time for each frame:

```bash
python carlamayo_open_loop.py \
  --mode vqa \
  --vqa-question "Is the road flooded?" \
  --temperature 0 \
  --data-root carla_data/town10hd_flood_hardrain_1000 \
  --output-video flood_vqa.mp4 \
  --output-json flood_vqa.json
```

Temperature zero selects deterministic VQA decoding. Trajectory modes retain
Alpamayo's positive sampling temperature (default `0.6`) because the upstream
trajectory generator produces multiple sampled rollouts and does not accept
zero. A VQA JSON checkpoint is atomically replaced after every completed frame
so an interrupted long run retains its finished answers.

VQA evaluates every recorded frame using edge-padded history. Normal and
navigation modes preserve the original behavior of starting only after a full
four-frame history exists. Result videos are written incrementally, so a
1,000-frame 1080p run does not retain the complete source and overlay videos in
memory.

Place `--output-video` and `--output-json` outside `--data-root`. The runner
rejects dataset-internal outputs so recorded camera, LiDAR, and trajectory files
cannot be overwritten by a result artifact.

### Open-loop image contract

Alpamayo receives four temporal images from each of four separate cameras:
front-left, front-wide, front-right, and front-telephoto. The resulting tensor
layout is `4 cameras x 4 frames x 3 RGB channels`, with camera IDs
`[0, 1, 2, 6]` supplied to match Alpamayo's training labels.

These individual JPEGs retain CARLA rendering post-processing. They are called
raw inputs only to distinguish them from the composed result MP4. A composed
video is not a model-equivalent RGB input: it collapses camera and time axes,
omits three views, and may contain trajectory or reasoning overlays.

Optional 4-bit quantized mode:

```bash
python carlamayo_open_loop.py --quantization
```

Optional OOM-free mode (full-precision CPU↔GPU demand layering; see
[OOM-Free Mode](oom-free-mode.md)):

```bash
python carlamayo_open_loop.py --oom-free
```

Output:

- `carla_alpamayo_open_loop_result.mp4`
- In VQA mode, the configured MP4 plus a frame-by-frame JSON result file

## 3. Closed-Loop Inference

Before running, make sure CARLA is running:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen
```

> Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

Set the CARLA PythonAPI path if needed:

```bash
export CARLA_ROOT=~/carla
```

or edit `module/config.py`:

```python
CARLA_AGENT_ROOT = "~/carla"
```

The CARLA town/map is configured in `module/config.py` with `CARLA_MAP`; the
closed-loop script does not expose a separate map CLI option.

Run closed-loop inference from the repository root:

```bash
source a1_5_carla_venv/bin/activate
python carlamayo_closed_loop.py
```

Optional pygame UI modes:

```bash
# Normal closed-loop trajectory control with camera UI.
python carlamayo_closed_loop.py --mode normal --pygame-ui
```

When `--pygame-ui` is enabled, the loop starts paused automatically so you can
enter navigation or VQA text before the first active tick.

Mode-specific usage guides:

- [Navigation Mode](navigation-mode.md)
- [VQA Mode](vqa-mode.md)

Optional 4-bit quantized mode:

```bash
python carlamayo_closed_loop.py --quantization
```

Optional OOM-free mode — keep CARLA running and offload only Alpamayo via
full-precision CPU↔GPU demand layering (see [OOM-Free Mode](oom-free-mode.md)).
On a 16 GB GPU this fits where `--quantization` still OOMs against a live CARLA:

```bash
python carlamayo_closed_loop.py --mode normal --oom-free
# off-tick inference is recommended for the slower offloaded model:
python carlamayo_closed_loop.py --mode normal --oom-free --async
```

Optional async inference mode:

```bash
python carlamayo_closed_loop.py --async
```

Output:

- `carla_alpamayo_closed_loop_result.mp4`

## 4. NVIDIA Original Test Script

The original Alpamayo test script is provided by the submodule. It downloads example data and model weights. The model weights are large and may take time depending on network speed.

```bash
source a1_5_venv/bin/activate
python third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py
```

To generate more trajectories and reasoning traces, increase `num_traj_samples` in that submodule script. Review the model card/license terms before downloading or using the model weights.
