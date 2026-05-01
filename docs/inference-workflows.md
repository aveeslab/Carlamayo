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

Optional pygame UI modes:

```bash
# Normal closed-loop trajectory control with camera UI.
python carla_alpamayo_closed_loop.py --mode normal --pygame-ui

# Navigation-controlled trajectory generation.
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --start-paused

# VQA over the current camera frames; ego vehicle is held braked.
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --start-paused

# Lower VQA generation memory further if needed.
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --start-paused \
  --vqa-camera-index 1 --vqa-num-frames 1 --vqa-max-generation-length 64
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
  The answer is shown in the pygame panel. By default VQA uses only the latest
  front-camera frame and 96 answer tokens because CARLA plus Alpamayo can exceed
  16 GB VRAM when all cameras/history frames are sent to the VLM generator.

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
