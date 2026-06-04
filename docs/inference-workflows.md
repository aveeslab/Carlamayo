# Data Collection and Inference Workflows

This guide covers CARLA 0.10.0 data collection, open-loop inference, and closed-loop inference.

## Start CARLA 0.10.0

```bash
cd ${CARLA_010_ROOT:-$HOME/Carla-0.10.0}
./CarlaUnreal.sh -RenderOffScreen
```

Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

## 1. Data Collection

```bash
.venv/bin/python data_collect.py
```

Outputs are written under `carla_data/`:

- `trajectory.json`
- `cam_*/<frame>.jpg`
- `lidar_top/<frame>.ply`

`data_collect.py` records only complete synchronous frames. Sensor messages are matched to the exact frame returned by `world.tick()` so camera, LiDAR, and trajectory data stay aligned.

## 2. Open-Loop Inference

Run Alpamayo on collected data:

```bash
.venv/bin/python carlamayo_open_loop.py
```

Default model loading is full precision. Use 4-bit quantization only when VRAM is limited:

```bash
.venv/bin/python carlamayo_open_loop.py --quantization
```

Output:

- `carla_alpamayo_open_loop_result.mp4`

## 3. Closed-Loop Inference

Set the CARLA PythonAPI root if it is not under `~/Carla-0.10.0`:

```bash
export CARLA_010_ROOT=/path/to/Carla-0.10.0
```

`module/config.py` controls the map and recording defaults. Current defaults include:

- `CARLA_MAP = "Town10HD_Opt"`
- `SAVE_VIDEO = True`
- `OUTPUT_VIDEO = "carla_alpamayo_closed_loop_result.mp4"`

Run normal closed-loop control:

```bash
.venv/bin/python carlamayo_closed_loop.py --device-map cuda:0
```

Use 4-bit quantization for lower VRAM:

```bash
.venv/bin/python carlamayo_closed_loop.py --quantization --device-map cuda:0
```

Use async inference when you want the CARLA tick loop to continue while the model worker runs:

```bash
.venv/bin/python carlamayo_closed_loop.py --async --device-map cuda:0
```

Closed-loop videos include the projected predicted trajectory/path overlay on the camera image and are written to:

- `carla_alpamayo_closed_loop_result.mp4`

Mode-specific guides:

- [Navigation Mode](navigation-mode.md)
- [VQA Mode](vqa-mode.md)

## 4. NVIDIA Original Test Script

The Alpamayo submodule includes the upstream test script. It downloads example data and model weights, so runtime depends on network speed and Hugging Face access.

```bash
.venv/bin/python third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py
```

Review the model card and license terms before downloading or using model weights.
