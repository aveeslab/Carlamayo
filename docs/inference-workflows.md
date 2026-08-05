# Data Collection and Inference Workflows

This guide covers CARLA data collection, open-loop inference, and closed-loop inference.

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

`data_collect.py` records the full Alpamayo 2 Super seven-camera source ring and only complete synchronous frames. Sensor messages are matched to the exact frame returned by `world.tick()` so camera/LiDAR files and trajectory poses remain aligned under slower Epic rendering or after map reloads.

## 2. Open-Loop Inference

Run open-loop inference on collected CARLA data:

```bash
source a2_venv/bin/activate
python carlamayo_open_loop.py
```

Optional 4-bit quantized mode (experimental for the 34B checkpoint):

```bash
python carlamayo_open_loop.py --quantization
```

Output:

- `carla_alpamayo_open_loop_result.mp4`

## 3. Closed-Loop Inference

Before running, make sure CARLA is running:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
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
source a2_carla_venv/bin/activate
python carlamayo_closed_loop.py
```

> Full-precision Alpamayo 2 Super needs roughly 70 GB of VRAM for the model
> alone. When one GPU is not enough for both the model and CARLA, run the CARLA
> server on a second GPU (`./CarlaUE4.sh -RenderOffScreen -graphicsadapter=1`)
> or pass an explicit `--device-map`.

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

Optional 4-bit quantized mode (experimental for the 34B checkpoint):

```bash
python carlamayo_closed_loop.py --quantization
```

Optional async inference mode:

```bash
python carlamayo_closed_loop.py --async
```

Output:

- `carla_alpamayo_closed_loop_result.mp4`

## 4. NVIDIA Original Test Script

The original Alpamayo 2 Super smoke script is provided by the submodule. It downloads example data and model weights. The model weights are large (~72 GB) and may take time depending on network speed.

```bash
source a2_venv/bin/activate
python -m alpamayo2_super.inference_smoke --help
```

Running the smoke end to end also requires access to the gated
[`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset; see `third_party/alpamayo2/README.md` for full usage. Review the model
card/license terms before downloading or using the model weights.
