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

`data_collect.py` records only complete synchronous frames. Sensor messages are
matched to the exact frame returned by `world.tick()` so camera/LiDAR files and
trajectory poses remain aligned under slower Epic rendering or after map reloads.

## 2. Open-Loop Inference

Run open-loop inference on collected CARLA data:

```bash
source ar1_venv/bin/activate
python carla_alpamayo_open_loop.py
```

Optional 4-bit quantized mode:

```bash
python carla_alpamayo_open_loop.py --quantization
```

Output:

- `carla_alpamayo_open_loop_result.mp4`

Smoke validation used a 4-frame subset and `--quantization` on an RTX 4080 SUPER
16 GB. Full-precision mode may require the larger VRAM budget described in the
README.

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
source ar1_carla_venv/bin/activate
python carla_alpamayo_closed_loop.py
```

Optional 4-bit quantized mode:

```bash
python carla_alpamayo_closed_loop.py --quantization
```

Optional async inference mode:

```bash
python carla_alpamayo_closed_loop.py --async
```

Output:

- `carla_alpamayo_closed_loop_result.mp4`

For lower VRAM machines, the validated command was:

```bash
source ar1_carla_venv/bin/activate
export CARLA_ROOT=~/carla
python carla_alpamayo_closed_loop.py --quantization --async
```

## 4. NVIDIA Original Test Script

This script downloads example data and model weights. The model weights are large and may take time depending on network speed.

```bash
source ar1_venv/bin/activate
python src/alpamayo1_5/test_inference.py
```

To generate more trajectories and reasoning traces, increase `num_traj_samples` in the script.
