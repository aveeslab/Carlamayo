# Data Collection and Inference Workflows

This guide covers CARLA data collection, open-loop inference, and closed-loop inference.

## 1. Data Collection

Start CARLA first:

```bash
cd ~/carla
./CarlaUE4.sh
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

## 3. Closed-Loop Inference

Before running, make sure CARLA is running:

```bash
cd ~/carla
./CarlaUE4.sh
```

Set the CARLA PythonAPI path if needed:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
```

or edit `module/config.py`:

```python
CARLA_AGENT_ROOT = "carla/CARLA_0.9.16"
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

## 4. NVIDIA Original Test Script

This script downloads example data and model weights. The model weights are large and may take time depending on network speed.

```bash
source ar1_venv/bin/activate
python src/alpamayo1_5/test_inference.py
```

To generate more trajectories and reasoning traces, increase `num_traj_samples` in the script.

## Project Structure

```text
~/carla/
├── Agents/
├── PythonAPI/
└── CarlaUE4.sh

<repo-root>/
├── data_collect.py
├── carla_alpamayo_open_loop.py
├── carla_alpamayo_closed_loop.py
├── module/
│   ├── config.py
│   ├── pid_controller.py
│   ├── visualization.py
│   ├── carla_interface.py
│   └── inference.py
├── src/
│   └── alpamayo1_5/
├── docs/
│   ├── environment-setup.md
│   └── inference-workflows.md
├── requirements-carla.txt
├── requirements-alpamayo.txt
└── README.md
```
