# CARLA + Alpamayo Closed-Loop (Modular)

![Closed-loop Demo](assets/carla_alpamayo_demo.gif)

This repo provides CARLA closed-loop control and open-loop inference scripts for Alpamayo.
The closed-loop pipeline is modularized so each concern is maintained in a separate Python module.

## Repository Layout

```text
<repo-root>/
├── carla_alpamayo_closed_loop.py      # main entrypoint (args + main loop orchestration)
├── carla_alpamayo_open_loop.py
├── data_collect.py
├── module/
│   ├── config.py                      # tunable constants and paths
│   ├── pid_controller.py              # VehiclePIDController integration
│   ├── visualization.py               # overlays + video writer
│   ├── carla_interface.py             # CARLA world/actors/sensors lifecycle
│   └── inference.py                   # model load + inference + trajectory selection
├── assets/
│   └── carla_alpamayo_demo.gif
├── requirements-carla.txt
└── requirements-alpamayo.txt
```

## Environment

Use a unified environment for CARLA + Alpamayo inference.

```bash
python -m pip install -r requirements-carla.txt -r requirements-alpamayo.txt
```

Install CARLA Python API matching your CARLA server version.

```bash
python -m pip install /path/to/carla-0.9.16-*.whl
```

## Required Path Config

Set your CARLA PythonAPI root in `module/config.py`:

```python
CARLA_AGENT_ROOT = "carla/CARLA_0.9.16"
```

This path must contain `PythonAPI/carla`.

## Run Closed-Loop

Start CARLA server first (`CarlaUE4.sh`), then run:

```bash
python carla_alpamayo_closed_loop.py
```

Options:

```bash
# 4-bit quantization
python carla_alpamayo_closed_loop.py --quantization

# Internal async inference mode
python carla_alpamayo_closed_loop.py --async

# Async + quantized
python carla_alpamayo_closed_loop.py --async --quantization
```

Output video:

- `carla_alpamayo_closed_loop_result.mp4`

## Run Open-Loop

```bash
python carla_alpamayo_open_loop.py
```

Option:

```bash
python carla_alpamayo_open_loop.py --quantization
```

Output video:

- `carla_alpamayo_open_loop_result.mp4`
