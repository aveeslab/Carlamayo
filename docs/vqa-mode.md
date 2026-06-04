# Closed-Loop VQA Mode

VQA mode asks Alpamayo a driving-scene question over the current CARLA camera frames. It does not produce trajectories, so the ego vehicle is held braked while this mode is active.

## Start CARLA 0.10.0

```bash
export CARLA_010_ROOT=${CARLA_010_ROOT:-$HOME/Carla-0.10.0}
./scripts/start_carla_010.sh
```

The wrapper launches `CarlaUnreal.sh -RenderOffScreen`. Do not pass `-quality-level=Low`.

## Run VQA Mode

From the repository root:

```bash
.venv/bin/python carlamayo_closed_loop.py --mode vqa --pygame-ui --device-map cuda:0
```

Default model loading is full precision. Add `--quantization` only when you need 4-bit loading to reduce VRAM:

```bash
.venv/bin/python carlamayo_closed_loop.py --mode vqa --pygame-ui --quantization --device-map cuda:0
```

You can provide the first question on the command line:

```bash
.venv/bin/python carlamayo_closed_loop.py --mode vqa --pygame-ui \
  --vqa-question "What traffic elements are visible?" \
  --device-map cuda:0
```

`--pygame-ui` starts paused automatically so you can enter the first VQA question before the CARLA loop begins ticking.

## Ask a Question

When the pygame UI opens, type a plain driving-scene question and press `Enter`.

Examples:

```text
What traffic elements are visible and how should they influence driving?
Is there a pedestrian or vehicle that affects the ego vehicle?
Describe the lane markings and traffic lights ahead.
```

The answer is shown in the pygame panel and printed to the terminal.

## UI Controls

- `Ctrl+P`: pause or resume the synchronous CARLA loop.
- `Enter`: apply the question in the input box.
- `Esc`: quit.
- Plain spaces and `p` characters are accepted in the text input.

## Useful Options

```bash
# Non-blocking inference worker.
.venv/bin/python carlamayo_closed_loop.py --mode vqa --pygame-ui --async --device-map cuda:0

# Lower VRAM model loading.
.venv/bin/python carlamayo_closed_loop.py --mode vqa --pygame-ui --quantization --device-map cuda:0
```

## Output

If `SAVE_VIDEO=True` in `module/config.py`, the script writes `carla_alpamayo_closed_loop_result.mp4`.
