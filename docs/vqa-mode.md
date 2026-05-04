# Closed-Loop VQA Mode

VQA mode asks Alpamayo a driving-scene question over the current CARLA camera frames. It is for visual question answering only; VQA does not produce trajectories, so the ego vehicle is held braked while this mode is active.

## Start CARLA

Start CARLA before launching the integration script:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

Set the CARLA PythonAPI root if it is not already configured:

```bash
export CARLA_ROOT=~/carla
```

## Run VQA Mode

From the repository root:

```bash
source a1_5_carla_venv/bin/activate
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --start-paused
```

Closed-loop loading defaults to full precision. On lower-VRAM machines, add `--quantization`:

```bash
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --start-paused --quantization
```

You can also provide the first question on the command line:

```bash
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --start-paused \
  --vqa-question "What traffic elements are visible?"
```

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
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --async

# Lower VRAM model loading.
python carla_alpamayo_closed_loop.py --mode vqa --pygame-ui --quantization
```

## Output

If video recording is enabled in `module/config.py`, the script writes:

- `carla_alpamayo_closed_loop_result.mp4`
