# Closed-Loop VQA Mode

VQA mode asks Alpamayo a driving-scene question over the current CARLA camera frames. It is for visual question answering only; VQA does not produce trajectories, so the ego vehicle is held braked while this mode is active.

## Start CARLA

Start CARLA before launching the integration script:

```bash
cd ~/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic
```

> Do not add `-quality-level=Low`; low-quality rendering can degrade camera inputs.

Set the CARLA PythonAPI root if it is not already configured:

```bash
export CARLA_ROOT=~/carla
```

## Run VQA Mode

From the repository root:

```bash
source a2_carla_venv/bin/activate
python carlamayo_closed_loop.py --mode vqa --pygame-ui
```

VQA uses Alpamayo 2 Super's six-camera VQA input profile (the four front/cross
cameras plus rear left/right and rear telephoto) selected from the seven-camera
source ring, with the no-special-token prompt format from the release notebooks.

Closed-loop loading defaults to full precision. On lower-VRAM machines, add `--quantization` (experimental for the 34B checkpoint):

```bash
python carlamayo_closed_loop.py --mode vqa --pygame-ui --quantization
```

You can also provide the first question on the command line:

```bash
python carlamayo_closed_loop.py --mode vqa --pygame-ui \
  --vqa-question "What traffic elements are visible?"
```

The pygame UI starts paused automatically so you can enter the first VQA question
before the CARLA loop begins ticking.

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
python carlamayo_closed_loop.py --mode vqa --pygame-ui --async

# Lower VRAM model loading (experimental).
python carlamayo_closed_loop.py --mode vqa --pygame-ui --quantization

```

## Output

If video recording is enabled in `module/config.py`, the script writes:

- `carla_alpamayo_closed_loop_result.mp4`
