# Closed-Loop Navigation Mode

Navigation mode runs Alpamayo closed-loop trajectory generation with a runtime text instruction. Use it when you want the ego vehicle to follow a natural-language driving command such as turning, lane choice, or route preference.

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

## Run Navigation Mode

From the repository root:

```bash
source a1_5_carla_venv/bin/activate
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --start-paused
```

Closed-loop loading defaults to full precision. On lower-VRAM machines, add `--quantization`:

```bash
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --start-paused --quantization
```

## Enter a Navigation Prompt

When the pygame UI opens, type a command in this format:

```text
Turn right in 30m | 1.0
```

Then press `Enter`.

- Text before `|` becomes the navigation instruction.
- The number after `|` becomes the navigation guidance weight.
- Weight `1.0` uses normal navigation conditioning.
- Weights other than `1.0` use Alpamayo classifier-free guidance navigation and may require more VRAM.

Examples:

```text
Turn right at the next intersection | 1.0
Stay in the left lane | 1.0
Prepare to stop at the traffic light | 1.0
```

## UI Controls

- `Ctrl+P`: pause or resume the synchronous CARLA loop.
- `Enter`: apply the text in the input box.
- `Esc`: quit.
- Plain spaces and `p` characters are accepted in the text input.

## Useful Options

```bash
# Non-blocking inference worker.
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --async

# Lower VRAM model loading.
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --quantization

# Exact returned-logits baseline for debugging memory changes.
python carla_alpamayo_closed_loop.py --mode navigation --pygame-ui --keep-generate-logits
```

## Output

If video recording is enabled in `module/config.py`, the script writes:

- `carla_alpamayo_closed_loop_result.mp4`
