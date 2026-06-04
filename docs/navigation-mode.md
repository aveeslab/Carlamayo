# Closed-Loop Navigation Mode

Navigation mode runs Alpamayo closed-loop trajectory generation with a runtime text instruction. Use it for natural-language route or maneuver preferences such as turning, lane choice, or preparing to stop.

## Start CARLA 0.10.0

```bash
export CARLA_010_ROOT=${CARLA_010_ROOT:-$HOME/Carla-0.10.0}
./scripts/start_carla_010.sh
```

The wrapper launches `CarlaUnreal.sh -RenderOffScreen`. Do not pass `-quality-level=Low`.

## Run Navigation Mode

From the repository root:

```bash
.venv/bin/python carlamayo_closed_loop.py --mode navigation --pygame-ui --device-map cuda:0
```

Default model loading is full precision. Add `--quantization` only when you need 4-bit loading to reduce VRAM:

```bash
.venv/bin/python carlamayo_closed_loop.py --mode navigation --pygame-ui --quantization --device-map cuda:0
```

`--pygame-ui` starts paused automatically so you can enter the first navigation prompt before the CARLA loop begins driving.

## Enter a Navigation Prompt

Type a command in this format and press `Enter`:

```text
Turn right in 30m | 1.0
```

- Text before `|` becomes the navigation instruction.
- The number after `|` becomes the navigation guidance weight.
- Weight `1.0` uses normal navigation conditioning.
- Weights other than `1.0` use classifier-free guidance navigation and may require more VRAM.

Example prompts:

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
.venv/bin/python carlamayo_closed_loop.py --mode navigation --pygame-ui --async --device-map cuda:0

# Lower VRAM model loading.
.venv/bin/python carlamayo_closed_loop.py --mode navigation --pygame-ui --quantization --device-map cuda:0

# Keep returned logits for debugging memory/behavior changes.
.venv/bin/python carlamayo_closed_loop.py --mode navigation --pygame-ui --keep-generate-logits --device-map cuda:0
```

## Output

If `SAVE_VIDEO=True` in `module/config.py`, the script writes `carla_alpamayo_closed_loop_result.mp4`. The video includes camera imagery, telemetry, Chain-of-Causation text, and the projected predicted trajectory/path overlay.
