# Open-Loop Modes and Flood VQA Evaluation Design

## Objective

Extend `carlamayo_open_loop.py` so one entry point supports normal trajectory
inference, navigation-conditioned trajectory inference, and per-frame VQA. Then
collect 1,000 synchronized frames in `Town10HD_Opt_Flood` under
`HardRainNoon` and evaluate whether Alpamayo recognizes flooding using two
fixed questions.

## User-visible CLI

`carlamayo_open_loop.py` gains these arguments:

- `--mode {normal,nav,navigation,vqa}`, default `normal`. `nav` is normalized
  to `navigation` so the short user-facing spelling and the closed-loop mode
  name are both accepted.
- `--navigation-text TEXT`, required in navigation mode and applied to every
  processed frame.
- `--navigation-weight FLOAT`, default `1.0`, validated as finite and
  non-negative.
- `--vqa-question TEXT`, required in VQA mode and applied to every processed
  frame.
- `--temperature FLOAT`, default `0.6` for trajectory modes and `0.0` for VQA.
  Zero uses deterministic VQA decoding; trajectory generation requires a
  positive value because Alpamayo produces multiple sampled rollouts.
- `--output-json PATH`, optional in VQA mode. When omitted, it is derived from
  `--output-video` by replacing the suffix with `.json`.

Existing normal-mode arguments and defaults remain compatible.

`data_collect.py` gains the minimum options needed for a reproducible finite
capture:

- `--map`, defaulting to the existing `Town02` behavior.
- `--weather`, accepting a CARLA weather preset name such as
  `HardRainNoon`.
- `--frames`, where a positive value stops after that many complete synchronized
  frames and the default preserves the current unlimited capture.
- `--output-dir`, defaulting to `carla_data`.

## Input Contract

Inference uses the recorded per-camera RGB images, not the rendered result MP4.
For every recorded frame, the loader constructs four temporal images from each
of these four cameras in this order:

1. front-left
2. front-wide
3. front-right
4. front-telephoto

This produces `4 cameras x 4 time steps x 3 RGB channels`. Early frames use the
loader's existing edge padding, so all 1,000 recorded frames can be processed.
The model input also carries camera IDs `[0, 1, 2, 6]`, matching Alpamayo's
front-left, front, front-right, and front-telephoto labels and preserving the
training prompt format.

The individual camera files may and should retain CARLA post-processing effects.
"Raw" here means uncomposited camera frames, not post-processing-disabled sensor
pixels. A composed result video is unsuitable even though it is also RGB because
it collapses camera and time axes, omits three model views, and may contain
trajectory or reasoning overlays that leak prior model output back into the
input.

## Mode Behavior

### Normal

Run the existing trajectory inference path and preserve its trajectory plus CoT
visualization.

### Navigation

Pass the same navigation text and guidance weight to `run_inference` for every
frame. Save the same trajectory plus CoT visualization as normal mode, augmented
with the active navigation instruction and weight.

### VQA

Call `run_vqa` once for every frame window using the same CLI question. Do not
run trajectory inference and do not render trajectory or CoT overlays. The video
shows only frame progress, inference latency, the question, and the answer over
the front-wide display image. With the default `--temperature 0`, VQA uses
greedy decoding (`do_sample=False`) instead of passing an invalid zero
temperature to a sampling configuration.

The JSON output contains run metadata and one result object per recorded frame:

```json
{
  "mode": "vqa",
  "data_root": "...",
  "question": "...",
  "frames": [
    {
      "frame_index": 1,
      "carla_frame_id": 123,
      "answer": "...",
      "inference_time_seconds": 1.23
    }
  ]
}
```

Results are checkpointed during the run using atomic replacement so an
interruption does not discard all completed answers. The final video is rendered
from the collected display images and answer records.

## Code Structure

- Keep model loading and dataset loading shared in `carlamayo_open_loop.py`.
- Add argument normalization and mode-specific validation before model loading.
- Keep the existing trajectory frame function, extending it with optional
  navigation conditioning.
- Add a focused VQA frame function that returns extracted answer text and
  latency.
- Add VQA visualization functions alongside the existing open-loop visualization
  helpers in `module/visualization.py`.
- Keep JSON serialization in a small standard-library helper so it is unit
  testable without loading Alpamayo.
- Update `docs/inference-workflows.md` with all three open-loop modes and finite
  data-collection examples.

No separate VQA entry point and no generic strategy-class hierarchy will be
introduced; both would duplicate or over-abstract a single linear inference
loop.

## Validation and Error Handling

- Reject navigation mode without non-empty `--navigation-text`.
- Reject VQA mode without non-empty `--vqa-question`.
- Reject invalid navigation weights before model loading.
- Reject negative or non-finite temperatures before model loading.
- Reject `--output-json` outside VQA mode to avoid misleading unused arguments.
- Reject non-positive finite capture limits when `--frames` is supplied.
- Reject unknown CARLA weather preset names with a clear message.
- Preserve current mutual exclusion between `--oom-free` and `--quantization`.

## Tests

Unit tests will cover:

- normal-mode backward-compatible defaults;
- `nav` alias normalization and navigation argument validation;
- VQA question and output-path validation;
- navigation arguments being forwarded to trajectory inference;
- VQA answer extraction, per-frame record construction, and JSON checkpointing;
- deterministic generation selection at temperature zero and sampling selection
  at positive temperatures;
- VQA video frames containing no projected trajectory;
- data-collection map, weather, frame-limit, and output-directory parsing;
- finite collection stopping only after the requested number of complete frames.

The repository test suite, targeted syntax checks, and a remote model-backed
smoke test will be run before the full experiment.

## Town10HD Flood Experiment

1. Use the packaged flood CARLA build already verified on `pro6000`.
2. Run CARLA with Epic-quality offscreen rendering and load the actual asset name
   `Town10HD_Opt_Flood`.
3. Apply `HardRainNoon` and verify the world reports the preset after a tick.
4. Record exactly 1,000 complete synchronized frames to a new dataset directory,
   retaining individual post-processed camera JPEGs and `trajectory.json`.
5. Validate that all 1,000 trajectory entries have the four required camera
   files. Inspect representative front-wide images to verify visible flooding and
   rain before inference.
6. On GPU 1, run VQA over all 1,000 frames for each question:
   - `Is the road or surrounding area flooded with standing water? Answer yes or no and briefly state the visual evidence.`
   - `Are there hazardous water conditions ahead that could affect safe driving? Describe what you see.`
7. Save one answer-overlay MP4 and one JSON result file per question.
8. Summarize exact-answer frequencies and a conservative recognition rate using
   explicit affirmative, negative, and ambiguous categories. Report sample
   answers and representative frames; do not infer recognition from CoT because
   VQA answers are the evaluated output.

The experiment is considered complete only when the dataset integrity checks,
both 1,000-frame VQA runs, output validation, and recognition summary are all
available.
