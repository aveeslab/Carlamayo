# Open-Loop Modes and Flood VQA Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add normal, navigation, and per-frame VQA modes to the open-loop runner, add reproducible finite CARLA capture options, and evaluate flood recognition on 1,000 `Town10HD_Opt_Flood`/`HardRainNoon` frames with two questions on pro6000 GPU 1.

**Architecture:** Keep `carlamayo_open_loop.py` as the single orchestration entry point and reuse the shared inference stack. Put checkpoint-safe VQA result serialization in a small dependency-light module, keep visualization in `module/visualization.py`, and add only the finite/map/weather/output options required to `data_collect.py`. The model consumes four separate post-processed camera histories with explicit Alpamayo camera IDs; result videos are output-only and are never fed back to the model.

**Tech Stack:** Python 3.12, argparse, PyTorch/Alpamayo 1.5, CARLA PythonAPI, NumPy, Pillow, OpenCV, pytest, JSON, ffmpeg, SSH/pro6000.

---

## File Map

- Modify `carlamayo_open_loop.py`: CLI contract, mode validation, per-frame dispatch, temperature forwarding, JSON checkpoints, and mode-specific video output.
- Modify `module/inference.py`: optional temperature forwarding and deterministic VQA generation at temperature zero.
- Modify `module/open_loop_dataset.py`: canonical Alpamayo camera IDs paired with the existing camera order.
- Create `module/open_loop_results.py`: dependency-light VQA result document construction and atomic JSON persistence.
- Modify `module/visualization.py`: navigation metadata on trajectory videos and VQA-only answer videos.
- Modify `data_collect.py`: map/weather/frame-count/output CLI and exact finite stopping.
- Create `tests/test_open_loop_modes.py`: lightweight CLI, dispatch, and checkpoint tests with heavy imports mocked.
- Modify `tests/test_inference_utils.py`: temperature and camera metadata contracts.
- Modify `tests/test_visualization_utils.py`: navigation and VQA rendering/video tests.
- Create `tests/test_data_collect_cli.py`: pure argument/weather/finite-stop tests with a fake CARLA module.
- Modify `docs/inference-workflows.md`: commands and output contracts for all modes and finite data collection.
- Remote artifacts only, not committed: 1,000-frame dataset, two MP4 files, two JSON files, logs, integrity report, and recognition summary.

### Task 1: Open-loop CLI and mode validation

**Files:**
- Create: `tests/test_open_loop_modes.py`
- Modify: `carlamayo_open_loop.py`

- [ ] **Step 1: Write failing CLI contract tests**

Add tests that import the script without loading the model and exercise an argv-aware parser:

```python
from pathlib import Path

import pytest

import carlamayo_open_loop as open_loop


def test_open_loop_defaults_to_normal_mode():
    args = open_loop.parse_args([])
    assert args.mode == "normal"
    assert args.temperature == 0.0
    assert args.output_json is None


def test_nav_alias_normalizes_to_navigation():
    args = open_loop.parse_args(
        ["--mode", "nav", "--navigation-text", "Keep right", "--navigation-weight", "1.25"]
    )
    assert args.mode == "navigation"
    assert args.navigation_text == "Keep right"
    assert args.navigation_weight == 1.25


@pytest.mark.parametrize("mode", ["nav", "navigation"])
def test_navigation_requires_nonempty_text(mode):
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--mode", mode, "--navigation-text", "  "])


def test_vqa_requires_question_and_derives_json_path():
    args = open_loop.parse_args(
        ["--mode", "vqa", "--vqa-question", "Is the road flooded?", "--output-video", "run.mp4"]
    )
    assert args.output_json == "run.json"


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_temperature_must_be_nonnegative_and_finite(value):
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--temperature", value])


def test_output_json_is_rejected_outside_vqa():
    with pytest.raises(SystemExit):
        open_loop.parse_args(["--output-json", "answers.json"])
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
python -m pytest tests/test_open_loop_modes.py -q
```

Expected: failures because `parse_args` accepts no argv and the new mode arguments do not exist.

- [ ] **Step 3: Implement the minimal parser and validation**

Change `parse_args` to `parse_args(argv=None)`, add mode arguments, normalize `nav`, validate mode-specific values before returning, and derive the JSON path with `pathlib.Path`:

```python
import math
from pathlib import Path


def _default_json_path(output_video):
    return str(Path(output_video).with_suffix(".json"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--mode", choices=("normal", "nav", "navigation", "vqa"), default="normal")
    parser.add_argument("--navigation-text", default="")
    parser.add_argument("--navigation-weight", type=float, default=1.0)
    parser.add_argument("--vqa-question", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json")
    # Preserve all existing arguments.
    args = parser.parse_args(argv)
    args.mode = "navigation" if args.mode == "nav" else args.mode
    args.navigation_text = args.navigation_text.strip()
    args.vqa_question = args.vqa_question.strip()
    if args.mode == "navigation" and not args.navigation_text:
        parser.error("--navigation-text is required in navigation mode.")
    if args.mode == "vqa" and not args.vqa_question:
        parser.error("--vqa-question is required in VQA mode.")
    if not math.isfinite(args.navigation_weight) or args.navigation_weight < 0:
        parser.error("--navigation-weight must be a non-negative finite number.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be a non-negative finite number.")
    if args.mode == "vqa":
        args.output_json = args.output_json or _default_json_path(args.output_video)
    elif args.output_json is not None:
        parser.error("--output-json is only valid in VQA mode.")
    if args.oom_free and args.use_quantization:
        parser.error("--oom-free and --quantization are mutually exclusive.")
    return args
```

- [ ] **Step 4: Run targeted tests**

Run `python -m pytest tests/test_open_loop_modes.py -q`.

Expected: all CLI tests pass without importing Alpamayo weights.

- [ ] **Step 5: Commit the CLI contract**

Stage only `carlamayo_open_loop.py` and `tests/test_open_loop_modes.py`, review the staged diff, then create a Lore-format commit whose intent is preserving one compatible entry point while validating mode-specific inputs. Record the targeted pytest command in `Tested:`.

### Task 2: Camera metadata and temperature-aware inference dispatch

**Files:**
- Modify: `module/open_loop_dataset.py`
- Modify: `module/inference.py`
- Modify: `carlamayo_open_loop.py`
- Modify: `tests/test_inference_utils.py`
- Modify: `tests/test_open_loop_modes.py`

- [ ] **Step 1: Write failing camera and dispatch tests**

Add:

```python
from unittest.mock import Mock, patch

import numpy as np
import torch

from module.open_loop_dataset import OPEN_LOOP_CAMERA_INDICES


def test_open_loop_camera_indices_match_training_camera_labels():
    assert OPEN_LOOP_CAMERA_INDICES == (0, 1, 2, 6)


def test_load_model_input_attaches_camera_indices(tmp_path, monkeypatch):
    arrays = {
        "image_frames": np.zeros((4, 4, 2, 3, 3), dtype=np.uint8),
        "history_xyz": np.zeros((16, 3), dtype=np.float32),
        "history_rot": np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
    }
    monkeypatch.setattr(open_loop, "load_open_loop_arrays", lambda *args, **kwargs: arrays)
    model_input = open_loop.load_model_input(tmp_path, {}, [1], 0)
    assert torch.equal(model_input["camera_indices"], torch.tensor([0, 1, 2, 6]))


def test_navigation_frame_forwards_prompt_weight_and_temperature(monkeypatch):
    fake = Mock(return_value=(torch.zeros((1, 1, 1, 2, 3)), {"cot": ["safe"]}))
    monkeypatch.setattr("module.inference.run_inference", fake)
    open_loop.run_trajectory_frame(
        object(), object(), {}, navigation_text="Turn left", navigation_weight=1.4, temperature=0.0
    )
    assert fake.call_args.kwargs["navigation_text"] == "Turn left"
    assert fake.call_args.kwargs["navigation_weight"] == 1.4
    assert fake.call_args.kwargs["temperature"] == 0.0
```

In `tests/test_inference_utils.py`, use a fake model/processor to assert that `run_vqa(..., temperature=0.0)` reaches the deterministic generation helper and positive temperature retains sampling.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_open_loop_modes.py tests/test_inference_utils.py -q
```

Expected: failures for the missing constant, missing frame function, and missing temperature parameters.

- [ ] **Step 3: Add canonical camera IDs and forward them**

In `module/open_loop_dataset.py` define:

```python
OPEN_LOOP_CAMERA_INDICES = (0, 1, 2, 6)
```

In `load_model_input`, attach a long tensor after `prepare_model_input`:

```python
model_input = prepare_model_input(...)
model_input["camera_indices"] = torch.tensor(OPEN_LOOP_CAMERA_INDICES, dtype=torch.long)
return model_input
```

- [ ] **Step 4: Add temperature-aware inference APIs**

Keep shared defaults backward compatible (`0.6`) and let the open-loop CLI explicitly pass its default `0.0`:

```python
def run_inference(..., temperature: float = 0.6, ...):
    ...
    pred_xyz, pred_rot, extra = inference_fn(
        ...,
        temperature=temperature,
        ...,
    )


def run_vqa(model, processor, data, question: str, temperature: float = 0.6):
    ...
```

In `_generate_vqa_text_with_partial_answer_fallback`, select decoding safely:

```python
do_sample = temperature > 0
generation_config.do_sample = do_sample
generation_config.temperature = temperature if do_sample else None
generation_config.top_p = top_p if do_sample else None
generation_config.top_k = top_k if do_sample else None
```

The real Alpamayo path has `model.vlm` and `model.tokenizer`; preserve the existing fallback for test doubles and older compatible models.

- [ ] **Step 5: Implement trajectory and VQA frame functions**

Rename the existing function without changing its return contract and add VQA:

```python
def run_trajectory_frame(
    model,
    processor,
    model_input,
    navigation_text=None,
    navigation_weight=1.0,
    temperature=0.0,
):
    pred_xyz, extra = run_inference(
        model,
        processor,
        model_input,
        navigation_text=navigation_text,
        navigation_weight=navigation_weight,
        temperature=temperature,
        disable_unused_generate_logits=True,
        vlm_image_pixels=cfg.VLM_IMAGE_PIXELS,
    )
    return extract_trajectory_samples(pred_xyz), extract_cot_text(extra), elapsed


def run_vqa_frame(model, processor, model_input, question, temperature=0.0):
    extra = run_vqa(model, processor, model_input, question=question, temperature=temperature)
    return extract_answer_text(extra), elapsed
```

- [ ] **Step 6: Run targeted tests**

Run `python -m pytest tests/test_open_loop_modes.py tests/test_inference_utils.py -q`.

Expected: camera IDs, temperature selection, and both frame dispatch paths pass.

- [ ] **Step 7: Commit inference contracts**

Stage only the five Task 2 files, inspect the diff, and commit with a Lore message explaining why separate camera/time metadata and safe zero-temperature decoding are required. Include targeted pytest output in `Tested:`.

### Task 3: Checkpoint-safe VQA JSON and answer-only video

**Files:**
- Create: `module/open_loop_results.py`
- Modify: `module/visualization.py`
- Modify: `carlamayo_open_loop.py`
- Modify: `tests/test_open_loop_modes.py`
- Modify: `tests/test_visualization_utils.py`

- [ ] **Step 1: Write failing JSON checkpoint tests**

Add:

```python
import json

from module.open_loop_results import append_vqa_frame, create_vqa_document, write_json_atomic


def test_vqa_document_records_full_frame_identity(tmp_path):
    document = create_vqa_document("dataset", "Is it flooded?", temperature=0.0)
    append_vqa_frame(document, 1, 77, "Yes, standing water is visible.", 1.25)
    output = tmp_path / "answers.json"
    write_json_atomic(document, output)
    saved = json.loads(output.read_text())
    assert saved["question"] == "Is it flooded?"
    assert saved["frames"] == [{
        "frame_index": 1,
        "carla_frame_id": 77,
        "answer": "Yes, standing water is visible.",
        "inference_time_seconds": 1.25,
    }]
    assert not (tmp_path / "answers.json.tmp").exists()
```

- [ ] **Step 2: Write failing VQA visualization tests**

Add a solid RGB image and verify shape/content plus a nonempty MP4:

```python
def test_create_open_loop_vqa_frame_renders_answer_without_trajectory():
    image = np.full((240, 640, 3), 80, dtype=np.uint8)
    frame = create_open_loop_vqa_frame(
        image, 1, 2, 0.5, "Is the road flooded?", "Yes, water covers the road."
    )
    assert frame.shape == image.shape
    assert np.any(frame != image)


def test_save_open_loop_vqa_video_writes_nonempty_video(tmp_path):
    output = tmp_path / "vqa.mp4"
    save_open_loop_vqa_video(
        [np.zeros((240, 640, 3), dtype=np.uint8)],
        ["Yes"],
        [0.5],
        "Is it flooded?",
        output,
        fps=5,
    )
    assert output.stat().st_size > 0
```

- [ ] **Step 3: Run tests and verify failure**

Run `python -m pytest tests/test_open_loop_modes.py tests/test_visualization_utils.py -q`.

Expected: import failures for the new result and visualization helpers.

- [ ] **Step 4: Implement dependency-light atomic JSON persistence**

Create `module/open_loop_results.py` using only `json`, `os`, and `pathlib`:

```python
def create_vqa_document(data_root, question, temperature):
    return {
        "mode": "vqa",
        "data_root": str(data_root),
        "question": question,
        "temperature": float(temperature),
        "frames": [],
    }


def append_vqa_frame(document, frame_index, carla_frame_id, answer, inference_time):
    document["frames"].append({
        "frame_index": int(frame_index),
        "carla_frame_id": int(carla_frame_id),
        "answer": str(answer),
        "inference_time_seconds": float(inference_time),
    })


def write_json_atomic(document, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
```

- [ ] **Step 5: Implement VQA-only visualization**

Add `create_open_loop_vqa_frame` and `save_open_loop_vqa_video` using the existing `VideoRecorder`, header style, `textwrap.wrap`, and RGB/BGR conventions. Do not call `project_trajectory_to_image`; render only frame count, latency, question, and answer. Extend `save_open_loop_video` with optional `navigation_text` and `navigation_weight` arguments and include them in the trajectory-mode text area.

- [ ] **Step 6: Dispatch modes in the open-loop main loop**

Process all recorded indices (`range(0, len(frame_ids))`) because the loader already edge-pads camera and ego history. In VQA mode:

```python
document = create_vqa_document(args.data_root, args.vqa_question, args.temperature)
answer, inference_time = run_vqa_frame(...)
append_vqa_frame(document, display_index, frame_id, answer, inference_time)
write_json_atomic(document, args.output_json)
```

Store only answers, times, and front-wide display images for VQA. In normal/navigation modes, retain predictions and CoT arrays and pass navigation metadata to the existing video saver. Print mode-specific summaries without referring to VQA output as CoT.

- [ ] **Step 7: Run targeted tests and syntax checks**

Run:

```bash
python -m pytest tests/test_open_loop_modes.py tests/test_visualization_utils.py -q
python -m py_compile carlamayo_open_loop.py module/open_loop_results.py module/visualization.py
```

Expected: all targeted tests pass and compilation exits zero.

- [ ] **Step 8: Commit VQA persistence and rendering**

Stage only Task 3 files and commit with a Lore message centered on preserving every expensive per-frame VQA answer and preventing trajectory overlays in VQA output.

### Task 4: Reproducible finite CARLA data collection

**Files:**
- Create: `tests/test_data_collect_cli.py`
- Modify: `data_collect.py`

- [ ] **Step 1: Write failing parser and weather tests**

Inject a minimal fake `carla` module before importing `data_collect`, then test:

```python
def test_data_collection_defaults_preserve_existing_behavior():
    args = data_collect.parse_args([])
    assert args.map_name == "Town02"
    assert args.weather is None
    assert args.frames is None
    assert args.output_dir == "carla_data"


def test_data_collection_accepts_finite_flood_capture():
    args = data_collect.parse_args([
        "--map", "Town10HD_Opt_Flood",
        "--weather", "HardRainNoon",
        "--frames", "1000",
        "--output-dir", "/tmp/town10-flood",
    ])
    assert args.map_name == "Town10HD_Opt_Flood"
    assert args.weather == "HardRainNoon"
    assert args.frames == 1000


def test_unknown_weather_is_rejected(fake_carla):
    with pytest.raises(ValueError, match="Unknown CARLA weather preset"):
        data_collect.resolve_weather("FloodMagic", fake_carla)
```

- [ ] **Step 2: Write a failing finite-stop unit test**

Extract a pure predicate:

```python
@pytest.mark.parametrize(
    ("recorded", "limit", "expected"),
    [(999, 1000, False), (1000, 1000, True), (1001, 1000, True), (1000, None, False)],
)
def test_capture_limit_counts_only_complete_frames(recorded, limit, expected):
    assert data_collect.capture_complete(recorded, limit) is expected
```

- [ ] **Step 3: Run tests and verify failure**

Run `python -m pytest tests/test_data_collect_cli.py -q`.

Expected: missing parser, resolver, and predicate failures.

- [ ] **Step 4: Implement CLI and pure validation helpers**

Add `parse_args(argv=None)`, `resolve_weather(name, carla_module=carla)`, and:

```python
def capture_complete(recorded_frames, frame_limit):
    return frame_limit is not None and recorded_frames >= frame_limit
```

Use `parser.error` for `--frames <= 0`. Resolve weather through an exact public attribute on `carla.WeatherParameters`; do not silently fall back to clear weather.

- [ ] **Step 5: Apply args in the collection loop**

Use `args.map_name`, `args.output_dir`, and `args.frames`. After synchronous mode is enabled, apply the requested weather and tick once; read back and print precipitation/cloud/wetness values. Set every RGB blueprint's `enable_postprocess_effects` attribute to `True`. Increment `recorded_frames` only after every expected sensor file exists and the pose is accepted into `trajectory_data`; break when `capture_complete` returns true so exactly 1,000 complete entries are written.

- [ ] **Step 6: Run targeted tests and syntax checks**

Run:

```bash
python -m pytest tests/test_data_collect_cli.py tests/test_data_collection_utils.py -q
python -m py_compile data_collect.py
```

Expected: all tests pass and the collector compiles.

- [ ] **Step 7: Commit finite collection support**

Stage only `data_collect.py` and its tests. Commit with a Lore message explaining that a complete synchronized frame, not a world tick, is the finite-count unit.

### Task 5: Documentation and full local verification

**Files:**
- Modify: `docs/inference-workflows.md`
- Modify only if necessary for test isolation: `tests/conftest.py`

- [ ] **Step 1: Update workflow documentation**

Document exact commands:

```bash
python data_collect.py \
  --map Town10HD_Opt_Flood \
  --weather HardRainNoon \
  --frames 1000 \
  --output-dir carla_data/town10hd_flood_hardrain_1000

python carlamayo_open_loop.py --mode normal --data-root DATASET --output-video normal.mp4

python carlamayo_open_loop.py --mode navigation \
  --navigation-text "Proceed cautiously through standing water" \
  --navigation-weight 1.0 \
  --data-root DATASET --output-video navigation.mp4

python carlamayo_open_loop.py --mode vqa \
  --vqa-question "Is the road flooded?" \
  --temperature 0 \
  --data-root DATASET --output-video flood_vqa.mp4 --output-json flood_vqa.json
```

State the `4 cameras x 4 frames x RGB` model contract and distinguish post-processed individual camera JPEGs from composed output videos.

- [ ] **Step 2: Initialize the worktree submodules and a Python 3.12 test environment**

Run `git submodule update --init --recursive`. Reuse an existing compatible project environment if available; otherwise run `uv sync --group dev` in the worktree. Do not install dependencies into the repository or track virtualenv contents.

- [ ] **Step 3: Run the complete verification set**

Run:

```bash
uv run python -m pytest -q
uv run python -m py_compile carlamayo_open_loop.py data_collect.py module/*.py
git diff --check origin/main...HEAD
git status --short
```

Expected: all tests pass; compilation and whitespace checks exit zero; status contains only the intended documentation change before its commit.

- [ ] **Step 4: Commit documentation**

Stage the workflow guide and any strictly necessary test-isolation change. Commit with a Lore message documenting the three modes, the raw-camera contract, and the exact flood capture command.

### Task 6: Deploy and run model-backed pro6000 smoke tests

**Files:**
- No additional source files expected.
- Remote logs/artifacts under a new timestamped directory on pro6000.

- [ ] **Step 1: Verify GPU 1 and remote process state**

Use `nvidia-smi` to map index 1 to its UUID and verify no CARLA/model process occupies it. Identify GPU 0 processes separately and do not stop or alter them.

- [ ] **Step 2: Deploy the verified branch without overwriting unrelated remote edits**

Inspect the remote Carlamayo status and diff first. Copy or fetch only the committed feature branch into a new timestamped runner directory, preserving the flood CARLA package and existing experiment artifacts. Record the deployed commit hash in the run log.

- [ ] **Step 3: Run CLI-only and small model-backed smoke tests**

With `CUDA_VISIBLE_DEVICES=1`, verify `--help`, invalid mode arguments, and a VQA run over a tiny copied dataset subset. Confirm the JSON has one record per selected frame, every answer is nonempty or explicitly empty, and the MP4 decodes to the same frame count.

- [ ] **Step 4: Commit any required compatibility fix**

If and only if the real remote environment exposes a reproducible code defect, add a regression test locally, make the smallest fix, rerun the full local suite plus remote smoke test, and commit with the Lore protocol. Do not patch only the remote copy.

### Task 7: Collect 1,000 Town10HD flood frames

**Files:**
- Remote dataset and logs only.

- [ ] **Step 1: Start the verified packaged CARLA build**

Launch the existing flood package offscreen at Epic quality on GPU 1 with unique CARLA and Traffic Manager ports if GPU 0 has a live CARLA instance. Do not use `-quality-level=Low`.

- [ ] **Step 2: Run finite collection**

Run the deployed collector with:

```bash
python data_collect.py \
  --map Town10HD_Opt_Flood \
  --weather HardRainNoon \
  --frames 1000 \
  --output-dir RUN_ROOT/dataset
```

Capture stdout/stderr to a log. The process must exit by itself after 1,000 complete synchronized frames.

- [ ] **Step 3: Validate dataset integrity and scene content**

Parse `trajectory.json` and assert exactly 1,000 keys. For each key, assert files exist in `cam_front_left`, `cam_front_wide`, `cam_front_right`, and `cam_front_tele`; open representative images with Pillow and verify consistent RGB dimensions. Extract contact sheets from early/middle/late front-wide frames and visually verify rain plus visible floodwater before starting the expensive VQA runs.

### Task 8: Run two 1,000-frame VQA evaluations and report recognition

**Files:**
- Remote MP4/JSON/log artifacts only.
- Local copied result summaries only if needed for delivery.

- [ ] **Step 1: Run question 1 on GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 python carlamayo_open_loop.py \
  --mode vqa \
  --data-root RUN_ROOT/dataset \
  --vqa-question "Is the road or surrounding area flooded with standing water? Answer yes or no and briefly state the visual evidence." \
  --temperature 0 \
  --output-video RUN_ROOT/vqa_flood_presence.mp4 \
  --output-json RUN_ROOT/vqa_flood_presence.json
```

- [ ] **Step 2: Validate question 1 artifacts**

Assert 1,000 JSON frame records, valid CARLA frame IDs, finite nonnegative inference times, and an H.264/yuv420p MP4 with 1,000 decodable frames. Preserve the run log and SHA-256 hashes.

- [ ] **Step 3: Run question 2 on GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 python carlamayo_open_loop.py \
  --mode vqa \
  --data-root RUN_ROOT/dataset \
  --vqa-question "Are there hazardous water conditions ahead that could affect safe driving? Describe what you see." \
  --temperature 0 \
  --output-video RUN_ROOT/vqa_water_hazard.mp4 \
  --output-json RUN_ROOT/vqa_water_hazard.json
```

- [ ] **Step 4: Validate question 2 artifacts**

Apply the same 1,000-record, timing, decode, hash, and log checks.

- [ ] **Step 5: Classify answers conservatively**

Normalize answer whitespace and categorize each answer using explicit evidence:

- affirmative: directly says flood, flooding, standing water, water covering the road, submerged roadway, or a hazardous water condition;
- negative: explicitly denies flooding/water hazard without contradictory water evidence;
- ambiguous: empty, hedged, describes only rain/wet road, or cannot determine.

Manually inspect a stratified sample from every category against its front-wide frame. Correct category rules if a keyword produces false positives such as `no standing water` being counted affirmative, then rerun the deterministic summary.

- [ ] **Step 6: Deliver evidence-backed result**

Report per question: affirmative/negative/ambiguous counts and percentages, representative answers with frame IDs, whether visible flood frames were recognized, average/median inference time, and any systematic confusion between rain, wet pavement, and standing floodwater. Include artifact locations and validation status. Copy final MP4/JSON/summary artifacts locally and send the requested concise Slack notification only after validation.
