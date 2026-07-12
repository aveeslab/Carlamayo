"""Checkpoint-safe result helpers for open-loop VQA runs."""

import json
import os
from pathlib import Path


def create_vqa_document(data_root, question, temperature):
    """Create the serializable result document for one fixed-question VQA run."""

    return {
        "mode": "vqa",
        "data_root": str(data_root),
        "question": str(question),
        "temperature": float(temperature),
        "frames": [],
    }


def append_vqa_frame(
    document,
    frame_index,
    carla_frame_id,
    answer,
    inference_time,
):
    """Append one completed VQA frame to a result document."""

    document["frames"].append(
        {
            "frame_index": int(frame_index),
            "carla_frame_id": int(carla_frame_id),
            "answer": str(answer),
            "inference_time_seconds": float(inference_time),
        }
    )


def write_json_atomic(document, output_path):
    """Persist a VQA checkpoint without exposing a partially written JSON file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
