#!/usr/bin/env python3
"""Compare normal-mode closed-loop latency benchmark JSON files."""

import argparse
import json
import sys
from pathlib import Path


def _load_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_time_per_frame(stats: dict) -> float:
    if "model_time_per_eligible_frame_sec" in stats:
        return float(stats["model_time_per_eligible_frame_sec"])
    eligible_frames = int(stats.get("eligible_frames", 0))
    if eligible_frames <= 0:
        raise ValueError("stats must include positive eligible_frames")
    return float(stats.get("total_model_time_sec", 0.0)) / eligible_frames


def _vlm_generate_time(stats: dict) -> float:
    if "avg_vlm_generate_time_sec" in stats:
        return float(stats["avg_vlm_generate_time_sec"])
    calls = int(stats.get("vlm_generate_calls", 0))
    if calls <= 0:
        raise ValueError("stats must include avg_vlm_generate_time_sec or positive vlm_generate_calls")
    return float(stats.get("total_vlm_generate_time_sec", 0.0)) / calls


def compare_latency(baseline: dict, optimized: dict, *, metric: str = "model-frame") -> dict[str, float]:
    if metric == "vlm-generate":
        baseline_time = _vlm_generate_time(baseline)
        optimized_time = _vlm_generate_time(optimized)
    else:
        baseline_time = _model_time_per_frame(baseline)
        optimized_time = _model_time_per_frame(optimized)
    if baseline_time <= 0:
        raise ValueError("baseline latency metric must be positive")
    latency_reduction = 1.0 - (optimized_time / baseline_time)
    baseline_refreshes = int(baseline.get("model_refreshes", 0))
    optimized_refreshes = int(optimized.get("model_refreshes", 0))
    call_reduction = 0.0
    if baseline_refreshes > 0:
        call_reduction = 1.0 - (optimized_refreshes / baseline_refreshes)
    return {
        "baseline_model_time_per_frame_sec": baseline_time,
        "optimized_model_time_per_frame_sec": optimized_time,
        "latency_reduction": latency_reduction,
        "vlm_call_reduction": call_reduction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_json", type=Path)
    parser.add_argument("optimized_json", type=Path)
    parser.add_argument(
        "--min-reduction",
        type=float,
        default=0.30,
        help="Minimum latency reduction required for a passing result. Default: 0.30.",
    )
    parser.add_argument(
        "--metric",
        choices=("model-frame", "vlm-generate"),
        default="model-frame",
        help=(
            "Latency metric to compare. Use 'vlm-generate' for single-call "
            "model.generate() hot-path benchmarking."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = _load_stats(args.baseline_json)
    optimized = _load_stats(args.optimized_json)
    metrics = compare_latency(baseline, optimized, metric=args.metric)
    passed = metrics["latency_reduction"] >= args.min_reduction
    status = "PASS" if passed else "FAIL"
    print(
        f"{status}: latency_reduction={metrics['latency_reduction'] * 100:.1f}% "
        f"(required>={args.min_reduction * 100:.1f}%), "
        f"baseline_{args.metric.replace('-', '_')}_time="
        f"{metrics['baseline_model_time_per_frame_sec']:.4f}s, "
        f"optimized_{args.metric.replace('-', '_')}_time="
        f"{metrics['optimized_model_time_per_frame_sec']:.4f}s, "
        f"vlm_call_reduction={metrics['vlm_call_reduction'] * 100:.1f}%"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
