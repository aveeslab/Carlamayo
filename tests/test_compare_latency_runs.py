import json
import subprocess
import sys


def test_compare_latency_runs_accepts_target_reduction(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text(json.dumps({
        "eligible_frames": 100,
        "model_refreshes": 100,
        "total_model_time_sec": 200.0,
        "model_time_per_eligible_frame_sec": 2.0,
    }))
    optimized.write_text(json.dumps({
        "eligible_frames": 100,
        "model_refreshes": 40,
        "total_model_time_sec": 80.0,
        "model_time_per_eligible_frame_sec": 0.8,
    }))

    result = subprocess.run(
        [
            sys.executable,
            "tools/compare_latency_runs.py",
            str(baseline),
            str(optimized),
            "--min-reduction",
            "0.30",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "latency_reduction=60.0%" in result.stdout
    assert "PASS" in result.stdout


def test_compare_latency_runs_can_compare_vlm_generate_time(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text(json.dumps({"avg_vlm_generate_time_sec": 2.0}))
    optimized.write_text(json.dumps({"avg_vlm_generate_time_sec": 1.2}))

    result = subprocess.run(
        [
            sys.executable,
            "tools/compare_latency_runs.py",
            str(baseline),
            str(optimized),
            "--metric",
            "vlm-generate",
            "--min-reduction",
            "0.30",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "latency_reduction=40.0%" in result.stdout
    assert "baseline_vlm_generate_time=2.0000s" in result.stdout
    assert "optimized_vlm_generate_time=1.2000s" in result.stdout


def test_compare_latency_runs_rejects_below_target(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text(json.dumps({"model_time_per_eligible_frame_sec": 2.0}))
    optimized.write_text(json.dumps({"model_time_per_eligible_frame_sec": 1.8}))

    result = subprocess.run(
        [
            sys.executable,
            "tools/compare_latency_runs.py",
            str(baseline),
            str(optimized),
            "--min-reduction",
            "0.30",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "latency_reduction=10.0%" in result.stdout
    assert "FAIL" in result.stdout
