from module.latency_control import NormalModeLatencyStats, should_refresh_normal_inference


def test_normal_inference_refreshes_initial_trajectory_when_ready():
    assert should_refresh_normal_inference(
        frame_ready=True,
        has_trajectory=False,
        pending_inference=False,
        frame_count=4,
        last_refresh_frame=None,
        min_interval_frames=10,
    )


def test_normal_inference_reuses_trajectory_until_frame_interval_elapses():
    assert not should_refresh_normal_inference(
        frame_ready=True,
        has_trajectory=True,
        pending_inference=False,
        frame_count=14,
        last_refresh_frame=10,
        min_interval_frames=10,
    )

    assert should_refresh_normal_inference(
        frame_ready=True,
        has_trajectory=True,
        pending_inference=False,
        frame_count=20,
        last_refresh_frame=10,
        min_interval_frames=10,
    )


def test_normal_inference_zero_interval_matches_baseline_every_ready_frame():
    assert should_refresh_normal_inference(
        frame_ready=True,
        has_trajectory=True,
        pending_inference=False,
        frame_count=11,
        last_refresh_frame=10,
        min_interval_frames=0,
    )


def test_latency_stats_reports_vlm_call_reduction_against_per_frame_baseline():
    stats = NormalModeLatencyStats()
    for _ in range(10):
        stats.record_eligible_frame()
    for _ in range(3):
        stats.record_model_refresh(2.0)
    for _ in range(7):
        stats.record_reuse_frame()

    assert stats.model_refreshes == 3
    assert stats.reuse_frames == 7
    assert stats.total_model_time_sec == 6.0
    assert stats.vlm_call_reduction == 0.7


def test_latency_stats_serializes_machine_readable_metrics():
    stats = NormalModeLatencyStats()
    for _ in range(4):
        stats.record_eligible_frame()
    stats.record_model_refresh(1.5)
    stats.record_reuse_frame()
    stats.record_reuse_frame()
    stats.record_reuse_frame()

    data = stats.to_dict(interval_frames=10, mode="normal")

    assert data["mode"] == "normal"
    assert data["normal_inference_interval_frames"] == 10
    assert data["eligible_frames"] == 4
    assert data["model_refreshes"] == 1
    assert data["trajectory_reuse_frames"] == 3
    assert data["total_model_time_sec"] == 1.5
    assert data["model_time_per_eligible_frame_sec"] == 0.375
    assert data["vlm_call_reduction_vs_per_frame_baseline"] == 0.75
