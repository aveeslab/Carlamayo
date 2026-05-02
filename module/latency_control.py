"""Latency-control helpers for closed-loop inference cadence."""

from dataclasses import dataclass


def should_refresh_normal_inference(
    *,
    frame_ready: bool,
    has_trajectory: bool,
    pending_inference: bool,
    frame_count: int,
    last_refresh_frame: int | None,
    min_interval_frames: int,
) -> bool:
    """Decide whether normal mode should issue a new model/VLM refresh."""

    if not frame_ready or pending_inference:
        return False
    if not has_trajectory or last_refresh_frame is None:
        return True
    if min_interval_frames <= 0:
        return True
    return (int(frame_count) - int(last_refresh_frame)) >= int(min_interval_frames)


@dataclass
class NormalModeLatencyStats:
    """Track model-refresh reduction against a per-ready-frame baseline."""

    eligible_frames: int = 0
    model_refreshes: int = 0
    reuse_frames: int = 0
    total_model_time_sec: float = 0.0

    def record_eligible_frame(self) -> None:
        self.eligible_frames += 1

    def record_model_refresh(self, inference_time_sec: float) -> None:
        self.model_refreshes += 1
        self.total_model_time_sec += float(inference_time_sec)

    def record_reuse_frame(self) -> None:
        self.reuse_frames += 1

    @property
    def vlm_call_reduction(self) -> float:
        if self.eligible_frames <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.model_refreshes / self.eligible_frames))
