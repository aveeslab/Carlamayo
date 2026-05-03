"""Auto-respawn decisions for CARLA closed-loop runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RespawnDecision:
    """Decision returned by respawn monitors."""

    should_respawn: bool
    reason: str = ""


class RespawnMonitor:
    """Detect collisions and high-throttle deadlocks that should restart the ego run."""

    def __init__(
        self,
        *,
        cooldown_frames: int,
        stuck_frames: int,
        stuck_speed_kmh: float = 0.5,
        stuck_throttle_threshold: float = 0.2,
        stuck_brake_threshold: float = 0.1,
    ):
        self.cooldown_frames = int(cooldown_frames)
        self.stuck_frames = int(stuck_frames)
        self.stuck_speed_kmh = float(stuck_speed_kmh)
        self.stuck_throttle_threshold = float(stuck_throttle_threshold)
        self.stuck_brake_threshold = float(stuck_brake_threshold)
        self._last_seen_collision_count = 0
        self._last_respawn_frame = None
        self._stuck_start_frame = None

    def mark_respawn(self, *, frame_count: int, collision_count: int | None = None):
        """Record that a respawn happened and clear pending trigger windows."""

        self._last_respawn_frame = int(frame_count)
        self._stuck_start_frame = None
        if collision_count is not None:
            self._last_seen_collision_count = int(collision_count)

    def check_collision(
        self,
        *,
        frame_count: int,
        collision_count: int,
        last_collision_event: dict | None,
    ) -> RespawnDecision:
        """Return a respawn decision when CARLA reports a new collision event."""

        collision_count = int(collision_count)
        if collision_count <= self._last_seen_collision_count:
            return RespawnDecision(False)

        self._last_seen_collision_count = collision_count
        if not self._cooldown_elapsed(frame_count):
            return RespawnDecision(False)

        details = self._format_collision_details(last_collision_event)
        return RespawnDecision(True, f"collision detected{details}")

    def check_stuck(
        self,
        *,
        frame_count: int,
        speed_kmh: float,
        throttle: float,
        brake: float,
        has_trajectory: bool,
    ) -> RespawnDecision:
        """Return a respawn decision after repeated commanded-motion zero-speed frames."""

        if self.stuck_frames <= 0:
            self._stuck_start_frame = None
            return RespawnDecision(False)

        stuck_now = (
            has_trajectory
            and float(speed_kmh) <= self.stuck_speed_kmh
            and float(throttle) >= self.stuck_throttle_threshold
            and float(brake) <= self.stuck_brake_threshold
        )
        if not stuck_now:
            self._stuck_start_frame = None
            return RespawnDecision(False)

        frame_count = int(frame_count)
        if self._stuck_start_frame is None:
            self._stuck_start_frame = frame_count
            return RespawnDecision(False)

        stuck_duration_frames = frame_count - self._stuck_start_frame + 1
        if stuck_duration_frames < self.stuck_frames or not self._cooldown_elapsed(frame_count):
            return RespawnDecision(False)

        return RespawnDecision(
            True,
            (
                f"stuck for {stuck_duration_frames} frames "
                f"at <= {self.stuck_speed_kmh:.1f} km/h with throttle command"
            ),
        )

    def _cooldown_elapsed(self, frame_count: int) -> bool:
        if self._last_respawn_frame is None:
            return True
        return int(frame_count) - self._last_respawn_frame >= self.cooldown_frames

    @staticmethod
    def _format_collision_details(event: dict | None) -> str:
        if not event:
            return ""
        other_actor = event.get("other_actor") or event.get("other_actor_type")
        intensity = event.get("intensity")
        pieces = []
        if other_actor:
            pieces.append(f"actor={other_actor}")
        if intensity is not None:
            pieces.append(f"impulse={float(intensity):.1f}")
        return f" ({', '.join(pieces)})" if pieces else ""
