"""Auto-respawn decisions for CARLA closed-loop runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RespawnDecision:
    """Decision returned by the respawn monitor."""

    should_respawn: bool
    reason: str = ""


class RespawnMonitor:
    """Detect new CARLA collision events that should restart the ego run."""

    def __init__(self, *, cooldown_frames: int):
        self.cooldown_frames = int(cooldown_frames)
        self._last_seen_collision_count = 0
        self._last_respawn_frame = None

    def mark_respawn(self, *, frame_count: int, collision_count: int | None = None):
        """Record that a respawn happened and mark handled collision events."""

        self._last_respawn_frame = int(frame_count)
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
