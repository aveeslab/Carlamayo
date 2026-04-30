"""Navigation prompt state for closed-loop CARLA control."""

from __future__ import annotations

from dataclasses import dataclass
import math


DEFAULT_NAVIGATION_WEIGHT = 1.0


@dataclass(frozen=True)
class NavigationCommand:
    """A parsed navigation instruction and guidance weight."""

    text: str
    weight: float = DEFAULT_NAVIGATION_WEIGHT
    revision: int = 0


def parse_navigation_command(
    raw: str,
    default_weight: float = DEFAULT_NAVIGATION_WEIGHT,
) -> NavigationCommand:
    """Parse a UI command of the form ``navigation text | weight``.

    The weight follows Alpamayo's classifier-free guidance convention:
    ``0`` ignores the route text, ``1`` is the normal conditioned path,
    and values greater than ``1`` amplify the navigation condition.
    """

    raw = raw.strip()
    if not raw:
        return NavigationCommand(text="", weight=float(default_weight))

    text = raw
    weight = float(default_weight)
    if "|" in raw:
        text_part, weight_part = raw.rsplit("|", 1)
        text = text_part.strip()
        try:
            weight = float(weight_part.strip())
        except ValueError as exc:
            raise ValueError("Navigation weight must be a non-negative number.") from exc

    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Navigation weight must be a non-negative number.")

    return NavigationCommand(text=text, weight=weight)


class NavigationControlState:
    """Mutable navigation prompt and pause state shared by the UI and loop."""

    def __init__(
        self,
        navigation_text: str = "",
        navigation_weight: float = DEFAULT_NAVIGATION_WEIGHT,
    ):
        initial = parse_navigation_command(f"{navigation_text} | {navigation_weight}")
        self.navigation_text = initial.text
        self.navigation_weight = initial.weight
        self.revision = 0
        self.paused = False
        self.input_text = ""
        self.last_error = ""

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def submit_command(self, raw: str) -> NavigationCommand:
        command = parse_navigation_command(raw, default_weight=self.navigation_weight)
        self.navigation_text = command.text
        self.navigation_weight = command.weight
        self.revision += 1
        self.input_text = ""
        self.last_error = ""
        return NavigationCommand(
            text=self.navigation_text,
            weight=self.navigation_weight,
            revision=self.revision,
        )

    def set_error(self, message: str) -> None:
        self.last_error = message
