"""Pygame UI for closed-loop CARLA navigation control."""

from __future__ import annotations

import textwrap
import time
from typing import Any

import cv2
import numpy as np

from .navigation_control import NavigationControlState


class ClosedLoopPygameUI:
    """Show the vehicle camera stream and collect live navigation prompts."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 900,
        title: str = "CarlaMayo",
        mode: str = "navigation",
    ):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        self.mode = mode
        self.width = width
        self.height = height
        self.panel_height = 190
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"{title} ({mode})")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("DejaVu Sans", 22)
        self.small_font = pygame.font.SysFont("DejaVu Sans", 18)
        self.last_draw_ts = time.time()

    def close(self) -> None:
        self.pygame.quit()

    def process_events(self, nav_state: NavigationControlState) -> bool:
        """Process keyboard events. Returns ``False`` when the user exits."""

        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return False
            event_mod = getattr(event, "mod", pygame.key.get_mods())
            ctrl_pressed = bool(event_mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
            if event.key == pygame.K_p and ctrl_pressed:
                nav_state.toggle_pause()
                continue
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                try:
                    nav_state.submit_command(nav_state.input_text)
                except ValueError as exc:
                    nav_state.set_error(str(exc))
                continue
            if event.key == pygame.K_BACKSPACE:
                nav_state.input_text = nav_state.input_text[:-1]
                continue
            if event.key == pygame.K_DELETE:
                nav_state.input_text = ""
                continue
            if event.unicode and event.unicode.isprintable():
                nav_state.input_text += event.unicode
        return True

    def draw(
        self,
        frame_rgb: np.ndarray | None,
        nav_state: NavigationControlState,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """Draw the latest RGB frame and navigation controls."""

        pygame = self.pygame
        telemetry = telemetry or {}
        self.screen.fill((8, 8, 8))

        video_height = self.height - self.panel_height
        if frame_rgb is not None:
            surface = self._frame_to_surface(frame_rgb, self.width, video_height)
            x = (self.width - surface.get_width()) // 2
            y = (video_height - surface.get_height()) // 2
            self.screen.blit(surface, (x, y))
        else:
            self._draw_text(
                "Waiting for CARLA camera frames...",
                24,
                24,
                self.font,
                (220, 220, 220),
            )

        self._draw_panel(nav_state, telemetry)
        pygame.display.flip()
        self.clock.tick(30)
        self.last_draw_ts = time.time()

    def _frame_to_surface(self, frame_rgb: np.ndarray, max_width: int, max_height: int):
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        scale = min(max_width / max(w, 1), max_height / max(h, 1))
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return self.pygame.surfarray.make_surface(np.ascontiguousarray(np.swapaxes(frame, 0, 1)))

    def _draw_panel(self, nav_state: NavigationControlState, telemetry: dict[str, Any]) -> None:
        pygame = self.pygame
        y0 = self.height - self.panel_height
        pygame.draw.rect(self.screen, (18, 18, 18), (0, y0, self.width, self.panel_height))
        pygame.draw.line(self.screen, (90, 90, 90), (0, y0), (self.width, y0), 2)

        status_color = (255, 210, 80) if nav_state.paused else (90, 220, 110)
        status = "PAUSED" if nav_state.paused else "RUNNING"
        speed = telemetry.get("speed_kmh", 0.0)
        frame = telemetry.get("frame", 0)
        inference_time = telemetry.get("inference_time", 0.0)
        steer = telemetry.get("steering", 0.0)

        status_text = (
            f"{status} | frame {frame} | {speed:.1f} km/h | "
            f"steer {steer:.2f} | inference {inference_time:.2f}s"
        )
        self._draw_text(status_text, 18, y0 + 14, self.font, status_color)
        if nav_state.mode == "navigation":
            nav_text = nav_state.navigation_text or "(no navigation text)"
            self._draw_wrapped(
                f"Active nav: {nav_text} | weight: {nav_state.navigation_weight:.2f}",
                18,
                y0 + 48,
                width=105,
                color=(230, 230, 230),
            )
            help_text = "Input: text | weight   Enter=apply, Ctrl+P=pause/resume, Esc=quit"
        elif nav_state.mode == "vqa":
            question = nav_state.vqa_question or "(no VQA question)"
            answer = nav_state.vqa_answer or "(answer pending after Enter/resume)"
            self._draw_wrapped(
                f"VQA question: {question}",
                18,
                y0 + 48,
                width=105,
                color=(230, 230, 230),
            )
            self._draw_wrapped(
                f"Answer: {answer}",
                18,
                y0 + 70,
                width=105,
                color=(190, 220, 255),
            )
            help_text = "Input: VQA question   Enter=ask, Ctrl+P=pause/resume, Esc=quit"
        else:
            self._draw_wrapped(
                "Normal closed-loop mode: no text prompt is applied.",
                18,
                y0 + 48,
                width=105,
                color=(230, 230, 230),
            )
            help_text = "Ctrl+P=pause/resume, Esc=quit"

        self._draw_text(help_text, 18, y0 + 88, self.small_font)
        input_rect = (18, y0 + 116, self.width - 36, 36)
        pygame.draw.rect(self.screen, (35, 35, 35), input_rect, border_radius=6)
        pygame.draw.rect(self.screen, (120, 120, 120), input_rect, width=1, border_radius=6)
        cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
        self._draw_text(nav_state.input_text + cursor, 28, y0 + 123, self.font, (255, 255, 255))

        if nav_state.last_error:
            self._draw_text(nav_state.last_error, 18, y0 + 160, self.small_font, (255, 100, 100))

    def _draw_wrapped(self, text: str, x: int, y: int, width: int, color=(255, 255, 255)) -> None:
        for idx, line in enumerate(textwrap.wrap(text, width=width)[:2]):
            self._draw_text(line, x, y + idx * 22, self.small_font, color)

    def _draw_text(self, text: str, x: int, y: int, font, color=(255, 255, 255)) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
