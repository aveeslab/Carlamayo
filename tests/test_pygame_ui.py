import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from module.navigation_control import NavigationControlState
from module.pygame_ui import ClosedLoopPygameUI


def test_pygame_ui_uses_ctrl_p_for_pause_resume():
    import pygame

    ui = ClosedLoopPygameUI(width=320, height=240)
    state = NavigationControlState()
    try:
        pygame.event.post(
            pygame.event.Event(
                pygame.KEYDOWN,
                key=pygame.K_p,
                unicode="\x10",
                mod=pygame.KMOD_CTRL,
            )
        )

        assert ui.process_events(state) is True
        assert state.paused is True
        assert state.input_text == ""
    finally:
        ui.close()


def test_pygame_ui_plain_p_is_inserted_into_empty_navigation_text():
    import pygame

    ui = ClosedLoopPygameUI(width=320, height=240)
    state = NavigationControlState()
    try:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, unicode="p"))

        assert ui.process_events(state) is True
        assert state.paused is False
        assert state.input_text == "p"
    finally:
        ui.close()


def test_pygame_ui_space_is_inserted_into_navigation_text():
    import pygame

    ui = ClosedLoopPygameUI(width=320, height=240)
    state = NavigationControlState()
    try:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t, unicode="t"))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, unicode=" "))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, unicode="r"))

        assert ui.process_events(state) is True
        assert state.paused is False
        assert state.input_text == "t r"
    finally:
        ui.close()


def test_pygame_ui_allows_p_inside_non_empty_navigation_text():
    import pygame

    ui = ClosedLoopPygameUI(width=320, height=240)
    state = NavigationControlState()
    state.input_text = "Kee"
    try:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, unicode="p"))

        assert ui.process_events(state) is True
        assert state.paused is False
        assert state.input_text == "Keep"
    finally:
        ui.close()


def test_pygame_ui_vqa_mode_enter_applies_question():
    import pygame

    ui = ClosedLoopPygameUI(width=320, height=240, mode="vqa")
    state = NavigationControlState(mode="vqa")
    try:
        for key, char in [(pygame.K_w, "w"), (pygame.K_h, "h"), (pygame.K_y, "y")]:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=char))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r"))

        assert ui.process_events(state) is True
        assert state.vqa_question == "why"
        assert state.revision == 1
        assert state.input_text == ""
    finally:
        ui.close()
