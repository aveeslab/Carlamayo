import math

import pytest

from module.navigation_control import NavigationControlState, parse_navigation_command


def test_parse_navigation_command_accepts_text_and_weight():
    command = parse_navigation_command("Turn right in 30m | 1.5")

    assert command.text == "Turn right in 30m"
    assert math.isclose(command.weight, 1.5)


def test_parse_navigation_command_defaults_weight_to_one():
    command = parse_navigation_command("Go straight for 50m")

    assert command.text == "Go straight for 50m"
    assert command.weight == 1.0


def test_parse_navigation_command_rejects_negative_or_non_numeric_weight():
    with pytest.raises(ValueError, match="Navigation weight"):
        parse_navigation_command("Turn left | -0.1")

    with pytest.raises(ValueError, match="Navigation weight"):
        parse_navigation_command("Turn left | heavy")


def test_navigation_control_state_tracks_pause_and_revision():
    state = NavigationControlState()

    assert state.paused is False
    assert state.revision == 0

    state.toggle_pause()
    applied = state.submit_command("Turn left in 20m | 0.8")

    assert state.paused is True
    assert state.navigation_text == "Turn left in 20m"
    assert state.navigation_weight == 0.8
    assert state.revision == 1
    assert applied.revision == 1

    state.toggle_pause()
    assert state.paused is False


def test_vqa_control_state_submits_question_and_tracks_answer():
    state = NavigationControlState(mode="vqa", vqa_question="Describe the scene.")

    assert state.mode == "vqa"
    assert state.vqa_question == "Describe the scene."
    assert state.revision == 0

    applied = state.submit_command("What traffic lights are visible?")

    assert state.vqa_question == "What traffic lights are visible?"
    assert state.revision == 1
    assert applied.text == "What traffic lights are visible?"

    state.set_vqa_answer("A green light is visible.")
    assert state.vqa_answer == "A green light is visible."
