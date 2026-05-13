import pytest

from module.navigation_control import NavigationControlState, parse_navigation_command


def test_parse_navigation_command_accepts_text_and_weight():
    command = parse_navigation_command("Turn right at the next light | 1.25")

    assert command.text == "Turn right at the next light"
    assert command.weight == 1.25
    assert command.revision == 0


def test_parse_navigation_command_uses_previous_weight_when_omitted():
    command = parse_navigation_command("Stay in lane", default_weight=0.8)

    assert command.text == "Stay in lane"
    assert command.weight == 0.8


@pytest.mark.parametrize("raw", ["Turn | -1", "Turn | nan", "Turn | not-a-number"])
def test_parse_navigation_command_rejects_invalid_weights(raw):
    with pytest.raises(ValueError, match="non-negative number"):
        parse_navigation_command(raw)


def test_navigation_state_submission_updates_revision_and_clears_input():
    state = NavigationControlState(mode="navigation")
    state.input_text = "draft"

    command = state.submit_command("Stop near the crosswalk | 1.5")

    assert command.text == "Stop near the crosswalk"
    assert command.weight == 1.5
    assert command.revision == 1
    assert state.navigation_text == command.text
    assert state.navigation_weight == command.weight
    assert state.input_text == ""
    assert state.last_error == ""


def test_normal_mode_ignores_text_but_preserves_state():
    state = NavigationControlState("Keep lane", 1.2, mode="normal")

    command = state.submit_command("Turn left | 0.3")

    assert command.text == ""
    assert command.weight == 1.2
    assert command.revision == 0
    assert state.navigation_text == "Keep lane"


def test_vqa_mode_stores_question_and_resets_answer():
    state = NavigationControlState(mode="vqa", vqa_question="old question")
    state.vqa_answer = "old answer"

    command = state.submit_command("What is ahead?")

    assert command.text == "What is ahead?"
    assert command.revision == 1
    assert state.vqa_question == "What is ahead?"
    assert state.vqa_answer == ""
