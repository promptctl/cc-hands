"""The rule of docs/claude-code-screen.md, conditions 1 to 3, on captures shaped as that document says Claude Code draws
each surface."""

import pytest

from hands.core.screen import typable

RULE = "─" * 40
NBSP = " "


@pytest.mark.parametrize(
    "shown",
    [
        pytest.param(f"{RULE}\n❯{NBSP}\n{RULE}\n  ? for shortcuts", id="an idle prompt"),
        pytest.param(f"● Done.\n\n{RULE}\n❯{NBSP}fix the tests\n  and run them\n{RULE}", id="a prompt holding a draft"),
        pytest.param(f"{'─' * 3} plan mode {'─' * 20}\n❯{NBSP}\n{RULE}", id="a top rule carrying a label"),
        pytest.param(f"{RULE}\n❯{NBSP}", id="a prompt row that captures as the pointer and U+00A0 alone"),
        pytest.param(f"> earlier text\n❯{NBSP}quoted in the transcript\n\n{RULE}\n❯{NBSP}\n{RULE}", id="the transcript imitating a prompt above the real one"),
    ],
)
def test_a_turn_goes_into_the_prompt(shown: str) -> None:
    assert typable(shown) is not None


@pytest.mark.parametrize(
    "shown",
    [
        pytest.param(" Write file\n Do you want to create notes.txt?\n ❯ 1. Yes\n   2. No, and tell Claude what to do differently", id="a permission dialog"),
        pytest.param(f"{RULE}\n❯{NBSP}/com\n{RULE}\n ❯ /compact   Clear history but keep a summary\n   /config", id="autocomplete below the prompt"),
        pytest.param(f"{RULE}\n❯{NBSP}fix\n{RULE}\n search prompts: fix", id="history search"),
        pytest.param(f"{RULE}\n❯{NBSP}\n{RULE}\n no matching prompt: zz", id="history search finding nothing"),
        pytest.param(f"● Earlier:\n❯{NBSP}a prompt quoted in the transcript\n Do you want to proceed?\n ❯ 1. Yes\n   2. No", id="transcript text imitating the prompt above a dialog"),
        pytest.param(f"some output\n❯{NBSP}\n{RULE}", id="a prompt row with no rule above it"),
        pytest.param(f"❯{NBSP}\n{RULE}", id="a prompt row on the screen's first row"),
        pytest.param(f"{RULE}\n❯ hello\n{RULE}", id="the pointer followed by an ordinary space"),
        pytest.param("", id="a blank screen"),
        pytest.param(f"{RULE}\n!{NBSP}ls\n{RULE}", id="bash mode, where the turn would run as a shell command"),
    ],
)
def test_a_turn_does_not_go_into_the_prompt(shown: str) -> None:
    assert typable(shown) is None


def test_the_box_is_what_settles_while_a_turn_draws_above_it() -> None:
    box = f"{RULE}\n❯{NBSP}\n{RULE}\n  esc to interrupt"
    assert typable(f"✻ Thinking… (3s)\n● streamed\n{box}") == typable(f"✽ Thinking… (4s)\n● streamed more\n{box}") == box
