"""The rule of docs/claude-code-screen.md, conditions 1 to 3, on captures shaped as that document says Claude Code draws
each surface."""

import pytest

from hands.core.screen import Box, plain, typable

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
    assert typable(f"✻ Thinking… (3s)\n● streamed\n{box}") == typable(f"✽ Thinking… (4s)\n● streamed more\n{box}") == Box(box, drafted=False)


DIM, INVERSE, RESET = "\x1b[2m", "\x1b[7m", "\x1b[0m"


def _boxed(*rows: str) -> str:
    return "\n".join([RULE, *rows, RULE, f"{DIM}  ? for shortcuts{RESET}"])


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(f"❯{NBSP}{INVERSE}T{RESET}{DIM}ry \"fix lint errors\"{RESET}", id="the placeholder, with the cursor on it"),
        pytest.param(f"❯{NBSP}{DIM}Try \"fix lint errors\"{RESET}", id="the placeholder, unfocused"),
        pytest.param(f"{DIM}❯{NBSP}Try{RESET}", id="dim set before the pointer"),
        pytest.param(f"❯{NBSP}{INVERSE} {RESET}", id="the cursor alone"),
        pytest.param(f"❯{NBSP}", id="nothing drawn"),
        pytest.param(f"❯{NBSP}\x1b[38;2;2;7;2m{DIM}Try{RESET}", id="a colour whose numbers read as dim and inverse"),
    ],
)
def test_an_empty_input_holds_no_words(row: str) -> None:
    box = typable(_boxed(row))
    assert box is not None and not box.drafted


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([f"❯{NBSP}fix it{INVERSE} {RESET}"], id="words, the cursor at their end"),
        pytest.param([f"❯{NBSP}fi{INVERSE}x{RESET} it"], id="the cursor mid-word"),
        pytest.param([f"❯{NBSP}{INVERSE}a{RESET}"], id="one character, the cursor on it"),
        pytest.param([f"❯{NBSP}", f"  second line{INVERSE} {RESET}"], id="words on a later row only"),
        pytest.param([f"❯{NBSP}\x1b[38;5;2mcoloured{RESET}"], id="words in a colour"),
    ],
)
def test_words_in_the_input_are_drafted(rows: list[str]) -> None:
    box = typable(_boxed(*rows))
    assert box is not None and box.drafted


def test_the_box_s_rows_are_plain_text() -> None:
    box = typable(_boxed(f"❯{NBSP}{DIM}Try{RESET}"))
    assert box is not None and "\x1b" not in box.rows and plain(f"{DIM}a{RESET}b") == "ab"
