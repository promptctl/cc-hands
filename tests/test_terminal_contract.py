"""The terminal contract (terminalcontract.py) held to the fake Claude Code the unit layer plays: in `make check`."""

from collections.abc import Sequence

import pytest

from claudescreen import ClaudeScreen
from terminalcontract import (
    CHORD,
    test_the_submit_chord_typed_into_a_held_permission_dialog_answers_nothing,  # pyright: ignore[reportUnusedImport]
    test_words_pasted_at_the_prompt_are_sent_by_the_submit_chord,  # pyright: ignore[reportUnusedImport]
)


class Fake:
    """The fake Claude Code, whose every change of screen is the step that caused it."""

    def __init__(self) -> None:
        self.screen = ClaudeScreen(CHORD)

    async def at_dialog(self) -> None:
        self.screen.at_dialog()

    async def paste(self, text: str) -> None:
        self.screen.paste(text)

    async def type(self, text: str) -> None:
        self.screen.type(text)

    async def key(self, name: str) -> None:
        self.screen.key(name)

    async def shows(self, text: str) -> str:
        assert text in (shown := self.screen.shown()), f"the screen showed:\n{shown}"
        return shown

    async def took(self, words: Sequence[str]) -> None:
        assert self.screen.took(words), f"Claude Code took {self.screen.turns!r}"


@pytest.fixture
def terminal() -> Fake:
    return Fake()
