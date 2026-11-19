"""The terminal contract (terminalcontract.py) held to the fake Claude Code the unit layer plays: in `make check`."""

from collections.abc import Sequence

import pytest

from claudescreen import ClaudeScreen
from terminalcontract import CHORD, TerminalContract


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

    async def ran(self) -> bool:
        return self.screen.ran


@pytest.fixture
def terminal() -> Fake:
    return Fake()


class TestTheFake(TerminalContract):
    pass
