"""The terminal contract (terminalcontract.py) held to the pinned Claude Code itself, in a terminal of hands' own against the
mock of Anthropic's API: in `make check-rest`."""

import json
from collections.abc import AsyncGenerator, Sequence

import pytest

from hands.core.brainwork import Idle
from pinnedbrain import RESULT, TURN, Hands, MockApi
from pinnedbrain import claude_code, hands, mock_api, mock_binary  # pyright: ignore[reportUnusedImport]
from terminalcontract import QUESTION, TerminalContract

pytestmark = pytest.mark.outside

# What the model's reply to the turn a paste is sent as says.
READ = "I read it."


class Pinned:
    """The pinned Claude Code in hands' brain, against the mock: its screen read until it shows what the case waits for."""

    def __init__(self, hands: Hands, mock_api: MockApi) -> None:
        self.hands = hands
        self.mock_api = mock_api

    async def at_dialog(self) -> None:
        # A symlink is a command no setting of the brain's allows, so Claude Code asks about it in default mode.
        self.mock_api.script("1-left", {"bodyContains": [TURN, RESULT]}, {"type": "text", "text": "I left it unmade."})
        self.mock_api.script("2-link", {"lastUserContains": ["link the file"], "bodyContains": [TURN]}, {"type": "tool_use", "name": "Bash", "input": {"command": f"ln -s target {self.hands.workdir / 'made'}"}}, stop="tool_use")
        await self.hands.connected()
        await self.hands.say("link the file")
        await self.hands.until(lambda: len(self.hands.held()) == 1 and QUESTION in self.hands.shown())

    async def paste(self, text: str) -> None:
        # The model's reply to a turn holding the first and the last line of what is pasted: the turn the chord sends.
        lines = text.splitlines()
        self.mock_api.script("1-read", {"lastUserContains": [lines[0], lines[-1]], "bodyContains": [TURN]}, {"type": "text", "text": READ})
        await self.hands.connected()
        await self.hands.pasted(text)

    async def type(self, text: str) -> None:
        await self.hands.keyed("-l", text)

    async def key(self, name: str) -> None:
        await self.hands.keyed(name)

    async def shows(self, text: str) -> str:
        await self.hands.until(lambda: text in self.hands.shown())
        return self.hands.shown()

    async def ran(self) -> bool:
        # The symlink the dialog's command makes: a chord that said yes would have it run.
        return (self.hands.workdir / "made").is_symlink()

    async def took(self, words: Sequence[str]) -> None:
        # What Claude Code took as a turn reaches the model as the last of what the brain sends, among its own turns.
        await self.hands.until(lambda: any(TURN in (body := json.dumps(request.body)) and all(word in body for word in words) for request in self.hands.sent))
        # The turn is heard to its end, so the run ends with Claude Code at its prompt.
        await self.hands.until(lambda: READ in "".join(self.hands.out.said()) and self.hands.stage.work() == Idle())

    async def finish(self) -> None:
        """Ends the turn a dialog holds, as the user would, so the run ends with Claude Code at its prompt."""
        await self.hands.until(lambda: any("Say yes to allow it." in said for said in self.hands.out.said()))
        await self.hands.say("no")
        await self.hands.until(lambda: "I left it unmade." in "".join(self.hands.out.said()))
        # The refusal kept the command from running.
        assert not await self.ran()


@pytest.fixture
async def terminal(hands: Hands, mock_api: MockApi) -> AsyncGenerator[Pinned]:
    pinned = Pinned(hands, mock_api)
    yield pinned
    # A dialog still held is ended as the user would, so the run ends at Claude Code's prompt.
    if hands.held():
        await pinned.finish()


class TestThePin(TerminalContract):
    pass
