"""The terminals contract (terminalscontract.py) held to the fake terminals and fritters: in `make check`."""

from collections.abc import Mapping, Sequence
from itertools import count
from pathlib import Path

import pytest

from fakeprocesses import USER
from faketerminals import Exited, FakeFritters, FakePane, FakeServer, FakeTerminals, Refusing, Wedged
from hands.core.effects import Fritter, Writer
from hands.core.tmux import Pane
from hands.sessions.panes import socket_directory
from hands.sessions.terminals import Opened
from terminalscontract import DRAWN, VALUE, VARIABLE, Ending, TerminalsContract

ENVIRONMENT = {"TMUX_TMPDIR": "/private/tmp/hands-contract"}


class Fake:
    """The fake terminals, in which a window's program draws DRAWN as it opens, and a fritter is an address put there."""

    def __init__(self) -> None:
        self.terminals = FakeTerminals(opened=self._drawn)
        self.fritters = FakeFritters()
        self._fritters = count(1)

    @staticmethod
    def _drawn(pane: FakePane, command: Sequence[str], environment: Mapping[str, str]) -> None:
        pane.screen = DRAWN

    @property
    def socket(self) -> Path:
        return self.directory / "default"

    @property
    def environment(self) -> Mapping[str, str]:
        return ENVIRONMENT

    @property
    def directory(self) -> Path:
        return socket_directory(ENVIRONMENT, USER)

    async def window(self, session: str, environment: Mapping[str, str] | None = None) -> Opened:
        opened = await self.terminals.open(session, Path("/code"), ("program",), {VARIABLE: VALUE}, {**ENVIRONMENT, **(environment or {})})
        assert isinstance(opened, Opened)
        return opened

    def given(self, window: Opened) -> str | None:
        return self._pane(window).environment.get(VARIABLE)

    def pid(self, window: Opened) -> int:
        return self._pane(window).pid

    def device(self, window: Opened) -> int:
        return self._pane(window).terminal

    async def end(self, window: Opened, ending: Ending) -> None:
        match ending:
            case "dead":
                self._pane(window).end()
            case "closed":
                self._server.close(self._pane(window))

    async def attach(self, window: Opened) -> int:
        client = 0x20000000 + len(self._server.clients)
        self._server.clients[client] = window.pane
        return client

    async def stop(self) -> None:
        self._server.answering = Exited()

    async def refuse(self) -> None:
        self._server.answering = Refusing(f"error connecting to {self.socket} (Permission denied)")

    async def wedge(self) -> None:
        self._server.answering = Wedged("was killed with SIGKILL, because it ran past 0.2s")

    async def wrapped(self) -> Fritter:
        fritter = Fritter(Path(f"/private/tmp/fritter-{next(self._fritters)}/fritter.sock"), 8000)
        self.fritters.listening[fritter.socket] = fritter.pid
        return fritter

    def unheard(self) -> Fritter:
        return Fritter(Path("/private/tmp/fritter-gone/fritter.sock"), 8001)

    async def stall(self, fritter: Fritter) -> None:
        self.fritters.wedged.add(fritter.socket)

    async def reads(self, writer: Writer, expected: str) -> str:
        match writer:
            case Fritter(socket=socket):
                read = self.fritters.read.get(socket, "")
            case Pane(id=id):
                pane = self._server.pane(id)
                assert pane is not None
                read = pane.read
        return read

    @property
    def _server(self) -> FakeServer:
        return self.terminals.servers[self.socket]

    def _pane(self, window: Opened) -> FakePane:
        pane = self.terminals.servers[window.socket].pane(window.pane)
        assert pane is not None
        return pane


@pytest.fixture
def world() -> Fake:
    return Fake()


class TestFakeTerminals(TerminalsContract):
    pass
