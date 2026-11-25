"""The terminals contract (terminalscontract.py) held to the fake terminals and fritters: in `make check`."""

from collections.abc import Mapping, Sequence
from itertools import count
from pathlib import Path

import pytest

from faketerminals import FakeFritters, FakePane, FakeServer, FakeTerminals, Refusing, Wedged
from hands.core.effects import Fritter, Writer
from hands.core.tmux import Pane
from hands.sessions.terminals import Opened
from terminalscontract import DRAWN, Ending, TerminalsContract

SOCKET = Path("/private/tmp/hands-contract/tmux-501/default")


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
        return SOCKET

    async def window(self, session: str) -> Opened:
        opened = await self.terminals.open(SOCKET, session, Path("/code"), ("program",), {})
        assert isinstance(opened, Opened)
        return opened

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
        del self.terminals.servers[SOCKET]
        self.terminals.left.append(SOCKET)

    async def refuse(self) -> None:
        self._server.answering = Refusing(f"error connecting to {SOCKET} (Permission denied)")

    async def wedge(self) -> None:
        self._server.answering = Wedged("did not exit in 2.0 s, so it was killed")

    async def wrapped(self) -> Fritter:
        fritter = Fritter(Path(f"/private/tmp/fritter-{next(self._fritters)}/fritter.sock"), 8000)
        self.fritters.listening[fritter.socket] = fritter.pid
        return fritter

    def unheard(self) -> Fritter:
        return Fritter(Path("/private/tmp/fritter-gone/fritter.sock"), 8001)

    async def reads(self, writer: Writer, expected: str) -> str:
        match writer:
            case Fritter(socket=socket):
                read = self.fritters.read.get(socket, "")
            case Pane(id=id):
                pane = self._server.pane(id)
                assert pane is not None
                read = pane.read
        if read != expected:
            raise TimeoutError(f"the program read {read!r}, not {expected!r}")
        return read

    @property
    def _server(self) -> FakeServer:
        return self.terminals.servers[SOCKET]

    def _pane(self, window: Opened) -> FakePane:
        pane = self._server.pane(window.pane)
        assert pane is not None
        return pane


@pytest.fixture
def world() -> Fake:
    return Fake()


class TestFakeTerminals(TerminalsContract):
    pass
