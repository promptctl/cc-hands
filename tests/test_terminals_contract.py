"""The terminals contract (terminalscontract.py) held to the fake terminals: in `make check`."""

from collections.abc import Mapping, Sequence
from itertools import count
from pathlib import Path

import pytest

from faketerminals import FakePane, FakeServer, FakeTerminals
from hands.core.effects import Fritter, Writer
from hands.core.tmux import Pane
from hands.sessions.terminals import Opened
from terminalscontract import DRAWN, TerminalsContract

SOCKET = Path("/private/tmp/hands-contract/tmux-501/default")


class Fake:
    """The fake terminals, in which a window's program draws DRAWN as it opens, and a fritter is an address put there."""

    def __init__(self) -> None:
        self.fake = FakeTerminals(opens_at=SOCKET, opened=self._drawn)
        self._fritters = count(1)

    @staticmethod
    def _drawn(pane: FakePane, command: Sequence[str], environment: Mapping[str, str]) -> None:
        pane.screen = DRAWN

    @property
    def terminals(self) -> FakeTerminals:
        return self.fake

    @property
    def socket(self) -> Path:
        return SOCKET

    async def window(self, session: str) -> Opened:
        opened = await self.fake.open(session, Path("/code"), ("program",), {})
        assert isinstance(opened, Opened)
        return opened

    def pid(self, window: Opened) -> int:
        return self._pane(window).pid

    def device(self, window: Opened) -> int:
        return self._pane(window).terminal

    async def end(self, window: Opened, kept: bool) -> None:
        if kept:
            self._pane(window).dead = True
        else:
            self._server.panes.remove(self._pane(window))

    async def attach(self, window: Opened) -> int:
        client = 0x20000000 + len(self._server.clients)
        self._server.clients[client] = window.pane
        return client

    async def stop(self) -> None:
        del self.fake.servers[SOCKET]
        self.fake.left.append(SOCKET)

    async def wrapped(self) -> Fritter:
        fritter = Fritter(Path(f"/private/tmp/fritter-{next(self._fritters)}/fritter.sock"), 8000)
        self.fake.fritters[fritter.socket] = fritter.pid
        return fritter

    def unheard(self) -> Fritter:
        return Fritter(Path("/private/tmp/fritter-gone/fritter.sock"), 8001)

    async def reads(self, writer: Writer, expected: str) -> str:
        match writer:
            case Fritter(socket=socket):
                return self.fake.read.get(socket, "")
            case Pane(id=id):
                pane = self._server.pane(id)
                assert pane is not None
                return pane.read

    @property
    def _server(self) -> FakeServer:
        return self.fake.servers[SOCKET]

    def _pane(self, window: Opened) -> FakePane:
        pane = self._server.pane(window.pane)
        assert pane is not None
        return pane


@pytest.fixture
def world() -> Fake:
    return Fake()


class TestFakeTerminals(TerminalsContract):
    pass

