"""The terminals contract (terminalscontract.py) held to tmux and fritter through `hands.sessions.tmux.TmuxTerminals`: a tmux
server of the test's own, its socket in a directory of the test's own, its windows running a Python that records what it
reads, and the fritter this checkout carries. Outside: it starts processes and connects to their sockets."""

import asyncio
import fcntl
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
from collections.abc import AsyncIterator, Callable
from itertools import count
from pathlib import Path

import pytest

from hands.clock import SYSTEM
from hands.core.effects import Fritter, Writer
from hands.core.tmux import Pane
from hands.sessions.child import Children
from hands.sessions.terminals import Opened
from hands.sessions.tmux import TmuxTerminals
from hands.sessions.typing import Typist, Untyped
from terminalscontract import DRAWN, TerminalsContract

pytestmark = pytest.mark.outside

# How long a program, a client or a server has to do what the world waits on.
UP_SECONDS = 10.0
# A window's program: it draws DRAWN, asks its terminal for bracketed paste, then reads its terminal raw, recording the
# device of its terminal, its pid, and every byte it reads, under the directory it is given.
RECORDER = r"""
import os, sys, tty
where, drawn = sys.argv[1], sys.argv[2]
os.makedirs(where, exist_ok=True)
os.write(1, b"\x1b[?2004h" + drawn.encode() + b"\r\n")
tty.setraw(0)
for name, value in (("tty", os.fstat(0).st_rdev), ("pid", os.getpid())):
    with open(os.path.join(where, name + ".part"), "w") as part:
        part.write(str(value))
    os.rename(os.path.join(where, name + ".part"), os.path.join(where, name))
with open(os.path.join(where, "read"), "ab", buffering=0) as read:
    while chunk := os.read(0, 4096):
        read.write(chunk)
"""


class Tmux:
    """A tmux server of the world's own, under `root`, started by the first window it opens; and the fritter at `fritter`."""

    def __init__(self, root: Path, fritter: Path) -> None:
        self._root = root
        self._fritter = fritter
        tmux = shutil.which("tmux")
        assert tmux is not None, "the contract is held to tmux, and no tmux is on the PATH"
        self._tmux = tmux
        # A server started with no config of the user's: none under this home, nor under its XDG one.
        self._environment = {"PATH": os.environ["PATH"], "TMUX_TMPDIR": str(root), "HOME": str(root), "XDG_CONFIG_HOME": str(root), "TERM": "xterm-256color"}
        self.terminals = TmuxTerminals(Children(SYSTEM, lambda _: None), os.environ.get("PATH"))
        self._names = count()
        # Where each window's program records, by its pane, and each fritter's, by its address.
        self._recorders: dict[str, Path] = {}
        self._wrapped: dict[Path, Path] = {}
        self._clients: list[subprocess.Popen[bytes]] = []
        # The other end of each client's terminal, held open while it is attached.
        self._primaries: list[int] = []

    @property
    def socket(self) -> Path:
        return self._root / f"tmux-{os.getuid()}" / "default"

    async def window(self, session: str) -> Opened:
        return (await self._opened(session, []))[0]

    def pid(self, window: Opened) -> int:
        return int((self._recorders[window.pane] / "pid").read_text())

    def device(self, window: Opened) -> int:
        return int((self._recorders[window.pane] / "tty").read_text())

    async def end(self, window: Opened, kept: bool) -> None:
        if kept:
            self._ask("set-option", "-p", "-t", window.pane, "remain-on-exit", "on")
        os.kill(self.pid(window), signal.SIGKILL)
        if kept:
            await self._until(lambda: self._ask("display-message", "-p", "-t", window.pane, "#{pane_dead}").strip() == "1")
        else:
            await self._until(lambda: window.pane not in self._ask("list-panes", "-a", "-F", "#{pane_id}").split())

    async def attach(self, window: Opened) -> int:
        self._ask("select-window", "-t", window.pane)
        primary, secondary = os.openpty()
        fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        self._clients.append(subprocess.Popen([self._tmux, "-S", str(self.socket), "attach-session", "-t", window.pane], stdin=secondary, stdout=secondary, stderr=secondary, env=self._environment, start_new_session=True))
        device = os.fstat(secondary).st_rdev
        os.close(secondary)
        self._primaries.append(primary)
        await self._until(lambda: len(self._ask("list-clients", "-F", "#{client_tty}").split()) >= len(self._clients))
        return device

    async def stop(self) -> None:
        server = int(self._ask("display-message", "-p", "#{pid}").strip())
        os.kill(server, signal.SIGKILL)
        await self._until(lambda: not _runs(server))

    async def wrapped(self) -> Fritter:
        at = self._root / f"f{next(self._names)}"
        at.mkdir()
        _, where = await self._opened("fritters", [str(self._fritter), "--socket-dir", str(at), "--"])
        await self._until(lambda: any(path.is_socket() for path in at.rglob("*")))
        [socket] = [path for path in at.rglob("*") if path.is_socket()]
        fritter = Fritter(socket, int((where / "pid").read_text()))
        self._wrapped[socket] = where
        await self._until(lambda: _pasting(Typist(fritter.socket, fritter.pid)))
        return fritter

    def unheard(self) -> Fritter:
        return Fritter(self._root / "gone.sock", 1)

    async def reads(self, writer: Writer, expected: str) -> str:
        match writer:
            case Fritter(socket=socket):
                read = self._wrapped[socket] / "read"
            case Pane(id=id):
                read = self._recorders[id] / "read"
        try:
            await self._until(lambda: read.exists() and read.read_bytes().decode(errors="replace") == expected)
        except TimeoutError:
            pass
        return read.read_bytes().decode(errors="replace") if read.exists() else ""

    def close(self) -> None:
        for client in self._clients:
            client.kill()
            client.wait()
        for primary in self._primaries:
            os.close(primary)
        subprocess.run([self._tmux, "-S", str(self.socket), "kill-server"], capture_output=True)

    async def _opened(self, session: str, wrapper: list[str]) -> tuple[Opened, Path]:
        where = self._root / f"w{next(self._names)}"
        opened = await self.terminals.open(session, self._root, [*wrapper, sys.executable, "-c", RECORDER, str(where), DRAWN], self._environment)
        assert isinstance(opened, Opened), opened
        self._recorders[opened.pane] = where
        await self._until(lambda: (where / "pid").exists() and (where / "tty").exists())
        if not wrapper:
            await self._until(lambda: DRAWN in self._ask("capture-pane", "-p", "-t", opened.pane))
        return opened, where

    def _ask(self, *arguments: str) -> str:
        return subprocess.run([self._tmux, "-S", str(self.socket), *arguments], capture_output=True, text=True, check=True).stdout

    async def _until(self, done: Callable[[], bool]) -> None:
        deadline = asyncio.get_running_loop().time() + UP_SECONDS
        while not done():
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("the world did not get there in time")
            await asyncio.sleep(0.02)


def _runs(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pasting(typist: Typist) -> bool:
    try:
        typist.pasting()
    except Untyped:
        return False
    return True


@pytest.fixture
async def world(fritter: Path) -> AsyncIterator[Tmux]:
    # Short, under /tmp by its real path: a unix socket's path is capped near 104 bytes, and tmux names its socket by the
    # directory's real path.
    root = Path(tempfile.mkdtemp(prefix="hc-", dir="/tmp")).resolve()
    tmux = Tmux(root, fritter)
    try:
        yield tmux
    finally:
        tmux.close()
        shutil.rmtree(root, ignore_errors=True)


class TestTmuxTerminals(TerminalsContract):
    pass
