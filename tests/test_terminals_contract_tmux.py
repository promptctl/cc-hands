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
from collections.abc import AsyncIterator, Callable, Mapping
from itertools import count
from pathlib import Path

import pytest

from hands.clock import SYSTEM
from hands.core.effects import Fritter, Writer
from hands.core.tmux import Pane
from hands.sessions.child import Children
from hands.sessions.panes import socket_directory
from hands.sessions.terminals import Opened, Unopened, Untyped
from hands.sessions.tmux import TYPING, TmuxExit, TmuxTerminals, ran_at
from hands.sessions.typing import FritterSockets, Typist
from terminalscontract import DRAWN, VALUE, VARIABLE, Ending, TerminalsContract

pytestmark = pytest.mark.outside

# How long tmux and fritter have to answer in this world: far past any load, inside the suite's own timeout, which is
# what bounds a wait here (pyproject.toml). The world's stopped ones are given up on at once: they never answer.
PATIENT, AT_ONCE = 60.0, 0.2
# A window's program: it draws DRAWN, asks its terminal for bracketed paste, then reads its terminal raw, recording what
# it was given as VARIABLE, the device of its terminal, its pid and its parent's, and every byte it reads, under the
# directory it is given.
RECORDER = r"""
import os, sys, tty
where, drawn, variable = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(where, exist_ok=True)
os.write(1, b"\x1b[?2004h" + drawn.encode() + b"\r\n")
tty.setraw(0)
for name, value in (("variable", os.environ.get(variable)), ("tty", os.fstat(0).st_rdev), ("pid", os.getpid()), ("parent", os.getppid())):
    if value is None:
        continue
    with open(os.path.join(where, name + ".part"), "w") as part:
        part.write(str(value))
    os.rename(os.path.join(where, name + ".part"), os.path.join(where, name))
with open(os.path.join(where, "read"), "ab", buffering=0) as read:
    while chunk := os.read(0, 4096):
        read.write(chunk)
"""


class Tmux:
    """A tmux server of the world's own, under `root`, a directory reached through a link (/tmp is /private/tmp), started
    by the first window it opens; and the fritter at `fritter`."""

    def __init__(self, root: Path, fritter: Path) -> None:
        self._root = root.resolve()
        self._fritter = fritter
        tmux = shutil.which("tmux")
        assert tmux is not None, "the contract is held to tmux, and no tmux is on the PATH"
        self._tmux = tmux
        # A server started with no config of the user's: none under this home, nor under its XDG one. Outside tmux.
        self._environment = {"PATH": os.environ["PATH"], "TMUX_TMPDIR": str(root), "HOME": str(root), "XDG_CONFIG_HOME": str(root), "TERM": "xterm-256color"}
        self.terminals = TmuxTerminals(Children(SYSTEM, lambda _: None), Path(tmux), PATIENT, PATIENT)
        self.fritters = FritterSockets(PATIENT)
        self._names = count()
        # Where each window's program records, by its pane, and each fritter's, by its address.
        self._recorders: dict[str, Path] = {}
        self._wrapped: dict[Path, Path] = {}
        self._clients: list[subprocess.Popen[bytes]] = []
        # The other end of each client's terminal, held open while it is attached.
        self._primaries: list[int] = []
        # The pid of each fritter, by its address; the server and fritters stopped, to go on once the world closes, and
        # whether the server's socket was made to refuse.
        self._fritters: dict[Path, int] = {}
        self._stopped: list[int] = []
        self._refused = False

    @property
    def socket(self) -> Path:
        return self._root / f"tmux-{os.getuid()}" / "default"

    @property
    def environment(self) -> dict[str, str]:
        return self._environment

    @property
    def directory(self) -> Path:
        return socket_directory(self._environment, os.getuid())

    async def window(self, session: str, environment: Mapping[str, str] | None = None) -> Opened:
        return (await self._opened(session, [], environment or {}))[0]

    def given(self, window: Opened) -> str | None:
        given = self._recorders[window.pane] / "variable"
        return given.read_text() if given.exists() else None

    def pid(self, window: Opened) -> int:
        return int((self._recorders[window.pane] / "pid").read_text())

    def device(self, window: Opened) -> int:
        return int((self._recorders[window.pane] / "tty").read_text())

    async def end(self, window: Opened, ending: Ending) -> None:
        match ending:
            case "dead":
                self._ask("set-option", "-p", "-t", window.pane, "remain-on-exit", "on")
                os.kill(self.pid(window), signal.SIGKILL)
                await self._until(lambda: self._ask("display-message", "-p", "-t", window.pane, "#{pane_dead}").strip() == "1")
            case "closed":
                os.kill(self.pid(window), signal.SIGKILL)
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

    async def refuse(self) -> None:
        self.socket.chmod(0)
        self._refused = True

    async def wedge(self) -> None:
        server = int(self._ask("display-message", "-p", "#{pid}").strip())
        os.kill(server, signal.SIGSTOP)
        self._stopped.append(server)
        self.terminals = TmuxTerminals(Children(SYSTEM, lambda _: None), Path(self._tmux), AT_ONCE, AT_ONCE)

    async def wrapped(self) -> Fritter:
        at = self._root / f"f{next(self._names)}"
        at.mkdir()
        # Under a shell that waits on it: tmux continues any program of a window's own that stops, so a fritter stopped
        # to stall it is one below the window's (measured with tmux 3.8a).
        wrapper = ["/bin/sh", "-c", '"$@"; exit $?', "sh", str(self._fritter), "--socket-dir", str(at), "--"]
        _, where = await self._opened("fritters", wrapper, {})
        await self._until(lambda: any(path.is_socket() for path in at.rglob("*")))
        [socket] = [path for path in at.rglob("*") if path.is_socket()]
        fritter = Fritter(socket, int((where / "pid").read_text()))
        self._wrapped[socket] = where
        # fritter runs the recorder, which names it as its parent.
        self._fritters[socket] = int((where / "parent").read_text())
        await self._until(lambda: _pasting(Typist(fritter.socket, fritter.pid)))
        return fritter

    def unheard(self) -> Fritter:
        return Fritter(self._root / "gone.sock", 1)

    async def stall(self, fritter: Fritter) -> None:
        os.kill(self._fritters[fritter.socket], signal.SIGSTOP)
        self._stopped.append(self._fritters[fritter.socket])
        self.fritters = FritterSockets(AT_ONCE)

    async def reads(self, writer: Writer, expected: str) -> str:
        match writer:
            case Fritter(socket=socket):
                read = self._wrapped[socket] / "read"
            case Pane(id=id):
                read = self._recorders[id] / "read"
        # Read once as much has arrived as was typed: what it is, whatever it is, is the program's answer.
        await self._until(lambda: read.exists() and len(read.read_bytes()) >= len(expected.encode()))
        return read.read_bytes().decode(errors="replace")

    def close(self) -> None:
        # The server and fritters answer again before the server is killed through its socket.
        for stopped in self._stopped:
            os.kill(stopped, signal.SIGCONT)
        if self._refused:
            self.socket.chmod(0o700)
        for client in self._clients:
            client.kill()
            client.wait()
        for primary in self._primaries:
            os.close(primary)
        subprocess.run([self._tmux, "-S", str(self.socket), "kill-server"], capture_output=True)

    async def _opened(self, session: str, wrapper: list[str], environment: Mapping[str, str]) -> tuple[Opened, Path]:
        where = self._root / f"w{next(self._names)}"
        opened = await self.terminals.open(session, self._root, [*wrapper, sys.executable, "-c", RECORDER, str(where), DRAWN, VARIABLE], {VARIABLE: VALUE}, {**self._environment, **environment})
        assert isinstance(opened, Opened), opened
        self._recorders[opened.pane] = where
        await self._until(lambda: (where / "pid").exists() and (where / "tty").exists())
        if not wrapper:
            await self._until(lambda: DRAWN in self._ask("capture-pane", "-p", "-t", opened.pane))
        return opened, where

    def _ask(self, *arguments: str) -> str:
        return subprocess.run([self._tmux, "-S", str(self.socket), *arguments], capture_output=True, text=True, check=True).stdout

    async def _until(self, done: Callable[[], bool]) -> None:
        # [LAW:no-ambient-temporal-coupling] no deadline of the world's own: the suite's timeout ends a wait that never does.
        while not done():
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
    root = Path(tempfile.mkdtemp(prefix="hc-", dir="/tmp"))
    tmux = Tmux(root, fritter)
    try:
        yield tmux
    finally:
        tmux.close()
        shutil.rmtree(root, ignore_errors=True)


class TestTmuxTerminals(TerminalsContract):
    pass


async def test_tmux_stops_a_list_at_its_first_failure_having_printed_what_ran_before_it(world: Tmux) -> None:
    # What TmuxTerminals.type tells a typing that failed after a key from one that failed before any by: TYPING printed.
    window = await world.window("play")
    tmux, children = shutil.which("tmux"), Children(SYSTEM, lambda _: None)
    assert tmux is not None
    partway = await ran_at(children, Path(tmux), world.socket, ["display-message", "-p", TYPING, ";", "paste-buffer", "-b", "hands-none", "-t", window.pane, ";", "send-keys", "-t", window.pane, "Enter"], PATIENT)
    before = await ran_at(children, Path(tmux), world.socket, ["send-keys", "-t", "%999999", "-l", "", ";", "display-message", "-p", TYPING], PATIENT)
    match partway, before:
        case TmuxExit(code=failed, out=out, err=err), TmuxExit(code=refused, out=nothing):
            assert (failed != 0, out.splitlines(), b"no buffer hands-none" in err) == (True, [TYPING.encode()], True)
            assert (refused != 0, nothing) == (True, b"")
        case other:
            pytest.fail(f"tmux answered {other}")


async def test_tmux_opens_no_window_where_its_socket_directory_is_one_another_user_could_reach() -> None:
    # [LAW:single-enforcer] tmux's own check, which it makes only where it picks the socket itself, as hands has it do.
    root = Path(tempfile.mkdtemp(prefix="hu-", dir="/tmp"))
    try:
        (shared := root / f"tmux-{os.getuid()}").mkdir()
        shared.chmod(0o777)
        tmux = shutil.which("tmux")
        assert tmux is not None
        match await TmuxTerminals(Children(SYSTEM, lambda _: None), Path(tmux), PATIENT, PATIENT).open("work", root, ("true",), {}, {"PATH": os.environ["PATH"], "TMUX_TMPDIR": str(root), "HOME": str(root)}):
            case Unopened(reason=reason):
                assert "unsafe permissions" in reason
            case other:
                pytest.fail(f"opened as {other}")
    finally:
        shutil.rmtree(root)
