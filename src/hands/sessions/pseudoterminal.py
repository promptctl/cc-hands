"""A Claude Code run on a pseudo-terminal hands holds: read as fast as it draws, and the last of what it showed kept.

The brain and the smoke test's working session both run so; what each runs, and in what environment,
is theirs to say.

[LAW:effects-at-boundaries] `Pseudoterminals` is the way hands runs a program so: `SpawnedPseudoterminals` on the
processes SpawnedChildren starts, named only at the root and where an exception is marked (test_children_boundary.py),
and tests/fakepseudoterminals.py in tests, the two held to one contract (tests/pseudoterminalcontract.py).
"""

import asyncio
import fcntl
import os
import pty
import queue
import re
import struct
import termios
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hands.core.session import ESCAPES
from hands.clock import Clock
from hands.sessions import wide
from hands.sessions.child import Command, Ending, Lost, NeverStarted, Running, SpawnedChildren, Terminal, Unreaped

# The terminal a Claude Code of hands' own draws on. Nobody looks at it; it is sized so a long line is not wrapped into many.
ROWS, COLS = 50, 200
# The terminal a program run on one is told it is on.
TERM = "xterm-256color"
# How much of what it last showed is kept, for the line that says it exited or never came up.
SHOWN_BYTES = 16 * 1024
SHOWN_LINES = 20
# How long a Claude Code told to stop has before it is killed.
STOP_SECONDS = 5.0

# What a terminal is told rather than shown, and the keys it is sent.
_CONTROL = re.compile(rf"{ESCAPES.pattern}|[\x00-\x09\x0b-\x1f\x7f]")


class _Terminal:
    """The pseudo-terminal's master side, held by hands: read as fast as it is written, and the last of it kept. Once
    no process holds the slave open its read ends, and `closed` is put to."""

    def __init__(self, master: int, closed: queue.SimpleQueue[None]) -> None:
        self._shown = b""
        self._lock = threading.Lock()

        def read() -> None:
            # A thread of its own: nothing is drawn when nothing reads, and a terminal's reads block.
            while True:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    data = b""
                if not data:
                    break
                with self._lock:
                    self._shown = (self._shown + data)[-SHOWN_BYTES:]
            os.close(master)
            closed.put(None)

        threading.Thread(target=read, name="a Claude Code's terminal", daemon=True).start()

    def last(self) -> str:
        """The last lines it showed, as text."""
        with self._lock:
            return last_lines(self._shown.decode(errors="replace"))


def last_lines(shown: str) -> str:
    """The last lines of what a terminal showed, as text: what it was told rather than shown taken out, and its blank
    lines."""
    lines = [line.rstrip() for line in _CONTROL.sub("", shown.replace("\r", "\n")).split("\n") if line.strip()]
    return "\n".join(lines[-SHOWN_LINES:])


class ClaudeCode(Protocol):
    """A Claude Code, or any program, running on a terminal hands holds, until it ends or is stopped."""

    @property
    def pid(self) -> int: ...

    @property
    def exit(self) -> asyncio.Future[Ending | Lost]:
        """How it ended, once it has and what it showed last is read to its end."""
        ...

    def shown(self) -> str:
        """The last lines it showed on its terminal."""
        ...

    async def stop(self) -> Ending | Lost | Unreaped:
        """Ask it alone to end, kill its group once it has not within STOP_SECONDS, and say how it ended."""
        ...


class Pseudoterminals(Protocol):
    """What runs a program on a pseudo-terminal of hands' own, as a terminal emulator runs one."""

    async def start(self, argv: Sequence[str], cwd: Path, environment: Mapping[str, str]) -> ClaudeCode | NeverStarted:
        """Run `argv` in `cwd` and `environment`, its TERM a terminal's, as a terminal's shell runs a command: a program
        the shell cannot run is said on the terminal, which ends 127 where it is not there and 126 where it cannot be
        run. Never started where the OS would not start the terminal's shell, in a directory that is not there among
        them."""
        ...


# [LAW:effects-at-boundaries] what makes the Pseudoterminals a unit runs on, as hands.sessions.child.ChildrenOn makes
# its Children.
type PseudoterminalsOn = Callable[[Clock, Callable[[wide.WideEvent], None]], Pseudoterminals]


class _Held:
    """A program running on a terminal hands holds, as SpawnedPseudoterminals starts it."""

    def __init__(self, running: Running, terminal: _Terminal) -> None:
        self._running = running
        self._terminal = terminal

    @property
    def pid(self) -> int:
        return self._running.pid

    @property
    def exit(self) -> asyncio.Future[Ending | Lost]:
        """How it ended, once it has and what it showed last is read to its end."""
        return self._running.ended

    def shown(self) -> str:
        """The last lines it showed on its terminal."""
        return self._terminal.last()

    async def stop(self) -> Ending | Lost | Unreaped:
        return await self._running.stopped(STOP_SECONDS)


@dataclass(frozen=True)
class SpawnedPseudoterminals:
    """Pseudoterminals on processes `children` starts, each on a pty of its own."""

    children: SpawnedChildren

    async def start(self, argv: Sequence[str], cwd: Path, environment: Mapping[str, str]) -> ClaudeCode | NeverStarted:
        # [LAW:single-enforcer] the program's own command refused as a run of it is, an empty one among them, which the
        # shell it runs under would take as nothing to run. Whether it can be run is the shell's exec's to say, alone.
        Command(tuple(argv), cwd, environment)
        master, slave = pty.openpty()
        closed: queue.SimpleQueue[None] = queue.SimpleQueue()
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
            # [LAW:no-ambient-temporal-coupling] read before the child starts, so nothing that can fail comes between its
            # start and the read of its terminal; the read closes the master once no process holds the slave.
            terminal = _Terminal(master, closed)
        except BaseException:
            os.close(master)
            os.close(slave)
            raise
        try:
            started = self.children.start(_holding_terminal(os.ttyname(slave), argv), cwd, {**environment, "TERM": TERM}, Terminal(slave, closed))
        finally:
            # The child holds its own: the read ends once it, and whatever it started, have let go of theirs.
            os.close(slave)
        match started:
            case NeverStarted():
                return started
            case Running():
                return _Held(started, terminal)


def _holding_terminal(terminal: str, argv: Sequence[str]) -> list[str]:
    """argv, run so that its terminal is its session's controlling terminal, as a terminal emulator's program's is: a
    Claude Code opens /dev/tty, which a process with no controlling terminal cannot.

    A session leader with no controlling terminal takes the first terminal it opens, so the shell opens it and execs
    argv in its place, keeping its pid. [LAW:no-ambient-temporal-coupling] no Python runs between fork and exec, where
    a lock another of hands' threads held at the fork would hang the child, and hands with it."""
    return ["/bin/sh", "-c", ': <>"$0"; exec "$@"', terminal, *argv]
