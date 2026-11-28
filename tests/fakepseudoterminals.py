"""Pseudoterminals as values: each program a test stands in for shows what it says and ends as it says, with no
terminal opened and no process started (hands.sessions.pseudoterminal.Pseudoterminals). tests/pseudoterminalcontract.py
holds it to SpawnedPseudoterminals."""

import asyncio
import errno
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fakechildren import Refused, never_started
from hands.sessions.child import Command, Ending, Exited, Lost, NeverStarted, Stop, Stopped, Unreaped
from hands.sessions.pseudoterminal import STOP_SECONDS, TERM, ClaudeCode, last_lines

# The pid the first start is given, the next the one after, and so on: no process of the machine's.
FIRST_PID = 90001


@dataclass(frozen=True)
class Asked:
    """A start, as its program is handed it: its argv, the directory it runs in, and its environment, the terminal's
    TERM in it. [LAW:single-enforcer] refused as SpawnedChildren refuses a start, by the same Command."""

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        Command(self.argv, self.cwd, self.environment)


@dataclass(frozen=True)
class Shows:
    """A program a test stands in for on a terminal: one that shows what `shown` makes of how it was started, and then
    exits `code`, or, with no code, runs until it is stopped."""

    shown: Callable[[Asked], str]
    code: int | None = None


# A program that is not there, and one that is there and cannot be run, as the terminal's shell says each and ends.
NOT_FOUND = Shows(lambda asked: f"exec: {asked.argv[0]}: not found", 127)
NOT_RUNNABLE = Shows(lambda asked: f"{asked.argv[0]}: Permission denied", 126)


class _Showing:
    """A program a test stands in for, on a terminal: what it showed, and how it ended once it has."""

    def __init__(self, pid: int, shown: str, code: int | None) -> None:
        self._pid = pid
        self._shown = shown
        self._ended: asyncio.Future[Ending | Lost] = asyncio.get_running_loop().create_future()
        if code is not None:
            self._ended.set_result(Exited(code))

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def exit(self) -> asyncio.Future[Ending | Lost]:
        # Shielded, as SpawnedChildren's is: a waiter cancelled leaves the end to the others.
        return asyncio.shield(self._ended)

    def shown(self) -> str:
        return last_lines(self._shown)

    async def stop(self) -> Ending | Lost | Unreaped:
        if not self._ended.done():
            # It ends when asked, within its grace.
            self._ended.set_result(Stopped(Stop(STOP_SECONDS)))
        return self._ended.result()


class FakePseudoterminals:
    """Pseudoterminals that open no terminal and start no process: a start of a program `programs` holds, by its argv[0],
    shows and ends as that program says, and a start in a directory that is not there is refused, as the OS refuses the
    terminal's shell one. A start of any other program is the test's mistake, raised as one: a program the test means to
    be missing is given as NOT_FOUND. Every start is kept in `asked`, oldest first."""

    def __init__(self, programs: Mapping[str, Shows] | None = None) -> None:
        self.programs = dict(programs or {})
        self.asked: list[Asked] = []

    async def start(self, argv: Sequence[str], cwd: Path, environment: Mapping[str, str]) -> ClaudeCode | NeverStarted:
        asked = Asked(tuple(argv), cwd, {**environment, "TERM": TERM})
        self.asked.append(asked)
        if not cwd.is_dir():
            return never_started(Refused(errno.ENOTDIR if cwd.exists() else errno.ENOENT), str(cwd))
        if (program := self.programs.get(asked.argv[0])) is None:
            raise AssertionError(f"the test stands in for no program {asked.argv[0]!r} on a terminal: give it in `programs`, as NOT_FOUND where the test means it to be missing")
        return _Showing(FIRST_PID + len(self.asked) - 1, program.shown(asked), program.code)
