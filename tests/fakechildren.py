"""Children as values: the programs a test stands in for, each answering the runs of it, and no process started
(hands.sessions.child.Children). tests/childrencontract.py holds it to the Runner."""

import asyncio
import errno
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hands.clock import Clock, wait_all
from hands.sessions.child import Captured, Command, Exited, Kill, Killed, Limit, Lost, NeverStarted, Outcome, Ran, Running, Stdio, Traced, Unreaped
from hands.tasks import cancel_and_wait


@dataclass(frozen=True)
class Asked:
    """A run, as its program is handed it: its argv, the directory and the environment it runs in (a traced run's with
    its TRACEPARENT), what is on its stdin, and the limit it is held to."""

    argv: tuple[str, ...]
    cwd: Path | None
    env: Mapping[str, str] | None
    given: bytes | None
    limit: float


# A program a test stands in for: the outcome every run of it ends in, or the answer it makes each run, in the time
# that run takes.
type Answer = Callable[[Asked], Awaitable[Outcome]]
type Program = Outcome | Answer


def exited(code: int, out: bytes = b"", err: bytes = b"") -> Ran:
    """A run that exited `code`, having written `out` and `err`."""
    return Ran(Exited(code), Captured(out), Captured(err))


class FakeChildren:
    """Children on `clock` that start no process. A run of a program `programs` holds, by its argv[0], is its answer
    within the run's limit, and is killed for it, as the Runner kills one, where the answer outlasts it; a run of any
    other never starts, as the OS answers for a program it cannot find. Every run is kept in `asked`, oldest first."""

    def __init__(self, clock: Clock, programs: Mapping[str, Program] | None = None) -> None:
        self.clock = clock
        self.programs = dict(programs or {})
        self.asked: list[Asked] = []

    async def run(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | Traced | None = None, given: bytes | None = None) -> Outcome:
        asked = self._asked(argv, limit, cwd, env.handed if isinstance(env, Traced) else env, given)
        match self.programs.get(asked.argv[0]):
            case None:
                return _unfound(asked.argv[0])
            case Ran() | NeverStarted() | Unreaped() | Lost() as outcome:
                return outcome
            case answer:
                answering = asyncio.ensure_future(answer(asked))
                try:
                    await wait_all(self.clock, (answering,), limit)
                finally:
                    # Cancelled, or past its limit: what the program was doing is ended with its run.
                    await cancel_and_wait(answering)
                return answering.result() if not answering.cancelled() else Ran(Killed(Kill(Limit(limit))), Captured(b""), Captured(b""))

    def run_blocking(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        asked = self._asked(argv, limit, cwd, env, None)
        match self.programs.get(asked.argv[0]):
            case None:
                return _unfound(asked.argv[0])
            case Ran() | NeverStarted() | Unreaped() | Lost() as outcome:
                return outcome
            case _:
                raise AssertionError(f"{asked.argv[0]} answers a run in a coroutine, which a blocking run cannot wait on: give it as the Outcome it ends in")

    def start(self, argv: Sequence[str], cwd: Path, env: Mapping[str, str], stdio: Stdio) -> Running | NeverStarted:
        raise AssertionError(f"FakeChildren starts no {argv[0]} to run on its own: a test of a started child runs the Runner (tests/test_child.py)")

    def _asked(self, argv: Sequence[str], limit: float, cwd: Path | None, env: Mapping[str, str] | None, given: bytes | None) -> Asked:
        # [LAW:single-enforcer] refused as the Runner refuses them, before anything runs.
        Command(tuple(argv), cwd, env)
        Limit(limit)
        asked = Asked(tuple(argv), cwd, env, given, limit)
        self.asked.append(asked)
        return asked


def _unfound(program: str) -> NeverStarted:
    """What the OS answers a start of a program it cannot find with."""
    return NeverStarted(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), program))
