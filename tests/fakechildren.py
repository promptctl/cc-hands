"""Children as values: the programs a test stands in for, each answering the runs of it, and no process started
(hands.sessions.child.Children). tests/childrencontract.py holds it to SpawnedChildren."""

import asyncio
import errno
import os
import queue
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.clock import Clock, wait_all
from hands.sessions import wide
from hands.sessions.child import Captured, Command, Exited, Kill, Killed, Limit, Lost, NeverStarted, Outcome, Ran, Traced, Unreaped
from hands.tasks import cancel_and_wait


@dataclass(frozen=True)
class Asked:
    """A run, as its program is handed it: its argv, the directory and the environment it runs in (a traced run's with
    its TRACEPARENT), what is on its stdin, and the limit it is held to. [LAW:single-enforcer] refused as SpawnedChildren
    refuses a run, by the same Command and Limit, so no run is asked that SpawnedChildren would not start."""

    argv: tuple[str, ...]
    cwd: Path | None
    env: Mapping[str, str] | None
    given: bytes | None
    limit: float

    def __post_init__(self) -> None:
        Command(self.argv, self.cwd, self.env)
        Limit(self.limit)


@dataclass(frozen=True)
class Refused:
    """A program the OS will not start, refusing it with `errno` as it refuses one that is not there (ENOENT) or one
    that is there and cannot be run (EACCES)."""

    errno: int


# A program that is not there.
NOT_THERE = Refused(errno.ENOENT)


@dataclass(frozen=True)
class Hung:
    """A program that writes `out` and `err` and then never ends of itself, so each run of it is killed at its limit
    with what it wrote."""

    out: bytes = b""
    err: bytes = b""


# [LAW:one-source-of-truth] how a run of a program a test stands in for ends: a run that never started given by the errno
# the OS refuses it with alone, so NeverStarted is the fake's to make, as the OS's refusal is SpawnedChildren's.
type Answer = Ran | Unreaped | Lost | Refused


@dataclass(frozen=True)
class Answers:
    """A program whose every run ends as `answer` says of it, in the time the answer takes: killed at its limit where
    the answer outlasts it, as SpawnedChildren kills a run."""

    answer: Callable[[Asked], Awaitable[Answer]]


# A program a test stands in for: how every run of it ends, or one of the ways a program behaves above.
type Program = Answer | Hung | Answers


def exited(code: int, out: bytes = b"", err: bytes = b"") -> Ran:
    """A run that exited `code`, having written `out` and `err`."""
    return Ran(Exited(code), Captured(out), Captured(err))


class FakeChildren:
    """Children on `clock` that start no process: a run of a program `programs` holds, by its argv[0], is as that
    program behaves. A run of any other is the test's mistake, raised as one: a program the test means to be missing is
    given as NOT_THERE. Every run is kept in `asked`, oldest first."""

    def __init__(self, clock: Clock, programs: Mapping[str, Program] | None = None) -> None:
        self.clock = clock
        self.programs = dict(programs or {})
        self.asked: list[Asked] = []

    async def run(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | Traced | None = None, given: bytes | None = None) -> Outcome:
        asked, program = self._asked(Asked(argv, cwd, env.handed if isinstance(env, Traced) else env, given, limit))
        match program:
            case Ran() | Unreaped() | Lost() | Refused():
                return _at_once(asked, program)
            case Hung(out=out, err=err):
                await self.clock.sleep(limit)
                return _killed(limit, out, err)
            case Answers(answer=answer):
                answering = asyncio.ensure_future(answer(asked))
                try:
                    await wait_all(self.clock, (answering,), limit)
                    if answering.done():
                        return _at_once(asked, answering.result())
                finally:
                    # Cancelled, or past its limit: what the program was doing is ended with its run.
                    await cancel_and_wait(answering)
                return _killed(limit, b"", b"")

    def run_blocking(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        asked, program = self._asked(Asked(argv, cwd, env, None, limit))
        match program:
            case Ran() | Unreaped() | Lost() | Refused():
                return _at_once(asked, program)
            case Hung(out=out, err=err):
                # The calling thread waits the limit out, as it does on SpawnedChildren: on a virtual clock, at once.
                try:
                    self.clock.take(queue.SimpleQueue[None](), limit)
                except queue.Empty:
                    pass
                return _killed(limit, out, err)
            case Answers():
                raise AssertionError(f"{asked.argv[0]} answers a run in a coroutine, which a blocking run cannot wait on: give it as the Answer it ends in")

    def _asked(self, asked: Asked) -> tuple[Asked, Program]:
        self.asked.append(asked)
        if (program := self.programs.get(asked.argv[0])) is None:
            raise AssertionError(f"the test stands in for no program {asked.argv[0]!r}: give it in `programs`, as NOT_THERE where the test means it to be missing")
        return asked, program


class Programs:
    """[LAW:effects-at-boundaries] what makes each unit's children in a test, as SpawnedChildren is made at the root: a
    FakeChildren on the unit's own clock, answering `programs`, each kept in `made`, oldest first."""

    def __init__(self, programs: Mapping[str, Program] | None = None) -> None:
        self.programs = dict(programs or {})
        self.made: list[FakeChildren] = []

    def __call__(self, clock: Clock, emit: Callable[[wide.WideEvent], None]) -> FakeChildren:
        made = FakeChildren(clock, self.programs)
        self.made.append(made)
        return made

    @property
    def asked(self) -> list[Asked]:
        """Every run the children it made were asked, theirs in the order they were made."""
        return [asked for made in self.made for asked in made.asked]


def _at_once(asked: Asked, answer: Answer) -> Outcome:
    """The outcome of a run that ends as `answer` says."""
    match answer:
        case Refused():
            return never_started(answer, asked.argv[0])
        case outcome:
            return outcome


def never_started(refused: Refused, program: str) -> NeverStarted:
    """A start of `program` the OS refused, as `refused` says: [LAW:one-source-of-truth] with the OSError the OS's code
    makes, FileNotFoundError for ENOENT among them."""
    return NeverStarted(OSError(refused.errno, os.strerror(refused.errno), program))


def _killed(limit: float, out: bytes, err: bytes) -> Ran:
    """A run killed at its limit, with what its program had written."""
    return Ran(Killed(Kill(Limit(limit))), Captured(out), Captured(err))
