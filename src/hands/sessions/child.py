"""Every child process hands starts: started in a session of its own, ended within the time its caller gives, and
never left behind by a caller that stops waiting on it, nor by hands' exit.

[LAW:single-enforcer] the one place hands starts a child. A start that is not this module's names its owner on the line
it starts, and tests/test_child.py keeps it so. Not covered: a SIGKILL of hands, or the watchdog's faulthandler exit,
runs no Python, and macOS gives a child no signal at its parent's death, so a child outlives those two exits.
"""

import asyncio
import atexit
import errno
import os
import queue
import signal
import subprocess
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, assert_never, overload

from loguru import logger

from hands.clock import Clock, wait
from hands.threads import settles_off_loop


type _Stream = int | IO[bytes]


class _Live:
    """The process group of every child started here and not yet reaped.

    [LAW:no-shared-mutable-globals] owned by this class alone, behind one lock: a start and its record are one step, so
    the sweep at hands' exit misses no child; a kill and the check that its group is still unreaped are one step, so a
    kill never reaches a group whose number has gone back to the OS.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: set[int] = set()
        self._swept = False

    def start(self, argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None, stdin: _Stream, stdout: _Stream, stderr: _Stream) -> subprocess.Popen[bytes]:
        with self._lock:
            if self._swept:
                raise OSError(errno.ECANCELED, "hands is exiting", argv[0])
            # A session of its own, so its process group is everything it starts: a git's filters, a lit's helpers.
            process = subprocess.Popen(
                argv, cwd=cwd, env=None if env is None else dict(env), stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=True
            )
            self._groups.add(process.pid)
        return process

    def reaped(self, pid: int) -> None:
        with self._lock:
            self._groups.discard(pid)

    def kill(self, pid: int) -> bool:
        """Whether the group was there to kill: one reaped already, or all of whose members had exited, was not."""
        with self._lock:
            if pid not in self._groups:
                return False
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Every process in the group had exited: gone, ProcessLookupError; exited and not yet reaped, macOS says
                # PermissionError.
                return False
            return True

    def sweep(self) -> None:
        """Kill every group still running, and start nothing after: hands is exiting, and no child outlives it."""
        with self._lock:
            self._swept = True
            for pid in self._groups:
                try:
                    os.killpg(pid, signal.SIGKILL)
                    logger.info(f"killed the child {pid} and everything it started, because hands is exiting")
                except (ProcessLookupError, PermissionError):
                    pass

    def forgotten(self) -> None:
        """In a fork's child: the groups are its parent's to end, and its lock may have been held at the fork."""
        self._lock = threading.Lock()
        self._groups = set()


_LIVE = _Live()
atexit.register(_LIVE.sweep)
os.register_at_fork(after_in_child=_LIVE.forgotten)


def sweep() -> None:
    """Kill every child still running: for an exit that skips atexit, an exec in place of hands."""
    _LIVE.sweep()


def start(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], stdin: int, stdout: int, stderr: int) -> subprocess.Popen[bytes]:
    """`argv` started for a caller that owns it to its end through `reaped`; raises the OSError of a start that failed."""
    return _LIVE.start(argv, cwd, env, stdin, stdout, stderr)


@dataclass(frozen=True)
class Child[T]:
    """A child started in a session of its own, and the outcome a daemon thread of its own settles once it has reaped it.

    [LAW:single-enforcer] the one place hands ends a child it started: stopped, killed, or its stopper cancelled, the
    child is reaped before the call returns or raises, so the loop never ends with a subprocess of hands' still running.

    Not asyncio's subprocesses, whose Python 3.12 transport cannot be cancelled while it starts: its exit is queued
    behind a task of its own that the loop's shutdown cancels alongside, and the shutdown then waits for ever on an exit
    nothing will deliver. A thread blocks on the exit instead: no task for the shutdown to cancel, no executor for the
    exit to join, and the loop never blocks.
    """

    name: str
    process: subprocess.Popen[bytes]
    ended: asyncio.Future[T]

    async def stopped(self, clock: Clock, grace: float) -> None:
        """Ask the child alone to end, and kill everything it started once it has not within `grace` on `clock`."""
        self.process.terminate()
        try:
            await wait(clock, (self.ended,), grace)
        except (asyncio.CancelledError, Exception) as stopped:
            # Cancelled, or its clock failed and never measured the grace: the child is ended all the same.
            await self.killed("its stopper stopped waiting" if isinstance(stopped, asyncio.CancelledError) else f"its wait failed: {stopped!r}")
            raise
        if not self.ended.done():
            await self.killed(f"it had not ended {grace:.1f}s after it was asked to")

    async def killed(self, why: str) -> None:
        """Kill the child's process group, and wait while its thread reaps it."""
        _killed(self.name, self.process.pid, why)
        # Reaped by its own thread, which the kill releases; the loop stays live while a child the kill cannot end at
        # once, one in uninterruptible I/O, finishes dying. A caller told twice to stop leaves without waiting, and
        # the thread reaps it all the same.
        await asyncio.wait((self.ended,))


def reaped[T](name: str, process: subprocess.Popen[bytes], outcome: Callable[[], T]) -> Child[T]:
    """`process`, with `outcome` run by a thread of its own, which must reap it, to settle `ended`."""
    return Child(name, process, _reaping(process, lambda reap: settles_off_loop(reap, name=f"{name} {process.pid}"), outcome))


def _reaping[T, R](process: subprocess.Popen[bytes], on_thread: Callable[[Callable[[], T]], R], outcome: Callable[[], T]) -> R:
    """`outcome` handed to `on_thread` to run, its group let go of once it has reaped; a thread that never starts leaves
    no child: the child is killed and reaped before the error leaves."""

    def reap() -> T:
        try:
            return outcome()
        finally:
            _LIVE.reaped(process.pid)

    try:
        return on_thread(reap)
    except BaseException:
        _LIVE.kill(process.pid)
        process.wait()
        _LIVE.reaped(process.pid)
        raise


def _killed(name: str, pid: int, why: str) -> None:
    if _LIVE.kill(pid):
        logger.info(f"killed {name} ({pid}) and everything it started, because {why}")
    else:
        logger.info(f"{name} ({pid}) had already ended when {why}")


@dataclass(frozen=True)
class Exited:
    """The child ran to its exit: its code, negative for the signal that killed it, and what it wrote, as text."""

    returncode: int
    # What it wrote may be a secret, a keychain's token: never in a repr, so never in a log line or an event.
    out: str = field(repr=False)
    err: str = field(repr=False)


@dataclass(frozen=True)
class NeverStarted:
    """The OS handed hands no process: the program, or the files its output goes to, could not be made; or hands is
    exiting. Its errno, text and filenames are the OS's own."""

    error: OSError


@dataclass(frozen=True)
class TimedOut:
    """The child ran past `limit`, and it and everything it started were killed and reaped; what it wrote till then."""

    limit: float
    out: str = field(repr=False)
    err: str = field(repr=False)


# [LAW:types-are-the-program] every way a child's run ends, closed: a caller matches on it, and sorts no exceptions.
type Ended = Exited | NeverStarted | TimedOut


def told(ended: Ended) -> str:
    """How the run ended, said without anything it wrote."""
    match ended:
        case Exited(returncode=code) if code < 0:
            return f"was killed by signal {-code}"
        case Exited(returncode=code):
            return f"exited {code}"
        case NeverStarted(error=error):
            return f"never started: {error}"
        case TimedOut(limit=limit):
            return f"ran past {limit:.1f}s"
        case _:
            assert_never(ended)


@dataclass(frozen=True)
class _Wrote:
    returncode: int
    out: str
    err: str

    def exited(self) -> Exited:
        return Exited(self.returncode, self.out, self.err)

    def past(self, limit: float) -> TimedOut:
        return TimedOut(limit, self.out, self.err)


@dataclass(frozen=True)
class _Started:
    name: str
    process: subprocess.Popen[bytes]
    out: IO[bytes]
    err: IO[bytes]

    def read(self) -> _Wrote:
        """Reap the child, then what it wrote: the files are this, on the reaper's thread, to close."""
        with self.out, self.err:
            returncode = self.process.wait()
            self.out.seek(0)
            self.err.seek(0)
            return _Wrote(returncode, self.out.read().decode(errors="replace"), self.err.read().decode(errors="replace"))


def _started(argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None) -> _Started | NeverStarted:
    # The files are closed by the stack if no child is made, and once one is they are let go of, to its thread.
    with ExitStack() as unspawned:
        try:
            out = unspawned.enter_context(tempfile.TemporaryFile())
            err = unspawned.enter_context(tempfile.TemporaryFile())
            process = _LIVE.start(argv, cwd, env, subprocess.DEVNULL, out, err)
        except OSError as error:
            logger.debug(f"{argv[0]} could not be started: {error}")
            return NeverStarted(error)
        unspawned.pop_all()
    return _Started(argv[0], process, out, err)


def _said(started: _Started, ended: Ended, took: float) -> Ended:
    logger.debug(f"{started.name} ({started.process.pid}) {told(ended)} in {took:.3f}s")
    return ended


async def run(*argv: str, timeout: float, cwd: Path | None = None, env: Mapping[str, str] | None = None, clock: Clock) -> Ended:
    """Run `argv` to its exit within `timeout` on `clock`, from a coroutine.

    Timed out, its caller cancelled or its clock failed, the child and every process it started are killed, and the
    child reaped, before the call returns or raises. Its output goes to files, which the child's own thread reads once
    it has reaped it.
    """
    started = _started(argv, cwd, env)
    if isinstance(started, NeverStarted):
        return started
    child = reaped(started.name, started.process, started.read)
    began = clock.now()
    try:
        # Waited on, not awaited: what the child's thread raised is hands' own failure, raised below as itself.
        await wait(clock, (child.ended,), timeout)
    except (asyncio.CancelledError, Exception) as stopped:
        # Cancelled, or its clock failed: whatever ends the wait early, the child does not outlive it. A coroutine closed
        # or an interpreter exiting is not waited in.
        await child.killed("its caller stopped waiting" if isinstance(stopped, asyncio.CancelledError) else f"its wait failed: {stopped!r}")
        raise
    if not child.ended.done():
        await child.killed(f"it ran past {timeout:.1f}s")
        return _said(started, child.ended.result().past(timeout), clock.now() - began)
    return _said(started, child.ended.result().exited(), clock.now() - began)


def run_blocking(*argv: str, timeout: float, cwd: Path | None = None, env: Mapping[str, str] | None = None, clock: Clock) -> Ended:
    """Run `argv` to its exit within `timeout` on `clock`, from a thread that blocks on it, or a command with no loop.

    [LAW:effects-at-boundaries] for a caller whose other work blocks too, the brain binary's hash among it: it stays on
    its thread, and nothing of it moves onto a loop. Timed out, or the wait ended by anything, a KeyboardInterrupt
    among them, the child and every process it started are killed, and the child reaped, before the call returns or
    raises.
    """
    started = _started(argv, cwd, env)
    if isinstance(started, NeverStarted):
        return started
    # What the reaper's thread gave, carried as a call, so what it raised is raised here, on the caller's thread.
    settled: queue.SimpleQueue[Callable[[], _Wrote]] = queue.SimpleQueue()

    def on_thread(reap: Callable[[], _Wrote]) -> None:
        threading.Thread(target=lambda: settled.put(_carried(reap)), name=f"{started.name} {started.process.pid}", daemon=True).start()

    _reaping(started.process, on_thread, started.read)
    began = clock.now()
    ended: Callable[[_Wrote], Ended] = _Wrote.exited
    try:
        try:
            given = clock.take(settled, timeout)
        except queue.Empty:
            _killed(started.name, started.process.pid, f"it ran past {timeout:.1f}s")
            ended = lambda wrote: wrote.past(timeout)
            # Untimed: the kill releases the reap.
            given = settled.get()
    except BaseException as stopped:
        _killed(started.name, started.process.pid, f"its caller stopped waiting: {stopped!r}")
        settled.get()
        raise
    # Outside the wait: what the reaper raised is hands' own failure, and the child is reaped already.
    return _said(started, ended(given()), clock.now() - began)


def _carried[T](work: Callable[[], T]) -> Callable[[], T]:
    try:
        result = work()
    except BaseException as error:

        def raised(failure: BaseException = error) -> T:
            raise failure

        return raised
    return lambda: result


@overload
async def settled[A, B](first: Awaitable[A], second: Awaitable[B], /) -> tuple[A, B]: ...
@overload
async def settled[A, B, C](first: Awaitable[A], second: Awaitable[B], third: Awaitable[C], /) -> tuple[A, B, C]: ...
@overload
async def settled[T](*sides: Awaitable[T]) -> tuple[T, ...]: ...
async def settled(*sides: Awaitable[object]) -> tuple[object, ...]:
    """Every side of a reading run together and waited out, then what each gave; where any raised, all they raised.

    [LAW:single-enforcer] the one way a reading runs its sides at once, each bounded by the reading's deadline. Waited
    out, so a side that raises does not end the reading while another runs on and writes to its event after it has
    ended [LAW:no-ambient-temporal-coupling]. Not a TaskGroup, nor a gather that cancels the rest: a child told twice to
    stop leaves its git unreaped.
    """
    gave = await asyncio.gather(*sides, return_exceptions=True)
    raised = [side for side in gave if isinstance(side, BaseException)]
    # A side cancelled on its own, as the loop's shutdown cancels every task at once, is the reading cancelled: raised
    # as a cancellation, it is told as one and goes on up, where in a group it would be told as a failure.
    if any(isinstance(side, asyncio.CancelledError) for side in raised):
        raise asyncio.CancelledError
    if raised:
        raise BaseExceptionGroup("a side of the reading raised", raised)
    return tuple(gave)
