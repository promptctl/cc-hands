"""Every child hands starts: run to its end inside its caller's limit, its whole process group ended with it, and none
left running once hands has ended, however it ended.

[LAW:single-enforcer] the one place hands starts a child. A start anywhere else names its owner on its line, as
`# [LAW:single-enforcer] exception: <owner and why>`, and tests/test_child.py fails on one that does not.
"""

import asyncio
import errno
import os
import queue
import select
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Literal, overload

from loguru import logger

from hands.clock import Clock, wait
from hands.sessions import wide
from hands.threads import settles_off_loop, tell


@dataclass(frozen=True)
class Exited:
    """The child exited, with `code`."""

    code: int


@dataclass(frozen=True)
class Signalled:
    """The child was ended by `signal`, sent by something other than hands."""

    signal: signal.Signals


type Status = Exited | Signalled


def status_of(returncode: int) -> Status:
    """The status of a child Popen reaped with `returncode`, which it gives as minus the signal that ended it."""
    return Exited(returncode) if returncode >= 0 else Signalled(signal.Signals(-returncode))


@dataclass(frozen=True)
class Ran:
    """The child ran to its own end: how it ended, everything it printed, and its errors as text."""

    status: Status
    out: bytes = field(repr=False)
    err: str = field(repr=False)


@dataclass(frozen=True)
class NeverStarted:
    """The OS made no process: the program, or the files its output goes to, could not be made. The OS's own error."""

    error: OSError


@dataclass(frozen=True)
class TimedOut:
    """The child ran past `limit` seconds, and hands killed it and everything in its group; what it printed until then."""

    limit: float
    out: bytes = field(repr=False)
    err: str = field(repr=False)


# [LAW:types-are-the-program] how a child hands ran ended, closed: a caller matches each arm, so "never ran", "ran past
# its limit" and "exited with N or by signal S" cannot be read as one another. A child hands killed for its caller is
# TimedOut or the caller's own interruption raised on, never a Signalled.
type Outcome = Ran | NeverStarted | TimedOut


def said(status: Status) -> str:
    """How a child ended by itself, after its program's name in a message."""
    match status:
        case Exited(code=code):
            return f"exited {code}"
        case Signalled(signal=ended):
            return f"was ended by {ended.name}"


def summary(outcome: Outcome) -> str:
    """[LAW:one-source-of-truth] the one wording of how a child ended, after its program's name in a message."""
    match outcome:
        case Ran(status=status):
            return said(status)
        case NeverStarted(error=error):
            return f"could not be started: {error.strerror or error}"
        case TimedOut(limit=limit):
            return f"ran past {limit:.1f}s"


@dataclass(frozen=True)
class Fields:
    """The facts of how a child ended, each a field of its own, for an event's consumer to query rather than read:
    `ending` says which way, and only that way's own fact is set."""

    ending: Literal["exited", "signalled", "never_started", "timed_out"]
    returncode: int | None = None
    signal: str | None = None
    limit: float | None = None
    # The OS's name for why it never started, ENOENT for a program that is not there; the event's error says it in words.
    errno: str | None = None


def fields(outcome: Outcome) -> Fields:
    """[LAW:one-source-of-truth] the one projection of an outcome onto an event's fields."""
    match outcome:
        case Ran(status=status):
            return _status_fields(status)
        case NeverStarted(error=error):
            return Fields("never_started", errno=None if error.errno is None else errno.errorcode.get(error.errno, str(error.errno)))
        case TimedOut(limit=limit):
            return Fields("timed_out", limit=limit)


def _status_fields(status: Status) -> Fields:
    match status:
        case Exited(code=code):
            return Fields("exited", returncode=code)
        case Signalled(signal=ended):
            return Fields("signalled", signal=ended.name)


class WardenGone(RuntimeError):
    """The warden, which ends hands' children once hands has ended, has ended first: a child started now could outlive hands."""


class Warden:
    """hands' end of the pipe to hands.sessions.warden, started on the first child hands records with it.

    [LAW:no-shared-mutable-globals] its one writer: every record goes through `record`, under its lock.
    """

    def __init__(self, audit: Path) -> None:
        self._audit = audit
        self._lock = threading.Lock()
        self._write: int | None = None
        self._gone: str | None = None

    def record(self, line: bytes) -> None:
        """Write `line`, `+<pgid>` or `-<pgid>`, to the warden, starting it first if it is not yet running."""
        with self._lock:
            if self._write is None:
                self._write = self._started()
            if self._gone is not None:
                # [LAW:no-silent-failure] never a child it cannot end once hands has ended.
                raise WardenGone(self._gone)
            try:
                os.write(self._write, line + b"\n")
            except BrokenPipeError:
                self._gone = "the warden stopped reading"
                raise WardenGone(self._gone) from None

    def _started(self) -> int:
        # Non-inheritable, as os.pipe makes both ends: no child, and no hands exec'd in this one's place, holds the write
        # end, so the kernel closes it when this hands ends, and the warden reads that as its end.
        read, write = os.pipe()
        try:
            # [LAW:single-enforcer] exception: the warden is the one child that outlives hands, the runner's own.
            warden = subprocess.Popen(
                [sys.executable, "-I", "-m", "hands.sessions.warden", str(self._audit), str(os.getpid())],
                stdin=read,
                stdout=subprocess.DEVNULL,
                # A session of its own: a terminal's SIGHUP or SIGINT to hands' process group does not reach it.
                start_new_session=True,
            )
        except OSError as error:
            os.close(write)
            # The child being recorded did start: its failure is the warden's, never a NeverStarted.
            raise WardenGone(f"the warden could not be started: {error}") from error
        finally:
            os.close(read)
        threading.Thread(target=self._reap, args=(warden,), name="the warden", daemon=True).start()
        return write

    def _reap(self, warden: subprocess.Popen[bytes]) -> None:
        code = warden.wait()
        with self._lock:
            self._gone = f"the warden ({warden.pid}) {said(status_of(code))} while hands ran"
        # [LAW:no-silent-failure] said at once, not only at the next child's start.
        logger.error(f"{self._gone}: no child hands starts from now can be ended with it")


@dataclass(frozen=True)
class Reaped:
    """A child its owner has reaped: how it ended, and whether anything of its process group was left to kill."""

    status: Status
    leftovers: bool


class Child:
    """A child and the one thread that owns it: starts it, records its process group with the warden, waits for its
    exit, kills what is left of its group, takes the record back, and reaps it, in that order.

    [LAW:no-ambient-temporal-coupling] the leader is reaped only once its group is killed and its record taken back: until
    then the unreaped leader keeps its pid, so no kill, hands' or the warden's, can reach a group of the same id made
    since. [LAW:single-enforcer] `kill` is the one kill of a child's group: on its limit, on its caller's interruption,
    and on its stop, before it starts as after.
    """

    def __init__(self, name: str, warden: Warden) -> None:
        self.name = name
        self._warden = warden
        self._lock = threading.Lock()
        self._pid: int | None = None
        self._killing = False
        self._reaping = False

    def kill(self) -> None:
        """Kill the child's process group now, or as soon as it is started; nothing once its owner is reaping it."""
        with self._lock:
            self._killing = True
            if self._pid is not None and not self._reaping:
                _killed(self._pid)

    def terminate(self) -> None:
        """Ask the child alone to end, with SIGTERM, while its owner has not reaped it and its pid is still its own."""
        with self._lock:
            if self._pid is not None and not self._reaping:
                os.kill(self._pid, signal.SIGTERM)

    def owned(self, popen: Callable[[], subprocess.Popen[bytes]], started: Callable[[int], None] = lambda pid: None) -> Reaped | NeverStarted:
        """Run the child `popen` starts to its end, on the calling thread, which owns it from here: `started` is told
        its pid once it is recorded. NeverStarted where `popen` made no process; what fails after it is raised."""
        # On the owner's thread, not the caller's: a signal's exception lands on the main thread alone, so none can come
        # between the start and the record, and leave a child the warden never heard of.
        try:
            process = popen()
        except OSError as error:
            return NeverStarted(error)
        with self._lock:
            self._pid = process.pid
            try:
                self._warden.record(b"+%d" % process.pid)
            except BaseException:
                # A child the warden never heard of would outlive hands: it is ended before the failure goes on up.
                _killed(process.pid)
                process.wait()
                raise
            if self._killing:
                _killed(process.pid)
        try:
            started(process.pid)
            _exited(process.pid)
        finally:
            with self._lock:
                self._reaping = True
                leftovers = _killed(process.pid)
            try:
                self._warden.record(b"-%d" % process.pid)
            finally:
                code = process.wait()
        return Reaped(status_of(code), leftovers)


def _exited(pid: int) -> None:
    """Once `pid`, a child of this process, has exited, without reaping it: kqueue's NOTE_EXIT, as macOS has no waitid."""
    kq = select.kqueue()
    try:
        [event] = kq.control([select.kevent(pid, select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 1, None)
    finally:
        kq.close()
    # One that had exited before it was watched comes back at once as an error, ESRCH: exited all the same.
    if event.flags & select.KQ_EV_ERROR and event.data != errno.ESRCH:
        raise OSError(event.data, os.strerror(event.data))


def _killed(group: int) -> bool:
    """SIGKILL the process group `group`; whether anything in it was still running."""
    try:
        os.killpg(group, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        # Gone, or nothing left in it but processes that have exited, which macOS refuses to signal.
        return False


@dataclass(frozen=True)
class _Owned:
    """What a child's owner settles a run with: the outcome, and whether its group had leftovers to kill."""

    outcome: Ran | NeverStarted
    leftovers: bool


# A wait the policy asks for: up to that many seconds, or None for until the owner has settled.
type _Wait = float | None
# What a wait gives back: what the owner settled with, raised or not, or None when the time was up first.
type _Settled = _Owned | BaseException | None


def _policy(child: Child, limit: float) -> Generator[_Wait, _Settled, tuple[Outcome, bool]]:
    """[LAW:single-enforcer] the one policy of a run's wait, written once for a coroutine and a blocking thread alike,
    which differ only in how they wait (a sans-I/O protocol): up to `limit`; past it, kill and wait for the reap; on any
    interruption of the wait, kill, wait for the reap, and raise the interruption on."""
    try:
        settled = yield limit
    except BaseException:
        child.kill()
        yield None
        raise
    timed_out = settled is None
    if timed_out:
        child.kill()
        settled = yield None
    match settled:
        case _Owned(outcome=Ran(out=out, err=err), leftovers=leftovers) if timed_out:
            return TimedOut(limit, out, err), leftovers
        case _Owned(outcome=outcome, leftovers=leftovers):
            return outcome, leftovers
        case BaseException():
            raise settled
        case None:
            raise AssertionError("an untimed wait ended with nothing settled")


class Children:
    """The runner of every child of one hands process: built once, where the process is put together.

    `audit` is the audit log's directory, which its warden writes its one event to once hands has ended.
    """

    def __init__(self, clock: Clock, emit: Callable[[wide.WideEvent], None], audit: Path) -> None:
        self.clock = clock
        self._emit = emit
        self._warden = Warden(audit)

    async def run(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        """Run `argv` to its end inside `limit` seconds; how it ended. Interrupted, its caller's cancellation or failure
        is raised on once the child and its group are killed and reaped. A second cancellation leaves at once, and the
        owner reaps the child all the same."""
        child = Child(argv[0], self._warden)
        began = wide.begun(self.clock)
        settles = settles_off_loop(lambda: _captured(child, argv, cwd, env), name=f"{argv[0]} (a child of hands)")
        policy = _policy(child, limit)
        try:
            wanted = next(policy)
            while True:
                try:
                    if wanted is None:
                        await asyncio.wait((settles,))
                    else:
                        await wait(self.clock, (settles,), wanted)
                except BaseException as stopped:
                    wanted = policy.throw(stopped)
                    continue
                wanted = policy.send(None if not settles.done() else settles.exception() or settles.result())
        except StopIteration as done:
            outcome, leftovers = done.value
            self._told(argv, began, outcome, leftovers)
            return outcome
        except BaseException as raised:
            wide.ended("child.run", self._emit, began, raised, None, program=argv[0])
            raise

    def run_blocking(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        """`run`, waited on by the calling thread rather than awaited: for work that blocks, off the loop."""
        child = Child(argv[0], self._warden)
        began = wide.begun(self.clock)
        settles: queue.SimpleQueue[_Owned | BaseException] = queue.SimpleQueue()
        threading.Thread(target=_settle_into, args=(settles, lambda: _captured(child, argv, cwd, env)), name=f"{argv[0]} (a child of hands)", daemon=True).start()
        policy = _policy(child, limit)
        try:
            wanted = next(policy)
            while True:
                try:
                    settled: _Settled = settles.get() if wanted is None else self.clock.take(settles, wanted)
                except queue.Empty:
                    settled = None
                except BaseException as stopped:
                    wanted = policy.throw(stopped)
                    continue
                wanted = policy.send(settled)
        except StopIteration as done:
            outcome, leftovers = done.value
            self._told(argv, began, outcome, leftovers)
            return outcome
        except BaseException as raised:
            wide.ended("child.run", self._emit, began, raised, None, program=argv[0])
            raise

    async def terminal(self, name: str, popen: Callable[[], subprocess.Popen[bytes]]) -> "Running":
        """The child `popen` starts, owned as every child is, with no limit: its end is its own, or its stop."""
        child = Child(name, self._warden)
        loop = asyncio.get_running_loop()
        started: asyncio.Future[int] = loop.create_future()
        ended = settles_off_loop(lambda: _status(child.owned(popen, lambda pid: tell(loop, started.set_result, pid))), name=f"{name} (a child of hands)")
        try:
            await asyncio.wait((started, ended), return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            child.kill()
            await asyncio.wait((ended,))
            raise
        if not started.done():
            # Ended before it was started: the OSError of a popen that made no process, or what the warden raised.
            ended.result()
        return Running(child, started.result(), ended)

    def _told(self, argv: Sequence[str], began: wide.Begun, outcome: Outcome, leftovers: bool) -> None:
        # [LAW:nothing-unseen] one event for each child, however it ended: a child that never ran or ran past its limit is a
        # failed run, said with how; one that exited, with any code, ran, and its fields say how it ended.
        failure = None if isinstance(outcome, Ran) else f"{argv[0]} {summary(outcome)}"
        wide.ended("child.run", self._emit, began, None, failure, program=argv[0], leftovers=leftovers, **asdict(fields(outcome)))


@dataclass(frozen=True)
class Running:
    """A child with no limit, running: its pid, and its status once its owner has reaped it."""

    child: Child
    pid: int
    ended: asyncio.Future[Status]

    async def stopped(self, clock: Clock, grace: float) -> None:
        """Ask the child alone to end, and kill everything it started once it has not within `grace` on `clock`."""
        self.child.terminate()
        try:
            await wait(clock, (self.ended,), grace)
        except BaseException:
            # Cancelled, or its clock failed and never measured the grace: the child is ended all the same.
            self.child.kill()
            await asyncio.wait((self.ended,))
            raise
        if not self.ended.done():
            logger.info(f"killed {self.child.name} ({self.pid}) and everything it started, because it had not ended {grace:.1f}s after it was asked to")
            self.child.kill()
            await asyncio.wait((self.ended,))


def _settle_into[T](settles: "queue.SimpleQueue[T | BaseException]", work: Callable[[], T]) -> None:
    try:
        settles.put(work())
    except BaseException as raised:
        settles.put(raised)


def _status(owned: Reaped | NeverStarted) -> Status:
    match owned:
        case Reaped(status=status):
            return status
        case NeverStarted(error=error):
            raise error


def _captured(child: Child, argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None) -> _Owned:
    """The owner's run of `argv`, its output going to files read once it is reaped; what fails once it ran is raised."""
    with ExitStack() as files:
        try:
            out = files.enter_context(tempfile.TemporaryFile())
            err = files.enter_context(tempfile.TemporaryFile())
        except OSError as error:
            # Nowhere for its output to go, so it was never started.
            return _Owned(NeverStarted(error), leftovers=False)
        match child.owned(lambda: _popen(argv, cwd, env, out, err)):
            case NeverStarted() as unstarted:
                return _Owned(unstarted, leftovers=False)
            case Reaped(status=status, leftovers=leftovers):
                out.seek(0)
                err.seek(0)
                return _Owned(Ran(status, out.read(), err.read().decode(errors="replace")), leftovers)


def _popen(argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None, out: IO[bytes], err: IO[bytes]) -> subprocess.Popen[bytes]:
    # A session of its own, so its process group is everything it starts: a git's filters, a lit's helpers.
    return subprocess.Popen(argv, cwd=cwd, env=None if env is None else dict(env), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)


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
