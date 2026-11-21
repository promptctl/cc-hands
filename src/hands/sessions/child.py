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
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from loguru import logger

from hands.clock import Clock, wait
from hands.sessions import warden, wide
from hands.threads import settles_off_loop, tell


@dataclass(frozen=True)
class Exited:
    """The child exited, with `code`."""

    code: int


@dataclass(frozen=True)
class Signalled:
    """The child was ended by `signal`."""

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
    """The OS made no process: the program, the files its output goes to, or the warden that would end it with hands
    could not be made. The OS's own error."""

    error: OSError


@dataclass(frozen=True)
class TimedOut:
    """The child ran past `limit` seconds, and hands killed it and everything in its group; what it printed until then."""

    limit: float
    out: bytes = field(repr=False)
    err: str = field(repr=False)


# [LAW:types-are-the-program] how a child hands ran ended, closed: a caller matches each arm, so "never ran", "ran past
# its limit" and "exited with N or by signal S" cannot be read as one another. A child hands killed for its caller is
# TimedOut or the caller's own interruption raised on, never a Ran.
type Outcome = Ran | NeverStarted | TimedOut


@dataclass(frozen=True)
class Reaped:
    """A child its supervisor has reaped: how it ended; whether hands killed it, at its limit, its caller's
    interruption, or its stop; and whether anything of its process group was left to kill once it had exited."""

    status: Status
    killed: bool
    leftovers: bool


def summary(ending: Outcome | Reaped | Status) -> str:
    """[LAW:one-source-of-truth] the one wording of how a child ended, after its program's name in a message."""
    match ending:
        case Exited(code=code):
            return f"exited {code}"
        case Signalled(signal=ended):
            return f"was ended by {ended.name}"
        case Reaped(killed=True):
            return "was killed by hands"
        case Reaped(status=status) | Ran(status=status):
            return summary(status)
        case NeverStarted(error=error):
            return f"could not be started: {error.strerror or error}"
        case TimedOut(limit=limit):
            return f"ran past {limit:.1f}s"


def fields(ending: Outcome | Reaped | Status) -> Mapping[str, wide.Fact]:
    """[LAW:one-source-of-truth] the one projection of how a child ended onto an event's facts: `ending` says which way,
    and beside it only that way's own facts, for an event's consumer to query rather than read."""
    match ending:
        case Exited(code=code):
            return {"ending": "exited", "returncode": code}
        case Signalled(signal=ended):
            return {"ending": "signalled", "signal": ended.name}
        case Reaped(killed=True):
            return {"ending": "killed"}
        case Reaped(status=status) | Ran(status=status):
            return fields(status)
        case NeverStarted(error=OSError(errno=int(number))):
            # The OS's name for why, ENOENT for a program that is not there; the event's error says it in words.
            return {"ending": "never_started", "errno": errno.errorcode.get(number, str(number))}
        case NeverStarted():
            return {"ending": "never_started"}
        case TimedOut(limit=limit):
            return {"ending": "timed_out", "limit": limit}


class WardenGone(RuntimeError):
    """No warden could be started again in place of one that ended: a child started now could outlive hands."""


@dataclass(frozen=True)
class _Running:
    """A warden that runs, and hands' end of its pipe."""

    process: subprocess.Popen[bytes]
    write: int


class Warden:
    """hands' end of the pipe to hands.sessions.warden: started before the first child hands records with it, and
    again, told every group still recorded, whenever the one before it has ended while hands ran.

    [LAW:no-shared-mutable-globals] its one writer: every record goes through `added` and `removed`, under its lock.
    [LAW:one-source-of-truth] the groups are hands' own; a warden holds a copy, written in full to each one started.
    """

    def __init__(self, audit: Path) -> None:
        self._audit = audit
        self._lock = threading.Lock()
        self._groups: set[int] = set()
        self._running: _Running | None = None

    def ready(self) -> None:
        """Start a warden, if none runs, before a child it would end is started; the OS's error where it cannot be."""
        with self._lock:
            if self._running is None:
                self._started()

    def added(self, group: int) -> None:
        """Record `group`, a child's process group that now exists."""
        with self._lock:
            self._groups.add(group)
            self._told(warden.added(group))

    def removed(self, group: int) -> None:
        """Take back `group`'s record, once what was left of it is killed."""
        with self._lock:
            self._groups.discard(group)
            self._told(warden.removed(group))

    def _told(self, line: bytes) -> None:
        try:
            if self._running is not None:
                try:
                    os.write(self._running.write, line)
                    return
                except BrokenPipeError:
                    self._lost(self._running, "stopped reading")
            # Told every group, this line's change among them.
            self._started()
        except OSError as error:
            # [LAW:no-silent-failure] never a child it cannot end once hands has ended.
            raise WardenGone(f"no warden could be started in place of the one that ended: {error}") from error

    def _started(self) -> None:
        # Non-inheritable, as os.pipe makes both ends: no child, and no hands exec'd in this one's place, holds the write
        # end, so the kernel closes it when this hands ends, and the warden reads that as its end.
        read, write = os.pipe()
        try:
            # [LAW:single-enforcer] exception: the warden is the one child that outlives hands, the runner's own.
            process = subprocess.Popen(
                [sys.executable, "-I", "-m", "hands.sessions.warden", str(self._audit), str(os.getpid())],
                stdin=read,
                stdout=subprocess.DEVNULL,
                # A session of its own: a terminal's SIGHUP or SIGINT to hands' process group does not reach it.
                start_new_session=True,
            )
        except OSError:
            os.close(write)
            raise
        finally:
            os.close(read)
        self._running = running = _Running(process, write)
        os.write(write, b"".join(map(warden.added, self._groups)))
        threading.Thread(target=self._reap, args=(running,), name="the warden", daemon=True).start()

    def _reap(self, running: _Running) -> None:
        code = running.process.wait()
        with self._lock:
            # One already found gone, by a write it stopped reading, is not lost twice.
            if self._running is running:
                self._lost(running, summary(status_of(code)))
                try:
                    # Started again at once, not at the next record: the groups recorded now are unwardened until then.
                    self._started()
                except OSError as error:
                    logger.error(f"no warden could be started in place of the one that ended: {error}; no child hands starts from now can be ended with it")

    def _lost(self, running: _Running, how: str) -> None:
        os.close(running.write)
        self._running = None
        # [LAW:no-silent-failure] said at once, however it is found.
        logger.error(f"the warden ({running.process.pid}) {how} while hands ran; one is started again, told every group still recorded")


@dataclass(frozen=True)
class _Unstarted:
    killed: bool


@dataclass(frozen=True)
class _Live:
    pid: int
    killed: bool


@dataclass(frozen=True)
class _Reaping:
    killed: bool


class Child:
    """A child and the one thread that supervises it: starts it, records its process group with the warden, waits for
    its exit, kills what is left of its group, takes the record back, and reaps it, in that order.

    [LAW:no-ambient-temporal-coupling] the leader is reaped only once its group is killed and its record taken back:
    until then the unreaped leader keeps its pid, so no kill, hands' or the warden's, can reach a group of the same id
    made since. [LAW:types-are-the-program] its lifecycle is one state: a child not yet started has no pid to signal, and
    one being reaped has none either. [LAW:single-enforcer] `kill` is the one kill of a child's group: on its limit, on
    its caller's interruption, and on its stop, before it starts as after.
    """

    def __init__(self, warden: Warden) -> None:
        self._warden = warden
        self._lock = threading.Lock()
        self._state: _Unstarted | _Live | _Reaping = _Unstarted(killed=False)

    def kill(self) -> None:
        """Kill the child's process group now, or as soon as it is started; nothing once its supervisor is reaping it."""
        with self._lock:
            match self._state:
                case _Unstarted():
                    self._state = _Unstarted(killed=True)
                case _Live(pid=pid):
                    self._state = _Live(pid, killed=True)
                    warden.killed(pid)
                case _Reaping():
                    pass

    def terminate(self) -> None:
        """Ask the child alone to end, with SIGTERM, while it is live and its pid is still its own."""
        with self._lock:
            match self._state:
                case _Live(pid=pid):
                    os.kill(pid, signal.SIGTERM)
                case _Unstarted() | _Reaping():
                    pass

    def supervise(self, popen: Callable[[], subprocess.Popen[bytes]], started: Callable[[int], None] = lambda pid: None) -> Reaped | NeverStarted:
        """Run the child `popen` starts to its end, on the calling thread, which supervises it from here: `started` is
        told its pid once it is recorded. NeverStarted where no process was made; what fails once one was is raised."""
        # Everything the watch needs is made before the child is, so a failure to make it is a child never started.
        try:
            self._warden.ready()
            watch = select.kqueue()
        except OSError as error:
            return NeverStarted(error)
        with closing(watch):
            # On the supervisor's thread, not the caller's: a signal's exception lands on the main thread alone, so none
            # can come between the start and the record, and leave a child the warden never heard of.
            try:
                process = popen()
            except OSError as error:
                return NeverStarted(error)
            with self._lock:
                killed = self._state.killed
                self._state = _Live(process.pid, killed)
                try:
                    self._warden.added(process.pid)
                except BaseException:
                    # [LAW:no-ambient-temporal-coupling] whatever raised, a child the warden never heard of would outlive
                    # hands: it is ended and reaped before the failure goes on up.
                    self._state = _Reaping(killed)
                    warden.killed(process.pid)
                    process.wait()
                    raise
                if killed:
                    warden.killed(process.pid)
            try:
                started(process.pid)
                _exited(watch, process.pid)
            finally:
                with self._lock:
                    killed = self._state.killed
                    self._state = _Reaping(killed)
                    leftovers = warden.killed(process.pid)
                try:
                    self._warden.removed(process.pid)
                finally:
                    code = process.wait()
        return Reaped(status_of(code), killed, leftovers)


def _exited(watch: select.kqueue, pid: int) -> None:
    """Once `pid`, a child of this process, has exited, without reaping it: kqueue's NOTE_EXIT, as macOS has no waitid."""
    [event] = watch.control([select.kevent(pid, select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 1, None)
    # One that had exited before it was watched comes back at once as an error, ESRCH: exited all the same.
    if event.flags & select.KQ_EV_ERROR and event.data != errno.ESRCH:
        raise OSError(event.data, os.strerror(event.data))


class _TimeUp:
    """The time a wait was given ran out before the child was reaped."""


_TIME_UP = _TimeUp()


def _policy[T](child: Child, limit: float) -> Generator[float | None, T | BaseException | _TimeUp, T]:
    """[LAW:single-enforcer] the one policy of a wait on a child, written once for a coroutine and a blocking thread,
    which differ only in how they wait: up to `limit`; past it, kill the child and wait for its reap; on any interruption
    of the wait, kill, wait for the reap, and raise the interruption on. What the supervisor raised is raised on.

    A sans-I/O protocol: it yields how long to wait, None for until the child is reaped, and is sent what the wait gave.
    """
    # A wait with no limit gives no _TimeUp, and each loop says so to the type, where one send would need an assertion.
    try:
        given = yield limit
    except BaseException:
        # [LAW:no-ambient-temporal-coupling] BaseException, not Exception: a cancellation or a KeyboardInterrupt that
        # went past this would leave the child running after its caller had stopped waiting for it.
        child.kill()
        while isinstance((yield None), _TimeUp):
            pass
        raise
    while isinstance(given, _TimeUp):
        child.kill()
        given = yield None
    if isinstance(given, BaseException):
        raise given
    return given


async def _awaited[T](clock: Clock, child: Child, limit: float, settles: "asyncio.Future[T]") -> T:
    """The policy, driven by a coroutine awaiting the supervisor's future."""
    policy: Generator[float | None, T | BaseException | _TimeUp, T] = _policy(child, limit)
    wanted = next(policy)
    while True:
        try:
            await (asyncio.wait((settles,)) if wanted is None else wait(clock, (settles,), wanted))
        except BaseException as stopped:
            wanted = policy.throw(stopped)
            continue
        try:
            wanted = policy.send(_given(settles))
        except StopIteration as done:
            return done.value


def _given[T](settles: "asyncio.Future[T]") -> T | BaseException | _TimeUp:
    """What a timed wait on `settles` gave: the supervisor's result, what it raised, or the time up first."""
    if not settles.done():
        return _TIME_UP
    raised = settles.exception()
    return settles.result() if raised is None else raised


def _blocked[T](clock: Clock, child: Child, limit: float, settles: "queue.SimpleQueue[T | BaseException]") -> T:
    """The policy, driven by a thread blocked on the supervisor's queue."""
    policy: Generator[float | None, T | BaseException | _TimeUp, T] = _policy(child, limit)
    wanted = next(policy)
    while True:
        try:
            given: T | BaseException | _TimeUp = settles.get() if wanted is None else clock.take(settles, wanted)
        except queue.Empty:
            given = _TIME_UP
        except BaseException as stopped:
            wanted = policy.throw(stopped)
            continue
        try:
            wanted = policy.send(given)
        except StopIteration as done:
            return done.value


@dataclass(frozen=True)
class _Captured:
    """A child reaped, and everything it printed while it ran."""

    reaped: Reaped
    out: bytes
    err: str


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
        supervisor reaps the child all the same."""
        child = Child(self._warden)
        began = wide.begun(self.clock)
        settles = settles_off_loop(lambda: _captured(child, argv, cwd, env), name=f"{argv[0]} (a child of hands)")
        try:
            captured = await _awaited(self.clock, child, limit, settles)
        except BaseException as raised:
            wide.ended("child.run", self._emit, began, raised, None, program=argv[0])
            raise
        return self._ran(argv[0], began, captured, limit)

    def run_blocking(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        """`run`, waited on by the calling thread rather than awaited: for work that blocks, off the loop."""
        child = Child(self._warden)
        began = wide.begun(self.clock)
        settles: queue.SimpleQueue[_Captured | NeverStarted | BaseException] = queue.SimpleQueue()
        # A daemon thread, as every supervisor is: a hands that ends leaves its children to the warden, never waits on them.
        threading.Thread(target=_settle_into, args=(settles, lambda: _captured(child, argv, cwd, env)), name=f"{argv[0]} (a child of hands)", daemon=True).start()
        try:
            captured = _blocked(self.clock, child, limit, settles)
        except BaseException as raised:
            wide.ended("child.run", self._emit, began, raised, None, program=argv[0])
            raise
        return self._ran(argv[0], began, captured, limit)

    async def terminal(self, name: str, popen: Callable[[], subprocess.Popen[bytes]]) -> "Running | NeverStarted":
        """The child `popen` starts, supervised as every child is, with no limit: its end is its own, or its stop."""
        child = Child(self._warden)
        began = wide.begun(self.clock)
        loop = asyncio.get_running_loop()
        started: asyncio.Future[int | NeverStarted] = loop.create_future()

        def supervised() -> Reaped | NeverStarted:
            ending = child.supervise(popen, lambda pid: tell(loop, started.set_result, pid))
            if isinstance(ending, NeverStarted):
                tell(loop, started.set_result, ending)
            self._told(name, began, ending, None if isinstance(ending, NeverStarted) else ending.leftovers)
            return ending

        ended = settles_off_loop(supervised, name=f"{name} (a child of hands)")
        try:
            await asyncio.wait((started, ended), return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            child.kill()
            await asyncio.wait((ended,))
            raise
        if not started.done():
            # Ended before its start was told: what the supervisor raised, raised here.
            ended.result()
        match started.result():
            case NeverStarted() as never:
                return never
            case int() as pid:
                return Running(child, pid, ended)

    def _ran(self, program: str, began: wide.Begun, captured: _Captured | NeverStarted, limit: float) -> Outcome:
        match captured:
            case NeverStarted():
                self._told(program, began, captured, None)
                return captured
            case _Captured(reaped=reaped, out=out, err=err):
                # Killed by hands, and no interruption raised on: the policy's kill at the limit.
                outcome = TimedOut(limit, out, err) if reaped.killed else Ran(reaped.status, out, err)
                self._told(program, began, outcome, reaped.leftovers)
                return outcome

    def _told(self, program: str, began: wide.Begun, ending: Outcome | Reaped, leftovers: bool | None) -> None:
        # [LAW:nothing-unseen] one event for each child, however it ended: a child that never ran, ran past its limit, or
        # was killed at its stop is a failed run, said with how; one that exited, with any code, ran, and its facts say how.
        # `leftovers` is None for a child that never ran, with no group to have left anything.
        ran = isinstance(ending, Ran) or (isinstance(ending, Reaped) and not ending.killed)
        wide.ended("child.run", self._emit, began, None, None if ran else f"{program} {summary(ending)}", program=program, leftovers=leftovers, **fields(ending))


@dataclass(frozen=True)
class Running:
    """A child with no limit, running: its pid, and how it ended once its supervisor has reaped it."""

    child: Child
    pid: int
    ended: "asyncio.Future[Reaped | NeverStarted]"

    async def stopped(self, clock: Clock, grace: float) -> None:
        """Ask the child alone to end, and kill everything it started once it has not within `grace` on `clock`: the
        one policy of a wait, with the grace for its limit."""
        self.child.terminate()
        await _awaited(clock, self.child, grace, self.ended)


def _settle_into[T](settles: "queue.SimpleQueue[T | BaseException]", work: Callable[[], T]) -> None:
    try:
        settles.put(work())
    except BaseException as raised:
        settles.put(raised)


def _captured(child: Child, argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None) -> _Captured | NeverStarted:
    """The supervisor's run of `argv`, its output going to files read once it is reaped; what fails once it ran is raised."""
    with ExitStack() as files:
        try:
            out = files.enter_context(tempfile.TemporaryFile())
            err = files.enter_context(tempfile.TemporaryFile())
        except OSError as error:
            # Nowhere for its output to go, so it was never started.
            return NeverStarted(error)
        match child.supervise(lambda: _popen(argv, cwd, env, out, err)):
            case NeverStarted() as unstarted:
                return unstarted
            case Reaped() as reaped:
                out.seek(0)
                err.seek(0)
                return _Captured(reaped, out.read(), err.read().decode(errors="replace"))


def _popen(argv: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None, out: IO[bytes], err: IO[bytes]) -> subprocess.Popen[bytes]:
    # A session of its own, so its process group is everything it starts: a git's filters, a lit's helpers. env None is
    # Popen's own: the child inherits hands' environment.
    return subprocess.Popen(argv, cwd=cwd, env=None if env is None else dict(env), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
