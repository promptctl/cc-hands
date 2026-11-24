"""Every child process hands starts: held to its caller's limit with its whole process group, reaped by hands, and told
to its caller as one closed outcome and to the audit log as one `child.run` event.

[LAW:single-enforcer] the one place hands starts, signals and reaps a child it waits on (tests/test_child.py holds
every other start to it, and each start outside it names its owner). Each child is the leader of a session of its own,
so its process group is everything it starts: a git's filters, a lit's helpers, a Claude Code's MCP servers. A daemon
thread of its own supervises it: a kqueue `NOTE_EXIT` says the leader has exited without reaping it, so while that
thread kills what is left of the group and reaps, the unreaped leader keeps its pid, and so the group's pgid, from
being given to another process.

Not asyncio's subprocesses, whose Python 3.12 transport cannot be cancelled while it starts: its exit is queued behind a
task of its own that the loop's shutdown cancels alongside, and the shutdown then waits for ever on an exit nothing will
deliver. A thread blocks on the exit instead: no task for the shutdown to cancel, and the loop never blocks.

A hands that is SIGKILLed or crashes leaves its children running; every other way it ends, its waits end, and a wait
that ends early kills its child.
"""

import asyncio
import math
import os
import queue
import select
import signal
# [LAW:single-enforcer] exception: the runner's own start, the one every other child of hands goes through.
import subprocess
import tempfile
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, Literal, Protocol

from hands.clock import Clock, wait
from hands.sessions import wide
from hands.threads import tell

# How long a child killed by hands has to be reaped before its caller's wait ends without it: a child in uninterruptible
# I/O, on a dead network mount or a stalled keychain, dies only once that I/O returns.
KILL_GRACE_SECONDS = 2.0
# How much of what a child writes to each of its outputs is read; a child that writes more to its stdout is Lost.
CAPTURE_BYTES = 16 * 1024 * 1024
# How long what a child showed on its terminal is read for, once it has been reaped, before a terminal another process
# still holds is given up on.
DRAIN_SECONDS = 1.0


@dataclass(frozen=True)
class Command:
    """What to run, where, and in what environment, which the OS can be handed: [LAW:parse-dont-validate] checked as it
    is made, so `Popen` raises nothing but an OSError. A NUL in an argument, the directory or the environment, which
    Popen refuses with ValueError, and what no file name can encode, which it refuses with UnicodeEncodeError, are
    refused here, in the caller, before any child is started."""

    argv: tuple[str, ...]
    cwd: Path | None
    env: Mapping[str, str] | None

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("a command needs a program to run")
        parts = (*self.argv, *(() if self.cwd is None else (str(self.cwd),)), *(part for pair in (self.env or {}).items() for part in pair))
        if any("\0" in part for part in parts):
            raise ValueError(f"{self.program} cannot be handed a NUL in its arguments, its directory or its environment")
        try:
            for part in parts:
                os.fsencode(part)
        except UnicodeEncodeError as error:
            raise ValueError(f"{self.program} cannot be handed {error.object[error.start:error.end]!r}, which no file name can encode") from None
        unnameable = [name for name in self.env or {} if not name or "=" in name]
        if unnameable:
            raise ValueError(f"{self.program} cannot be handed an environment variable named {unnameable[0]!r}")

    @property
    def program(self) -> str:
        return self.argv[0]


@dataclass(frozen=True)
class Limit:
    """Why a child was killed: it ran past the `seconds` its caller gave it, a finite time after it started."""

    seconds: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.seconds) and self.seconds > 0):
            raise ValueError(f"a child's limit is a finite time after it starts, not {self.seconds!r}")


@dataclass(frozen=True)
class Stop:
    """hands asking a child alone to end, with SIGTERM, and giving it `seconds` to before its group is killed; and, once
    they are up, why it was."""

    seconds: float


@dataclass(frozen=True)
class Cancelled:
    """Why a child was killed: its wait was cancelled, interrupted from the keyboard, or its coroutine closed."""


@dataclass(frozen=True)
class WaitFailed:
    """Why a child was killed: its wait failed of itself, its clock most often, or hands could no longer watch it."""

    error: str


type Why = Limit | Stop | Cancelled | WaitFailed


@dataclass(frozen=True)
class Kill:
    """[LAW:single-enforcer] a SIGKILL hands sent a child's whole group, and why."""

    why: Why


# What hands sends a child: a kill of its group, or a stop asked of the child alone, which may end what it started its
# own way.
type Sent = Kill | Stop


@dataclass(frozen=True)
class Exited:
    code: int


@dataclass(frozen=True)
class Signalled:
    """Ended by a signal hands did not send it."""

    signal: signal.Signals


@dataclass(frozen=True)
class Killed:
    """Ended by a kill hands sent its group, which says why."""

    kill: Kill


@dataclass(frozen=True)
class Stopped:
    """Ended by the stop hands asked of it, within its grace."""

    stop: Stop


type Ending = Exited | Signalled | Killed | Stopped


def ending(returncode: int, sent: tuple[Sent, ...]) -> Ending:
    """How a child reaped with `returncode`, as `Popen` decodes the wait status, ended, given what hands sent it while it
    was unreaped: ended by the first of those whose signal it died of, else exited or signalled on its own. A child
    that exits 0 while its kill is on its way has exited."""
    if returncode >= 0:
        return Exited(returncode)
    died = signal.Signals(-returncode)
    for each in sent:
        match each:
            case Kill() if died == signal.SIGKILL:
                return Killed(each)
            case Stop() if died == signal.SIGTERM:
                return Stopped(each)
            case Kill() | Stop():
                pass
    return Signalled(died)


@dataclass(frozen=True)
class Captured:
    """What a child wrote to one of its outputs, up to CAPTURE_BYTES."""

    data: bytes

    def text(self) -> str:
        """What it wrote, as text, where it is not text replaced as `errors="replace"` replaces it."""
        return self.data.decode(errors="replace")


@dataclass(frozen=True)
class Ran:
    """A child that ran, how it ended, and what it wrote: its whole stdout, and its stderr up to CAPTURE_BYTES."""

    ending: Ending
    out: Captured
    err: Captured


@dataclass(frozen=True)
class NeverStarted:
    """The OS gave hands no process, so nothing ran: its errno, text and filenames are the OS's own."""

    error: OSError


@dataclass(frozen=True)
class Unreaped:
    """A child hands killed that had not been reaped KILL_GRACE_SECONDS later. Its caller's wait ends here; its own
    thread reaps it once it dies, and its event says how it ended."""

    sent: tuple[Sent, ...]
    pid: int


@dataclass(frozen=True)
class Lost:
    """A child hands started and reaped, whose answer hands could not give its caller: hands' own following of it
    failed, or it wrote more to its stdout than CAPTURE_BYTES. Its event says how it ended, and why it was lost."""

    why: str


type Outcome = Ran | NeverStarted | Unreaped | Lost


def why_said(why: Why) -> str:
    match why:
        case Limit(seconds=seconds):
            return f"it ran past {seconds:.1f}s"
        case Stop(seconds=seconds):
            return f"it was told to stop, with {seconds:.1f}s to end"
        case Cancelled():
            return "its caller stopped waiting"
        case WaitFailed(error=error):
            return f"its wait failed: {error}"


def sent_said(sent: Sent) -> str:
    match sent:
        case Kill(why=why):
            return f"SIGKILL, because {why_said(why)}"
        case Stop(seconds=seconds):
            return f"SIGTERM, to stop within {seconds:.1f}s"


def said(outcome: Outcome | Ending) -> str:
    """[LAW:one-source-of-truth] how a child ended, in the one wording its event and its callers' errors use, said of
    the child as its program's name would begin it: a child that exited nonzero as `wide.exited` says a command's exit,
    with what it wrote to its stderr."""
    match outcome:
        case Ran(ending=Exited(code=code), err=err):
            return wide.exited(code, err.text().strip()[:300])
        case Ran(ending=ended):
            return said(ended)
        case Exited(code=code):
            return f"exited {code}"
        case Signalled(signal=died):
            return f"was ended by {died.name}"
        case Killed(kill=kill):
            return f"was killed with {sent_said(kill)}"
        case Stopped(stop=Stop(seconds=seconds)):
            return f"stopped as it was told to, within {seconds:.1f}s"
        case NeverStarted(error=error):
            return f"could not be started: {error.strerror or error}"
        case Unreaped(pid=pid):
            return f"was killed, and still not reaped {KILL_GRACE_SECONDS:.1f}s later, as {pid}"
        case Lost(why=why):
            return f"ran, and hands lost its answer: {why}"


def facts(ended: Ending) -> dict[str, wide.Fact]:
    """[LAW:one-source-of-truth] the facts a child's event carries of how it ended, as values a query groups by: one
    `ending` naming the arm, and the arm's own, `exit_code` in the name a hands command's own exit is written under."""
    match ended:
        case Exited(code=code):
            return {"ending": "exited", "exit_code": code}
        case Signalled(signal=died):
            return {"ending": "signalled", "signal": died.name}
        case Killed(kill=Kill(why=WaitFailed(error=error))):
            return {"ending": "killed", "signal": signal.SIGKILL.name, "why": "wait_failed", "why_error": error}
        case Killed(kill=Kill(why=why)):
            return {"ending": "killed", "signal": signal.SIGKILL.name, "why": _kind(why)}
        case Stopped():
            return {"ending": "stopped", "signal": signal.SIGTERM.name}


def _kind(why: Why) -> str:
    match why:
        case Limit():
            return "limit"
        case Stop():
            return "stop"
        case Cancelled():
            return "cancelled"
        case WaitFailed():
            return "wait_failed"


def _token(sent: Sent) -> str:
    """What hands sent a child, as its event lists it: the signal, and why."""
    match sent:
        case Kill(why=why):
            return f"{signal.SIGKILL.name}:{_kind(why)}"
        case Stop():
            return f"{signal.SIGTERM.name}:stop"


def _how(over: Ending | NeverStarted, lost: Lost | None = None) -> tuple[BaseException | None, str | None]:
    """[LAW:single-enforcer] how every child's event says its run ended, as `wide.ended` reads it: cancelled where its
    caller stopped waiting; failed where it never started, where it hung, so hands killed it, or where a signal nobody
    in hands sent ended it; and otherwise failed only where its answer was `lost`. An exit of any code is the child's
    answer, which its caller reads, and a child that stopped when asked to has done as it was told."""
    match over:
        case Killed(kill=Kill(why=Cancelled())):
            return asyncio.CancelledError(), None
        case Killed() | Signalled() | NeverStarted():
            return None, said(over)
        case Exited() | Stopped():
            return None, (None if lost is None else said(lost))


@dataclass(frozen=True)
class Null:
    """A child given nothing to read and nowhere to write."""


@dataclass(frozen=True)
class Terminal:
    """A child run on the slave side of a pseudo-terminal whose master hands reads: `closed` is put to once that read
    has reached its end, which comes once no process holds the slave open."""

    slave: int
    closed: queue.SimpleQueue[None]


type Stdio = Null | Terminal


class _Process(Protocol):
    """The part of a `Popen` the supervisor uses: the leader's pid, and the wait that reaps it and sets its returncode."""

    @property
    def pid(self) -> int: ...

    def wait(self) -> int: ...


class _Supervised:
    """A started child, what hands sent it, and its reap: [LAW:no-ambient-temporal-coupling] a signal is sent only under
    the lock, and only until the reap has killed what is left of the group, while the leader is still unreaped, so no
    signal can reach a pid already given away. No path signals through Popen, whose `send_signal` polls and so reaps
    on the caller's thread."""

    def __init__(self, process: _Process) -> None:
        self.process = process
        self.pid = process.pid
        # Reentrant: a coroutine closed on this child's own thread, as the collector may close one, sends its kill
        # from inside the lock's hold.
        self._lock = threading.RLock()
        self._sent: list[Sent] = []
        self._phase: Literal["live", "reaping", "reaped"] = "live"

    def signal(self, sent: Sent) -> None:
        """Send `sent`, unless the reap has already killed what is left of the group."""
        with self._lock:
            if self._phase != "live":
                return
            match sent:
                case Kill():
                    try:
                        os.killpg(self.pid, signal.SIGKILL)
                    except PermissionError:
                        # [LAW:no-silent-failure] exception: macOS refuses to signal a group whose every process has
                        # exited and is not yet reaped, and so would refuse a group holding a live process of another
                        # user's, which no child hands starts can make. Nothing was sent, so nothing is recorded.
                        return
                case Stop():
                    os.kill(self.pid, signal.SIGTERM)
            self._sent.append(sent)

    def sent(self) -> tuple[Sent, ...]:
        with self._lock:
            return tuple(self._sent)

    @property
    def reaped(self) -> bool:
        with self._lock:
            return self._phase == "reaped"

    def reap(self) -> tuple[int, tuple[Sent, ...], bool]:
        """Once the leader has exited, or been killed: kill what is left of its group, send it nothing from then on, and
        reap it; then say its returncode, what hands sent it, and whether anything was left in the group to kill."""
        with self._lock:
            try:
                os.killpg(self.pid, signal.SIGKILL)
                left = True
            except PermissionError:
                # [LAW:no-silent-failure] exception: macOS's answer for a group whose every process, the leader with
                # them, had exited: nothing was left to kill.
                left = False
            self._phase = "reaping"
            sent = tuple(self._sent)
        # [LAW:single-enforcer] reaped through Popen, so its returncode is set: Popen's own clean-up of children it
        # thinks unreaped would otherwise wait on this pid once it belongs to another child of hands'. Outside the lock,
        # which no signal needs any longer: a leader killed in uninterruptible I/O is reaped once it dies, and its
        # caller's wait goes on meanwhile.
        returncode = self.process.wait()
        with self._lock:
            self._phase = "reaped"
        return returncode, sent, left


@dataclass(frozen=True)
class _Wait:
    """A step of the policy: wait up to `seconds` for the child to be over."""

    seconds: float


type _Step = _Wait | Sent
# Each step is answered with whether the child is over yet, and the policy returns whether it ever was.
type _Policy = Generator[_Step, bool, bool]


def _bounded(why: Limit | Stop) -> _Policy:
    """[LAW:single-enforcer] the one policy every wait on a child follows, however it waits: up to the time `why` gives;
    past it, killed with its group for `why`; then up to KILL_GRACE_SECONDS for the reap. A wait that ends early,
    cancelled or failed, kills the child the same way and waits the same grace before what ended it goes on. A policy
    closed unfinished cannot wait: its driver kills the child as it is closed."""
    try:
        if (yield _Wait(why.seconds)):
            return True
        yield Kill(why)
        return (yield _Wait(KILL_GRACE_SECONDS))
    except GeneratorExit:
        raise
    except BaseException as stopped:
        yield Kill(WaitFailed(repr(stopped)) if isinstance(stopped, Exception) else Cancelled())
        yield _Wait(KILL_GRACE_SECONDS)
        raise


def _stopping(stop: Stop) -> _Policy:
    """Asked to end alone, then the bounded policy with the stop's grace as its limit."""
    yield stop
    return (yield from _bounded(stop))


async def _awaited[T](policy: _Policy, child: _Supervised, ended: asyncio.Future[T], clock: Clock) -> T | Unreaped:
    """Run `policy` on the loop, waiting on `ended`, which the child's thread settles once it has followed it to its end:
    what it settled once the child is over, or what was sent a child never reaped."""
    step = next(policy)
    try:
        while True:
            match step:
                case Kill() | Stop():
                    child.signal(step)
                case _Wait(seconds=seconds):
                    try:
                        await wait(clock, (ended,), seconds)
                    except GeneratorExit:
                        # A coroutine being closed cannot wait: its child is killed with its group, and its thread reaps it.
                        child.signal(Kill(Cancelled()))
                        raise
                    except BaseException as stopped:
                        step = policy.throw(stopped)
                        continue
            # [LAW:one-source-of-truth] over once reaped, whatever its thread has still to read; or once its thread,
            # failing, has settled `ended` without a reap.
            step = policy.send(child.reaped or ended.done())
    except StopIteration as returned:
        return await asyncio.shield(ended) if returned.value else Unreaped(child.sent(), child.pid)


def _waited[T](policy: _Policy, child: _Supervised, ended: queue.SimpleQueue[T], clock: Clock) -> T | Unreaped:
    """`_awaited`, on this thread, waiting on `ended`, which the child's thread puts to once it has followed it to its end."""
    taken: list[T] = []
    step = next(policy)
    try:
        while True:
            match step:
                case Kill() | Stop():
                    child.signal(step)
                case _Wait(seconds=seconds):
                    try:
                        taken.append(clock.take(ended, seconds))
                    except queue.Empty:
                        pass
                    except BaseException as stopped:
                        step = policy.throw(stopped)
                        continue
            step = policy.send(child.reaped or bool(taken))
    except StopIteration as returned:
        return (taken[0] if taken else ended.get()) if returned.value else Unreaped(child.sent(), child.pid)


@dataclass(frozen=True)
class Running:
    """A child started on a terminal or with no stdio, for as long as it runs."""

    pid: int
    _ended: asyncio.Future[Ending | Lost]
    _child: _Supervised
    _clock: Clock

    @property
    def ended(self) -> asyncio.Future[Ending | Lost]:
        """How it ended, once it is reaped and, on a terminal, what it showed last is read: [LAW:single-enforcer] a
        view its holder may cancel, which leaves the one its thread settles alone."""
        return asyncio.shield(self._ended)

    async def stopped(self, grace: float) -> Ending | Lost | Unreaped:
        """Ask the child alone to end, kill its group once it has not within `grace`, and say how it ended."""
        return await _awaited(_stopping(Stop(grace)), self._child, self._ended, self._clock)


# What the child's thread does once the child is reaped: what its caller is told, and the facts its event adds.
type _Finish[T] = Callable[[Ending], tuple[T | Lost, dict[str, wide.Fact]]]


class _Event(Protocol):
    """A child's `child.run` event, written as it ends, as `wide.ended` writes one."""

    def __call__(self, raised: BaseException | None, failure: str | None, /, **told: wide.Fact) -> None: ...


@dataclass(frozen=True)
class Traced:
    """An environment a child runs in as a span of hands' trace: `began`, the span its `child.run` event is written
    under, handed to it as W3C Trace Context's TRACEPARENT, so what the child records of its run joins hands' record of
    the same run. Its caller holds `began` to find that record by."""

    environment: Mapping[str, str]
    began: wide.Begun

    @property
    def handed(self) -> Mapping[str, str]:
        return {**self.environment, "TRACEPARENT": self.began.span.traceparent}


def _as(env: Mapping[str, str] | Traced | None, clock: Clock) -> tuple[wide.Begun, Mapping[str, str] | None]:
    """The span a child runs as, and the environment it is handed: a traced child's own, and any other's begun now."""
    match env:
        case Traced(began=began) as traced:
            return began, traced.handed
        case _:
            return wide.begun(clock), env


@dataclass(frozen=True)
class Children:
    """[LAW:effects-at-boundaries] what starts hands' children: a value made from its caller's one clock, which measures
    every child's limit, and the edge its caller's events leave through, which its children's events leave through too."""

    clock: Clock
    emit: Callable[[wide.WideEvent], None]

    async def run(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | Traced | None = None) -> Outcome:
        """Run `argv` to its end, within `limit`, from a coroutine: raises ValueError for a command no OS can be
        handed or a limit that is no time, and what ended the wait early, once it is killed."""
        bound = Limit(limit)
        loop = asyncio.get_running_loop()
        ended: asyncio.Future[Ran | Lost] = loop.create_future()
        began, handed = _as(env, self.clock)
        started = self._captured(began, Command(argv, cwd, handed), limit, lambda result: tell(loop, ended.set_result, result))
        if isinstance(started, NeverStarted):
            return started
        return await _awaited(_bounded(bound), started, ended, self.clock)

    def run_blocking(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | Traced | None = None) -> Outcome:
        """`run`, waited on by the calling thread, for code that runs off any event loop."""
        bound = Limit(limit)
        ended: queue.SimpleQueue[Ran | Lost] = queue.SimpleQueue()
        began, handed = _as(env, self.clock)
        started = self._captured(began, Command(argv, cwd, handed), limit, ended.put)
        if isinstance(started, NeverStarted):
            return started
        return _waited(_bounded(bound), started, ended, self.clock)

    def start(self, argv: Sequence[str], cwd: Path, env: Mapping[str, str], stdio: Stdio) -> Running | NeverStarted:
        """Start `argv` to run until it ends or is stopped, from a coroutine, with no limit of its own."""
        loop = asyncio.get_running_loop()
        ended: asyncio.Future[Ending | Lost] = loop.create_future()

        def finish(ended_as: Ending) -> tuple[Ending, dict[str, wide.Fact]]:
            return ended_as, self._after(stdio)

        started = self._spawned(Command(tuple(argv), cwd, env), _streams(stdio), wide.begun(self.clock), None, finish, lambda result: tell(loop, ended.set_result, result))
        return started if isinstance(started, NeverStarted) else Running(started.pid, ended, started, self.clock)

    def _captured(self, began: wide.Begun, parsed: Command, limit: float, settle: Callable[[Ran | Lost], None]) -> _Supervised | NeverStarted:
        """`parsed` started as `began` with its outputs to files hands reads once it is reaped, and `settle` given what it wrote."""
        with ExitStack() as unstarted:
            try:
                out = unstarted.enter_context(tempfile.TemporaryFile())
                err = unstarted.enter_context(tempfile.TemporaryFile())
            except OSError as error:
                return self._never(began, parsed, limit, error)

            def finish(ended: Ending) -> tuple[Ran | Lost, dict[str, wide.Fact]]:
                with out, err:
                    (wrote, out_cut), (complained, err_cut) = _read(out), _read(err)
                sizes: dict[str, wide.Fact] = {"out_bytes": len(wrote), "out_cut": out_cut, "err_bytes": len(complained), "err_cut": err_cut}
                if out_cut:
                    return Lost(f"it wrote more than the {CAPTURE_BYTES} bytes hands reads of its output"), sizes
                return Ran(ended, Captured(wrote), Captured(complained)), sizes

            started = self._spawned(parsed, (subprocess.DEVNULL, out, err), began, limit, finish, settle)
            if not isinstance(started, NeverStarted):
                # The files are the child's thread's from here, and it closes them once it has read them.
                unstarted.pop_all()
            return started

    def _spawned[T](self, parsed: Command, streams: tuple[int | IO[bytes], int | IO[bytes], int | IO[bytes]], began: wide.Begun, limit: float | None, finish: _Finish[T], settle: Callable[[T | Lost], None]) -> _Supervised | NeverStarted:
        """`parsed` started as the leader of a session of its own, and handed to a thread of its own that follows it to
        its end. [LAW:no-ambient-temporal-coupling] the thread is started first, so nothing that can fail comes between
        the start of a child and the hold its thread takes of it: the watch on its exit, and the thread, are made first."""
        handed: queue.SimpleQueue[_Supervised | None] = queue.SimpleQueue()
        try:
            watch = select.kqueue()
        except OSError as error:
            return self._never(began, parsed, limit, error)
        try:
            threading.Thread(target=_supervise, args=(watch, handed, finish, partial(self._event, began, parsed, limit), settle), name=f"a child: {parsed.program}", daemon=True).start()
        except BaseException:
            # No thread, so no child: the watch is closed, and hands' failure to make one goes on up.
            watch.close()
            raise
        child: _Supervised | None = None
        try:
            stdin, stdout, stderr = streams
            child = _Supervised(subprocess.Popen(parsed.argv, cwd=parsed.cwd, env=parsed.env, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=True))
            return child
        except OSError as error:
            return self._never(began, parsed, limit, error)
        finally:
            # On every path the thread is handed the child, or told there is none.
            handed.put(child)

    def _never(self, began: wide.Begun, parsed: Command, limit: float | None, error: OSError) -> NeverStarted:
        never = NeverStarted(error)
        self._event(began, parsed, limit, *_how(never), ending="never_started", errno=error.errno)
        return never

    def _after(self, stdio: Stdio) -> dict[str, wide.Fact]:
        """What a child's stdio says once it is reaped: whether what it showed on its terminal was read to its end."""
        match stdio:
            case Terminal(closed=closed):
                try:
                    self.clock.take(closed, DRAIN_SECONDS)
                except queue.Empty:
                    return {"drained": False}
                return {"drained": True}
            case Null():
                return {}

    def _event(self, began: wide.Begun, parsed: Command, limit: float | None, raised: BaseException | None, failure: str | None, /, **told: wide.Fact) -> None:
        """[LAW:nothing-unseen] the one `child.run` event each child writes, as it is reaped or as it fails to start."""
        wide.ended("child.run", self.emit, began, raised, failure, program=parsed.program, argv=parsed.argv, limit=limit, **told)


def _supervise[T](watch: select.kqueue, handed: queue.SimpleQueue[_Supervised | None], finish: _Finish[T], event: _Event, settle: Callable[[T | Lost], None]) -> None:
    """Follow the child handed to this thread to its end: reaped, its event written, and its caller settled."""
    with closing(watch):
        match handed.get():
            case None:
                # It never started: its caller wrote its event.
                return
            case _Supervised() as child:
                try:
                    _exited(watch, child.pid)
                except OSError as error:
                    # hands can no longer see it exit: killed with its group, as a wait that failed kills it, and
                    # reaped once it dies.
                    child.signal(Kill(WaitFailed(repr(error))))
    returncode, sent, left = child.reap()
    ended = ending(returncode, sent)
    told: T | Lost
    try:
        told, more = finish(ended)
    except Exception as error:
        # [LAW:no-silent-failure] hands' own reading of its answer failed: its caller is told so, as its event is.
        told, more = Lost(repr(error)), {}
    lost = told if isinstance(told, Lost) else None
    try:
        # [LAW:nothing-unseen] the loss is on the event however the child ended, beside what decides its outcome.
        event(*_how(ended, lost), pid=child.pid, kills=tuple(_token(each) for each in sent), leftovers=left, lost=None if lost is None else lost.why, **facts(ended), **more)
    finally:
        settle(told)


def _exited(watch: select.kqueue, pid: int) -> None:
    """Return once the leader `pid` has exited, without reaping it.

    macOS attaches no `NOTE_EXIT` to a process that has exited and is not yet reaped: it answers ESRCH, as for a process
    that is gone. hands alone reaps its children, so its child answering so is one that has exited already."""
    try:
        watch.control([select.kevent(pid, select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 0)
    except ProcessLookupError:
        return
    watch.control(None, 1)


def _streams(stdio: Stdio) -> tuple[int, int, int]:
    match stdio:
        case Terminal(slave=slave):
            return slave, slave, slave
        case Null():
            return subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL


def _read(output: IO[bytes]) -> tuple[bytes, bool]:
    """What a child wrote to `output`, up to CAPTURE_BYTES, and whether it wrote more."""
    output.seek(0)
    data = output.read(CAPTURE_BYTES + 1)
    return data[:CAPTURE_BYTES], len(data) > CAPTURE_BYTES
