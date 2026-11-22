"""Every child process hands starts: held to its caller's limit with its whole process group, reaped by hands, and told
to its caller as one closed outcome and to the audit log as one `child.run` event.

[LAW:single-enforcer] the one place hands starts, signals and reaps a child (tests/test_child.py holds every other module
to it). Each child is the leader of a session of its own, so its process group is everything it starts: a git's
filters, a lit's helpers, a Claude Code's MCP servers. A daemon thread of its own supervises it: a kqueue `NOTE_EXIT`
says the leader has exited without reaping it, so while that thread kills what is left of the group and reaps, the
unreaped leader keeps its pid, and so the group's pgid, from being given to another process.

Not asyncio's subprocesses, whose Python 3.12 transport cannot be cancelled while it starts: its exit is queued behind a
task of its own that the loop's shutdown cancels alongside, and the shutdown then waits for ever on an exit nothing will
deliver. A thread blocks on the exit instead: no task for the shutdown to cancel, and the loop never blocks.

A hands that is SIGKILLed or crashes leaves its children running; every other way it ends, its waits end, and a wait
that ends early kills its child.
"""

import asyncio
import os
import queue
import select
import signal
# [LAW:single-enforcer] exception: the runner's own start, the one every other child of hands goes through.
import subprocess
import tempfile
import threading
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, overload

from hands.clock import Clock, wait
from hands.sessions import wide
from hands.threads import tell

# How long a child killed by hands has to be reaped before its caller's wait ends without it: a child in uninterruptible
# I/O, on a dead network mount or a stalled keychain, dies only once that I/O returns.
KILL_GRACE_SECONDS = 2.0
# How much of what a child writes to each of its outputs is kept; what it writes past this is cut, and said to be.
CAPTURE_BYTES = 16 * 1024 * 1024
# How long what a child showed on its terminal is read for, once it has been reaped, before a terminal another process
# still holds is given up on.
DRAIN_SECONDS = 1.0


@dataclass(frozen=True)
class Command:
    """What to run, where, and in what environment, parsed so the OS refuses nothing but the start itself.

    [LAW:parse-dont-validate] built only by `command`, so `Popen` raises nothing but an OSError: a NUL in an argument or
    the environment, which Popen refuses with ValueError, is refused here, in the caller, before any child is started.
    """

    argv: tuple[str, ...]
    cwd: Path | None
    env: Mapping[str, str] | None

    @property
    def program(self) -> str:
        return self.argv[0]


def command(argv: Sequence[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Command:
    """`argv` run in `cwd` with `env` (hands' own, where None); raises ValueError for what no OS can be handed."""
    if not argv:
        raise ValueError("a command needs a program to run")
    parts = (*argv, *(() if env is None else (part for pair in env.items() for part in pair)))
    if any("\0" in part for part in parts):
        raise ValueError(f"{argv[0]} cannot be handed a NUL in its arguments or environment")
    unnameable = [name for name in env or {} if not name or "=" in name]
    if unnameable:
        raise ValueError(f"{argv[0]} cannot be handed an environment variable named {unnameable[0]!r}")
    return Command(tuple(argv), cwd, None if env is None else dict(env))


@dataclass(frozen=True)
class Limit:
    """Why a child was killed: it ran past the time its caller gave it."""

    seconds: float


@dataclass(frozen=True)
class Stop:
    """Why a child was signalled: hands told it to stop, and gave it `grace` to end before it is killed."""

    grace: float


@dataclass(frozen=True)
class Cancelled:
    """A wait cancelled, or interrupted from the keyboard or by the coroutine waiting being closed."""


@dataclass(frozen=True)
class WaitFailed:
    """A wait that failed of itself, its clock most often, before its child had ended."""

    error: str


@dataclass(frozen=True)
class Interrupted:
    """Why a child was killed: whoever waited on it stopped waiting before it ended."""

    cause: Cancelled | WaitFailed


type Why = Limit | Stop | Interrupted


@dataclass(frozen=True)
class Kill:
    """A signal hands sent a child, and why. [LAW:single-enforcer] a SIGKILL goes to the child's whole group; any other
    signal asks the child alone, which may end what it started its own way."""

    signal: signal.Signals
    why: Why


@dataclass(frozen=True)
class Exited:
    code: int


@dataclass(frozen=True)
class Signalled:
    """Ended by a signal hands did not send it."""

    signal: signal.Signals


@dataclass(frozen=True)
class Killed:
    """Ended by a signal hands sent it, which says why."""

    kill: Kill


type Ending = Exited | Signalled | Killed


def ending(returncode: int, kills: Sequence[Kill]) -> Ending:
    """How a child reaped with `returncode`, as `Popen` decodes the wait status, ended, given the kills hands sent it
    while it was unreaped: killed by the first of them whose signal it died of, else exited or signalled on its own.
    A child that exits 0 while its kill is on its way has exited."""
    if returncode >= 0:
        return Exited(returncode)
    died = signal.Signals(-returncode)
    return next((Killed(kill) for kill in kills if kill.signal == died), Signalled(died))


@dataclass(frozen=True)
class Captured:
    """What a child wrote to one of its outputs, up to CAPTURE_BYTES, and whether it wrote more, which is not kept."""

    data: bytes
    cut: bool

    def text(self) -> str:
        """What it wrote, as text, where it is not text replaced as `errors="replace"` replaces it."""
        return self.data.decode(errors="replace")


@dataclass(frozen=True)
class Ran:
    """A child that ran, how it ended, and what it wrote."""

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

    kills: tuple[Kill, ...]
    pid: int


type Outcome = Ran | NeverStarted | Unreaped


def why_said(why: Why) -> str:
    match why:
        case Limit(seconds=seconds):
            return f"it ran past {seconds:.1f}s"
        case Stop(grace=grace):
            return f"it was told to stop, with {grace:.1f}s to end"
        case Interrupted(cause=cause):
            match cause:
                case Cancelled():
                    return "its caller stopped waiting"
                case WaitFailed(error=error):
                    return f"its wait failed: {error}"


def kill_said(kill: Kill) -> str:
    return f"{kill.signal.name}, because {why_said(kill.why)}"


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
            return f"was killed with {kill_said(kill)}"
        case NeverStarted(error=error):
            return f"could not be started: {error.strerror or error}"
        case Unreaped(pid=pid):
            return f"was killed, and still not reaped {KILL_GRACE_SECONDS:.1f}s later, as {pid}"


def facts(ended: Ending) -> dict[str, wide.Fact]:
    """[LAW:one-source-of-truth] the facts a child's event carries of how it ended: one `ending` naming the arm, and
    the arm's own, `exit_code` in the name a hands command's own exit is written under."""
    match ended:
        case Exited(code=code):
            return {"ending": "exited", "exit_code": code}
        case Signalled(signal=died):
            return {"ending": "signalled", "signal": died.name}
        case Killed(kill=kill):
            return {"ending": "killed", "signal": kill.signal.name, "why": why_said(kill.why)}


def _how(ended: Ending) -> tuple[BaseException | None, str | None]:
    """How a child's event says its run ended, as `wide.ended` reads it: cancelled where its caller stopped waiting, and
    failed where it hung, so hands killed it, or where a signal nobody in hands sent ended it. An exit of any code is
    the child's answer, which its caller reads, and a child that ended as a stop asked it to has done as it was told."""
    match ended:
        case Exited() | Killed(kill=Kill(signal=signal.SIGTERM, why=Stop())):
            return None, None
        case Killed(kill=Kill(why=Interrupted(cause=Cancelled()))):
            return asyncio.CancelledError(), None
        case Killed() | Signalled():
            return None, said(ended)


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
    """A started child, its kills, and its reap: [LAW:no-ambient-temporal-coupling] a kill is sent only under the
    lock the reap holds, and only while the leader is unreaped, so a kill can never reach a pid already given away.
    No path signals through Popen, whose `send_signal` polls and so reaps on the caller's thread."""

    def __init__(self, process: "_Process") -> None:
        self.process = process
        self.pid = process.pid
        # Reentrant: a coroutine closed on this child's own thread, as the collector may close one, sends its kill
        # from inside the reap's hold, and a kill after the reap sends nothing.
        self._lock = threading.RLock()
        self._kills: list[Kill] = []
        self._reaped = False

    def signal(self, kill: Kill) -> None:
        """Send `kill`, unless the leader is reaped already; a group whose every process has exited is sent nothing."""
        with self._lock:
            if self._reaped:
                return
            try:
                (os.killpg if kill.signal == signal.SIGKILL else os.kill)(self.pid, kill.signal)
            except PermissionError:
                # [LAW:no-silent-failure] exception: macOS refuses to signal a group whose every process has exited and is
                # not yet reaped, and so would refuse a group holding a live process of another user's, which no child
                # hands starts can make. Nothing was sent, so nothing is recorded to say how the child ended.
                return
            self._kills.append(kill)

    def kills(self) -> tuple[Kill, ...]:
        with self._lock:
            return tuple(self._kills)

    def reaped(self) -> tuple[int, tuple[Kill, ...], bool]:
        """Once the leader has exited: kill what is left of its group, reap it, and say its returncode, the kills sent it,
        and whether anything was left in the group to kill."""
        with self._lock:
            try:
                os.killpg(self.pid, signal.SIGKILL)
                left = True
            except PermissionError:
                # Every process in the group had exited, the leader with them: nothing was left to kill.
                left = False
            # [LAW:single-enforcer] reaped through Popen, so its returncode is set: Popen's own clean-up of children it
            # thinks unreaped would otherwise wait on this pid once it belongs to another child of hands'.
            returncode = self.process.wait()
            self._reaped = True
            return returncode, tuple(self._kills), left


@dataclass(frozen=True)
class _Wait:
    """A step of the policy: wait up to `seconds` for the child to be reaped."""

    seconds: float


type _Step = _Wait | Kill
# Each step is answered with whether the child is reaped yet, and the policy returns whether it ever was.
type _Policy = Generator[_Step, bool, bool]


def _bounded(limit: float, why: Why) -> _Policy:
    """[LAW:single-enforcer] the one policy every wait on a child follows, however it waits: up to `limit`; past it,
    killed with its group for `why`; then up to KILL_GRACE_SECONDS for the reap. A wait that ends early, cancelled or
    failed, kills the child the same way and waits the same grace before what ended it goes on."""
    try:
        if (yield _Wait(limit)):
            return True
        yield Kill(signal.SIGKILL, why)
        return (yield _Wait(KILL_GRACE_SECONDS))
    except BaseException as stopped:
        yield Kill(signal.SIGKILL, Interrupted(_cause(stopped)))
        yield _Wait(KILL_GRACE_SECONDS)
        raise


def _stopping(grace: float) -> _Policy:
    """Asked to end alone, then the bounded policy with the stop's grace as its limit."""
    yield Kill(signal.SIGTERM, Stop(grace))
    return (yield from _bounded(grace, Stop(grace)))


def _cause(stopped: BaseException) -> Cancelled | WaitFailed:
    return WaitFailed(repr(stopped)) if isinstance(stopped, Exception) else Cancelled()


async def _awaited[T](policy: _Policy, send: Callable[[Kill], None], ended: asyncio.Future[T], clock: Clock) -> T | None:
    """Run `policy` on the loop, waiting on `ended`, which the child's thread settles once it has reaped it: what it
    settled, or None for a child not reaped."""
    step = next(policy)
    try:
        while True:
            match step:
                case Kill() as kill:
                    send(kill)
                    step = policy.send(ended.done())
                case _Wait(seconds=seconds):
                    try:
                        await wait(clock, (ended,), seconds)
                    except BaseException as stopped:
                        step = policy.throw(stopped)
                    else:
                        step = policy.send(ended.done())
    except StopIteration as returned:
        reaped: bool = returned.value
        return ended.result() if reaped else None


def _waited[T](policy: _Policy, send: Callable[[Kill], None], ended: queue.SimpleQueue[T], clock: Clock) -> T | None:
    """`_awaited`, on this thread, waiting on `ended`, which the child's thread puts to once it has reaped it."""
    taken: list[T] = []
    step = next(policy)
    try:
        while True:
            match step:
                case Kill() as kill:
                    send(kill)
                    step = policy.send(bool(taken))
                case _Wait(seconds=seconds):
                    try:
                        taken.append(clock.take(ended, seconds))
                    except queue.Empty:
                        step = policy.send(False)
                    except BaseException as stopped:
                        step = policy.throw(stopped)
                    else:
                        step = policy.send(True)
    except StopIteration as returned:
        reaped: bool = returned.value
        return taken[0] if reaped else None


@dataclass(frozen=True)
class Running:
    """A child started on a terminal or with no stdio, for as long as it runs: `ended` settles once it is reaped and,
    on a terminal, what it showed last is read."""

    pid: int
    ended: asyncio.Future[Ending]
    _child: _Supervised
    _clock: Clock

    async def stopped(self, grace: float) -> Ending | Unreaped:
        """Ask the child alone to end, kill its group once it has not within `grace`, and say how it ended."""
        match await _awaited(_stopping(grace), self._child.signal, self.ended, self._clock):
            case None:
                return Unreaped(self._child.kills(), self.pid)
            case ended:
                return ended


@dataclass(frozen=True)
class Children:
    """[LAW:effects-at-boundaries] what starts hands' children: built at each composition root from its one clock,
    which measures every child's limit, and the edge its events leave through."""

    clock: Clock
    emit: Callable[[wide.WideEvent], None]

    async def run(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        """Run `argv` to its end, within `limit`, from a coroutine: raises ValueError for a command no OS can be
        handed, OSError for what it wrote that could not be read, and what ended the wait early, once it is killed."""
        loop = asyncio.get_running_loop()
        ended: asyncio.Future[Ran | OSError] = loop.create_future()
        started = self._captured(command(argv, cwd, env), limit, lambda result: tell(loop, _settle, ended, result))
        if isinstance(started, NeverStarted):
            return started
        return _told(await _awaited(_bounded(limit, Limit(limit)), started.signal, ended, self.clock), started)

    def run_blocking(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Outcome:
        """`run`, waited on by the calling thread, for code that runs off any event loop."""
        ended: queue.SimpleQueue[Ran | OSError] = queue.SimpleQueue()
        started = self._captured(command(argv, cwd, env), limit, ended.put)
        if isinstance(started, NeverStarted):
            return started
        return _told(_waited(_bounded(limit, Limit(limit)), started.signal, ended, self.clock), started)

    def start(self, argv: Sequence[str], cwd: Path, env: Mapping[str, str], stdio: Stdio) -> Running | NeverStarted:
        """Start `argv` to run until it ends or is stopped, from a coroutine, with no limit of its own."""
        loop = asyncio.get_running_loop()
        ended: asyncio.Future[Ending] = loop.create_future()
        parsed = command(argv, cwd, env)
        began = wide.begun(self.clock)

        def reaped(child: _Supervised) -> None:
            returncode, kills, left = child.reaped()
            ended_as = ending(returncode, kills)
            self._event(began, parsed, None, *_how(ended_as), **_reaped(child.pid, kills, left, ended_as), **self._after(stdio))
            tell(loop, _settle, ended, ended_as)

        started = self._spawned(parsed, _streams(stdio), began, None, reaped)
        return started if isinstance(started, NeverStarted) else Running(started.pid, ended, started, self.clock)

    def _captured(self, parsed: Command, limit: float, settle: Callable[[Ran | OSError], None]) -> "_Supervised | NeverStarted":
        """`parsed` started with its outputs to files hands reads once it is reaped, and `settle` given its Ran, or
        what reading them raised, once its event is written."""
        began = wide.begun(self.clock)
        with ExitStack() as unstarted:
            try:
                out = unstarted.enter_context(tempfile.TemporaryFile())
                err = unstarted.enter_context(tempfile.TemporaryFile())
            except OSError as error:
                self._event(began, parsed, limit, None, None, **_never(error))
                return NeverStarted(error)

            def reaped(child: _Supervised) -> None:
                returncode, kills, left = child.reaped()
                ended_as = ending(returncode, kills)
                result: Ran | OSError
                try:
                    with out, err:
                        result = Ran(ended_as, _read(out), _read(err))
                except OSError as unread:
                    result = unread
                raised, failure = (result, None) if isinstance(result, OSError) else _how(ended_as)
                self._event(began, parsed, limit, raised, failure, **_reaped(child.pid, kills, left, ended_as), **_sizes(result))
                settle(result)

            started = self._spawned(parsed, (subprocess.DEVNULL, out, err), began, limit, reaped)
            if not isinstance(started, NeverStarted):
                # The files are the child's thread's from here, and it closes them once it has read them.
                unstarted.pop_all()
            return started

    def _spawned(self, parsed: Command, streams: tuple[int | IO[bytes], int | IO[bytes], int | IO[bytes]], began: wide.Begun, limit: float | None, reaped: Callable[[_Supervised], None]) -> "_Supervised | NeverStarted":
        """`parsed` started as the leader of a session of its own, and a thread of its own that runs `reaped` once it has
        exited. [LAW:no-ambient-temporal-coupling] the start and the watch on its exit are made on the caller's thread,
        so a child exists to kill, and is watched, before the caller can stop waiting on it."""
        try:
            watch = select.kqueue()
        except OSError as error:
            self._event(began, parsed, limit, None, None, **_never(error))
            return NeverStarted(error)
        try:
            stdin, stdout, stderr = streams
            process = subprocess.Popen(parsed.argv, cwd=parsed.cwd, env=parsed.env, stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError as error:
            watch.close()
            self._event(began, parsed, limit, None, None, **_never(error))
            return NeverStarted(error)
        child = _Supervised(process)
        try:
            watching = _watched(watch, child.pid)
            threading.Thread(target=_supervise, args=(watch, watching, child, reaped), name=f"{parsed.program} {child.pid}", daemon=True).start()
        except BaseException:
            # A child made and not watched would have no owner: its group killed and it reaped here, before the failure goes on.
            watch.close()
            child.reaped()
            raise
        return child

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

    def _event(self, began: wide.Begun, parsed: Command, limit: float | None, raised: BaseException | None, failure: str | None, **told: wide.Fact) -> None:
        """[LAW:nothing-unseen] the one `child.run` event each child writes, as it is reaped or as it fails to start."""
        wide.ended("child.run", self.emit, began, raised, failure, program=parsed.program, argv=parsed.argv, limit=limit, **told)


def _watched(watch: select.kqueue, pid: int) -> bool:
    """Whether the leader `pid` is watched for its exit, or had already exited when the watch was made.

    macOS attaches no `NOTE_EXIT` to a process that has exited and is not yet reaped: it answers ESRCH, as for a process
    that is gone. hands alone reaps its children, so its child answering so is one that has exited already."""
    try:
        watch.control([select.kevent(pid, select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 0)
    except ProcessLookupError:
        return False
    return True


def _supervise(watch: select.kqueue, watching: bool, child: _Supervised, reaped: Callable[[_Supervised], None]) -> None:
    try:
        if watching:
            watch.control(None, 1)
    finally:
        watch.close()
    reaped(child)


def _settle[T](ended: asyncio.Future[T], result: T) -> None:
    if not ended.done():
        ended.set_result(result)


def _told(result: Ran | OSError | None, child: _Supervised) -> Outcome:
    """The outcome of a captured child: what its thread put, what reading its outputs raised raised here in its caller,
    or, where it was never reaped, the kills that were sent it."""
    match result:
        case Ran() as ran:
            return ran
        case OSError() as unread:
            raise unread
        case None:
            return Unreaped(child.kills(), child.pid)


def _streams(stdio: Stdio) -> tuple[int, int, int]:
    match stdio:
        case Terminal(slave=slave):
            return slave, slave, slave
        case Null():
            return subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL


def _reaped(pid: int, kills: tuple[Kill, ...], left: bool, ended: Ending) -> dict[str, wide.Fact]:
    return {"pid": pid, "kills": tuple(kill_said(kill) for kill in kills), "leftovers": left, **facts(ended)}


def _read(output: IO[bytes]) -> Captured:
    output.seek(0)
    data = output.read(CAPTURE_BYTES + 1)
    return Captured(data[:CAPTURE_BYTES], len(data) > CAPTURE_BYTES)


def _never(error: OSError) -> dict[str, wide.Fact]:
    return {"ending": "never_started", "errno": error.errno, "error_said": said(NeverStarted(error))}


def _sizes(result: Ran | OSError) -> dict[str, wide.Fact]:
    match result:
        case Ran(out=out, err=err):
            return {"out_bytes": len(out.data), "out_cut": out.cut, "err_bytes": len(err.data), "err_cut": err.cut}
        case OSError():
            return {}


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
