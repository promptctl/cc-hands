"""Every child hands starts: held to its caller's limit with its whole process group, reaped, and told as one outcome."""

import ast
import asyncio
import errno
import math
import os
import re
import select
import signal
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

import pytest

import hands
from conftest import BrokenClock, VirtualClock, events
from hands.clock import SYSTEM
from imports import imported, qualified
from hands.sessions import child, wide
from hands.sessions.child import (
    KILL_GRACE_SECONDS,
    Cancelled,
    Captured,
    Children,
    Command,
    Exited,
    Kill,
    Killed,
    Limit,
    Lost,
    NeverStarted,
    Null,
    Ran,
    Signalled,
    Stop,
    Stopped,
    Terminal,
    Unreaped,
    WaitFailed,
    ending,
    facts,
    said,
)
from hands.sessions.wide import WideEvent

KILL = signal.SIGKILL
TERM = signal.SIGTERM


def stepped(policy: child._Policy, answers: list[bool]) -> tuple[list[child._Step], bool]:  # pyright: ignore[reportPrivateUsage]
    """Every step `policy` takes, each answered in turn from `answers`, and whether it says the child was over."""
    steps = [next(policy)]
    try:
        for answer in answers:
            steps.append(policy.send(answer))
    except StopIteration as returned:
        return steps, returned.value
    raise AssertionError(f"the policy took more steps than it was answered: {steps}")


def wait(seconds: float) -> child._Wait:  # pyright: ignore[reportPrivateUsage]
    return child._Wait(seconds)  # pyright: ignore[reportPrivateUsage]


def test_a_child_reaped_inside_its_limit_is_only_waited_on() -> None:
    assert stepped(child._bounded(Limit(5)), [True]) == ([wait(5)], True)  # pyright: ignore[reportPrivateUsage]


def test_a_child_past_its_limit_is_killed_with_its_group_and_its_reap_waited_a_grace() -> None:
    steps, reaped = stepped(child._bounded(Limit(5)), [False, False, True])  # pyright: ignore[reportPrivateUsage]
    assert steps == [wait(5), Kill(Limit(5)), wait(KILL_GRACE_SECONDS)]
    assert reaped


def test_a_killed_child_not_reaped_within_the_grace_is_left_to_its_thread() -> None:
    assert stepped(child._bounded(Limit(5)), [False, False, False])[1] is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("stopped", "why"),
    [(asyncio.CancelledError(), Cancelled()), (KeyboardInterrupt(), Cancelled()), (RuntimeError("no timer"), WaitFailed("RuntimeError('no timer')"))],
)
def test_a_wait_that_ends_early_kills_its_child_waits_the_grace_and_goes_on(stopped: BaseException, why: Cancelled | WaitFailed) -> None:
    policy = child._bounded(Limit(5))  # pyright: ignore[reportPrivateUsage]
    assert next(policy) == wait(5)
    assert policy.throw(stopped) == Kill(why)
    assert policy.send(False) == wait(KILL_GRACE_SECONDS)
    with pytest.raises(type(stopped)):
        policy.send(True)


def test_a_wait_stopped_twice_goes_on_without_its_grace() -> None:
    policy = child._bounded(Limit(5))  # pyright: ignore[reportPrivateUsage]
    next(policy)
    policy.throw(asyncio.CancelledError())
    policy.send(False)
    with pytest.raises(asyncio.CancelledError):
        policy.throw(asyncio.CancelledError())


def test_a_policy_closed_unfinished_takes_no_step() -> None:
    policy = child._bounded(Limit(5))  # pyright: ignore[reportPrivateUsage]
    next(policy)
    policy.close()


def test_a_stop_asks_the_child_alone_then_kills_its_group_once_the_grace_is_up() -> None:
    steps, reaped = stepped(child._stopping(Stop(5)), [False, False, False, True])  # pyright: ignore[reportPrivateUsage]
    assert steps == [Stop(5), wait(5), Kill(Stop(5)), wait(KILL_GRACE_SECONDS)]
    assert reaped


@pytest.mark.parametrize(
    ("returncode", "sent", "ended"),
    [
        (0, (), Exited(0)),
        (3, (), Exited(3)),
        # Its kill was on its way as it exited: it exited.
        (0, (Kill(Limit(5)),), Exited(0)),
        (-9, (Kill(Limit(5)),), Killed(Kill(Limit(5)))),
        # A SIGKILL from outside hands, an OOM or a `kill -9`, is not hands' kill.
        (-9, (), Signalled(KILL)),
        (-15, (Kill(Limit(5)),), Signalled(TERM)),
        # Killed for its limit, then its caller cancelled before the reap: the limit is why.
        (-9, (Kill(Limit(5)), Kill(Cancelled())), Killed(Kill(Limit(5)))),
        (-15, (Stop(5), Kill(Stop(5))), Stopped(Stop(5))),
        (-9, (Stop(5), Kill(Stop(5))), Killed(Kill(Stop(5)))),
    ],
)
def test_how_a_child_ended_is_read_from_its_code_and_what_hands_sent_it(returncode: int, sent: tuple[child.Sent, ...], ended: child.Ending) -> None:
    assert ending(returncode, sent) == ended


@pytest.mark.parametrize(
    ("ended", "told", "worded"),
    [
        (Exited(3), {"ending": "exited", "exit_code": 3}, "exited 3"),
        (Signalled(KILL), {"ending": "signalled", "signal": "SIGKILL"}, "was ended by SIGKILL"),
        (Killed(Kill(Limit(2))), {"ending": "killed", "signal": "SIGKILL", "why": "limit"}, "was killed with SIGKILL, because it ran past 2.0s"),
        (Killed(Kill(Stop(2))), {"ending": "killed", "signal": "SIGKILL", "why": "stop"}, "was killed with SIGKILL, because it was told to stop, with 2.0s to end"),
        (Killed(Kill(Cancelled())), {"ending": "killed", "signal": "SIGKILL", "why": "cancelled"}, "was killed with SIGKILL, because its caller stopped waiting"),
        (Killed(Kill(WaitFailed("E()"))), {"ending": "killed", "signal": "SIGKILL", "why": "wait_failed", "why_error": "E()"}, "was killed with SIGKILL, because its wait failed: E()"),
        (Stopped(Stop(5)), {"ending": "stopped", "signal": "SIGTERM"}, "stopped as it was told to, within 5.0s"),
    ],
)
def test_each_ending_has_its_own_facts_and_words(ended: child.Ending, told: dict[str, wide.Fact], worded: str) -> None:
    assert facts(ended) == told
    assert said(ended) == worded


def test_a_child_that_exited_nonzero_is_said_with_what_it_wrote_to_stderr() -> None:
    assert said(Ran(Exited(128), Captured(b""), Captured(b"fatal: not a git repository\n"))) == wide.exited(128, "fatal: not a git repository")


def test_a_child_that_never_started_was_never_reaped_or_was_lost_is_said_so() -> None:
    assert said(NeverStarted(FileNotFoundError(errno.ENOENT, "No such file or directory", "lit"))) == "could not be started: No such file or directory"
    assert said(Unreaped((Kill(Limit(1)),), 41)) == f"was killed, and still not reaped {KILL_GRACE_SECONDS:.1f}s later, as 41"
    assert said(Lost("OSError(5)")) == "ran, and hands lost its answer: OSError(5)"


@pytest.mark.parametrize("seconds", [0, -1.0, math.nan, math.inf])
def test_a_limit_that_is_no_finite_time_cannot_be_made(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite time"):
        Limit(seconds)


async def test_a_limit_that_is_no_time_is_refused_before_anything_starts() -> None:
    recorded: list[WideEvent] = []
    with pytest.raises(ValueError, match="finite time"):
        await Children(VirtualClock(), recorded.append).run("true", limit=math.nan)
    with pytest.raises(ValueError, match="finite time"):
        Children(VirtualClock(), recorded.append).run_blocking("true", limit=-1)
    assert recorded == []


@pytest.mark.parametrize(
    ("argv", "cwd", "env", "refused"),
    [
        (("git", "log\0"), None, None, "NUL"),
        (("git",), Path("/tmp/a\0b"), None, "NUL"),
        (("git",), None, {"A": "b\0"}, "NUL"),
        (("git",), None, {"A=B": "c"}, "named 'A=B'"),
        (("git",), None, {"": "c"}, "named ''"),
        (("git", "\ud800"), None, None, "no file name can encode"),
        (("git",), None, {"A": "\ud800"}, "no file name can encode"),
        ((), None, None, "a program"),
    ],
)
def test_a_command_no_os_can_be_handed_cannot_be_made(argv: tuple[str, ...], cwd: Path | None, env: dict[str, str] | None, refused: str) -> None:
    with pytest.raises(ValueError, match=refused):
        Command(argv, cwd, env)


async def test_a_directory_with_a_nul_is_refused_in_the_caller_before_anything_starts() -> None:
    recorded: list[WideEvent] = []
    with pytest.raises(ValueError, match="NUL"):
        await Children(VirtualClock(), recorded.append).run("true", limit=5, cwd=Path("/tmp/a\0b"))
    assert recorded == []


def test_what_a_child_wrote_past_the_capture_is_cut_and_said_to_be(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(child, "CAPTURE_BYTES", 4)
    with (tmp_path / "out").open("w+b") as output:
        output.write(b"12345")
        assert child._read(output) == (b"1234", True)  # pyright: ignore[reportPrivateUsage]
        output.truncate(4)
        assert child._read(output) == (b"1234", False)  # pyright: ignore[reportPrivateUsage]


class Leader:
    """A leader whose reap is held at a gate, so a test can send a kill while the supervisor reaps it."""

    pid = 4242

    def __init__(self) -> None:
        self.waiting = threading.Event()
        self.gate = threading.Event()

    def wait(self) -> int:
        self.waiting.set()
        self.gate.wait(5)
        return 0


@pytest.fixture
def kernel(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, signal.Signals]]:
    """Every signal hands sends, to which pid and how, sent nowhere."""
    sent: list[tuple[str, int, signal.Signals]] = []

    def sender(how: str) -> Callable[[int, int], None]:
        return lambda pid, signum: sent.append((how, pid, signal.Signals(signum)))

    monkeypatch.setattr(child.os, "kill", sender("kill"))
    monkeypatch.setattr(child.os, "killpg", sender("killpg"))
    return sent


def supervised() -> tuple[child._Supervised, Leader]:  # pyright: ignore[reportPrivateUsage]
    leader = Leader()
    return child._Supervised(leader), leader  # pyright: ignore[reportPrivateUsage]


async def test_a_child_past_its_limit_on_its_caller_s_clock_is_killed_and_its_reap_awaited(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    clock = VirtualClock()
    held, _ = supervised()
    ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(child._awaited(child._bounded(Limit(5)), held, ended, clock))  # pyright: ignore[reportPrivateUsage]
    await clock.sleeping(5)
    assert held.sent() == ()
    clock.advance(5)
    await clock.sleeping(5 + KILL_GRACE_SECONDS)
    assert held.sent() == (Kill(Limit(5)),)
    ended.set_result("reaped")
    assert await waiting == "reaped"


async def test_a_killed_child_whose_reap_does_not_come_ends_its_caller_s_wait_without_it(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    clock = VirtualClock()
    held, _ = supervised()
    ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(child._awaited(child._bounded(Limit(5)), held, ended, clock))  # pyright: ignore[reportPrivateUsage]
    await clock.sleeping(5)
    clock.advance(5)
    await clock.sleeping(5 + KILL_GRACE_SECONDS)
    clock.advance(KILL_GRACE_SECONDS)
    assert await waiting == Unreaped((Kill(Limit(5)),), 4242)


async def test_a_child_reaped_whose_answer_its_thread_is_still_reading_is_waited_for_past_the_grace(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    clock = VirtualClock()
    held, leader = supervised()
    ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(child._awaited(child._bounded(Limit(5)), held, ended, clock))  # pyright: ignore[reportPrivateUsage]
    await clock.sleeping(5)
    clock.advance(5)
    await clock.sleeping(5 + KILL_GRACE_SECONDS)
    leader.gate.set()
    held.reap()
    clock.advance(KILL_GRACE_SECONDS)
    await asyncio.sleep(0)
    assert not waiting.done(), "a reaped child was told as unreaped while its thread read what it wrote"
    ended.set_result("drained")
    assert await waiting == "drained"


async def test_a_caller_cancelled_while_it_waits_kills_its_child_and_is_cancelled_once_it_is_reaped(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    clock = VirtualClock()
    held, _ = supervised()
    ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(child._awaited(child._bounded(Limit(5)), held, ended, clock))  # pyright: ignore[reportPrivateUsage]
    await clock.sleeping(5)
    waiting.cancel()
    await clock.sleeping(KILL_GRACE_SECONDS)
    assert held.sent() == (Kill(Cancelled()),)
    assert not waiting.done(), "the cancelled caller left before its child was reaped"
    ended.set_result("reaped")
    with pytest.raises(asyncio.CancelledError):
        await waiting


async def test_a_caller_whose_clock_fails_kills_its_child_before_the_failure_goes_on(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    held, _ = supervised()
    ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    ended.get_loop().call_soon(ended.set_result, "reaped")
    with pytest.raises(RuntimeError, match="no timer"):
        await child._awaited(child._bounded(Limit(5)), held, ended, BrokenClock())  # pyright: ignore[reportPrivateUsage]
    assert held.sent() == (Kill(WaitFailed("RuntimeError('no timer')")),)


async def test_a_caller_closed_while_it_waits_kills_its_child_as_it_closes(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    clock = VirtualClock()
    held, _ = supervised()
    ended: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    waiting = child._awaited(child._bounded(Limit(5)), held, ended, clock)  # pyright: ignore[reportPrivateUsage]
    waiting.send(None)
    waiting.close()
    assert held.sent() == (Kill(Cancelled()),)


def test_a_blocking_caller_interrupted_while_it_waits_kills_its_child_and_goes_on(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    class Interrupting(VirtualClock):
        def take[T](self, source: "child.queue.SimpleQueue[T] | child.queue.Queue[T]", seconds: float) -> T:
            if seconds == 5:
                raise KeyboardInterrupt
            return super().take(source, seconds)

    held, _ = supervised()
    ended: child.queue.SimpleQueue[str] = child.queue.SimpleQueue()
    ended.put("reaped")
    with pytest.raises(KeyboardInterrupt):
        child._waited(child._bounded(Limit(5)), held, ended, Interrupting())  # pyright: ignore[reportPrivateUsage]
    assert held.sent() == (Kill(Cancelled()),)


async def test_a_stop_not_reaped_within_its_grace_and_the_kill_s_is_told_unreaped(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    clock = VirtualClock()
    held, _ = supervised()
    running = child.Running(4242, asyncio.get_running_loop().create_future(), held, clock)
    stopping = asyncio.ensure_future(running.stopped(5))
    await clock.sleeping(5)
    clock.advance(5)
    await clock.sleeping(5 + KILL_GRACE_SECONDS)
    clock.advance(KILL_GRACE_SECONDS)
    assert await stopping == Unreaped((Stop(5), Kill(Stop(5))), 4242)
    assert kernel == [("kill", 4242, TERM), ("killpg", 4242, KILL)]


async def test_a_stop_whose_clock_fails_kills_its_child_before_the_failure_goes_on(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    held, _ = supervised()
    ended: asyncio.Future[child.Ending | Lost] = asyncio.get_running_loop().create_future()
    ended.get_loop().call_soon(ended.set_result, Exited(0))
    with pytest.raises(RuntimeError, match="no timer"):
        await child.Running(4242, ended, held, BrokenClock()).stopped(5)
    assert held.sent() == (Stop(5), Kill(WaitFailed("RuntimeError('no timer')")))


def test_a_kill_sent_while_the_leader_is_being_reaped_sends_nothing(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    held, leader = supervised()
    reaping = threading.Thread(target=held.reap)
    reaping.start()
    assert leader.waiting.wait(5)
    held.signal(Kill(Limit(5)))
    leader.gate.set()
    reaping.join(5)
    # The reap's own sweep of the group, and nothing after it.
    assert kernel == [("killpg", 4242, KILL)]
    assert held.sent() == ()
    assert held.reaped


def test_a_kill_after_the_reap_sends_nothing(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    held, leader = supervised()
    leader.gate.set()
    held.reap()
    held.signal(Kill(Limit(5)))
    held.signal(Stop(5))
    assert kernel == [("killpg", 4242, KILL)]


def test_a_stop_asks_the_leader_alone_and_a_kill_goes_to_its_group(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    held, _ = supervised()
    held.signal(Stop(5))
    held.signal(Kill(Stop(5)))
    assert kernel == [("kill", 4242, TERM), ("killpg", 4242, KILL)]
    assert held.sent() == (Stop(5), Kill(Stop(5)))


def test_a_kill_of_a_group_every_process_of_which_has_exited_sends_nothing_and_leaves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def exited(_pgid: int, _signum: int) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(child.os, "killpg", exited)
    held, leader = supervised()
    held.signal(Kill(Limit(5)))
    assert held.sent() == ()
    leader.gate.set()
    assert held.reap() == (0, (), False)


type Written = tuple[BaseException | None, str | None, dict[str, wide.Fact]]


def followed(finish: "child._Finish[str]", monkeypatch: pytest.MonkeyPatch, exited: Callable[[object, int], None] = lambda _watch, _pid: None) -> tuple[list[str | Lost], list[Written]]:  # pyright: ignore[reportPrivateUsage]
    """What a child's thread tells its caller, and the event it writes, for a leader whose exit `exited` watches."""
    monkeypatch.setattr(child, "_exited", exited)
    settled: list[str | Lost] = []
    written: list[Written] = []
    handed: child.queue.SimpleQueue[child._Supervised | None] = child.queue.SimpleQueue()  # pyright: ignore[reportPrivateUsage]
    held, leader = supervised()
    leader.gate.set()
    handed.put(held)

    def event(raised: BaseException | None, failure: str | None, /, **said_of: wide.Fact) -> None:
        written.append((raised, failure, said_of))

    child._supervise(select.kqueue(), handed, finish, event, settled.append)  # pyright: ignore[reportPrivateUsage]
    return settled, written


def test_a_child_whose_answer_cannot_be_read_is_lost_and_its_event_says_so(kernel: list[tuple[str, int, signal.Signals]], monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(_ended: child.Ending) -> tuple[str, dict[str, wide.Fact]]:
        raise OSError(errno.EIO, "Input/output error")

    settled, written = followed(unreadable, monkeypatch)
    assert settled == [Lost("OSError(5, 'Input/output error')")]
    # What the reap learned is kept beside what was lost.
    assert written == [(None, "ran, and hands lost its answer: OSError(5, 'Input/output error')", {"pid": 4242, "kills": (), "leftovers": True, "lost": "OSError(5, 'Input/output error')", "ending": "exited", "exit_code": 0})]


def test_a_cancelled_caller_s_child_whose_answer_is_lost_is_still_said_cancelled(kernel: list[tuple[str, int, signal.Signals]], monkeypatch: pytest.MonkeyPatch) -> None:
    class Killed(Leader):
        def wait(self) -> int:
            return -signal.SIGKILL

    def exited(_watch: object, _pid: int) -> None:
        pass

    monkeypatch.setattr(child, "_exited", exited)
    held = child._Supervised(Killed())  # pyright: ignore[reportPrivateUsage]
    held.signal(Kill(Cancelled()))
    handed: child.queue.SimpleQueue[child._Supervised | None] = child.queue.SimpleQueue()  # pyright: ignore[reportPrivateUsage]
    handed.put(held)
    written: list[Written] = []

    def event(raised: BaseException | None, failure: str | None, /, **said_of: wide.Fact) -> None:
        written.append((raised, failure, said_of))

    child._supervise(select.kqueue(), handed, lambda ended: (Lost("cut"), {}), event, lambda _told: None)  # pyright: ignore[reportPrivateUsage]
    [(raised, failure, said_of)] = written
    assert isinstance(raised, asyncio.CancelledError) and failure is None
    # What was lost is on its event all the same.
    assert (said_of["ending"], said_of["lost"]) == ("killed", "cut")


def test_a_child_hands_cannot_watch_is_killed_with_its_group_and_reaped(kernel: list[tuple[str, int, signal.Signals]], monkeypatch: pytest.MonkeyPatch) -> None:
    def unwatchable(_watch: object, _pid: int) -> None:
        raise OSError(errno.EMFILE, "Too many open files")

    settled, [(_raised, _failure, said_of)] = followed(lambda ended: ("answered", {}), monkeypatch, unwatchable)
    assert settled == ["answered"]
    assert kernel == [("killpg", 4242, KILL), ("killpg", 4242, KILL)]
    assert said_of["kills"] == ("SIGKILL:wait_failed",)


def test_a_child_s_event_is_written_and_its_caller_told_where_it_exited(kernel: list[tuple[str, int, signal.Signals]], monkeypatch: pytest.MonkeyPatch) -> None:
    settled, written = followed(lambda ended: ("answered", {"drained": True}), monkeypatch)
    assert settled == ["answered"]
    assert written == [(None, None, {"pid": 4242, "kills": (), "leftovers": True, "lost": None, "ending": "exited", "exit_code": 0, "drained": True})]


def test_a_child_s_terminal_read_to_its_end_or_not_is_on_its_event() -> None:
    children = Children(VirtualClock(), lambda _: None)
    closed: child.queue.SimpleQueue[None] = child.queue.SimpleQueue()
    assert children._after(Terminal(0, closed)) == {"drained": False}  # pyright: ignore[reportPrivateUsage]
    closed.put(None)
    assert children._after(Terminal(0, closed)) == {"drained": True}  # pyright: ignore[reportPrivateUsage]
    assert children._after(Null()) == {}  # pyright: ignore[reportPrivateUsage]


async def test_no_thread_to_follow_a_child_starts_no_child_and_hands_failure_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(_thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    def started(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a child was started with no thread to follow it")

    monkeypatch.setattr(child.threading.Thread, "start", refused)
    monkeypatch.setattr(child.subprocess, "Popen", started)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        await Children(VirtualClock(), lambda _: None).run("true", limit=5)


async def test_a_watch_that_cannot_be_made_is_a_child_never_started_and_said_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused() -> None:
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(child.select, "kqueue", refused)
    recorded: list[WideEvent] = []
    outcome = await Children(VirtualClock(), recorded.append).run("true", limit=5)
    assert isinstance(outcome, NeverStarted) and outcome.error.errno == errno.EMFILE
    [event] = events(recorded, "child.run")
    assert (event.outcome, event.error, event.facts["ending"]) == ("failed", "could not be started: Too many open files", "never_started")


async def test_a_started_child_s_ending_is_kept_whoever_cancels_their_view_of_it(kernel: list[tuple[str, int, signal.Signals]]) -> None:
    held, _ = supervised()
    own: asyncio.Future[child.Ending | Lost] = asyncio.get_running_loop().create_future()
    running = child.Running(4242, own, held, VirtualClock())
    running.ended.cancel()
    own.set_result(Exited(0))
    assert await running.ended == Exited(0)


async def test_output_files_that_cannot_be_made_are_a_child_that_never_started_and_none_is_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[IO[bytes]] = []
    real = child.tempfile.TemporaryFile

    def second_refused() -> IO[bytes]:
        if made:
            raise OSError(errno.EMFILE, "Too many open files")
        made.append(file := real())
        return file

    monkeypatch.setattr(child.tempfile, "TemporaryFile", second_refused)
    recorded: list[WideEvent] = []
    outcome = await Children(VirtualClock(), recorded.append).run("true", limit=5)
    assert isinstance(outcome, NeverStarted) and outcome.error.errno == errno.EMFILE
    [first] = made
    assert first.closed
    [event] = events(recorded, "child.run")
    assert (event.outcome, event.facts["ending"]) == ("failed", "never_started")


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def until_gone(pid: int) -> None:
    # Killed, a grandchild is reparented to launchd and reaped there, a moment later.
    for _ in range(100):
        if gone(pid):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"{pid} is still running")


@pytest.mark.outside
async def test_a_child_s_output_and_exit_are_what_it_ran_to_and_its_event_is_in_its_caller_s_span() -> None:
    recorded: list[WideEvent] = []
    children = Children(SYSTEM, recorded.append)
    with wide.unit("test.caller", recorded.append, wide.begun(SYSTEM)):
        ran = await children.run("sh", "-c", "echo out; echo err >&2; exit 3", limit=5)
    assert ran == Ran(Exited(3), Captured(b"out\n"), Captured(b"err\n"))
    [run] = events(recorded, "child.run")
    [caller] = events(recorded, "test.caller")
    assert (run.trace_id, run.parent_id) == (caller.trace_id, caller.span_id)
    assert (run.outcome, run.facts["exit_code"], run.facts["program"], run.facts["kills"]) == ("ok", 3, "sh", ())


@pytest.mark.outside
async def test_a_child_that_wrote_more_than_hands_reads_of_its_output_is_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(child, "CAPTURE_BYTES", 4)
    recorded: list[WideEvent] = []
    outcome = await Children(SYSTEM, recorded.append).run("sh", "-c", "echo 12345", limit=5)
    assert outcome == Lost("it wrote more than the 4 bytes hands reads of its output")
    [event] = events(recorded, "child.run")
    assert (event.outcome, event.facts["exit_code"], event.facts["out_cut"]) == ("failed", 0, True)


@pytest.mark.outside
async def test_a_child_past_its_limit_is_killed_with_what_it_started(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    recorded: list[WideEvent] = []
    started = time.monotonic()
    ran = await Children(SYSTEM, recorded.append).run("sh", "-c", f"sleep 30 & echo $! > {pid}; wait", limit=0.3)
    assert time.monotonic() - started < 0.3 + KILL_GRACE_SECONDS
    assert isinstance(ran, Ran) and ran.ending == Killed(Kill(Limit(0.3)))
    await until_gone(int(pid.read_text()))
    [event] = events(recorded, "child.run")
    assert (event.outcome, event.error) == ("failed", "was killed with SIGKILL, because it ran past 0.3s")


@pytest.mark.outside
async def test_what_a_child_leaves_in_its_group_as_it_exits_is_killed(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    recorded: list[WideEvent] = []
    ran = await Children(SYSTEM, recorded.append).run("sh", "-c", f"sleep 600 & echo $! > {pid}", limit=5)
    assert isinstance(ran, Ran) and ran.ending == Exited(0)
    await until_gone(int(pid.read_text()))
    [event] = events(recorded, "child.run")
    assert event.facts["leftovers"] is True


@pytest.mark.outside
async def test_a_process_that_moved_to_a_session_of_its_own_is_left_alone() -> None:
    # Under /tmp: a socket's path holds 104 bytes on macOS, which pytest's tmp_path leaves no room in.
    sockets = tempfile.TemporaryDirectory(dir="/tmp")
    socket = Path(sockets.name) / "tmux"
    children = Children(SYSTEM, lambda _: None)
    try:
        assert (await children.run("tmux", "-S", str(socket), "new-session", "-d", "-s", "kept", "sleep 600", limit=10)) == Ran(Exited(0), Captured(b""), Captured(b""))
        listed = await children.run("tmux", "-S", str(socket), "list-sessions", "-F", "#{session_name}", limit=10)
        assert isinstance(listed, Ran) and (listed.ending, listed.out.data) == (Exited(0), b"kept\n")
    finally:
        # The test's own server, on its own socket: its one session ended, it exits.
        await children.run("tmux", "-S", str(socket), "kill-session", "-t", "kept", limit=10)
        sockets.cleanup()


@pytest.mark.outside
async def test_a_child_whose_caller_stops_waiting_is_killed_and_its_event_says_cancelled(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    recorded: list[WideEvent] = []
    running = asyncio.ensure_future(Children(SYSTEM, recorded.append).run("sh", "-c", f"echo $$ > {pid}; exec sleep 30", limit=30))
    while not pid.exists() or not pid.read_text().strip():
        await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert gone(int(pid.read_text())), "its caller left before it was reaped"
    [event] = events(recorded, "child.run")
    assert (event.outcome, event.facts["why"]) == ("cancelled", "cancelled")


@pytest.mark.outside
async def test_a_program_that_is_not_there_never_started_and_the_os_says_why() -> None:
    recorded: list[WideEvent] = []
    outcome = await Children(SYSTEM, recorded.append).run("/nonexistent/hands-no-such-program", limit=5)
    assert isinstance(outcome, NeverStarted)
    assert (outcome.error.errno, outcome.error.filename) == (errno.ENOENT, "/nonexistent/hands-no-such-program")
    [event] = events(recorded, "child.run")
    assert (event.outcome, event.error) == ("failed", "could not be started: No such file or directory")


@pytest.mark.outside
async def test_a_directory_that_is_gone_is_a_child_that_never_started_there(tmp_path: Path) -> None:
    gone_dir = tmp_path / "gone"
    outcome = await Children(SYSTEM, lambda _: None).run("true", limit=5, cwd=gone_dir)
    assert isinstance(outcome, NeverStarted)
    assert (outcome.error.errno, str(outcome.error.filename)) == (errno.ENOENT, str(gone_dir))


@pytest.mark.outside
def test_a_child_run_from_a_thread_is_held_to_its_limit_there(tmp_path: Path) -> None:
    pid = tmp_path / "pid"
    children = Children(SYSTEM, lambda _: None)
    outcomes: list[child.Outcome] = []
    thread = threading.Thread(target=lambda: outcomes.append(children.run_blocking("sh", "-c", f"sleep 30 & echo $! > {pid}; wait", limit=0.3)))
    thread.start()
    thread.join(5)
    [ran] = outcomes
    assert isinstance(ran, Ran) and ran.ending == Killed(Kill(Limit(0.3)))
    assert children.run_blocking("sh", "-c", "exit 4", limit=5) == Ran(Exited(4), Captured(b""), Captured(b""))


@pytest.mark.outside
async def test_a_started_child_that_ignores_its_stop_is_killed_once_the_grace_is_up(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    recorded: list[WideEvent] = []
    started = Children(SYSTEM, recorded.append).start(["sh", "-c", f"trap '' TERM; touch {ready}; while :; do sleep 0.05; done"], tmp_path, dict(os.environ), Null())
    assert isinstance(started, child.Running)
    while not ready.exists():
        await asyncio.sleep(0.01)
    assert await started.stopped(0.2) == Killed(Kill(Stop(0.2)))
    [event] = events(recorded, "child.run")
    assert event.outcome == "failed"
    assert event.facts["kills"] == ("SIGTERM:stop", "SIGKILL:stop")


@pytest.mark.outside
async def test_a_started_child_that_ends_as_its_stop_asks_ended_by_the_stop(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    started = Children(SYSTEM, lambda _: None).start(["sh", "-c", f"touch {ready}; exec sleep 30"], tmp_path, dict(os.environ), Null())
    assert isinstance(started, child.Running)
    while not ready.exists():
        await asyncio.sleep(0.01)
    assert await started.stopped(5) == Stopped(Stop(5))


# --- the one place hands starts a child: every other start is marked with its owner and reason ------------------------

# Modules every facility of which is a way to start a process, so importing one is reaching for it.
STARTING = ("subprocess", "multiprocessing")

# The calls that start a process, or replace this one with another, by the name they were imported as.
STARTS = re.compile(
    r"^(os\.(fork|forkpty|posix_spawnp?|spawn\w*|system|popen|exec\w*)|pty\.(fork|spawn)|asyncio\.create_subprocess_\w+|concurrent\.futures\.ProcessPoolExecutor)$"
)

# A loop's own starts, on whatever loop they are called on.
LOOP_STARTS = ("subprocess_exec", "subprocess_shell")

# Modules a star import brings a start of in by its bare name.
STAR_STARTING = ("os", "pty", "asyncio", "concurrent.futures", *STARTING)

# What a start outside the runner carries, on its statement or in the comment above it.
OWNED = "[LAW:single-enforcer] exception"


def starts(source: str, package: str = "hands") -> set[str]:
    """Each start `source` makes outside the runner unmarked, as `line: spelling`, in code it runs and in code it writes
    out for another interpreter to run."""
    tree = ast.parse(source)
    named = imported(tree, package)
    lines = source.splitlines()
    held = _statements(tree)
    return {
        f"{node.lineno}: {spelling}"
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt | ast.expr) and (spelling := _start(node, named)) is not None and not _owned(lines, held.get(node, node.lineno), node.lineno)
    }


def _start(node: ast.AST, named: dict[str, str]) -> str | None:
    """How `node` starts a process, by the name it reaches the start with, or None for a node that starts none."""
    match node:
        case ast.JoinedStr() | ast.Constant(value=str()) if (text := _code(node)) is not None and (written := _started(text)):
            return f"written {sorted(written)}"
        case ast.Import(names=aliases) if reaching := [alias.name for alias in aliases if alias.name.split(".")[0] in STARTING]:
            return reaching[0]
        case ast.ImportFrom(module=str(module), names=[ast.alias(name="*")], level=0) if module in STAR_STARTING:
            return f"{module}.*"
        case ast.ImportFrom(module=str(module), level=0) if module.split(".")[0] in STARTING:
            return module
        case ast.Call(func=func) if STARTS.match(qualified(func, named)):
            return f"{qualified(func, named)}()"
        case ast.Call(func=ast.Attribute(attr=attr)) if attr in LOOP_STARTS:
            return f".{attr}()"
        case _:
            return None


def _statements(tree: ast.Module) -> dict[ast.AST, int]:
    """The line each node's innermost statement begins on."""
    held: dict[ast.AST, int] = {}
    for statement in ast.walk(tree):
        if isinstance(statement, ast.stmt):
            held.update({node: statement.lineno for node in ast.walk(statement)})
    return held


def _owned(lines: list[str], begins: int, at: int) -> bool:
    """A start marked on its statement, from where the statement begins to the start's own line, or in the comment
    block just above it."""
    above = begins - 1
    while above >= 1 and lines[above - 1].lstrip().startswith("#"):
        above -= 1
    return any(OWNED in line for line in lines[above:at])


def _code(node: ast.expr) -> str | None:
    """The code a string literal holds, if it reads as code: an f-string's holes are read as a name."""
    match node:
        case ast.JoinedStr(values=values):
            return "".join(str(v.value) if isinstance(v, ast.Constant) else "x" for v in values)
        case ast.Constant(value=str(text)) if "\n" in text or "import" in text:
            return text
        case _:
            return None


def _started(text: str) -> set[str]:
    """What code written out starts, marked or not: the string it is in is marked or not as a whole."""
    try:
        return {spelling.split(": ", 1)[1] for spelling in starts(textwrap.dedent(text).replace(OWNED, ""))}
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()


def test_hands_starts_no_child_but_through_the_runner_or_a_start_marked_with_its_owner() -> None:
    root = Path(hands.__file__).parent
    unowned = {
        path.relative_to(root).as_posix(): found
        for path in sorted(root.rglob("*.py"))
        if (found := starts(path.read_text(), ".".join([root.name, *path.relative_to(root).parent.parts])))
    }
    assert unowned == {}


@pytest.mark.parametrize(
    ("source", "spelling"),
    [
        ("import subprocess\n", "subprocess"),
        ("import subprocess as sp\n", "subprocess"),
        ("from subprocess import Popen\n", "subprocess"),
        ("import multiprocessing.pool\n", "multiprocessing.pool"),
        ("from os import *\n", "os.*"),
        ("from concurrent.futures import *\n", "concurrent.futures.*"),
        ("import os\nos.fork()\n", "os.fork()"),
        ("import os as o\no.posix_spawnp('true', ['true'], {})\n", "os.posix_spawnp()"),
        ("from os import execv\nexecv('/bin/true', ['true'])\n", "os.execv()"),
        ("from os import spawnlp as s\ns(0, 'true')\n", "os.spawnlp()"),
        ("import os\nos.system('true')\n", "os.system()"),
        ("import os\nos.popen('true')\n", "os.popen()"),
        ("import pty\npty.fork()\n", "pty.fork()"),
        ("from pty import spawn\nspawn(['true'])\n", "pty.spawn()"),
        ("import asyncio\nasyncio.create_subprocess_exec('true')\n", "asyncio.create_subprocess_exec()"),
        ("loop.subprocess_shell(factory, 'true')\n", ".subprocess_shell()"),
        ("from concurrent.futures import ProcessPoolExecutor\nProcessPoolExecutor()\n", "concurrent.futures.ProcessPoolExecutor()"),
        ("from ..sessions import x\nimport os\nos.forkpty()\n", "os.forkpty()"),
    ],
)
def test_the_scan_refuses_each_way_of_starting_a_process(source: str, spelling: str) -> None:
    assert spelling in {found.split(": ", 1)[1] for found in starts(source, "hands.daemon")}


def test_the_scan_refuses_a_start_written_for_another_interpreter() -> None:
    assert starts('launch = "import os\\nos.execve(path, argv, env)\\n"\n') == {"1: written ['os.execve()']"}


def test_a_start_marked_on_its_statement_or_just_above_it_is_owned() -> None:
    marked = textwrap.dedent(
        f"""
        import os
        # {OWNED}: the indicator's process, started apart from every child.
        os.posix_spawn("x", ["x"], {{}})
        os.execv(  # {OWNED}: the restart.
            "x",
            ["x"],
        )
        launch = (  # {OWNED}: the pane's own exec.
            "import os\\n"
            "os.execve(a, b, c)\\n"
        )
        """
    )
    assert starts(marked) == set()


def test_a_mark_further_up_than_the_comment_above_owns_nothing() -> None:
    unmarked = textwrap.dedent(
        f"""
        import os
        # {OWNED}: the line below.
        value = 1
        os.fork()
        """
    )
    assert starts(unmarked) == {"5: os.fork()"}
