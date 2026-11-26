"""The `close_session` tool: a session's claude ended, and the session gone from the registry, as the user asked: a session
they named whatever it is doing, and of the sessions asked about as done only those at their prompt. Each session's claude is
a process in a fake table, which a SIGTERM ends unless it catches it, and the registry is kept by the liveness sweep, as the
daemon keeps it, on a clock the test moves."""

import asyncio
import errno
import json
import os
import signal
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from fakeprocesses import KILL, PROC_PID, FakeProcesses, Running, processes
from hands.core.events import Event, Joined, PermissionRequested, Prompted, StatusReported
from hands.core.session import Membership, Permission, PromptId, RequestId, SessionId
from hands.core.status import Busy, Idle, Report, Shell, Stamp, Status, Waiting
from hands.sessions import closesession
from hands.sessions.audit import AuditLog, segment
from hands.sessions.home import Home
from hands.sessions.liveness import sweep
from hands.sessions.membership import write_membership
from hands.sessions.registry import Sessions
from hands.virtualclock import VirtualClock
from hands.voice.tool import Result
from hands.voice.tools import close_session_tool
from test_startsession import home_in
from conftest import new_sessions


async def run[T](table: FakeProcesses, home: Home, sessions: Sessions, clock: VirtualClock, body: Callable[[], Awaitable[T]]) -> T:
    """`body` run while the liveness sweep keeps `sessions`, as the daemon's does: swept once each time the close looks at
    the registry, and the clock moved on to its next look only once the close waits on it
    [LAW:no-ambient-temporal-coupling]."""

    async def swept() -> T:
        closing = asyncio.ensure_future(body())
        unfiled: frozenset[SessionId] = frozenset()
        while not closing.done():
            unfiled = await sweep(table, home, sessions, unfiled)
            looking = asyncio.ensure_future(clock.sleeping(clock.now() + closesession.LOOK_SECONDS))
            await asyncio.wait([closing, looking], return_when=asyncio.FIRST_COMPLETED)
            looking.cancel()
            if not closing.done():
                clock.advance(closesession.LOOK_SECONDS)
        return closing.result()

    return await swept()


def close(table: FakeProcesses, home: Home, sessions: Sessions, clock: VirtualClock, ids: list[str], asked: str) -> Callable[[], Awaitable[Result]]:
    """The tool called as the model calls it."""
    close_session = close_session_tool(table, home, AuditLog(home.audit, clock=clock.wall).record, sessions)
    return lambda: close_session.body(sessions=ids, asked=asked)


def each(result: Result) -> list[Any]:
    """What a call said of each session it was given, in the order given."""
    return cast(list[Any], result["sessions"])


def closes(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.close"]


def claudes(*pids: int) -> FakeProcesses:
    """A table in which each of `pids` is a claude, running since long before any session here wrote its file."""
    return processes(*(Running(pid) for pid in pids))


async def joined(home: Home, sessions: Sessions, tmp_path: Path, pid: int, name: str, *events: Callable[[SessionId], Event]) -> SessionId:
    """A session in its own folder whose claude is `pid`, as its first hook writes it, then `events` of it applied."""
    folder = tmp_path / name
    folder.mkdir()
    member = Membership(SessionId(name), pid, folder, folder / "t.jsonl")
    write_membership(home, member)

    async def applied() -> None:
        for event in (Joined(member, "startup"), *(event(member.id) for event in events)):
            await sessions.apply(event)

    await applied()
    return member.id


def status(said: Status) -> Callable[[SessionId], Event]:
    return lambda session: StatusReported(session, Report(said, Stamp(1)), at=1.0)


async def test_only_the_sessions_that_are_done_are_closed_of_those_asked_about_as_done(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    clock = VirtualClock()
    sessions = new_sessions(clock=clock)
    pids = idle, busy, waiting, shelled, held, prompted, unasked = range(4001, 4008)
    table = claudes(*pids)
    ids = {
        "idle": await joined(home, sessions, tmp_path, idle, "idle", status(Idle())),
        "busy": await joined(home, sessions, tmp_path, busy, "busy", status(Busy())),
        "waiting": await joined(home, sessions, tmp_path, waiting, "waiting", status(Waiting("permission prompt"))),
        "shelled": await joined(home, sessions, tmp_path, shelled, "shelled", status(Shell())),
        # At a dialog whose request is heard before the status that says it waits.
        "held": await joined(home, sessions, tmp_path, held, "held", status(Idle()), lambda session: PermissionRequested(session, at=2.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default")),
        # A prompt its hook opened a turn for, before the status that says it is busy is read.
        "prompted": await joined(home, sessions, tmp_path, prompted, "prompted", status(Idle()), lambda session: Prompted(session, at=2.0, mode=None, prompt=PromptId("p1"), text="")),
    }
    await joined(home, sessions, tmp_path, unasked, "unasked", status(Idle()))

    results = dict(zip(ids, each(await run(table, home, sessions, clock, close(table, home, sessions, clock, list(ids.values()), "done"))), strict=True))

    assert results["idle"] == {"closed": "idle"}
    assert {name: result["state"] for name, result in results.items() if name != "idle"} == {
        "busy": "working",
        "waiting": "waiting at a dialog: permission prompt",
        "shelled": "idle, with a shell command it started in the background still running",
        "held": "waiting for permission to use Bash",
        "prompted": "working",
    }
    assert all(results[name]["left_running"] == name for name in ("busy", "waiting", "shelled", "held", "prompted"))
    # Only the done one was told to end and left the registry; the rest, and the session nobody asked about, run on.
    assert table.signalled == [(idle, signal.SIGTERM)]
    assert sorted(table.running) == [1, busy, waiting, shelled, held, prompted, unasked]
    assert sorted(sessions.live_ids()) == ["busy", "held", "prompted", "shelled", "unasked", "waiting"]
    # [LAW:nothing-unseen] each close says which session, why it was asked, whether it was done, and whether it was signalled.
    facts = {event["facts"]["session"]: event["facts"] for event in closes(home)}
    assert facts["idle"] == {"session": "idle", "asked": "done", "done": True, "signalled": True}
    assert facts["busy"] == {"session": "busy", "asked": "done", "done": False}


async def test_a_named_session_is_closed_whatever_it_is_doing(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    clock = VirtualClock()
    sessions = new_sessions(clock=clock)
    busy, other = 4001, 4002
    table = claudes(busy, other)
    id = await joined(home, sessions, tmp_path, busy, "billing", status(Busy()))
    await joined(home, sessions, tmp_path, other, "docs", status(Idle()))
    assert each(await run(table, home, sessions, clock, close(table, home, sessions, clock, [id], "named"))) == [{"closed": "billing"}]
    assert table.signalled == [(busy, signal.SIGTERM)]
    assert sessions.live_ids() == ["docs"] and other in table.running
    assert closes(home)[0]["facts"] == {"session": "billing", "asked": "named", "done": False, "signalled": True}


async def test_a_session_that_is_not_running_and_one_that_will_not_end_are_said(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    clock = VirtualClock()
    sessions = new_sessions(clock=clock)
    # A claude that does not end on SIGTERM, as one wedged would not.
    stuck = 4001
    table = processes(Running(stuck, caught=frozenset({signal.SIGTERM})))
    assert each(await run(table, home, sessions, clock, close(table, home, sessions, clock, ["nobody"], "named"))) == [{"error": "no session nobody is running"}]
    id = await joined(home, sessions, tmp_path, stuck, "stuck", status(Idle()))
    assert each(await run(table, home, sessions, clock, close(table, home, sessions, clock, [id], "named"))) == [{"error": "session stuck was told to end and is still running 10 seconds later"}]
    assert table.signalled == [(stuck, signal.SIGTERM)]
    assert stuck in table.running and sessions.live_ids() == ["stuck"]
    assert [event["outcome"] for event in closes(home)] == ["failed", "failed"]


async def test_a_session_whose_claude_no_longer_holds_it_leaves_with_nothing_signalled(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    clock = VirtualClock()
    sessions = new_sessions(clock=clock)
    cleared, reused, unfiled = 4001, 4002, 4003
    # Each file here is written at `written`, and the process that took the reused pid started after it.
    written = 1000.0
    table = processes(Running(cleared), Running(reused, start=written + 60), Running(unfiled))
    moved = await joined(home, sessions, tmp_path, cleared, "moved", status(Idle()))
    os.utime(home.membership(moved), (written, written))
    # A /clear in it: the same claude holds a new session, written after the one it left.
    after = await joined(home, sessions, tmp_path, cleared, "after", status(Idle()))
    os.utime(home.membership(after), (written + 5, written + 5))
    # A pid a later process took: the file naming it was written before that process started.
    taken = await joined(home, sessions, tmp_path, reused, "taken", status(Idle()))
    os.utime(home.membership(taken), (written, written))
    # Its file gone, as its end hook removes it first.
    gone = await joined(home, sessions, tmp_path, unfiled, "gone", status(Idle()))
    home.membership(gone).unlink()
    assert each(await run(table, home, sessions, clock, close(table, home, sessions, clock, [moved, taken, gone], "named"))) == [{"closed": "moved"}, {"closed": "taken"}, {"closed": "gone"}]
    # No process was sent anything: the claude that moved on runs on in the session it holds.
    assert table.signalled == []
    assert sessions.live_ids() == ["after"]
    assert [event["facts"]["signalled"] for event in closes(home)] == [False, False, False]


async def test_a_session_whose_claude_the_kernel_will_not_signal_is_said_and_left_running(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    clock = VirtualClock()
    sessions = new_sessions(clock=clock)
    pid = 4001
    table = processes(Running(pid, refused={KILL: errno.EPERM}))
    id = await joined(home, sessions, tmp_path, pid, "held", status(Idle()))
    assert each(await run(table, home, sessions, clock, close(table, home, sessions, clock, [id], "named"))) == [
        {"error": f"session held's claude, pid {pid}, could not be signalled to end: [Errno 1] Operation not permitted"}
    ]
    assert pid in table.running and sessions.live_ids() == ["held"]


async def test_a_session_whose_claude_the_kernel_will_not_say_the_start_of_is_said_and_left_running(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    clock = VirtualClock()
    sessions = new_sessions(clock=clock)
    pid = 4001
    id = await joined(home, sessions, tmp_path, pid, "held", status(Idle()))
    # Not swept: the sweep asks the kernel the same, and is refused alike.
    table = processes(Running(pid, refused={PROC_PID: errno.EPERM}))
    assert each(await close(table, home, sessions, clock, [id], "named")()) == [
        {"error": f"session held's claude, pid {pid}, could not be looked for: [Errno 1] sysctl could not say when pid {pid} started: Operation not permitted"}
    ]
    assert table.signalled == [] and sessions.live_ids() == ["held"]
