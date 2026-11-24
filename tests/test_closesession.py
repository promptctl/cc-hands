"""The `close_session` tool: a session's claude ended, and the session gone from the registry, as the user asked: a session
they named whatever it is doing, and of the sessions asked about as done only those at their prompt. Each session's claude is
a process in a fake table, which a SIGTERM ends unless it catches it, and the registry is kept by the liveness sweep, as the
daemon keeps it."""

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from fakeprocesses import FakeProcesses, Running, processes
from hands.clock import SYSTEM
from hands.core.events import Event, Joined, PermissionRequested, Prompted, StatusReported
from hands.core.session import Membership, Permission, PromptId, RequestId, SessionId
from hands.core.status import Busy, Idle, Report, Shell, Stamp, Status, Waiting
from hands.sessions import closesession
from hands.sessions.audit import AuditLog, segment
from hands.sessions.home import Home
from hands.sessions.liveness import keep_sweeping
from hands.sessions.membership import write_membership
from hands.sessions.registry import Sessions
from hands.voice.tool import Result
from hands.voice.tools import close_session_tool
from test_startsession import home_in
from conftest import new_sessions


def run[T](table: FakeProcesses, home: Home, sessions: Sessions, body: Callable[[], Awaitable[T]]) -> T:
    """`body` run while the liveness sweep keeps `sessions`, as the daemon's does."""

    async def swept() -> T:
        sweeping = asyncio.create_task(keep_sweeping(table, home, sessions, 0.05))
        try:
            return await body()
        finally:
            sweeping.cancel()

    return asyncio.run(swept())


def close(table: FakeProcesses, home: Home, sessions: Sessions, ids: list[str], asked: str) -> Callable[[], Awaitable[Result]]:
    """The tool called as the model calls it."""
    close_session = close_session_tool(table, home, AuditLog(home.audit, clock=datetime.now).record, sessions)
    return lambda: close_session.body(sessions=ids, asked=asked)


def each(result: Result) -> list[Any]:
    """What a call said of each session it was given, in the order given."""
    return cast(list[Any], result["sessions"])


def closes(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.close"]


def claudes(*pids: int) -> FakeProcesses:
    """A table in which each of `pids` is a claude, running since long before any session here wrote its file."""
    return processes(*(Running(pid) for pid in pids))


def joined(home: Home, sessions: Sessions, tmp_path: Path, pid: int, name: str, *events: Callable[[SessionId], Event]) -> SessionId:
    """A session in its own folder whose claude is `pid`, as its first hook writes it, then `events` of it applied."""
    folder = tmp_path / name
    folder.mkdir()
    member = Membership(SessionId(name), pid, folder, folder / "t.jsonl")
    write_membership(home, member)

    async def applied() -> None:
        for event in (Joined(member, "startup"), *(event(member.id) for event in events)):
            await sessions.apply(event)

    asyncio.run(applied())
    return member.id


def status(said: Status) -> Callable[[SessionId], Event]:
    return lambda session: StatusReported(session, Report(said, Stamp(1)), at=1.0)


def test_only_the_sessions_that_are_done_are_closed_of_those_asked_about_as_done(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    sessions = new_sessions(clock=SYSTEM)
    pids = idle, busy, waiting, shelled, held, prompted, unasked = range(4001, 4008)
    table = claudes(*pids)
    ids = {
        "idle": joined(home, sessions, tmp_path, idle, "idle", status(Idle())),
        "busy": joined(home, sessions, tmp_path, busy, "busy", status(Busy())),
        "waiting": joined(home, sessions, tmp_path, waiting, "waiting", status(Waiting("permission prompt"))),
        "shelled": joined(home, sessions, tmp_path, shelled, "shelled", status(Shell())),
        # At a dialog whose request is heard before the status that says it waits.
        "held": joined(home, sessions, tmp_path, held, "held", status(Idle()), lambda session: PermissionRequested(session, at=2.0, request=RequestId("r"), on=Permission("Bash", {}), mode="default")),
        # A prompt its hook opened a turn for, before the status that says it is busy is read.
        "prompted": joined(home, sessions, tmp_path, prompted, "prompted", status(Idle()), lambda session: Prompted(session, at=2.0, mode=None, prompt=PromptId("p1"), text="")),
    }
    joined(home, sessions, tmp_path, unasked, "unasked", status(Idle()))

    results = dict(zip(ids, each(run(table, home, sessions, close(table, home, sessions, list(ids.values()), "done"))), strict=True))

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


def test_a_named_session_is_closed_whatever_it_is_doing(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    sessions = new_sessions(clock=SYSTEM)
    busy, other = 4001, 4002
    table = claudes(busy, other)
    id = joined(home, sessions, tmp_path, busy, "billing", status(Busy()))
    joined(home, sessions, tmp_path, other, "docs", status(Idle()))
    assert each(run(table, home, sessions, close(table, home, sessions, [id], "named"))) == [{"closed": "billing"}]
    assert table.signalled == [(busy, signal.SIGTERM)]
    assert sessions.live_ids() == ["docs"] and other in table.running
    assert closes(home)[0]["facts"] == {"session": "billing", "asked": "named", "done": False, "signalled": True}


def test_a_session_that_is_not_running_and_one_that_will_not_end_are_said(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(closesession, "END_SECONDS", 0.5)
    home = home_in(tmp_path)
    sessions = new_sessions(clock=SYSTEM)
    # A claude that does not end on SIGTERM, as one wedged would not.
    stuck = 4001
    table = processes(Running(stuck, caught=frozenset({signal.SIGTERM})))
    assert each(run(table, home, sessions, close(table, home, sessions, ["nobody"], "named"))) == [{"error": "no session nobody is running"}]
    id = joined(home, sessions, tmp_path, stuck, "stuck", status(Idle()))
    assert each(run(table, home, sessions, close(table, home, sessions, [id], "named"))) == [{"error": "session stuck was told to end and is still running 0.5 seconds later"}]
    assert table.signalled == [(stuck, signal.SIGTERM)]
    assert stuck in table.running and sessions.live_ids() == ["stuck"]
    assert [event["outcome"] for event in closes(home)] == ["failed", "failed"]


def test_a_session_whose_claude_no_longer_holds_it_leaves_with_nothing_signalled(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    sessions = new_sessions(clock=SYSTEM)
    cleared, reused, unfiled = 4001, 4002, 4003
    # Each file here is written at `written`, and the process that took the reused pid started after it.
    written = 1000.0
    table = processes(Running(cleared), Running(reused, start=written + 60), Running(unfiled))
    moved = joined(home, sessions, tmp_path, cleared, "moved", status(Idle()))
    os.utime(home.membership(moved), (written, written))
    # A /clear in it: the same claude holds a new session, written after the one it left.
    after = joined(home, sessions, tmp_path, cleared, "after", status(Idle()))
    os.utime(home.membership(after), (written + 5, written + 5))
    # A pid a later process took: the file naming it was written before that process started.
    taken = joined(home, sessions, tmp_path, reused, "taken", status(Idle()))
    os.utime(home.membership(taken), (written, written))
    # Its file gone, as its end hook removes it first.
    gone = joined(home, sessions, tmp_path, unfiled, "gone", status(Idle()))
    home.membership(gone).unlink()
    assert each(run(table, home, sessions, close(table, home, sessions, [moved, taken, gone], "named"))) == [{"closed": "moved"}, {"closed": "taken"}, {"closed": "gone"}]
    # No process was sent anything: the claude that moved on runs on in the session it holds.
    assert table.signalled == []
    assert sessions.live_ids() == ["after"]
    assert [event["facts"]["signalled"] for event in closes(home)] == [False, False, False]
