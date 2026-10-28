"""The shim against a real hook socket: it records membership and posts; with no daemon it is silent when hands is
off and loud when hands is broken."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import socket
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.core.effects import Summarise
from hands.core.session import Membership, Opened, PromptId, Session, SessionId, Unreported
from hands.sessions import heartbeat
from hands.sessions.hookconfig import LAUNCHER, SHIM_MODULE
from hands.sessions.shim import OVERVIEW
from hands.sessions.home import Home
from hands.sessions.liveness import sweep
from hands.sessions.membership import read_membership, write_membership
from hands.sessions.registry import Sessions
from hands.core.events import Located, StatusReported
from hands.core.status import Busy, Report, Stamp
from hands.sessions.audit import Entry
from hands.sessions.names import NameGiven, NameUnread, NameWithheld, Names
from hands.sessions.reader import Reader
from hands.sessions.server import serve_hooks
from hands.sessions.tail import Tails
from hands.sessions.wide import Fact, WideEvent
from conftest import SYSTEM_PATH

SID = SessionId("0f1e2d3c-aaaa-bbbb-cccc-000000000001")
COMMON = {"session_id": SID, "transcript_path": "/nowhere/t.jsonl", "cwd": "/code/a"}
START = {**COMMON, "hook_event_name": "SessionStart", "source": "startup"}
PROMPT = {**COMMON, "hook_event_name": "UserPromptSubmit", "prompt": "hi", "prompt_id": "p1"}
STOP = {**COMMON, "hook_event_name": "Stop", "stop_hook_active": False, "last_assistant_message": "done", "prompt_id": "p1"}
END = {**COMMON, "hook_event_name": "SessionEnd", "reason": "other"}
ASK = {**COMMON, "hook_event_name": "PermissionRequest", "tool_name": "Bash", "tool_input": {"command": "ls"}}
# What a start prints for Claude Code, whatever the daemon answers.
STARTED = json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": OVERVIEW}})


@pytest.fixture
def home() -> Iterator[Home]:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    yield Home(root)
    shutil.rmtree(root)


@pytest.fixture
async def sessions(home: Home) -> AsyncIterator[Sessions]:
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None)
    runner = await serve_hooks(home, registry, Names(), lambda _: None, Reader(registry, Tails(registry), registry.apply))
    yield registry
    await runner.cleanup()


async def shim(home: Home, payload: Mapping[str, object], fritter: str | None = None, given: Mapping[str, str] = {}) -> tuple[int | None, str, str]:
    """The shim's exit, stdout, and stderr for one hook, run as the plugin runs it: with only HANDS_HOME to go on."""
    environment = os.environ | dict(given)
    if fritter is not None:
        environment["FRITTER_SOCKET"] = fritter
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        SHIM_MODULE,
        env={**environment, "HANDS_HOME": str(home.root)},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(json.dumps(payload).encode())
    return process.returncode, stdout.decode(), stderr.decode()


def hooks(recorded: list[Entry]) -> list[tuple[str, str | None, dict[str, Fact]]]:
    """Each hook event recorded: how it ended, why it failed, and the facts naming the hook and the branch it took."""
    return [
        (entry.outcome, entry.error, {name: entry.facts[name] for name in ("hook", "stop", "reply", "name") if name in entry.facts})
        for entry in recorded
        if isinstance(entry, WideEvent) and entry.event == "hook"
    ]


def beat(home: Home, pid: int, written_ago: timedelta, pipeline: heartbeat.PipelineState = "running") -> None:
    now = datetime.now(UTC)
    heartbeat.write(home.status, heartbeat.Status(pid, now, now - written_ago, heartbeat.HEARTBEAT, pipeline, None, 0, False, (), None))


@pytest.mark.parametrize("payload", [PROMPT, ASK], ids=["prompt", "permission"])
async def test_a_hands_that_never_ran_costs_the_session_nothing(home: Home, payload: Mapping[str, object]) -> None:
    # Nothing on stdout, so a permission request falls through to Claude Code's own dialog.
    assert await shim(home, payload) == (0, "", "")


async def test_a_session_is_told_as_it_starts_to_end_every_turn_on_a_speakable_overview_whether_or_not_hands_runs(home: Home) -> None:
    """What a session says at the end of a turn is what the model is handed to say aloud."""
    code, stdout, stderr = await shim(home, START)
    assert (code, stderr) == (0, "")
    assert stdout == STARTED
    assert "End every turn with a concise, speakable overview" in OVERVIEW


async def test_a_hands_that_was_stopped_costs_the_session_nothing(home: Home) -> None:
    beat(home, os.getpid(), timedelta(minutes=5), pipeline="stopped")
    assert await shim(home, ASK) == (0, "", "")


async def test_a_session_started_while_hands_is_off_is_still_recorded_for_when_it_starts(home: Home) -> None:
    await shim(home, START)
    assert read_membership(home, SID).pid == os.getpid()


async def test_a_hands_that_died_is_reported_with_the_socket_and_the_heartbeat(home: Home, dead_pid: Callable[[], int]) -> None:
    beat(home, dead_pid(), timedelta(seconds=1))
    code, stdout, stderr = await shim(home, PROMPT)
    assert (code, stdout) == (1, "")
    assert f"cannot reach the hands daemon at {home.socket}" in stderr
    assert "hands is down" in stderr


async def test_a_hands_that_hung_is_reported(home: Home) -> None:
    beat(home, os.getpid(), timedelta(minutes=1))
    code, _, stderr = await shim(home, ASK)
    assert code == 1
    assert "hands is not responding" in stderr


async def test_a_heartbeat_nothing_can_read_is_reported(home: Home) -> None:
    home.status.write_text("not json")
    code, _, stderr = await shim(home, PROMPT)
    assert code == 1
    assert "hands is unknown" in stderr


async def test_a_hands_still_starting_has_not_served_its_socket_yet_and_costs_the_session_nothing(home: Home) -> None:
    beat(home, os.getpid(), timedelta(seconds=0), pipeline="starting")
    assert await shim(home, ASK) == (0, "", "")


async def test_a_relative_home_is_refused_rather_than_made_in_every_project(home: Home, tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", SHIM_MODULE, cwd=tmp_path, env={**os.environ, "HANDS_HOME": "relhome"},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate(json.dumps(START).encode())
    assert process.returncode == 1
    assert "HANDS_HOME must be an absolute path, got 'relhome'" in stderr.decode()
    assert list(tmp_path.iterdir()) == []


async def test_a_hands_that_says_it_is_up_but_does_not_answer_is_reported(home: Home) -> None:
    beat(home, os.getpid(), timedelta(seconds=0))
    code, _, stderr = await shim(home, PROMPT)
    assert code == 1
    assert "cannot reach the hands daemon" in stderr and "hands is up" in stderr


def launch(plugin: Path, home: Home, cwd: Path, **environment: str) -> subprocess.CompletedProcess[bytes]:
    """The plugin's hook as Claude Code spawns it: its launcher, run directly in the session's directory."""
    environment = os.environ | {"HANDS_HOME": str(home.root), "PATH": SYSTEM_PATH, **environment}
    return subprocess.run([plugin / LAUNCHER, "-m", SHIM_MODULE], input=json.dumps(START).encode(), env=environment, cwd=cwd, capture_output=True)


def test_the_plugin_launcher_runs_the_shim_on_hands_own_python_as_the_process_claude_code_spawned(plugin: Path, home: Home, tmp_path: Path) -> None:
    ran = launch(plugin, home, tmp_path)
    assert (ran.returncode, ran.stdout, ran.stderr) == (0, STARTED.encode(), b"")
    # Spawned directly, as Claude Code spawns an exec-form hook, so the pid recorded must be this process's.
    assert read_membership(home, SID).pid == os.getpid()


def test_a_project_s_own_modules_cannot_stand_in_for_the_shim_s(plugin: Path, home: Home, tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "hands").mkdir(parents=True)
    (project / "json.py").write_text("raise SystemExit('shadowed json')\n")
    (project / "hands" / "__init__.py").write_text("raise SystemExit('shadowed hands')\n")
    # Neither from the session's directory nor from a PYTHONPATH the user's shell set.
    ran = launch(plugin, home, project, PYTHONPATH=str(project))
    assert (ran.returncode, ran.stdout, ran.stderr) == (0, STARTED.encode(), b"")


async def test_a_start_records_membership_and_joins_the_registry(home: Home, sessions: Sessions) -> None:
    assert await shim(home, START) == (0, STARTED, "")
    assert await shim(home, PROMPT) == (0, "", "")
    membership = Membership(SID, pid=os.getpid(), cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl"))
    assert [listing.session for listing in sessions.live()] == [Session(membership, Unreported(), mode=None, turn=Opened(PromptId("p1")))]
    assert home.membership(SID).exists()


async def test_a_session_running_before_the_plugin_joins_on_its_first_hook_and_its_turn_is_told(home: Home, sessions: Sessions) -> None:
    """/reload-plugins in a running session fires no start hook: its first prompt is where hands hears of it."""
    assert await shim(home, PROMPT) == (0, "", "")
    membership = Membership(SID, pid=os.getpid(), cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl"))
    assert [listing.session for listing in sessions.live()] == [Session(membership, Unreported(), mode=None, turn=Opened(PromptId("p1")))]
    assert read_membership(home, SID) == membership
    assert await shim(home, STOP) == (0, "", "")
    assert await asyncio.wait_for(sessions.story(), 1.0) == Summarise(SID, PromptId("p1"), "done")


async def test_a_stop_is_decided_on_the_status_and_transcript_claude_code_wrote_before_it_however_soon_it_came(home: Home, tmp_path: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] no read on the period came between the prompt and the Stop: the Stop names the
    # id Claude went on under, which only the transcript says, and it is read through the hook first.
    config = tmp_path / "claude"
    transcript = config / "projects" / "-code-a" / f"{SID}.jsonl"
    transcript.parent.mkdir(parents=True)
    located = {"transcript_path": str(transcript)}
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None)
    runner = await serve_hooks(home, registry, Names(), lambda _: None, Reader(registry, Tails(registry), registry.apply))
    try:
        await shim(home, {**START, **located})
        await shim(home, {**PROMPT, **located})
        # Under the process the shim recorded the session as.
        pid = read_membership(home, SID).pid
        status = config / "sessions" / f"{pid}.json"
        status.parent.mkdir()
        status.write_text(json.dumps({"pid": pid, "sessionId": SID, "status": "busy", "statusUpdatedAt": 1000}))
        said = [("p1", "hi", "On it."), ("p2", "and then?", "done")]
        # Compact, as Claude Code writes its records.
        transcript.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for prompt, asked, answer in said for record in (
            {"type": "user", "promptId": prompt, "message": {"role": "user", "content": asked}},
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]}},
        )))
        assert await shim(home, {**STOP, **located, "prompt_id": "p2"}) == (0, "", "")
        assert await asyncio.wait_for(registry.story(), 1.0) == Summarise(SID, PromptId("p2"), "done")
    finally:
        await runner.cleanup()


async def test_a_later_hook_leaves_the_membership_its_start_wrote(home: Home, sessions: Sessions) -> None:
    await shim(home, START, fritter="/tmp/fritter-abc.sock")
    await shim(home, {**PROMPT, "cwd": "/code/a/sub"})
    assert read_membership(home, SID).fritter == Path("/tmp/fritter-abc.sock")
    assert read_membership(home, SID).cwd == Path("/code/a")


def _entered_worktree(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """A session's start, its transcript written, and the EnterWorktree call's PostToolUse once Claude Code has renamed
    the transcript into the worktree's project, naming it there."""
    started, moved = tmp_path / "-code-a" / "t.jsonl", tmp_path / "-code-a--claude-worktrees-lane" / "t.jsonl"
    started.parent.mkdir()
    started.write_text("{}\n")
    moved.parent.mkdir()
    return {**START, "transcript_path": str(started)}, {**COMMON, "hook_event_name": "PostToolUse", "tool_name": "EnterWorktree", "tool_input": {"name": "lane"}, "transcript_path": str(moved), "cwd": "/code/a/.claude/worktrees/lane"}


async def test_a_hook_naming_the_transcript_claude_code_moved_moves_the_session_there_and_says_so(home: Home, tmp_path: Path) -> None:
    """Claude Code renames the transcript into the worktree's project as a session enters one, and every hook
    after names it there, the EnterWorktree call's own PostToolUse first. The file and the registry follow it; the cwd
    stays the start's, since a hook's cwd is the shell's, which a `cd` moves too."""
    recorded: list[Entry] = []
    registry = Sessions(clock=lambda: 10.0, record=recorded.append)
    runner = await serve_hooks(home, registry, Names(), lambda _: None, Reader(registry, Tails(registry), registry.apply))
    start, entered = _entered_worktree(tmp_path)
    moved = str(entered["transcript_path"])
    try:
        await shim(home, start, fritter="/tmp/fritter-abc.sock")
        Path(str(start["transcript_path"])).rename(moved)
        assert await shim(home, entered, fritter="/tmp/fritter-abc.sock") == (0, "", "")
    finally:
        await runner.cleanup()
    membership = Membership(SID, pid=os.getpid(), cwd=Path("/code/a"), transcript=Path(moved), fritter=Path("/tmp/fritter-abc.sock"))
    assert read_membership(home, SID) == membership
    assert [listing.session.membership for listing in registry.live()] == [membership]
    # [LAW:nothing-unseen] the move is applied as its own event, naming where the session's transcript is now.
    assert [entry.facts["applied"] for entry in recorded if isinstance(entry, WideEvent) and isinstance(entry.facts.get("applied"), Located)] == [Located(membership)]


async def test_a_background_hook_landing_after_the_move_leaves_the_session_where_its_transcript_is(home: Home, sessions: Sessions, tmp_path: Path) -> None:
    """PostToolUse runs in the background, so the hook of a call made before EnterWorktree can land after the hook that
    named the moved transcript, naming the path the rename left. The file, and the session, stay where the transcript is."""
    start, entered = _entered_worktree(tmp_path)
    await shim(home, start)
    Path(str(start["transcript_path"])).rename(str(entered["transcript_path"]))
    await shim(home, entered)
    assert await shim(home, {**start, "hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}}) == (0, "", "")
    assert read_membership(home, SID).transcript == Path(str(entered["transcript_path"]))
    assert [listing.session.membership.transcript for listing in sessions.live()] == [Path(str(entered["transcript_path"]))]


async def test_a_late_hook_of_a_session_its_process_has_moved_on_from_writes_nothing(home: Home, sessions: Sessions) -> None:
    """After a /clear the start's file names the process, and it must stay the newest file naming it, or the sweep would
    take the old session for the one the process holds and end the new one."""
    cleared = SessionId("0f1e2d3c-aaaa-bbbb-cccc-00000000000c")
    await shim(home, {**START, "session_id": cleared})
    await shim(home, {**END, "session_id": cleared})
    await shim(home, START)
    assert await shim(home, {**PROMPT, "session_id": cleared}) == (0, "", "")
    assert not home.membership(cleared).exists()
    assert [listing.session.membership.id for listing in sessions.live()] == [SID]


async def test_a_hook_that_lands_after_its_session_ended_does_not_bring_it_back(home: Home, sessions: Sessions) -> None:
    """A shim racing its session's end can write the file again; the registry holds the session as gone, so attaching it
    is nothing, and the sweep that finds its process over says nothing either."""
    await shim(home, START)
    await shim(home, END)
    assert await shim(home, PROMPT) == (0, "", "")
    await sweep(home, sessions, frozenset())
    assert sessions.live() == []
    assert sessions.live_session(SID) is None


async def test_a_file_a_dead_process_left_under_this_session_is_replaced_on_its_first_hook(home: Home, sessions: Sessions, dead_pid: Callable[[], int]) -> None:
    """A session killed while hands was off, resumed with the plugin off, and joined by /reload-plugins."""
    write_membership(home, Membership(SID, pid=dead_pid(), cwd=Path("/code/a"), transcript=Path("/nowhere/t.jsonl")))
    assert await shim(home, PROMPT) == (0, "", "")
    assert read_membership(home, SID).pid == os.getpid()
    assert [listing.session.membership.pid for listing in sessions.live()] == [os.getpid()]


async def test_a_file_an_ended_session_left_under_this_process_pid_does_not_stop_it_joining(home: Home, sessions: Sessions) -> None:
    """A session that ended while hands was off left its file, and its pid came round to this one."""
    ended = SessionId("0f1e2d3c-aaaa-bbbb-cccc-00000000000e")
    write_membership(home, Membership(ended, pid=os.getpid(), cwd=Path("/code/old"), transcript=Path("/nowhere/old.jsonl")))
    os.utime(home.membership(ended), (0, 0))
    assert await shim(home, PROMPT) == (0, "", "")
    assert read_membership(home, SID).pid == os.getpid()


async def test_an_end_removes_membership_and_leaves_the_listing(home: Home, sessions: Sessions) -> None:
    await shim(home, START)
    assert await shim(home, END) == (0, "", "")
    assert not home.membership(SID).exists()
    assert sessions.live() == []


async def test_a_hook_the_daemon_refuses_exits_nonzero_with_its_reason(home: Home, sessions: Sessions) -> None:
    code, _, stderr = await shim(home, {**COMMON, "hook_event_name": "Notification"})
    assert code == 1
    assert "refused this hook (400): hook event 'Notification' is not one hands handles" in stderr
    assert sessions.live() == []


async def test_each_hook_posted_is_one_event_saying_how_it_was_answered(home: Home) -> None:
    recorded: list[Entry] = []
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None, stop_hold=0.2)
    runner = await serve_hooks(home, registry, Names(), recorded.append, Reader(registry, Tails(registry), registry.apply))
    try:
        await shim(home, START)
        await shim(home, {**COMMON, "hook_event_name": "Notification"})
        await shim(home, PROMPT)
        # Nothing is waiting to be read: the Stop is decided as it is heard.
        await shim(home, STOP)
        await shim(home, {**PROMPT, "prompt_id": "p2"})
        # A Stop for a turn whose prompt hands never heard waits for the transcript to say whose it is, read once a
        # status is; no record comes, so the hold passes.
        await registry.apply(StatusReported(SID, Report(Busy(), Stamp(1000)), at=10.0))
        await shim(home, {**STOP, "prompt_id": "p3"})
    finally:
        await runner.cleanup()
    rejected = "rejected hook: hook event 'Notification' is not one hands handles; a session that loaded hands' hooks before hands stopped hooking it takes the current ones with /reload-plugins"
    assert hooks(recorded) == [
        ("ok", None, {"hook": "SessionStart"}),
        ("failed", rejected, {}),
        ("ok", None, {"hook": "UserPromptSubmit", "name": None}),
        ("ok", None, {"hook": "Stop", "stop": "decided"}),
        ("ok", None, {"hook": "UserPromptSubmit", "name": None}),
        ("ok", None, {"hook": "Stop", "stop": "let go"}),
    ]
    [let_go] = [entry for entry in recorded if isinstance(entry, WideEvent) and entry.facts.get("stop") == "let go"]
    # The event lasts as long as the hook held its session: the whole hold.
    assert let_go.duration_ms >= 200 and let_go.facts["session"] == SID


async def test_a_socket_left_by_a_dead_daemon_is_reclaimed(home: Home) -> None:
    # A daemon that died without cleaning up leaves a socket file nothing listens on.
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(home.socket))
    dead.close()
    assert home.socket.exists()
    registry = Sessions(clock=lambda: 0.0, record=lambda _: None)
    runner = await serve_hooks(home, registry, Names(), lambda _: None, Reader(registry, Tails(registry), registry.apply))
    try:
        assert await shim(home, START) == (0, STARTED, "")
        assert len(registry.live()) == 1
    finally:
        await runner.cleanup()


async def test_a_session_started_under_fritter_records_where_to_type_into_it(home: Home, sessions: Sessions) -> None:
    # fritter publishes its socket in the environment of the process it wrapped, and this
    # hook is a child of that process, so the address arrives without either side naming
    # a path the other has to guess.
    assert await shim(home, START, fritter="/tmp/fritter-abc.sock") == (0, STARTED, "")
    [listing] = sessions.live()
    assert listing.session.membership.fritter == Path("/tmp/fritter-abc.sock")


async def test_a_session_started_outside_fritter_records_no_way_to_type_into_it(home: Home, sessions: Sessions) -> None:
    assert await shim(home, START) == (0, STARTED, "")
    [listing] = sessions.live()
    assert listing.session.membership.fritter is None


async def test_an_empty_fritter_address_is_no_address(home: Home, sessions: Sessions) -> None:
    # An exported-but-empty variable is how a shell hands on a value it does not have.
    assert await shim(home, START, fritter="") == (0, STARTED, "")
    [listing] = sessions.live()
    assert listing.session.membership.fritter is None


TAP = "http://127.0.0.1:40000"
TAPPED = {"FRITTER_TAP": TAP, "HTTPS_PROXY": TAP, "https_proxy": TAP, "HTTP_PROXY": TAP, "http_proxy": TAP, "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem", "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128"}


async def test_a_tapped_session_s_commands_run_with_what_its_tap_replaced_given_back(home: Home, tmp_path: Path) -> None:
    # Claude Code sources this file before each of the session's commands, in the session's environment.
    commands = tmp_path / "sessionstart-hook-0.sh"
    assert await shim(home, START, given={**TAPPED, "CLAUDE_ENV_FILE": str(commands)}) == (0, STARTED, "")
    ran = subprocess.run(
        ["/bin/sh", "-c", f'. {commands}; echo "$HTTPS_PROXY ${{http_proxy-unset}} ${{NODE_EXTRA_CA_CERTS-unset}} ${{FRITTER_TAP-unset}}"'],
        env=TAPPED, capture_output=True, text=True, check=True,
    )
    assert ran.stdout == "http://corp:3128 unset unset unset\n"


async def test_an_untapped_session_s_commands_are_given_nothing(home: Home, tmp_path: Path) -> None:
    commands = tmp_path / "sessionstart-hook-0.sh"
    assert await shim(home, START, given={"HTTPS_PROXY": "http://corp:3128", "CLAUDE_ENV_FILE": str(commands)}) == (0, STARTED, "")
    assert not commands.exists()


async def test_a_finished_turn_has_its_session_named_and_the_name_is_handed_to_claude_code_at_the_next_prompt_once(home: Home) -> None:
    names = Names()
    given: list[Entry] = []
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None, stop_hold=0.05)
    runner = await serve_hooks(home, registry, names, given.append, Reader(registry, Tails(registry), registry.apply))
    try:
        await shim(home, START)
        await shim(home, PROMPT)
        assert await shim(home, STOP) == (0, "", "")
        # The closing reply is what the name is judged from, with the transcript that holds the name it has now.
        finished = await asyncio.wait_for(names.next_finished(), 1.0)
        assert (finished.membership.id, finished.membership.transcript, finished.closing) == (SID, Path("/nowhere/t.jsonl"), "done")
        names.rename(SID, "naming fix", None)
        named = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "sessionTitle": "naming fix"}}
        code, stdout, stderr = await shim(home, {**PROMPT, "prompt_id": "p2"})
        assert (code, json.loads(stdout), stderr) == (0, named, "")
        # Given once: the next prompt sets nothing, and Claude Code keeps the name it holds.
        assert await shim(home, {**PROMPT, "prompt_id": "p3"}) == (0, "", "")
        prompts = [facts for _, _, facts in hooks(given) if facts["hook"] == "UserPromptSubmit"]
        assert prompts == [{"hook": "UserPromptSubmit", "name": name} for name in (None, NameGiven("naming fix"), None)]
    finally:
        await runner.cleanup()


async def test_a_name_set_since_hands_decided_one_is_not_overwritten_by_it(home: Home, tmp_path: Path) -> None:
    names = Names()
    given: list[Entry] = []
    transcript = tmp_path / "t.jsonl"
    # The user gave the session a name with /rename after hands decided its own against the session having none.
    transcript.write_text('{"type":"custom-title","customTitle":"my thing","sessionId":"s"}\n')
    at = {"transcript_path": str(transcript)}
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None)
    runner = await serve_hooks(home, registry, names, given.append, Reader(registry, Tails(registry), registry.apply))
    try:
        await shim(home, {**START, **at})
        names.rename(SID, "naming fix", None)
        assert await shim(home, {**PROMPT, **at}) == (0, "", "")
        assert hooks(given)[-1] == ("ok", None, {"hook": "UserPromptSubmit", "name": NameWithheld("naming fix", None, "my thing")})
        assert names.due(SID) is None
    finally:
        await runner.cleanup()


async def test_a_name_is_not_given_over_a_title_that_cannot_be_read(home: Home, tmp_path: Path) -> None:
    names = Names()
    given: list[Entry] = []
    # A directory where the transcript should be: reading the session's title fails.
    at = {"transcript_path": str(tmp_path)}
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None)
    runner = await serve_hooks(home, registry, names, given.append, Reader(registry, Tails(registry), registry.apply))
    try:
        await shim(home, {**START, **at})
        names.rename(SID, "naming fix", None)
        assert await shim(home, {**PROMPT, **at}) == (0, "", "")
        [(outcome, error, facts)] = hooks(given)[-1:]
        assert (outcome, facts) == ("failed", {"hook": "UserPromptSubmit", "name": NameUnread("naming fix", None)})
        assert error is not None and error.startswith(f"cannot read the name of session {SID} from {tmp_path}, so 'naming fix' is not given:")
    finally:
        await runner.cleanup()


async def test_a_turn_that_closed_on_no_reply_is_not_named(home: Home) -> None:
    names = Names()
    registry = Sessions(clock=lambda: 10.0, record=lambda _: None, stop_hold=0.05)
    runner = await serve_hooks(home, registry, names, lambda _: None, Reader(registry, Tails(registry), registry.apply))
    try:
        await shim(home, START)
        await shim(home, PROMPT)
        await shim(home, {key: value for key, value in STOP.items() if key != "last_assistant_message"})
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(names.next_finished(), 0.2)
    finally:
        await runner.cleanup()
