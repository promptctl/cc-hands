"""The brain: hands' tools over MCP as a Claude Code client reaches them, the brain's process as hands drives it, and the
side questions hands asks of a Claude Code of their own."""

from types import SimpleNamespace
import asyncio
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest
from loguru import logger

from hands.core.playback import Bookmark
from hands.brain import pinned, process
from hands.brain.mcp import TOOL_USE_ID, CallSpans, McpServer, serve_mcp
from hands.brain.asides import AsideFailed, Deadline, Unanswered, Within
from hands.sessions.reader import Reader
from hands.sessions.tail import Tails
from hands.brain.process import ClaudeTurns, BROKEN, HOOKS, LET_GO, NOBODY, STOPPED, SLIM, INTERRUPT_SECONDS, TAKE_SECONDS, UNANSWERED, UNREAD, UNVOICED, Asked, Brain, BrainAnswered, BrainEnded, BrainGone, BrainStopped, Fresh, Launch, NotLoggedIn, Resumed, Station, StopUnrecorded, Unstartable, Untaken, _listen, _Turn, Mode, account_kept_out, command, conversation, model, settings_mode, environment, logged_in, spawn, start, unanswered, workdir  # pyright: ignore[reportPrivateUsage]
from hands.core.effects import Allow, Deny, Effect, HookReply, Submitted, Summarise
from hands.sessions.hooks import hook_output
from hands.core.permissions import heard
from hands.core.session import DenyAndInterrupt, Held, Permission, RequestId
from hands.core.stopfailure import StopFailed
from hands.core.attention import Attention
from hands.sessions.hookconfig import PERMISSION_HOOK_TIMEOUT_SECONDS, POST_TIMEOUT_SECONDS
from hands.sessions.audit import Entry, segment
from conftest import events, onboard, patiently, pin, pinned_to, unread
from turns import RECORDINGS, prompts, stops

from hands.brain.stage import BrainStage
from hands.core.events import Abandoned, Dequeued, Ending, Located, Prompted, Removed, StatusReported, Stopped, Taken, Unqueued
from hands.core.status import Busy, Idle, Report, Stamp
from hands.core.session import Keystroke, Membership, PromptId, PromptText, SessionId, pasted
from hands.core.wire import Exchanged, Fork, Heard, Kind, MainTurn, Message, MessageStarted, Reached, Sent, Written
from hands.core.wire import Text as Said
from hands.daemon.cli import main
from hands.daemon import run
from hands.daemon.run import mind
from hands.daemon.starting import CannotStart
from hands.sessions.proxy import Wire
from hands.sessions.sentences import Sentences
from hands.sessions.typing import Typist
from hands.sessions import wrapper
from hands.sessions.wrapper import MARK
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.voice.refocus import Refocus
from hands.voice.sentences import SummaryStore
from hands.voice.backends import Account, ClaudeCodeBackend
from hands.voice.pipeline import VoiceConfig
from hands.voice.summary import SummaryFailed, aside
from hands.sessions.wide import Fact, WideEvent, begun, here, root, unit, within
from hands.voice.tool import Result, tool
from hands.voice.tools import Called, audited
from hands.voice import voices


def never_cut() -> asyncio.Future[Bookmark]:
    """The speaker's reading at a barge-in, which a test with no speaker never has."""
    return asyncio.get_running_loop().create_future()


async def echo(text: str, times: int = 1) -> Result:
    """Say it back.

    Args:
        text: What to say.
        times: How many times.
    """
    return {"said": " ".join([text] * times)}


async def broken() -> Result:
    """Fail."""
    raise RuntimeError("the transcript went away")


async def rpc(server: McpServer, message: dict[str, object]) -> tuple[int, object]:
    async with aiohttp.ClientSession() as client, client.post(server.url, json=message, headers={"Authorization": f"Bearer {server.token}"}) as reply:
        return reply.status, (await reply.json() if reply.status == 200 else None)


def ran(recorded: Sequence[Entry]) -> list[tuple[str, str, Mapping[str, Fact]]]:
    return [(entry.event, entry.outcome, entry.facts) for entry in recorded if isinstance(entry, WideEvent)]


async def test_a_client_opens_lists_and_calls_the_tools_and_each_request_is_one_event() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append, CallSpans())
    try:
        status, opened = await rpc(server, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "claude-code"}}})
        assert status == 200 and opened == {
            "jsonrpc": "2.0", "id": 0,
            "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "hands", "version": "0"}},
        }
        assert (await rpc(server, {"jsonrpc": "2.0", "method": "notifications/initialized"}))[0] == 202
        _, listed = await rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert listed == {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{
            "name": "echo", "description": "Say it back.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string", "description": "What to say."}, "times": {"type": "integer", "description": "How many times."}}, "required": ["text"]},
        }]}}
        _, called = await rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi", "times": 2}}})
        assert called == {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": '{"said": "hi hi"}'}], "isError": False}}
        assert ran(recorded) == [
            ("mcp.request", "ok", {"method": "initialize", "client": "claude-code", "client_version": None, "protocol": "2025-06-18"}),
            ("mcp.request", "ok", {"method": "tools/list"}),
            ("tool.run", "ok", {"tool": "echo", "called": Called({"text": "hi", "times": 2}, {"said": "hi hi"})}),
        ]
    finally:
        await server.close()


async def test_a_call_the_tool_cannot_answer_is_told_to_the_model_and_one_the_server_cannot_is_an_error() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append), audited(tool(broken), recorded.append)], recorded.append, CallSpans())
    errors, sink = failures()
    try:
        _, wrong = await rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"words": "hi"}}})
        refusal = "echo was called with the wrong arguments: missing a required argument: 'text'"
        # A refusal is a result, as Claude Code hands it to the model unwrapped; its event is what ends failed.
        assert wrong == {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps({"error": refusal})}], "isError": False}}
        _, failed = await rpc(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "broken"}})
        assert failed == {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "broken failed: RuntimeError: the transcript went away"}], "isError": True}}
        # What the tool raised ends where the model is told, so it is said there, on the terminal, once.
        assert errors == ["broken failed: RuntimeError: the transcript went away"]
        _, unknown = await rpc(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "resume"}})
        assert unknown == {"jsonrpc": "2.0", "id": 3, "error": {"code": -32602, "message": "no tool resume taking arguments None"}}
        _, unshaped = await rpc(server, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "echo", "arguments": '{"text": "hi"}'}})
        assert unshaped == {"jsonrpc": "2.0", "id": 4, "error": {"code": -32602, "message": """no tool echo taking arguments '{"text": "hi"}'"""}}
        _, unasked = await rpc(server, {"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        assert unasked == {"jsonrpc": "2.0", "id": 5, "error": {"code": -32601, "message": "no method resources/list"}}
        # Claude Code asks it of every server it opens: the known no is no failure.
        _, discovered = await rpc(server, {"jsonrpc": "2.0", "id": 6, "method": "server/discover"})
        assert discovered == {"jsonrpc": "2.0", "id": 6, "error": {"code": -32601, "message": "no method server/discover"}}
        async with aiohttp.ClientSession() as client, client.post(server.url, data=b"{not json", headers={"Authorization": f"Bearer {server.token}"}) as garbled:
            assert (await garbled.json())["error"]["code"] == -32700
        async with aiohttp.ClientSession() as client, client.get(server.url, headers={"Authorization": f"Bearer {server.token}"}) as stream:
            assert stream.status == 405
        assert [(event, outcome, facts.get("tool", facts.get("method"))) for event, outcome, facts in ran(recorded)] == [
            ("tool.run", "failed", "echo"),
            ("tool.run", "failed", "broken"),
            ("mcp.request", "failed", "tools/call"),
            ("mcp.request", "failed", "tools/call"),
            ("mcp.request", "failed", "resources/list"),
            ("mcp.request", "ok", "server/discover"),
            ("mcp.request", "failed", None),
        ]
        [garbled_event] = [entry for entry in recorded if isinstance(entry, WideEvent)][-1:]
        assert garbled_event.error is not None and garbled_event.error.startswith("not JSON: ")
    finally:
        logger.remove(sink)
        await server.close()


async def test_a_call_a_turn_is_running_is_run_as_a_part_of_the_turns_call_and_one_no_turn_is_running_is_a_root() -> None:
    recorded: list[Entry] = []
    spans = CallSpans()
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append, spans)
    with unit("voice.turn", recorded.append):
        call = within(here())
    spans.opened("toolu_1", call)
    try:
        for id, used in enumerate(("toolu_1", "toolu_2")):
            await rpc(server, {"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}, "_meta": {TOOL_USE_ID: used}}})
        spans.ended(["toolu_1"])
        await rpc(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}, "_meta": {TOOL_USE_ID: "toolu_1"}}})
    finally:
        await server.close()
    _, in_turn, outside, after = recorded
    assert isinstance(in_turn, WideEvent) and (in_turn.trace_id, in_turn.parent_id) == (call.trace_id, call.span_id)
    assert all(isinstance(run, WideEvent) and run.parent_id is None and run.trace_id != call.trace_id for run in (outside, after))


async def test_a_request_without_the_brains_token_reaches_no_tool() -> None:
    recorded: list[Entry] = []
    server = await serve_mcp([audited(tool(echo), recorded.append)], recorded.append, CallSpans())
    call ={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}}
    try:
        async with aiohttp.ClientSession() as client:
            # What a web page can send with no preflight: a text/plain POST, with no token or the wrong one.
            for headers in ({"Content-Type": "text/plain"}, {"Authorization": "Bearer guessed"}):
                async with client.post(server.url, data=json.dumps(call), headers=headers) as reply:
                    assert reply.status == 401
        assert recorded == []
    finally:
        await server.close()


def station(tmp: Path) -> Station:
    return Station(tmp / "brain", workdir(tmp / "brain"), "claude-sonnet-5", "http://127.0.0.1:1", dict(os.environ))


def unasked(asked: Asked) -> None:
    """A turn's teller of permissions, for a turn that asks none."""
    pytest.fail(f"a turn that asks nothing held {asked.permission}")


def launch(tmp: Path) -> Launch:
    return Launch(station(tmp), Account("claude.ai", "brain@example.com"), "You are hands.", '{"mcpServers": {}}', Fresh(SessionId("b1"), None))


def typed(tmp: Path) -> list[list[str]]:
    """What the fake claude read from its terminal, in order."""
    path = tmp / "typed.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def typings_answered(tmp: Path) -> Callable[[float], Awaitable[None]]:
    """A take's time for a brain the stand-in is: the n-th is up once the stand-in is done with n typings, however long
    that takes and never on a clock, so a turn not taken is one it did not take. pytest's timeout keeps it from hanging."""
    begun = 0

    async def pause(seconds: float) -> None:
        nonlocal begun
        if seconds != TAKE_SECONDS:
            # A stop's wait for Claude Code to hold nothing of what it stopped, on its own bound: no typing's.
            await asyncio.sleep(seconds)
            return
        begun += 1
        waited, path = begun, tmp / "answered"
        while (len(path.read_text()) if path.exists() else 0) < waited:
            await asyncio.sleep(0.01)

    return pause


async def until(check: Callable[[], bool]) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("never")


def running(tmp: Path) -> list[str]:
    """The fake Claude Codes still running."""
    return subprocess.run(["pgrep", "-f", str(tmp / "bin" / "claude")], capture_output=True, text=True).stdout.split()


def answered(session: str, words: str, stop: str = "end_turn", exchange: str = "x1", kind: Kind = Fork()) -> Exchanged:
    message = Message("m1", "claude-sonnet-5", (Said(words),) if words else (), stop, {})
    return Exchanged(exchange, SessionId(session), kind, None, "POST", "/v1/messages", 10, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 10, Written(message, True)), False, root())


def test_the_brain_is_interactive_on_its_own_setup_beside_hands_server_and_its_own_login_through_the_proxy(tmp_path: Path) -> None:
    argv = command(launch(tmp_path), Path("/real/claude"), "http://127.0.0.1:7", None)
    # The real claude, interactive: no -p, and no pipe to speak over.
    assert argv[0] == "/real/claude" and "-p" not in argv and "--print" not in argv
    # Its tools, what it may do without asking, and its MCP servers are its config directory's: nothing here narrows them.
    assert not {"--tools", "--disallowedTools", "--permission-mode", "--strict-mcp-config", "--system-prompt", "--bare"} & set(argv)
    assert argv[argv.index("--allowedTools") + 1 : argv.index("--plugin-dir")] == ["mcp__hands", "Skill(hands:chat)", "Skill(hands:close)", "Skill(hands:prompt)", "Skill(hands:recall)", "Skill(hands:start)", "Skill(hands:update)"]
    assert argv[argv.index("--mcp-config") + 1] == launch(tmp_path).mcp_config
    # Beside its setup's skills, hands gives it its own: how it talks with the user, and how it writes a session's prompt.
    plugin = Path(argv[argv.index("--plugin-dir") + 1])
    assert json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())["name"] == "hands"
    for skill in ("chat", "prompt"):
        assert (plugin / "skills" / skill / "SKILL.md").read_text().startswith(f"---\nname: {skill}\n")
    assert [argv[argv.index(flag) + 1] for flag in ("--setting-sources", "--append-system-prompt", "--session-id")] == ["user", "You are hands.", "b1"]
    # The instruction this hands writes goes on every request, never the one its conversation first had.
    assert argv[argv.index("--system-prompt-snapshot") + 1] == "off"
    settings = json.loads(argv[argv.index("--settings") + 1])
    # hands' tools are offered to the model directly, never deferred behind ToolSearch: set where neither the brain's
    # settings.json env nor hands' environment outranks it.
    assert settings["env"] == {"ENABLE_TOOL_SEARCH": "false"}
    assert "permissions" not in settings
    hooks = settings["hooks"]
    assert hooks.pop("Elicitation") == [{"hooks": [{"type": "http", "url": "http://127.0.0.1:7/Elicitation"}]}]
    # The order of the brain's turns rests on its prompt hook and its Stop being waited on: a prompt's as long as a typed
    # turn has to be taken, and a Stop's past the hold the model puts on it, as the shim's outlives a working session's.
    waits = {"UserPromptSubmit": TAKE_SECONDS + POST_TIMEOUT_SECONDS, "Stop": TAKE_SECONDS + POST_TIMEOUT_SECONDS, "StopFailure": TAKE_SECONDS + POST_TIMEOUT_SECONDS}
    assert {event: hooks.pop(event) for event in waits} == {event: [{"hooks": [{"type": "http", "url": f"http://127.0.0.1:7/{event}", "timeout": wait}]}] for event, wait in waits.items()}
    # A permission is held while the user is asked, as long as a working session's is.
    assert hooks == {"PermissionRequest": [{"hooks": [{"type": "http", "url": "http://127.0.0.1:7/PermissionRequest", "timeout": PERMISSION_HOOK_TIMEOUT_SECONDS}]}]}
    env = environment(tmp_path / "brain", "http://127.0.0.1:1", {
        "PATH": "/bin",
        # A command its setup's skill runs, firecrawl among them, is logged in by its own key or by what it stored under HOME.
        "HOME": "/home/u",
        "FIRECRAWL_API_KEY": "fc",
        "ANTHROPIC_API_KEY": "sk",
        "CLAUDE_CODE_OAUTH_TOKEN": "t",
        "ANTHROPIC_BASE_URL": "http://elsewhere",
        # A provider other than Anthropic's API is the brain's settings' to choose, never the shell hands was started in.
        "CLAUDE_CODE_USE_BEDROCK": "1",
        # A daemon started inside a tapped session inherits the session's tap; the brain is not that session.
        "FRITTER_TAP": "http://127.0.0.1:40000",
        "HTTPS_PROXY": "http://127.0.0.1:40000",
        "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem",
        "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128",
    })
    assert env == {"PATH": "/bin", "HOME": "/home/u", "FIRECRAWL_API_KEY": "fc", "HTTPS_PROXY": "http://corp:3128", **SLIM, **wrapper.EVERY_RUN, "CLAUDE_CONFIG_DIR": str(tmp_path / "brain"), "ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}
    # The account's claude.ai connectors stay out of every request, whatever the brain's own setup names.
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    # A background task's notification opens a turn of its own, which `ClaudeTurns` owns; no scheduled prompt opens one.
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" not in env and env["CLAUDE_CODE_DISABLE_CRON"] == "1"


@pytest.mark.parametrize("given", [{}, {"DISABLE_AUTOUPDATER": "0"}], ids=["unset", "on"])
async def test_every_claude_code_of_hands_own_runs_with_its_auto_updater_off(tmp_path: Path, given: dict[str, str]) -> None:
    # The brain and each side question are spawned on a terminal of hands' own; `hands login` and the brain's
    # `claude auth status` run in the same environment.
    recorder = tmp_path / "claude"
    recorder.write_text('#!/bin/sh\nprintf "updater=%s\\n" "${DISABLE_AUTOUPDATER-unset}"\n')
    recorder.chmod(0o755)
    inherited = {"PATH": "/usr/bin:/bin", **given}
    assert environment(tmp_path / "brain", "http://127.0.0.1:1", inherited)["DISABLE_AUTOUPDATER"] == "1"
    running = await spawn(Station(tmp_path / "brain", tmp_path, "claude-sonnet-5", "http://127.0.0.1:1", inherited), [str(recorder)])
    assert await running.exit == 0
    assert running.shown() == "updater=1"


async def test_a_turn_is_typed_behind_a_space_and_ends_at_its_stop_hook_and_the_brains_launch_turns_and_run_are_one_event_each(tmp_path: Path, fake_claude: Path, fritter: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await brain.ask("what is running?", unasked) == BrainAnswered("p1", None)
        assert await brain.ask("/and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert (tmp_path / "brain" / "cwd").is_dir()
    # Behind a space, so a turn that opens with a slash is the words it is and not a command.
    assert typed(tmp_path) == [["prompt", " what is running?"], ["prompt", " /and now?"]]
    # Each event the model applies to the brain's session is one of its own (`applied`), as a working session's is, beside these.
    launched, first, second, ran = [entry for entry in recorded if not (isinstance(entry, WideEvent) and entry.event == "applied")]
    assert isinstance(launched, WideEvent) and isinstance(first, WideEvent) and isinstance(second, WideEvent) and isinstance(ran, WideEvent)
    # Each turn is its own, with the prompt Claude Code took it as.
    assert [(turn.event, turn.outcome, turn.facts) for turn in (first, second)] == [
        ("brain.turn", "ok", {"taken": "p1", "queued": False, "offered": (), "told": "p1", "order_lost": None, "ids": ("p1",), "matched": "id"}),
        ("brain.turn", "ok", {"taken": "p2", "queued": False, "offered": (), "told": "p2", "order_lost": None, "ids": ("p2",), "matched": "id"}),
    ]
    assert (launched.event, launched.outcome, launched.parent_id) == ("brain.launch", "ok", None)
    assert launched.facts == {
        "session": "b1", "account": Account("claude.ai", "brain@example.com"), "model": "claude-sonnet-5", "config_dir": tmp_path / "brain", "cwd": tmp_path / "brain" / "cwd",
        "conversation": "fresh", "untranscribed": None, "fritter": fritter, "claude": fake_claude, "mode": None, "pid": brain.pid,
    }
    # Held once the brain is up, for the next start to resume, as Claude Code keeps the transcript of its turns.
    assert conversation(tmp_path / "brain", SessionId("b2")) == Resumed(SessionId("b1"))
    # Its run is a part of its launch, from its input coming up to its process's end.
    assert (ran.event, ran.outcome, ran.trace_id, ran.parent_id) == ("brain.run", "ok", launched.trace_id, launched.span_id)
    assert ran.facts["pid"] == brain.pid and isinstance(ran.facts["code"], int) and isinstance(ran.facts["shown"], str)


def transcript(config_dir: Path, session: str) -> None:
    """Claude Code's transcript of `session`, under the project directory it names the brain's cwd by."""
    project = config_dir / "projects" / "-brain-cwd"
    project.mkdir(parents=True, exist_ok=True)
    (project / f"{session}.jsonl").write_text("{}\n")


def test_a_brain_starts_new_until_one_has_held_a_conversation_and_resumes_it_while_claude_code_keeps_its_transcript(tmp_path: Path) -> None:
    brain = tmp_path / "brain"
    brain.mkdir()
    assert conversation(brain, SessionId("n1")) == Fresh(SessionId("n1"), None)
    (brain / "conversation").write_text("c1\n")
    # A conversation Claude Code has no transcript of has nothing to resume: the one held is named as untranscribed.
    assert conversation(brain, SessionId("n1")) == Fresh(SessionId("n1"), SessionId("c1"))
    transcript(brain, "c1")
    assert conversation(brain, SessionId("n1")) == Resumed(SessionId("c1"))
    # Resumed under its own session, which Claude Code keeps: never a session id of its own beside it.
    argv = command(replace(launch(tmp_path), conversation=Resumed(SessionId("c1"))), Path("/real/claude"), "http://127.0.0.1:7", None)
    assert argv[argv.index("--resume") + 1] == "c1" and "--session-id" not in argv


def test_the_mode_the_brains_settings_name_is_said_on_its_command_line_too_so_claude_code_offers_no_other(tmp_path: Path) -> None:
    """Claude Code offers to make auto mode the default to a session whose mode its user settings alone name, and the first
    turn typed into the offer answers Yes (promptctl-brain-lq3.mcw): the same mode said in --settings means no offer."""
    argv = command(launch(tmp_path), Path("/real/claude"), "http://127.0.0.1:7", "acceptEdits")
    assert json.loads(argv[argv.index("--settings") + 1])["permissions"] == {"defaultMode": "acceptEdits"}


def test_the_mode_the_brains_settings_name_is_read_as_claude_code_reads_it_and_one_it_does_not_take_refuses_the_start(tmp_path: Path) -> None:
    brain = tmp_path / "brain"
    brain.mkdir()
    settings = brain / "settings.json"
    assert settings_mode(brain) is None
    settings.write_text(json.dumps({"syncClaudeAiSkills": False}))
    assert settings_mode(brain) is None
    settings.write_text(json.dumps({"permissions": {"defaultMode": "plan", "allow": ["Bash(ls:*)"]}}))
    assert settings_mode(brain) == "plan"
    # Claude Code's other name for its default mode.
    settings.write_text(json.dumps({"permissions": {"defaultMode": "manual"}}))
    assert settings_mode(brain) == "default"
    settings.write_text(json.dumps({"permissions": {"defaultMode": "acceptedits"}}))
    with pytest.raises(Unstartable, match='name a defaultMode Claude Code does not take, "acceptedits"'):
        settings_mode(brain)
    settings.write_text(json.dumps({"permissions": {"defaultMode": 3}}))
    with pytest.raises(Unstartable, match="the brain's settings could not be read"):
        settings_mode(brain)
    settings.unlink()
    settings.mkdir()
    with pytest.raises(Unstartable, match="the brain's settings could not be read"):
        settings_mode(brain)


async def test_the_brain_starts_with_the_mode_its_settings_name_on_its_command_line_and_its_launch_says_which(tmp_path: Path, fake_claude: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir(tmp_path / "brain")
    (tmp_path / "brain" / "settings.json").write_text(json.dumps({"permissions": {"defaultMode": "manual"}}))
    told: list[Mode | None] = []

    def told_of(launched: Launch, claude: Path, hooks: str, mode: Mode | None) -> list[str]:
        told.append(mode)
        return command(launched, claude, hooks, mode)

    monkeypatch.setattr(process, "command", told_of)
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    await brain.stop()
    assert told == ["default"]
    [launched] = events(recorded, "brain.launch")
    assert launched.facts["mode"] == "default"


def permissions(recorded: Sequence[Entry]) -> list[tuple[Fact, Fact, Fact]]:
    """The permissions the brain's setup asked about, as the log says each was settled."""
    return [(event.facts["prompt"], event.facts["tool"], event.facts["decision"]) for event in events(recorded, "brain.permission")]


async def test_a_permission_the_brains_setup_asks_about_holds_its_turn_until_answered_and_only_a_yes_runs_the_tool(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    held: list[Asked] = []
    notes = tmp_path / "notes.txt"
    no = heard("No, leave it.")
    assert isinstance(no, Deny)
    brain = await start(launch(tmp_path), recorded.append)
    try:
        turn = asyncio.create_task(brain.ask("write", held.append))
        await until(lambda: len(held) == 1)
        assert held[0].permission == Permission("Write", {"file_path": str(notes), "content": "hello"})
        # Held: the turn waits on the user, and the tool has not run.
        await asyncio.sleep(0.3)
        assert not turn.done() and not notes.exists()
        held[0].settle(Allow())
        assert await asyncio.wait_for(turn, 10) == BrainAnswered("p1", None)
        assert notes.read_text() == "hello"
        notes.unlink()
        turn = asyncio.create_task(brain.ask("write", held.append))
        await until(lambda: len(held) == 2)
        held[1].settle(no)
        assert await asyncio.wait_for(turn, 10) == BrainAnswered("p2", None)
        assert not notes.exists()
    finally:
        await brain.stop()
    # The brain heard each decision, the no with the user's words; nothing was typed into it while it held one.
    assert typed(tmp_path) == [
        ["prompt", " write"], ["permission", "allow", "", False], ["elicitation", "decline"],
        ["prompt", " write"], ["permission", "deny", no.message, False], ["elicitation", "decline"],
    ]
    assert permissions(recorded) == [("p1", "Write", Allow()), ("p2", "Write", no)]
    assert [event.facts for event in events(recorded, "brain.elicitation")] == [{"prompt": "p1", "server": "probe"}, {"prompt": "p2", "server": "probe"}]


async def test_a_permission_nobody_answers_is_refused_before_its_hooks_day_ends(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.brain.process.BRAIN_REPLY_BY_SECONDS", 0.3)
    recorded: list[Entry] = []
    held: list[Asked] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        with unit("voice.turn", recorded.append):
            asking = here()
            turn = asyncio.create_task(brain.ask("write", held.append))
            await until(lambda: len(held) == 1)
            # The turn's request, read off the wire while the turn holds its permission.
            brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), None, {"messages": [], "tools": [{"name": "Write"}, {"name": "mcp__hands__read_session"}]}))
            assert await asyncio.wait_for(turn, 10) == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    assert (await held[0].decided()) == Deny(UNANSWERED)
    assert permissions(recorded) == [("p1", "Write", Deny(UNANSWERED))]
    # The brain's turn is a part of the voice turn that asked it, with the tools its request offered; the permission is a
    # part of the brain's turn it was held for, timed from its post to its answer.
    [brain_turn] = events(recorded, "brain.turn")
    assert (brain_turn.trace_id, brain_turn.parent_id) == (asking.trace_id, asking.span_id)
    assert brain_turn.facts == {"taken": "p1", "queued": False, "offered": ("Write", "mcp__hands__read_session"), "told": "p1", "order_lost": None, "ids": ("p1",), "matched": "id"}
    [permission] = events(recorded, "brain.permission")
    assert (permission.trace_id, permission.parent_id) == (asking.trace_id, brain_turn.span_id) and permission.duration_ms >= 300
    assert not (tmp_path / "notes.txt").exists()


async def test_a_permission_posted_under_another_turns_prompt_is_refused_and_never_asked(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    held: list[Asked] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await asyncio.wait_for(brain.ask("stray", held.append), 10) == BrainAnswered("p1", None)
    finally:
        await brain.stop()
    assert held == []
    assert permissions(recorded) == [("p0", "Write", Deny(NOBODY))]
    # No turn's own, so a part of the brain's launch.
    [launched] = events(recorded, "brain.launch")
    [permission] = events(recorded, "brain.permission")
    assert permission.parent_id == launched.span_id


async def test_a_dialog_between_turns_or_with_a_body_that_does_not_parse_is_answered_no_and_said() -> None:
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", recorded.append):
        launched = here()
    brain._launched = launched  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    listener, url = await _serving(brain)

    async def answered(event: str, body: bytes) -> dict[str, object]:
        async with aiohttp.ClientSession() as client, client.post(f"{url}/{event}", data=body) as reply:
            assert reply.status == 200
            return (await reply.json())["hookSpecificOutput"]

    try:
        assert await answered("Elicitation", b"not json") == {"hookEventName": "Elicitation", "action": "decline"}
        assert await answered("PermissionRequest", b"not json") == {"hookEventName": "PermissionRequest", "decision": {"behavior": "deny", "message": UNREAD}}
        # An MCP server asking while it connects, and a permission with no turn in flight to ask the user in: no prompt id.
        elicited = {"hook_event_name": "Elicitation", "session_id": "b1", "mcp_server_name": "probe", "message": "Which?"}
        assert (await answered("Elicitation", json.dumps(elicited).encode()))["action"] == "decline"
        permission = {"hook_event_name": "PermissionRequest", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
        assert (await answered("PermissionRequest", json.dumps(permission).encode()))["decision"] == {"behavior": "deny", "message": NOBODY}
        # A dialog of questions is never put to the user by voice, in a turn or not: the brain is told to ask in words.
        questions: dict[str, object] = {**permission, "tool_name": "AskUserQuestion", "tool_input": {"questions": []}}
        assert (await answered("PermissionRequest", json.dumps(questions).encode()))["decision"] == {"behavior": "deny", "message": UNVOICED}
    finally:
        await listener.cleanup()
    # Each is one event, a part of the brain's launch, the unreadable ones failed saying so.
    unread, elicited_event = events(recorded, "brain.elicitation")
    assert unread.outcome == "failed" and unread.error is not None and unread.error.startswith("hands could not read the Elicitation hook")
    assert (elicited_event.outcome, elicited_event.facts) == ("ok", {"prompt": None, "server": "probe"})
    unreadable, *read = events(recorded, "brain.permission")
    assert unreadable.outcome == "failed" and unreadable.error is not None and unreadable.error.startswith("hands could not read the permission request")
    assert permissions(read) == [(None, "Write", Deny(NOBODY)), (None, "AskUserQuestion", Deny(UNVOICED))]
    assert {event.parent_id for event in events(recorded, "brain.permission") + events(recorded, "brain.elicitation")} == {launched.span_id}


async def test_a_turn_the_api_fails_ends_at_its_stop_failure_hook_saying_what_failed_it(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert await brain.ask("fail", unasked) == BrainAnswered("p1", StopFailed("unknown", "API Error: 400 refused"))
        # An asker that stops waiting leaves the turn to run to its end, and its event still says what failed it.
        asking = asyncio.create_task(brain.ask("fail", unasked))
        await until(lambda: sum(line[0] == "prompt" for line in typed(tmp_path)) == 2)
        asking.cancel()
        await until(lambda: len(events(recorded, "brain.turn")) == 2)
        assert await brain.ask("and now?", unasked) == BrainAnswered("p3", None)
    finally:
        await brain.stop()
    failed, left, _ = events(recorded, "brain.turn")
    assert [(turn.outcome, turn.error, turn.facts["told"]) for turn in (failed, left)] == [
        ("failed", "the brain's turn ended in error: unknown: API Error: 400 refused", "p1"),
        ("failed", "the brain's turn ended in error: unknown: API Error: 400 refused", "p2"),
    ]


async def test_a_stop_failure_whose_kind_or_text_is_missing_empty_or_not_text_still_ends_its_turn(tmp_path: Path, fake_claude: Path) -> None:
    # Both only narrate the failure: refusing the hook over either would leave the turn waiting on an end already posted.
    brain = await start(launch(tmp_path), list[Entry]().append)
    said, sink = failures()
    try:
        assert await brain.ask("fail blank", unasked) == BrainAnswered("p1", StopFailed("authentication_failed", None))
        assert await brain.ask("fail odd", unasked) == BrainAnswered("p2", StopFailed("server_error", None))
        assert await brain.ask("fail unnamed", unasked) == BrainAnswered("p3", StopFailed(None, None))
    finally:
        logger.remove(sink)
        await brain.stop()
    assert said == ["StopFailure carried last_assistant_message as dict, not a string, so its turn is told without it"]


async def test_an_interrupt_stops_the_turn_in_flight_as_escape_does_and_with_none_types_nothing(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        assert brain.interrupt(brain.answering()) is False
        # A turn that runs until Escape stops it.
        asking = asyncio.ensure_future(brain.ask("wait", unasked))
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt"])
        assert brain.interrupt(brain.answering()) is True
        assert await asyncio.wait_for(asking, 10) == BrainStopped()
        assert brain.interrupt(brain.answering()) is False
        # Escape, then Ctrl-C, which empties whatever Escape put back in the input: read by the fake as it reads them.
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt", "escape", "ctrl_c"])
    finally:
        await brain.stop()
    [turn] = events(recorded, "brain.turn")
    # The turn its Escape stopped was told, under the id it ran under, before it was over.
    assert (turn.outcome, turn.facts["stopped"], turn.facts["taken"], turn.facts["told"]) == ("ok", True, "p1", "p1")


class _HeldClear:
    """A typist whose Ctrl-C goes in only once `going` is set, so the model reads what the Escape did before the stop is
    done pressing keys."""

    def __init__(self, typist: Typist, going: threading.Event) -> None:
        self._typist = typist
        self._going = going

    def type(self, text: PromptText) -> None:
        self._typist.type(text)

    def press(self, key: Keystroke) -> None:
        if key == "ctrl_c":
            assert self._going.wait(10)
        self._typist.press(key)


async def test_a_stopped_turn_ends_with_the_id_it_was_told_under_however_far_the_model_reads_ahead_of_its_telling(tmp_path: Path, fake_claude: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] the model has decided the stopped turn told, and the telling is still to be
    # performed, when the stop is done pressing keys: the turn is over only once the telling is heard, under its id.
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    going = threading.Event()
    brain._typist = _HeldClear(brain._typist, going)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    performing = asyncio.Event()
    try:
        asking = asyncio.ensure_future(brain.ask("wait", unasked))
        await until(lambda: brain._turn is not None and brain._turn.taken.done())  # pyright: ignore[reportPrivateUsage]
        perform = brain._sessions._perform  # pyright: ignore[reportPrivateUsage]

        async def told_late(effect: Effect) -> None:
            # A telling the model decided is performed only once the test lets it, and what is decided after it waits.
            if isinstance(effect, Summarise):
                await performing.wait()
            await perform(effect)

        brain._sessions._perform = told_late  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        cleared = brain._cleared  # pyright: ignore[reportPrivateUsage]
        assert brain.interrupt(brain.answering()) is True
        await until(brain._all_told)  # pyright: ignore[reportPrivateUsage]
        going.set()
        # The stop's last key is in, and with it the stop has asked whether Claude Code holds anything of the turn.
        await until(lambda: brain._cleared != cleared)  # pyright: ignore[reportPrivateUsage]
        assert not asking.done()
        performing.set()
        assert await asyncio.wait_for(asking, 10) == BrainStopped()
    finally:
        going.set()
        performing.set()
        await brain.stop()
    [turn] = events(recorded, "brain.turn")
    assert (turn.outcome, turn.facts["stopped"], turn.facts["taken"], turn.facts["told"]) == ("ok", True, "p1", "p1")


async def test_a_stop_claude_code_does_not_carry_out_fails_its_turn_saying_what_the_model_holds_and_what_it_showed(tmp_path: Path, fake_claude: Path) -> None:
    # [LAW:no-silent-failure] a turn Claude Code runs on after its Escape is no stopped turn: its asker is told so, once
    # the stop's time has passed, and the next turn is not typed into it as though it were over.
    recorded: list[Entry] = []
    stopping = asyncio.Event()

    async def pause(seconds: float) -> None:
        # A take waits as long as one does; the stop's time passes once the stand-in has read its keys.
        await (stopping.wait() if seconds == INTERRUPT_SECONDS else asyncio.sleep(seconds))

    brain = await start(launch(tmp_path), recorded.append, pause)
    try:
        asking = asyncio.ensure_future(brain.ask("stuck", unasked))
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt"])
        assert brain.interrupt(brain.answering()) is True
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt", "escape", "ctrl_c"])
        assert not asking.done()
        stopping.set()
        with pytest.raises(StopUnrecorded, match=rf"did not stop what hands pressed Escape for in {re.escape(f'{INTERRUPT_SECONDS:g}')}s: the model holds Session\(.*turn=Opened\(turn='p1'.*it showed:") as raised:
            await asyncio.wait_for(asking, 10)
    finally:
        await brain.stop()
    # The model's account says what it holds of the brain's dialogs: none, each refused as the stop went, and the stop
    # still ending what Claude Code runs.
    assert f"dialogs=(), ending=DenyAndInterrupt(message={STOPPED!r})" in str(raised.value)
    [turn] = events(recorded, "brain.turn")
    assert (turn.outcome, turn.facts["taken"]) == ("failed", "p1") and turn.error is not None and turn.error.startswith("the brain failed the turn: the brain's Claude Code did not stop")


class _Timed:
    """A typist that notes when each Ctrl-C it pressed had gone in, on the monotonic clock."""

    def __init__(self, typist: Typist) -> None:
        self._typist = typist
        self.cleared: list[float] = []

    def type(self, text: PromptText) -> None:
        self._typist.type(text)

    def press(self, key: Keystroke) -> None:
        self._typist.press(key)
        if key == "ctrl_c":
            self.cleared.append(time.monotonic())


async def test_no_stop_presses_ctrl_c_within_claude_codes_exit_window_of_the_last(tmp_path: Path, fake_claude: Path) -> None:
    # A Ctrl-C that finds the input empty arms Claude Code's exit, which a second within 800ms takes. Each turn taken as
    # other words runs on, and the turn after it stops it before it is typed: two stops a take's wait apart, which the
    # stand-in ends as soon as it is done with each typing.
    brain = await start(launch(tmp_path), lambda _entry: None, typings_answered(tmp_path))
    timed = brain._typist = _Timed(brain._typist)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    try:
        for _ in range(2):
            with pytest.raises(Untaken):
                await brain.ask("garbled", unasked)
        await brain.ask("and now?", unasked)
    finally:
        await brain.stop()
    first, second = timed.cleared
    assert second - first > 0.8


def test_text_passed_on_to_the_brain_is_typed_as_the_characters_it_shows() -> None:
    # What a terminal is told is dropped, line ends are newlines, a tab is its spaces, any other control is spelled out,
    # and a closing backslash is kept from the Return that sends it.
    assert pasted("\x1b[31mred\x1b[0m\r\nnext\rline\n\tgo\tthere\x07 C:\\") == "red\nnext\nline\n    go  there\\x07 C:\\ "
    assert pasted("plain\nwords") == "plain\nwords"
    # Half of an emoji that was cut in two is spelled out; a whole one is itself.
    assert pasted("cut \ud83d, whole \U0001f600") == "cut \\ud83d, whole \U0001f600"


async def test_a_hook_with_a_field_that_does_not_parse_is_passed_over_and_hooks_are_still_heard() -> None:
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(message.record["message"]), level="WARNING")
    try:
        # A prompt id that is not text names no turn: the hook is logged and passed over, never raised out of the loop that hears hooks.
        failed = {"hook_event_name": "StopFailure", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": {"id": "p1"}, "error": "unknown"}
        await brain.posted("StopFailure", json.dumps(failed).encode())
    finally:
        logger.remove(sink)
    assert [line.split(":")[0] for line in warned] == ["the brain posted a hook that does not parse"]


def test_a_brain_turn_without_hands_tools_is_an_error() -> None:
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    errors, sink = failures()
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Read"}]}
        with_hands = {**body, "tools": [{"name": "Read"}, {"name": "mcp__hands__read_session"}]}
        brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), None, with_hands))
        brain.hear(Sent("x2", SessionId("b1"), MainTurn(None), None, with_hands))
        brain.hear(Sent("x3", SessionId("elsewhere"), MainTurn(None), None, body))
        assert errors == []
        brain.hear(Sent("x4", SessionId("b1"), MainTurn(None), None, body))
        # Deferred, as Claude Code offers MCP tools under tool search: the server may be connected, and is not blamed.
        deferred = {**body, "tools": [{"name": "Read"}, {"name": "ToolSearch"}, {"name": "DeferredToolPlaceholder", "defer_loading": True}]}
        brain.hear(Sent("x5", SessionId("b1"), MainTurn(None), None, deferred))
    finally:
        logger.remove(sink)
    assert errors == [
        "the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server (('Read',))",
        "the brain's turn went to the model without hands' tools and with ToolSearch: tool search is on though hands' --settings"
        " turn it off (('Read', 'ToolSearch', 'DeferredToolPlaceholder'))",
    ]
    # What a turn's requests offered is on its brain turn's event.
    assert recorded == []


async def test_a_brain_that_dies_mid_turn_fails_the_turn_and_says_once_how_it_ended(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        # Bounded, so a brain that never dies fails here, naming the wait it hung in.
        async with asyncio.timeout(TAKE_SECONDS * 2):
            with pytest.raises(BrainGone, match="exited"):
                await brain.ask("die", unasked)
            assert await brain.exited() == 3
            with pytest.raises(BrainGone, match="before it was asked"):
                await brain.ask("anyone?", unasked)
    finally:
        # The watch that saw it die and this stop both wait on the one exit; a brain that did not die is ended here.
        await brain.stop()
    [ran] = events(recorded, "brain.run")
    shown = ran.facts["shown"]
    assert ran.facts["code"] == 3 and isinstance(shown, str) and "bye" in shown


async def test_a_brain_slow_to_read_its_terminal_is_typed_into_once_its_input_is_up(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A loaded machine starts it slowly: a turn typed before it reads raw has its Return made a newline, and is untaken.
    monkeypatch.setenv("STARTS_AFTER", "1")
    brain = await start(launch(tmp_path), lambda _entry: None)
    try:
        async with asyncio.timeout(TAKE_SECONDS * 2):
            assert await brain.ask("what is running?", unasked) == BrainAnswered("p1", None)
    finally:
        await brain.stop()


async def test_a_turn_never_taken_fails_naming_hands_login_and_the_next_turn_is_its_own(
    tmp_path: Path, fake_claude: Path
) -> None:
    brain = await start(launch(tmp_path), lambda _entry: None, typings_answered(tmp_path))
    try:
        with pytest.raises(Untaken, match="`hands login` answers them"):
            await brain.ask("deaf", unasked)
        assert await brain.ask("and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()


async def test_a_turn_taken_as_other_words_is_typed_once_and_what_it_left_running_is_stopped_before_the_next_is_typed(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append, typings_answered(tmp_path))
    try:
        # What Claude Code took is not what was typed, so the turn is never taken; typed once, as the same words would be
        # taken the same way again.
        with pytest.raises(Untaken):
            await brain.ask("garbled", unasked)
        # What it took runs on as a turn of Claude Code's that answers no turn of the brain's: the model says so, and it
        # is stopped with Escape and Ctrl-C before the next turn is typed.
        assert await asyncio.wait_for(brain.ask("and now?", unasked), 10) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert typed(tmp_path) == [["prompt", " garbled"], ["escape", ""], ["ctrl_c", " garbled"], ["prompt", " and now?"]]
    untaken, answered = events(recorded, "brain.turn")
    assert (untaken.facts["taken"], untaken.facts["ids"]) == (None, None)
    assert (answered.facts["taken"], answered.facts["ids"], answered.facts["matched"], answered.facts["told"]) == ("p2", ("p2",), "id", "p2")


async def test_a_turn_typed_as_the_last_ones_stop_hooks_end_is_taken_once_by_the_prompt_claude_code_queued_it_as(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The next turn is typed the moment the last one's Stop is heard, while Claude Code's other Stop hooks still run: it is
    # queued, its UserPromptSubmit comes under the last prompt's id with the queued words, and it runs as a prompt of its own.
    monkeypatch.setenv("STOP_TAIL", "0.3")
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(message.record["message"]), level="WARNING")
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append, typings_answered(tmp_path))
    try:
        assert await asyncio.wait_for(brain.ask("first", unasked), 10) == BrainAnswered("p1", None)
        answered = await asyncio.wait_for(brain.ask("second", unasked), 10)
    finally:
        await brain.stop()
        logger.remove(sink)
    _, second = events(recorded, "brain.turn")
    # Taken under the last prompt's id, and run once, as p2, with no Escape pressed into the brain and nothing said about
    # it: the turn behind the one told, which no hook opened, named by its own telling.
    assert (answered, typed(tmp_path), second.facts["taken"], second.facts["queued"], second.facts["told"], second.facts["ids"], second.facts["matched"], warned) == (
        BrainAnswered("p2", None),
        [["prompt", " first"], ["prompt", " second"]],
        "p1",
        True,
        "p2",
        ("p2",),
        "behind",
        [],
    )


async def test_a_turn_is_taken_as_claude_code_keeps_what_it_typed(tmp_path: Path, fake_claude: Path) -> None:
    brain = await start(launch(tmp_path), lambda _entry: None)
    try:
        # A long paste comes back in its tags, and a closing backslash's space comes back trimmed.
        assert await asyncio.wait_for(brain.ask("one\ntwo\nthree\nfour", unasked), 10) == BrainAnswered("p1", None)
        assert await asyncio.wait_for(brain.ask("path C:\\", unasked), 10) == BrainAnswered("p2", None)
    finally:
        await brain.stop()


class _Jammed:
    """A typist whose every key fails as a terminal gone from under it does: not the Untyped a stale socket raises."""

    def type(self, text: object) -> None:
        raise OSError("jammed")

    press = type


def _played(tmp: Path) -> int:
    """How much of a recorded turn the stand-in has written to the brain's transcripts so far."""
    return sum(path.stat().st_size for path in (tmp / "brain" / "projects").glob("*/*.jsonl"))


@pytest.mark.parametrize("name", RECORDINGS)
async def test_a_recorded_claude_code_turn_is_answered_under_the_stop_that_ends_it_and_the_next_is_typed_with_nothing_before_it(tmp_path: Path, fake_claude: Path, name: str) -> None:
    brain = await start(launch(tmp_path), lambda _: None)
    asking = asyncio.create_task(brain.ask(f"play {name}", unasked))
    following: asyncio.Task[BrainEnded] | None = None
    try:
        # Played to its end, every hook of it answered: whatever the brain makes of the turn, it has made by then.
        await patiently(lambda: ["played", name] in typed(tmp_path), lambda: (len(typed(tmp_path)), _played(tmp_path)))
        # Its Stop was answered before its last line was played, so the answer has only to reach the asker, within the
        # patience: one never given is the expected failure, not a stall. The typed turn ends at the first Stop posted.
        with suppress(TimeoutError):
            await patiently(asking.done, lambda: None)
        assert asking.done() and asking.result() == BrainAnswered(stops(RECORDINGS[name])[0], None)
        following = asyncio.create_task(brain.ask("and then?", unasked))
        # [LAW:no-silent-failure] a stall is a TimeoutError, never the expected failure.
        await patiently(following.done, lambda: len(typed(tmp_path)))
        assert following.result() == BrainAnswered(f"p{prompts(RECORDINGS[name]) + 1}", None)
    finally:
        asking.cancel()
        if following is not None:
            following.cancel()
        await brain.stop()
    assert typed(tmp_path) == [["prompt", f" play {name}"], ["played", name], ["prompt", " and then?"]]


def failures() -> tuple[list[str], int]:
    """What is said on the terminal as an error, as it is said, and the sink that hears it."""
    said: list[str] = []
    return said, logger.add(lambda message: said.append(message.record["message"]), level="ERROR")


async def test_a_turn_whose_typing_fails_unexpectedly_fails_its_asker_says_so_once_and_the_next_turn_is_its_own(
    tmp_path: Path, fake_claude: Path
) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    said, sink = failures()
    try:
        typist = brain._typist  # pyright: ignore[reportPrivateUsage]
        brain._typist = _Jammed()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        with pytest.raises(OSError, match="jammed"):
            await asyncio.wait_for(brain.ask("what is running?", unasked), 10)
        brain._typist = typist  # pyright: ignore[reportPrivateUsage]
        assert await asyncio.wait_for(brain.ask("and now?", unasked), 10) == BrainAnswered("p1", None)
    finally:
        logger.remove(sink)
        await brain.stop()
    assert said == ["the brain's own work failed: OSError('jammed')"]
    failed, _ = events(recorded, "brain.turn")
    assert (failed.outcome, failed.error) == ("failed", "OSError: jammed")


async def _fails_at_once(_take: float) -> None:
    raise RuntimeError("no clock")


async def _fails_as_it_is_ended(_take: float) -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        raise RuntimeError("no clock") from None


@pytest.mark.parametrize(("pause", "typed"), [(_fails_at_once, "deaf"), (_fails_as_it_is_ended, "wait")])
async def test_a_take_whose_time_fails_to_pass_fails_its_asker_says_so_once_and_is_never_called_untaken(
    tmp_path: Path, fake_claude: Path, pause: Callable[[float], Awaitable[None]], typed: str
) -> None:
    # A pause that fails as the turn's taking ends it fails as surely as one that fails while it runs.
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append, pause)
    said, sink = failures()
    try:
        with pytest.raises(RuntimeError, match="no clock"):
            await brain.ask(typed, unasked)
    finally:
        logger.remove(sink)
        await brain.stop()
    # Each time the broken clock fails is said: the take's, and, for a turn that was taken, the wait of the stop that
    # ends it, which waits on the same clock.
    assert said and set(said) == {"the brain's own work failed: RuntimeError('no clock')"}
    [failed] = events(recorded, "brain.turn")
    assert (failed.outcome, failed.error) == ("failed", "RuntimeError: no clock")


async def test_a_permission_that_fails_unexpectedly_is_refused_stops_its_turn_and_says_so_once(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []

    def asks(held: Asked) -> None:
        asked.append(held)
        raise RuntimeError("no voice")

    brain = await start(launch(tmp_path), recorded.append)
    said, sink = failures()
    try:
        with pytest.raises(RuntimeError, match="no voice"):
            await asyncio.wait_for(brain.ask("write", asks), 10)
        # The brain heard the refusal, so its dialog did not hold it: its Stop comes, and the next turn is its own.
        await until(lambda: ["permission", "deny", BROKEN, True] in typed(tmp_path))
        assert await asyncio.wait_for(brain.ask("and now?", unasked), 10) == BrainAnswered("p2", None)
    finally:
        logger.remove(sink)
        await brain.stop()
    assert said == ["the brain's own work failed: RuntimeError('no voice')"]
    # The user's side is told the refusal the brain heard, which ends the turn, and the turn is stopped as an Escape stops it.
    assert [await held.decided() for held in asked] == [DenyAndInterrupt(BROKEN)]
    assert ["escape", ""] in typed(tmp_path)
    [permission] = events(recorded, "brain.permission")
    assert (permission.outcome, permission.error, permission.facts["decision"]) == ("failed", "RuntimeError: no voice", DenyAndInterrupt(BROKEN))
    assert not (tmp_path / "notes.txt").exists()


async def test_a_stop_whose_keys_fail_unexpectedly_ends_its_turn_says_so_once_and_is_never_pressed_again(tmp_path: Path, fake_claude: Path) -> None:
    brain = await start(launch(tmp_path), lambda _entry: None)
    pressed: list[object] = []

    class Jammed(_Jammed):
        def press(self, key: object) -> None:
            pressed.append(key)
            raise OSError("jammed")

    def asks(_held: Asked) -> None:
        # The turn breaks once it is taken, and the keys that would stop it are jammed.
        brain._typist = Jammed()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        raise RuntimeError("no voice")

    said, sink = failures()
    try:
        with pytest.raises(RuntimeError, match="no voice"):
            await asyncio.wait_for(brain.ask("write", asks), 5)
        await until(lambda: len(said) == 2)
    finally:
        logger.remove(sink)
        await brain.stop()
    # A second stop would press Escape again, and its Ctrl-C would land inside Claude Code's exit window.
    assert (pressed, said) == (["escape"], ["the brain's own work failed: RuntimeError('no voice')", "the brain's own work failed: OSError('jammed')"])


def _modelled(brain: Brain) -> None:
    """A brain built bare given the model it reads its turns with, and what it waits on that model with, as `Brain`
    builds it; a part of a launch of its own where the test gave it none."""
    brain._sessions = model(lambda _entry: None, brain._told, brain._took)  # pyright: ignore[reportPrivateUsage]
    brain._reader = Reader(brain._sessions, Tails(brain._sessions), brain._apply)  # pyright: ignore[reportPrivateUsage]
    brain._moved = asyncio.Event()  # pyright: ignore[reportPrivateUsage]
    brain._claude_turns = ClaudeTurns()  # pyright: ignore[reportPrivateUsage]
    brain.__dict__.setdefault("_launched", root())


async def _serving(brain: Brain) -> tuple[web.AppRunner, str]:
    """The listener a bare brain's hooks are posted to, answered by its `posted` as `start` wires it."""
    posting: asyncio.Future[Callable[[str, bytes], Awaitable[Mapping[str, object]]]] = asyncio.get_running_loop().create_future()
    posting.set_result(brain.posted)
    return await _listen(posting)  # pyright: ignore[reportPrivateUsage]


def _hooked(event: str, id: str, **fields: object) -> bytes:
    """A hook of the brain's session under prompt `id`, as Claude Code fills it."""
    return json.dumps({"hook_event_name": event, "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": id, **fields}).encode()


@pytest.mark.parametrize(("delivered", "answer", "matched"), [(True, "p1", "id"), (False, "p2", "behind")], ids=["taken into the turn running", "run after it"])
async def test_words_filed_under_a_running_turn_are_answered_by_the_turn_that_answers_them(delivered: bool, answer: str, matched: str) -> None:
    # [LAW:one-source-of-truth] which turn answers them is the model's to say, from Claude Code's queue records: the turn
    # they were typed into, when a remove took them out of the queue before its Stop, or the turn they ran as after it.
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(), TAKE_SECONDS, PromptText(" and add 3?"))  # pyright: ignore[reportPrivateUsage]
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    # Read on its period, as a running brain's model is: a Stop with words waiting is held until the transcript is read
    # through it, and let go unread, the order of the brain's turns is lost (`Brain._ordered`).
    reading = asyncio.ensure_future(brain._reader.keep_reading(0.01))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="count to ten"))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    # Typed while p1 runs: Claude Code files the words under its id.
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="and add 3?"))  # pyright: ignore[reportPrivateUsage]
    assert (turn.taken.done(), turn.taken_under, turn.queued) == (True, "p1", True)
    if delivered:
        await brain._apply(Unqueued(SessionId("b1"), Removed("and add 3?")))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Ten.", stop_hook_active=False))  # pyright: ignore[reportPrivateUsage]
    if not delivered:
        assert not turn.answered.done()
        await brain._apply(Unqueued(SessionId("b1"), Dequeued()))  # pyright: ignore[reportPrivateUsage]
        await brain._apply(Taken(SessionId("b1"), PromptId("p2"), Stamp(3100), 3.1))  # pyright: ignore[reportPrivateUsage]
        await brain.posted("Stop", _hooked("Stop", "p2", last_assistant_message="Thirteen.", stop_hook_active=False))  # pyright: ignore[reportPrivateUsage]
    reading.cancel()
    assert turn.running is not None
    assert (turn.answered.result(), turn.running.ids, turn.running.matched) == (BrainAnswered(answer, None), frozenset({PromptId(answer)}), matched)


async def test_a_permission_of_a_turn_dequeued_under_a_new_id_is_asked_in_it_though_only_the_transcript_says_it_opened(tmp_path: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] the permission is owned once Claude Code is read through now, not when a period
    # next reads the record that opened the turn it asks in.
    transcript = tmp_path / "projects" / "-brain-cwd" / "b1.jsonl"
    transcript.parent.mkdir(parents=True)
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    recorded: list[Entry] = []
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", lambda _entry: None):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    asked: list[Asked] = []

    def allowed(held: Asked) -> None:
        asked.append(held)
        held.settle(Allow())

    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), allowed, begun(), TAKE_SECONDS, PromptText(" and add 3?"))  # pyright: ignore[reportPrivateUsage]
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    path = str(transcript)
    # Read on its period, as a running brain's model is, so the Stop with words waiting is decided and not let go.
    reading = asyncio.ensure_future(brain._reader.keep_reading(0.01))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="count to ten", transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="and add 3?", transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Ten.", stop_hook_active=False, transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Unqueued(SessionId("b1"), Dequeued()))  # pyright: ignore[reportPrivateUsage]
    # Claude Code runs the words as p2, which no hook has named, and the model asks for a tool before any read since.
    transcript.write_text('{"type":"user","promptId":"p2","timestamp":"1970-01-01T00:00:03.100Z","message":{"role":"user","content":"and add 3?"}}\n')
    permission = {**json.loads(_hooked("PermissionRequest", "p2", transcript_path=path)), "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    reply = await brain.posted("PermissionRequest", json.dumps(permission).encode())
    reading.cancel()
    assert (len(asked), reply) == (1, hook_output(Allow()))


async def test_a_brain_whose_reading_failed_still_lets_go_of_what_it_held_as_it_stops(tmp_path: Path) -> None:
    # What stopped the reading is `exited`'s to raise, to whoever waits on the brain: the stop ends it and lets go anyway.
    ended = asyncio.Event()

    class Ended:
        async def stop(self) -> None:
            ended.set()

    class Listener:
        cleaned = False

        async def cleanup(self) -> None:
            self.cleaned = True

    async def exited() -> int:
        # Its end is heard a moment after its process ends, as `_run_out` hears it.
        await ended.wait()
        await asyncio.sleep(0.05)
        return 0

    async def reading() -> None:
        raise OSError("the transcript went")

    brain = object.__new__(Brain)
    brain._claude = Ended()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    brain._exit = asyncio.ensure_future(exited())  # pyright: ignore[reportPrivateUsage]
    brain._reading = asyncio.ensure_future(reading())  # pyright: ignore[reportPrivateUsage]
    await asyncio.wait({brain._reading})  # pyright: ignore[reportPrivateUsage]
    listener = brain._listener = Listener()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    brain._sockets = tmp_path / "sockets"  # pyright: ignore[reportPrivateUsage]
    brain._sockets.mkdir()  # pyright: ignore[reportPrivateUsage]
    await brain.stop()
    assert (listener.cleaned, brain._sockets.exists()) == (True, False)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(OSError, match="the transcript went"):
        brain._reading.result()  # pyright: ignore[reportPrivateUsage]


async def test_a_hook_whose_hearing_fails_unexpectedly_is_said_once_and_the_hooks_after_it_are_still_heard() -> None:
    recorded: list[Entry] = []

    def record(entry: Entry) -> None:
        # The first event the log is handed cannot be written.
        if not recorded:
            recorded.append(entry)
            raise ValueError("the log is full")
        recorded.append(entry)

    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = record  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", lambda _entry: None):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    # The elicitation that breaks is a part of the turn in flight: it was declined before it was heard, so the turn runs on.
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(), TAKE_SECONDS, PromptText(" probe"))  # pyright: ignore[reportPrivateUsage]
    turn.taken.set_result(None)
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    # Claude Code's turn p1 in flight, as the model holds it: the turn taken.
    _modelled(brain)
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path("/brain/projects/-brain-cwd/b1.jsonl"))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "probe"))  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    listener, url = await _serving(brain)
    said, sink = failures()

    async def answered(event: str, body: Mapping[str, object]) -> dict[str, object]:
        async with aiohttp.ClientSession() as client, client.post(f"{url}/{event}", data=json.dumps(body).encode()) as reply:
            assert reply.status == 200
            return (await reply.json())["hookSpecificOutput"]

    try:
        elicited = {"hook_event_name": "Elicitation", "session_id": "b1", "prompt_id": "p1", "mcp_server_name": "probe", "message": "Which?"}
        assert (await answered("Elicitation", elicited))["action"] == "decline"
        permission = {"hook_event_name": "PermissionRequest", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
        assert (await answered("PermissionRequest", permission))["decision"] == {"behavior": "deny", "message": NOBODY}
    finally:
        logger.remove(sink)
        await listener.cleanup()
    assert said == ["the brain's own work failed: ValueError('the log is full')"]
    assert (turn.broken, turn.answered.done(), brain._turn) == (None, False, turn)  # pyright: ignore[reportPrivateUsage]
    assert [event.event for event in recorded if isinstance(event, WideEvent)] == ["brain.elicitation", "brain.permission"]


async def test_a_break_of_a_turn_already_over_ends_nothing_the_turn_in_flight_runs() -> None:
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    brain._turn, brain._typing = None, set()  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path("/brain/projects/-brain-cwd/b1.jsonl"))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p2"), "next"))  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    over, current = (_Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(), TAKE_SECONDS, PromptText(f" {typed}")) for typed in ("first", "next"))  # pyright: ignore[reportPrivateUsage]
    brain._turn = current  # pyright: ignore[reportPrivateUsage]
    said, sink = failures()
    try:
        # What the turn before it left running breaks late, once the next turn's prompt has started new work.
        brain._broke(over, RuntimeError("late"))  # pyright: ignore[reportPrivateUsage]
    finally:
        logger.remove(sink)
    held = brain._modelled()  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    assert (said, held.ending, current.broken, brain._typing) == (["the brain's own work failed: RuntimeError('late')"], None, None, set())  # pyright: ignore[reportPrivateUsage]


async def test_a_permission_another_session_posts_to_the_brain_is_refused_and_is_one_event_naming_that_session() -> None:
    recorded: list[Entry] = []
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    permission = {"hook_event_name": "PermissionRequest", "session_id": "b2", "transcript_path": "/brain/projects/-brain-cwd/b2.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": "p9", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    assert await brain.posted("PermissionRequest", json.dumps(permission).encode()) == hook_output(Deny(NOBODY))
    assert [(event.facts["session"], event.facts["prompt"], event.facts["tool"], event.facts["decision"]) for event in events(recorded, "brain.permission")] == [("b2", "p9", "Write", Deny(NOBODY))]


async def test_an_asker_that_stops_waiting_leaves_the_turn_to_its_stop_and_the_next_turn_gets_its_own(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    brain = await start(launch(tmp_path), recorded.append)
    try:
        asked = asyncio.create_task(brain.ask("slow", unasked))
        await asyncio.sleep(0.1)
        asked.cancel()
        assert await brain.ask("and now?", unasked) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert [event.event for event in events(recorded, "brain.launch") + events(recorded, "brain.run")] == ["brain.launch", "brain.run"]


# How a hands starts a Claude Code of its own, and the pids of what it started: its brain under fritter.
BRAIN = """
brain = await start(launch, lambda _entry: None)
print(brain.pid, _child_of(brain.pid), flush=True)
"""
@pytest.mark.parametrize("started", [BRAIN], ids=["brain"])
async def test_a_hands_that_dies_without_stopping_its_claude_code_leaves_nothing_it_started_running(tmp_path: Path, fake_claude: Path, started: str) -> None:
    # hands killed outright: no stop, no cleanup, only the kernel's hangup of the terminal it held.
    # Its temp dir is the test's, so what a dead hands leaves there - the brain's socket dir - goes with the test.
    temp = Path(tempfile.mkdtemp(dir="/tmp"))  # short: the sockets in it are held to the unix socket path limit
    pids: list[int] = []

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):  # gone, or the pid is now another user's
            return False
        return True

    script = pinned_to(fake_claude) + "import asyncio, os, pickle, sys\nfrom hands.brain.process import _child_of, brain_claude, spawn, start\nasync def main():\n    launch = pickle.load(sys.stdin.buffer)\n"
    script += "".join(f"    {line}\n" for line in started.strip().splitlines()) + "    os._exit(0)\nasyncio.run(main())\n"
    hands = await asyncio.create_subprocess_exec(sys.executable, "-c", script, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "TMPDIR": str(temp)})
    try:
        out, err = await asyncio.wait_for(hands.communicate(pickle.dumps(launch(tmp_path))), 30)
        assert hands.returncode == 0, err.decode()
        pids = [int(pid) for pid in out.split()]
        assert pids

        await until(lambda: not any(map(alive, pids)))
    finally:
        # Whatever failed, nothing it started outlives the test: killing a hands that hangs is itself its brain's hangup.
        if hands.returncode is None:
            hands.kill()
            await hands.wait()
        for pid in filter(alive, pids):
            os.kill(pid, signal.SIGKILL)
        shutil.rmtree(temp)


# A loop that ends as the brain starts: asyncio.run cancels every task at once, the launch's included, wherever it is.
# Ended a tick later each time, so one ends in the tick the brain is spawned in, which no step in time is sure to hit;
# then a tenth of a second later each time, so others end past the spawn: in the launch's own stop of what it spawned, or
# in the stop of the brain it started, as the daemon stops the brain it holds.
# Run in a process of its own, since a loop that never closes would take the test run down with it.
SHUT_DOWN = """
import asyncio, contextlib, math, os, pickle, sys
from hands.brain.process import start
from hands.sessions import pseudoterminal

launch = pickle.load(sys.stdin.buffer)
# A close that only a fallback would end, the stop's kill or giving up on the terminal, is a hang here, and so fails,
# where on the real clock it was a close seconds late that only a bound on the time could tell from a loaded machine.
pseudoterminal.STOP_SECONDS = pseudoterminal.DRAIN_SECONDS = math.inf

class Ended(Exception):
    pass

async def run() -> None:
    brain = await start(launch, lambda _entry: None)
    await brain.stop()

async def ends(ticks: int, seconds: float) -> None:
    asyncio.create_task(run())
    for _ in range(ticks):
        await asyncio.sleep(0)
    await asyncio.sleep(seconds)
    raise Ended

for ticks, seconds in [*((ticks, 0.0) for ticks in range(40)), *((0, tenths / 10) for tenths in range(1, 16))]:
    with contextlib.suppress(Ended):
        asyncio.run(ends(ticks, seconds))
    try:
        os.waitpid(-1, os.WNOHANG)
        sys.exit(f"a fritter was left unreaped when the loop ended {ticks} ticks and {seconds}s into the launch")
    except ChildProcessError:
        pass
    # This loop closed, with nothing it started left unreaped.
    print(ticks, seconds, flush=True)
"""
# One loop is a launch and its close, each in a process of its own: 0.4 s at most at load 50. A loop this long without
# closing is not closing.
LOOP_SECS = 30.0


# The suite's timeout is for a test that hangs; this one's 55 loops each have LOOP_SECS, and the run must outlast them.
@pytest.mark.timeout(600)
async def test_a_loop_ended_as_the_brain_starts_closes_and_leaves_nothing_it_started_running(tmp_path: Path, fake_claude: Path) -> None:
    """The daemon's shutdown cancels the brain's launch wherever it is, and must not then wait for ever on what it spawned.

    Python 3.12's asyncio subprocesses did: cancelled before the transport's own task first ran, they waited for an exit
    nothing would deliver."""
    temp = Path(tempfile.mkdtemp(dir="/tmp"))  # short: the sockets in it are held to the unix socket path limit
    hands = await asyncio.create_subprocess_exec(sys.executable, "-c", pinned_to(fake_claude) + SHUT_DOWN, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "TMPDIR": str(temp)})
    assert hands.stdin is not None and hands.stdout is not None and hands.stderr is not None
    errors = asyncio.ensure_future(hands.stderr.read())
    closed: list[str] = []
    try:
        hands.stdin.write(pickle.dumps(launch(tmp_path)))
        await hands.stdin.drain()
        hands.stdin.close()
        # [LAW:no-ambient-temporal-coupling] a bound on each loop's close, never on the fifty-five of them together.
        while line := await asyncio.wait_for(hands.stdout.readline(), LOOP_SECS):
            closed.append(line.decode().strip())
        await asyncio.wait_for(hands.wait(), LOOP_SECS)
    except TimeoutError:
        hands.kill()
        await hands.wait()
        said = (await errors).decode()[-2000:]
        raise AssertionError(f"a loop ended as the brain started never finished closing: the one after {closed[-1:] or 'none'}\n{said}") from None
    finally:
        if hands.returncode is None:
            hands.kill()
            await hands.wait()
        shutil.rmtree(temp)
    assert hands.returncode == 0, (await errors).decode()[-2000:]
    # The claude each fritter started ends with the terminal its fritter's end hangs up.
    await until(lambda: not running(tmp_path))


async def test_a_fritter_that_cannot_be_run_is_refused_with_what_its_terminal_showed(tmp_path: Path, fake_claude: Path, fritter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unrunnable = tmp_path / "fritter"
    shutil.copy(fritter, unrunnable)
    unrunnable.chmod(0o644)
    monkeypatch.setattr(wrapper, "PACKAGED", unrunnable)
    with pytest.raises(Unstartable, match=r"(?s)fritter exited \(126\).*[Pp]ermission denied"):
        await start(launch(tmp_path), lambda _entry: None)


async def test_a_brain_that_never_turns_its_input_on_is_refused_with_what_it_showed_and_left_not_running(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.START_SECONDS", 1.0)
    fake_claude.write_text("#!/bin/sh\necho stuck on a screen\nsleep 30\n")
    pin(monkeypatch, fake_claude)
    with pytest.raises(Unstartable, match=r"(?s)had not turned its input on \(.*bracketed paste.*\) after 1s.*stuck on a screen"):
        await start(launch(tmp_path), lambda _entry: None)
    await until(lambda: not running(tmp_path))


async def test_a_conversation_the_brain_cannot_come_up_in_is_let_go_so_the_next_start_begins_anew(
    tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hands.brain.process.START_SECONDS", 1.0)
    # As Claude Code does with a transcript it cannot read.
    fake_claude.write_text("#!/bin/sh\necho No conversation found with session ID: c1\n")
    pin(monkeypatch, fake_claude)
    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / "conversation").write_text("c1\n")
    transcript(brain, "c1")
    recorded: list[Entry] = []
    with pytest.raises(Unstartable, match="No conversation found"):
        await start(replace(launch(tmp_path), conversation=conversation(brain, SessionId("n1"))), recorded.append)
    [launched] = events(recorded, "brain.launch")
    assert (launched.outcome, launched.facts["conversation"], launched.facts["let_go"]) == ("failed", "resumed", "c1")
    assert conversation(brain, SessionId("n2")) == Fresh(SessionId("n2"), None)


def test_a_brain_with_no_login_is_refused_naming_the_command(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ) == Account("claude.ai", "brain@example.com")
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(NotLoggedIn, match="`hands login` gives it one"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ)


def logins(home: Path) -> list[tuple[str, dict[str, object]]]:
    """Each `hands login`'s event in the audit log of `home`, as its outcome and its facts."""
    events = (json.loads(line) for line in segment(home / "audit", 0).read_text().splitlines())
    return [(event["outcome"], event["facts"]) for event in events if event["type"] == "WideEvent" and event["event"] == "brain.login"]


def test_hands_login_sets_a_new_brain_home_up_with_its_settings_then_claude_codes_first_run_where_the_brain_runs(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not the brain's")
    # A hands shim ahead of the real claude would run the first run at this terminal as a session, under fritter.
    shim = tmp_path / "shim" / "claude"
    shim.parent.mkdir()
    shim.write_text(f"#!/bin/sh\n{MARK}\nexit 99\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")
    home = tmp_path / "home"
    assert main(["--home", str(home), "login"]) == 0
    brain = home / "brain"
    assert json.loads((brain / "settings.json").read_text()) == {"syncClaudeAiSkills": False, "syncClaudeAiPlugins": False, "permissions": {"defaultMode": "default"}}
    account_kept_out(brain)
    # Claude Code's own first run, on the brain's settings sources, in the directory the brain runs in, where its screens,
    # the trust of that directory among them, are answered once at this terminal; its settings there before it started,
    # so that it never syncs the account's skills or plugins; with no credential of this shell's beside it.
    assert json.loads((brain / "login.json").read_text()) == {"argv": ["--setting-sources", "user"], "cwd": str(workdir(brain)), "settings": True, "credentials": []}
    assert capsys.readouterr().out.splitlines()[0] == f"the brain at {brain} is logged in as brain@example.com (claude.ai)"
    assert logins(home) == [("ok", {"asked": "claudeai", "settings_written": True, "first_run": f"no {brain / '.claude.json'}", "account": {"type": "Account", "method": "claude.ai", "holder": "brain@example.com"}})]


def test_hands_login_on_a_brain_home_logs_it_in_again_leaving_its_settings_as_they_are(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    brain = tmp_path / "brain"
    said = b'{"permissions": {"allow": ["Bash(lit:*)"], "defaultMode": "acceptEdits"},\n "syncClaudeAiSkills": false, "syncClaudeAiPlugins": false}'
    onboard(brain, said)
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert (brain / "settings.json").read_bytes() == said
    assert json.loads((brain / "login.json").read_text())["argv"] == ["auth", "login", "--claudeai"]
    assert capsys.readouterr().out.splitlines() == [f"the brain at {brain} is logged in as brain@example.com (claude.ai)", "a hands already running started its brain on the login before: restart it to start the brain on this one"]
    assert logins(tmp_path) == [("ok", {"asked": "claudeai", "settings_written": False, "first_run": None, "account": {"type": "Account", "method": "claude.ai", "holder": "brain@example.com"}})]


def test_hands_login_on_a_brain_home_claude_code_never_finished_its_first_run_on_runs_it(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    # What `claude auth status` leaves, asked by a `hands run` before any `hands login`: its state, and no onboarding.
    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / ".claude.json").write_text("{}")
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((brain / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    account_kept_out(brain)
    assert logins(tmp_path) == [("ok", {"asked": "claudeai", "settings_written": True, "first_run": "its onboarding unfinished", "account": {"type": "Account", "method": "claude.ai", "holder": "brain@example.com"}})]


def test_hands_login_on_a_brain_home_whose_directory_claude_code_was_never_told_to_trust_runs_its_first_run(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    # Onboarded, then quit at the trust screen: the brain would start on that screen.
    onboard(tmp_path / "brain", trusted=False)
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert logins(tmp_path)[0][1]["first_run"] == f"{(tmp_path / 'brain' / 'cwd').resolve()} untrusted"


def test_hands_login_on_a_brain_home_whose_state_claude_code_cannot_have_written_runs_its_first_run(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    onboard(tmp_path / "brain")
    (tmp_path / "brain" / ".claude.json").write_text("[]")
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert str(logins(tmp_path)[0][1]["first_run"]).startswith(f"{tmp_path / 'brain' / '.claude.json'} unreadable: ")


def test_hands_login_whose_first_run_was_quit_before_its_last_screen_exits_1_saying_so(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("LOGIN_UNANSWERED", "1")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err == f"hands login: the brain has not been through Claude Code's first screens (no {tmp_path / 'brain' / '.claude.json'}): `hands login` answers them\n"


def test_hands_login_on_a_brain_whose_settings_would_sync_its_accounts_skills_runs_no_claude_code(tmp_path: Path, fake_claude: Path, capsys: pytest.CaptureFixture[str]) -> None:
    onboard(tmp_path / "brain", b'{"permissions": {"defaultMode": "default"}}')
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert not (tmp_path / "brain" / "login.json").exists()
    assert "its account's syncClaudeAiSkills and syncClaudeAiPlugins" in capsys.readouterr().err


def test_hands_login_after_a_first_run_that_failed_runs_it_again(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("LOGIN_EXIT", "130")
    assert main(["--home", str(tmp_path), "login"]) == 1
    monkeypatch.delenv("LOGIN_EXIT")
    assert main(["--home", str(tmp_path), "login"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert [outcome for outcome, _ in logins(tmp_path)] == ["failed", "ok"]


def test_hands_login_on_a_brain_path_that_is_no_directory_exits_1_saying_so(tmp_path: Path, fake_claude: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "brain").write_text("")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err.startswith(f"hands login: [Errno 17] File exists: '{tmp_path / 'brain'}'")


def test_hands_login_console_logs_a_brain_in_with_an_anthropic_console_key(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    onboard(tmp_path / "brain")
    assert main(["--home", str(tmp_path), "login", "--console"]) == 0
    assert json.loads((tmp_path / "brain" / "login.json").read_text())["argv"] == ["auth", "login", "--console"]
    assert capsys.readouterr().out.splitlines()[0] == f"the brain at {tmp_path / 'brain'} is logged in as brain@example.com (api_key)"
    assert logins(tmp_path) == [("ok", {"asked": "console", "settings_written": False, "first_run": None, "account": {"type": "Account", "method": "api_key", "holder": "brain@example.com"}})]


@pytest.mark.parametrize(("method", "account"), [("api_key", Account("api_key", "brain@example.com")), ("oauth_token", Account("oauth_token", None))])
def test_a_brain_on_any_anthropic_login_starts(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, method: str, account: Account) -> None:
    monkeypatch.setenv("AUTH_METHOD", method)
    assert logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ) == account


def test_hands_login_that_leaves_the_brain_on_another_login_than_asked_exits_1_naming_both(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    # A token its settings.json sets ahead of the plan `claude auth login` made.
    monkeypatch.setenv("AUTH_METHOD", "oauth_token")
    onboard(tmp_path / "brain")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err == "hands login: the brain holds a token (oauth_token), not the claude.ai login asked for\n"
    assert [(outcome, facts["asked"]) for outcome, facts in logins(tmp_path)] == [("failed", "claudeai")]


def test_hands_login_on_a_brain_whose_settings_set_a_key_claude_code_never_asked_about_runs_its_first_run(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    monkeypatch.setenv("AUTH_METHOD", "api_key")
    brain = tmp_path / "brain"
    onboard(brain, b'{"syncClaudeAiSkills": false, "syncClaudeAiPlugins": false, "env": {"ANTHROPIC_API_KEY": "sk-ant-api03-0123456789abcdefghijklmn"}}')
    # The brain would start on Claude Code's screen asking whether to use the key, with nobody at its keyboard.
    assert unanswered(brain) == f"the API key {brain / 'settings.json'} sets unanswered"
    assert main(["--home", str(tmp_path), "login", "--console"]) == 0
    assert json.loads((brain / "login.json").read_text())["argv"] == ["--setting-sources", "user"]
    assert unanswered(brain) is None


def test_a_brain_reaching_claude_through_a_cloud_provider_is_refused_naming_it(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PROVIDER", "bedrock")
    with pytest.raises(NotLoggedIn, match="through bedrock, not the Anthropic API hands' proxy forwards to"):
        logged_in(tmp_path / "brain", "http://127.0.0.1:1", os.environ)


@pytest.mark.parametrize(("onboarded", "said"), [(False, "the brain's first run of Claude Code exited 3"), (True, "`claude auth login` for the brain exited 3")], ids=["first", "again"])
def test_hands_login_that_claude_code_fails_exits_1_saying_so(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], onboarded: bool, said: str) -> None:
    monkeypatch.setenv("LOGIN_EXIT", "3")
    if onboarded:
        onboard(tmp_path / "brain")
    assert main(["--home", str(tmp_path), "login"]) == 1
    assert capsys.readouterr().err == f"hands login: {said}\n"
    assert logins(tmp_path) == [("failed", {"asked": "claudeai", "settings_written": not onboarded})]
    # The command fails saying why, in the words it said it.
    [command] = [event for event in map(json.loads, segment(tmp_path / "audit", 0).read_text().splitlines()) if event.get("event") == "hands.command"]
    assert command["error"] == f"exited 1: {said}"


def test_the_brain_config_is_one_line_of_json_naming_only_hands() -> None:
    server = McpServer(url="http://127.0.0.1:9/mcp", token="t", runner=None)  # pyright: ignore[reportArgumentType]
    assert json.loads(server.config()) == {"mcpServers": {"hands": {"type": "http", "url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer t"}}}}


async def test_the_summariser_under_the_brain_asks_the_turn_as_a_side_question_within_its_time_and_says_a_failed_one() -> None:
    asked: list[tuple[str, Within]] = []

    async def ask(question: str, within: Within) -> str:
        asked.append((question, within))
        if "fail" in question:
            raise AsideFailed(Unanswered.TIMED_OUT, "no answer by its deadline")
        return "  The tests ran. "

    summarise = aside(ask, "Sum it up.", 0.1)
    assert await summarise("the tests ran") == "The tests ran."
    # The summary's own time is the side question's deadline, over the ones asked before it and its answer alike.
    assert asked == [("Sum it up.\n\nthe tests ran", Deadline(0.1))]
    with pytest.raises(SummaryFailed, match="no answer by its deadline"):
        await summarise("fail")


async def test_the_run_starts_the_brain_beside_hands_mcp_server(tmp_path: Path, fake_claude: Path) -> None:
    recorded: list[Entry] = []
    wire = Wire(lambda _observed: None)
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)

    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT)
    async with mind(claude, [tool(echo)], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", wire, store, tmp_path / "audit", "hands", recorded.append, os.environ, never_cut, Attention) as minded:
        assert isinstance(minded.llm, BrainStage)
        assert [watch.name for watch in minded.watches] == ["the brain", "the brain's turns", "the brain's context"]
        # The brain is behind the proxy, so it is given its usage read off the wire.
        assert list(minded.llm._tools) == ["mcp__hands__echo", "mcp__hands__context_usage"]  # pyright: ignore[reportPrivateUsage]
        [launched] = events(recorded, "brain.launch")
        assert (launched.facts["cwd"], launched.facts["account"]) == (tmp_path / "brain" / "cwd", Account("claude.ai", "brain@example.com"))
        # The stage speaks from the wire while the brain runs, so a second one cannot join it.
        with pytest.raises(RuntimeError, match="joined the wire"), wire.joined(minded.llm):
            pass
    assert isinstance(ran := recorded[-1], WideEvent) and ran.event == "brain.run"
    # Gone with the brain: the wire forwards everything again.
    with wire.joined(minded.llm):
        pass
    # The next run's brain takes the conversation up again, while Claude Code keeps its transcript.
    transcript(tmp_path / "brain", str(launched.facts["session"]))
    async with mind(claude, [tool(echo)], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", wire, store, tmp_path / "audit", "hands", recorded.append, os.environ, never_cut, Attention) as minded:
        # Its usage is heard under the session it resumed, not one begun for it.
        wire.observe(Sent("x", SessionId(str(launched.facts["session"])), MainTurn(None), None, {}))
        wire.observe(Heard("x", MessageStarted("msg_x", "claude-sonnet-5", {"input_tokens": 7, "output_tokens": 1})))
        assert isinstance(minded.llm, BrainStage)
        reported = await minded.llm._tools["mcp__hands__context_usage"].body()  # pyright: ignore[reportPrivateUsage]
        assert reported["in_context_tokens"] == 8
    [_, again] = events(recorded, "brain.launch")
    assert (again.facts["session"], again.facts["conversation"]) == (launched.facts["session"], "resumed")


async def test_the_brain_is_told_the_personality_the_run_was_configured_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)

    def minding(llm: ClaudeCodeBackend):
        return mind(VoiceConfig(llm=llm, voice=voices.DEFAULT, personality="Dry and wry."), [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", Wire(lambda _observed: None), store, tmp_path / "audit", "hands", recorded.append, os.environ, never_cut, Attention)

    launched: list[Launch] = []

    async def starting(launch: Launch, _record: object) -> Brain:
        launched.append(launch)
        raise Unstartable("seen")

    monkeypatch.setattr(run, "start_brain", starting)
    with pytest.raises(CannotStart, match="^seen$"):
        async with minding(ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com"))):
            pass
    [launch] = launched
    assert "\n\nDry and wry.\n\n" in launch.instruction


async def test_a_brain_that_cannot_start_refuses_the_run_saying_why(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)

    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT)
    with pytest.raises(CannotStart, match=f"^Claude Code {pinned.VERSION}, the brain's, is not installed at {tmp_path / 'brain'}/"):
        async with mind(claude, [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", Wire(lambda _observed: None), store, tmp_path / "audit", "hands", recorded.append, {"PATH": str(tmp_path)}, never_cut, Attention):
            pass
    # The launch that failed is one event, saying why; no brain ran, so there is no run.
    [launched] = events(recorded, "brain.launch")
    assert launched.outcome == "failed" and launched.error == f"Unstartable: Claude Code {pinned.VERSION}, the brain's, is not installed at {pinned.claude(tmp_path / 'brain')}: run `hands install-brain`"
    assert launched.facts["account"] == Account("claude.ai", "brain@example.com") and "pid" not in launched.facts
    assert events(recorded, "brain.run") == []


async def test_a_hands_built_without_its_fritter_refuses_the_run_naming_the_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    refocus = Refocus(Sessions(clock=lambda: 0.0, record=recorded.append), Home(tmp_path), recorded.append)
    missing = tmp_path / "package" / "bin" / "fritter"
    monkeypatch.setattr(wrapper, "PACKAGED", missing)

    claude = VoiceConfig(llm=ClaudeCodeBackend(model="claude-sonnet-5", config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT)
    # The words `hands install-fritter` refuses the same hands with.
    refused = f"hands' package carries no fritter at {missing}: install hands again, or in a checkout, `uv sync --reinstall-package hands`"
    with pytest.raises(CannotStart, match=f"^{re.escape(refused)}$"):
        async with mind(claude, [], [], lambda: "", unread, lambda: "screen", lambda: "held key", refocus, "http://127.0.0.1:1", Wire(lambda _observed: None), store, tmp_path / "audit", "hands", recorded.append, dict(os.environ), never_cut, Attention):
            pass
    [launched] = events(recorded, "brain.launch")
    assert launched.outcome == "failed" and launched.error == f"Unstartable: {refused}"
    assert launched.facts["account"] == Account("claude.ai", "brain@example.com") and "fritter" not in launched.facts and "pid" not in launched.facts
    assert events(recorded, "brain.run") == []


TRANSCRIPT = "/brain/projects/-brain-cwd/b1.jsonl"


def _permission_hook(prompt: str, path: str = "/tmp/x") -> bytes:
    """A PermissionRequest of the brain's session under prompt `prompt`, as Claude Code fills it."""
    return _hooked("PermissionRequest", prompt, tool_name="Write", tool_input={"file_path": path})


async def _in_turn(recorded: list[Entry], asks: Callable[[Asked], None]) -> tuple[Brain, _Turn]:  # pyright: ignore[reportPrivateUsage]
    """A brain built bare, with a turn in flight that Claude Code took as p1, which the model holds running."""
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", lambda _entry: None):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), asks, begun(), TAKE_SECONDS, PromptText(" write"))  # pyright: ignore[reportPrivateUsage]
    turn.taken.set_result(None)
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "write"))  # pyright: ignore[reportPrivateUsage]
    return brain, turn


def _dialogs(brain: Brain) -> tuple[Held, ...]:
    held = brain._modelled()  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    return held.dialogs


async def test_a_permission_request_is_held_in_the_model_before_anything_is_awaited_while_a_stop_posted_before_it_is_still_open() -> None:
    """[LAW:no-ambient-temporal-coupling] each hook is handled as it arrives: a dialog posted behind a Stop the model holds
    for its record is held in the model while the Stop's post is still open, before its own read of Claude Code."""
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _ = await _in_turn(recorded, asked.append)
    seen: list[tuple[tuple[Held, ...], bool]] = []
    read = brain._reader.read  # pyright: ignore[reportPrivateUsage]
    # A Stop under an id no record names yet, with the turn read busy: held until the transcript says whose it is.
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    stopping = asyncio.ensure_future(brain.posted("Stop", _hooked("Stop", "p9", last_assistant_message="", stop_hook_active=False)))

    async def reading() -> None:
        # What the model holds as the permission's own read of Claude Code begins, and whether the Stop is still open.
        seen.append((_dialogs(brain), stopping.done()))
        await read()

    await until(lambda: any(stop.prompt == "p9" for stop in brain._modelled().stops))  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    brain._reader.read = reading  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    asking = asyncio.ensure_future(brain.posted("PermissionRequest", _permission_hook("p1")))
    await until(lambda: len(asked) == 1)
    [(held, stopped)] = seen
    assert [dialog.on for dialog in held] == [Permission("Write", {"file_path": "/tmp/x"})] and not stopped
    asked[0].settle(Allow())
    assert await asyncio.wait_for(asking, 10) == hook_output(Allow())
    stopping.cancel()
    await asyncio.wait({stopping})


async def test_a_dialog_asked_after_hands_stopped_its_turn_is_refused_with_interrupt_and_never_put_to_the_user_whatever_the_brain_still_holds() -> None:
    """The model, not the brain's turn in flight, says what hands is ending: once the stop's Escape went, a dialog Claude
    Code asks before a prompt starts new work ends its turn, though the brain let go of its turn already (a stop
    StopUnrecorded ended), where it would otherwise be nobody's and refused only as an answer."""
    recorded: list[Entry] = []
    brain, turn = await _in_turn(recorded, unasked)
    brain._sessions.hear(Ending(SessionId("b1"), DenyAndInterrupt(STOPPED)))  # pyright: ignore[reportPrivateUsage]
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(DenyAndInterrupt(STOPPED))
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    assert await brain.posted("PermissionRequest", _permission_hook("p1", "/tmp/y")) == hook_output(DenyAndInterrupt(STOPPED))
    assert _dialogs(brain) == ()
    assert permissions(recorded) == [("p1", "Write", DenyAndInterrupt(STOPPED)), ("p1", "Write", DenyAndInterrupt(STOPPED))]
    owned, unowned = events(recorded, "brain.permission")
    assert (owned.parent_id, unowned.parent_id) == (turn.began.span.span_id, brain._launched.span_id)  # pyright: ignore[reportPrivateUsage]


async def test_a_dialog_claude_code_went_on_past_unanswered_is_refused_and_its_asker_and_event_say_so() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _ = await _in_turn(recorded, asked.append)
    asking = asyncio.ensure_future(brain.posted("PermissionRequest", _permission_hook("p1")))
    await until(lambda: len(asked) == 1)
    # Claude Code at its prompt, the dialog never answered: the model lets it go, and the brain refuses it.
    await brain._apply(StatusReported(SessionId("b1"), Report(Idle(), Stamp(9000)), 9.0))  # pyright: ignore[reportPrivateUsage]
    assert await asyncio.wait_for(asking, 10) == hook_output(Deny(LET_GO))
    assert await asked[0].decided() == Deny(LET_GO)
    assert permissions(recorded) == [("p1", "Write", Deny(LET_GO))]


async def test_a_permission_whose_read_breaks_is_no_turns_and_is_refused_said_once_and_left_held_by_nothing() -> None:
    recorded: list[Entry] = []
    brain, turn = await _in_turn(recorded, unasked)

    async def broken() -> None:
        raise OSError("the transcript went")

    brain._reader.read = broken  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    said, sink = failures()
    try:
        assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    finally:
        logger.remove(sink)
    assert said == ["the brain's own work failed: OSError('the transcript went')"]
    assert (_dialogs(brain), turn.broken) == ((), None)
    [permission] = events(recorded, "brain.permission")
    assert (permission.outcome, permission.error, permission.facts["decision"]) == ("failed", "OSError: the transcript went", Deny(BROKEN))
    assert permission.trace


async def test_a_permission_the_model_could_not_hold_is_refused_at_its_hook_and_said_once() -> None:
    recorded: list[Entry] = []
    brain, _ = await _in_turn(recorded, unasked)

    def unheld(_hook: object) -> "asyncio.Future[HookReply]":
        raise RuntimeError("the model is gone")

    brain._reader.asked = unheld  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    said, sink = failures()
    try:
        assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    finally:
        logger.remove(sink)
    assert said == ["the brain's own work failed: RuntimeError('the model is gone')"]
    assert permissions(recorded) == [("p1", "Write", Deny(BROKEN))]


async def test_a_dialog_of_a_turn_hands_broke_is_refused_and_never_put_to_the_user() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, turn = await _in_turn(recorded, asked.append)
    # Broken once Claude Code took it, and its prompt cleared any ending the model held: its stop is still to come.
    turn.broken = RuntimeError("no voice")
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    assert (asked, permissions(recorded)) == ([], [("p1", "Write", Deny(BROKEN))])


async def test_a_permission_whose_event_cannot_be_written_is_refused_at_its_hook_and_said_once() -> None:
    """Never left unanswered, which would leave it to Claude Code's own dialog on a screen nobody sees."""
    recorded: list[Entry] = []
    brain, _ = await _in_turn(recorded, unasked)

    def full(entry: Entry) -> None:
        if isinstance(entry, WideEvent) and entry.event == "brain.permission":
            raise ValueError("the log is full")
        recorded.append(entry)

    brain._record = full  # pyright: ignore[reportPrivateUsage]
    other = {"hook_event_name": "PermissionRequest", "session_id": "b2", "transcript_path": "/brain/projects/-brain-cwd/b2.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": "p9", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    said, sink = failures()
    try:
        assert await brain.posted("PermissionRequest", json.dumps(other).encode()) == hook_output(Deny(BROKEN))
    finally:
        logger.remove(sink)
    assert said == ["the brain's own work failed: ValueError('the log is full')"]


def test_the_brain_is_posted_no_hook_that_says_a_tool_ran() -> None:
    """What a withdrawn brain dialog means rests on this: the brain's model hears a tool run only from these hooks, so a
    dialog it lets go is one Claude Code went on past unanswered, never one whose tool ran (`refused`). A hook added
    here is a decision about that."""
    assert {"PostToolUse", "PostToolUseFailure"}.isdisjoint(HOOKS)


@pytest.mark.parametrize("asking", ["ask and wait", "ask twice and wait"], ids=["one dialog held", "two dialogs stacked"])
async def test_a_turn_stopped_while_its_dialogs_are_held_refuses_each_with_interrupt_ends_stopped_and_holds_none(tmp_path: Path, fake_claude: Path, asking: str) -> None:
    recorded: list[Entry] = []
    held: list[Asked] = []
    count = 2 if asking == "ask twice and wait" else 1
    brain = await start(launch(tmp_path), recorded.append)
    try:
        turn = asyncio.ensure_future(brain.ask(asking, held.append))
        await until(lambda: len(held) == count)
        assert brain.interrupt(brain.answering()) is True
        assert await asyncio.wait_for(turn, 10) == BrainStopped()
        await until(lambda: [line for line in typed(tmp_path) if line[0] == "permission"] == [["permission", "deny", STOPPED, True]] * count)
        assert _dialogs(brain) == ()
        assert [await asked.decided() for asked in held] == [DenyAndInterrupt(STOPPED)] * count
        # A yes after the stop answers nothing.
        held[0].settle(Allow())
        assert await held[0].decided() == DenyAndInterrupt(STOPPED)
    finally:
        await brain.stop()
    assert permissions(recorded) == [("p1", "Write", DenyAndInterrupt(STOPPED))] * count


async def test_the_turn_typed_after_a_clear_of_what_was_left_running_is_asked_its_permissions(tmp_path: Path, fake_claude: Path) -> None:
    """The clear's refusal lasts until a prompt starts new work, and the next turn's own prompt is that: its dialog is
    held and put to the user, never refused as the work the clear stopped."""
    held: list[Asked] = []

    def allowed(asked: Asked) -> None:
        held.append(asked)
        asked.settle(Allow())

    brain = await start(launch(tmp_path), lambda _entry: None, typings_answered(tmp_path))
    try:
        with pytest.raises(Untaken):
            await brain.ask("garbled", unasked)
        assert await asyncio.wait_for(brain.ask("write", allowed), 10) == BrainAnswered("p2", None)
    finally:
        await brain.stop()
    assert len(held) == 1
    assert ["permission", "allow", "", False] in typed(tmp_path)


async def test_a_turn_asked_while_claude_code_still_runs_what_a_stop_did_not_stop_is_never_typed_into_it(tmp_path: Path, fake_claude: Path) -> None:
    stopping = asyncio.Event()

    async def pause(seconds: float) -> None:
        await (stopping.wait() if seconds == INTERRUPT_SECONDS else asyncio.sleep(seconds))

    brain = await start(launch(tmp_path), lambda _entry: None, pause)
    try:
        asking = asyncio.ensure_future(brain.ask("stuck", unasked))
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt"])
        assert brain.interrupt(brain.answering()) is True
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt", "escape", "ctrl_c"])
        stopping.set()
        with pytest.raises(StopUnrecorded):
            await asyncio.wait_for(asking, 10)
        stopping.clear()
        # The next turn clears what still runs first, and waits for the clear to take: it never does, so nothing is typed.
        following = asyncio.ensure_future(brain.ask("write", unasked))
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt", "escape", "ctrl_c", "escape", "ctrl_c"])
        stopping.set()
        with pytest.raises(StopUnrecorded, match="did not stop what hands pressed Escape for"):
            await asyncio.wait_for(following, 10)
    finally:
        await brain.stop()
    assert [line[0] for line in typed(tmp_path)].count("prompt") == 1


async def test_a_brain_that_exits_while_a_stop_waits_ends_the_turn_gone_not_unrecorded_after_the_bound(tmp_path: Path, fake_claude: Path) -> None:
    async def pause(seconds: float) -> None:
        # The stop's bound never passes: only the brain's exit ends its wait.
        await (asyncio.Event().wait() if seconds == INTERRUPT_SECONDS else asyncio.sleep(seconds))

    brain = await start(launch(tmp_path), lambda _entry: None, pause)
    try:
        asking = asyncio.ensure_future(brain.ask("stuck", unasked))
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt"])
        assert brain.interrupt(brain.answering()) is True
        await until(lambda: [line[0] for line in typed(tmp_path)] == ["prompt", "escape", "ctrl_c"])
        os.kill(brain.pid, signal.SIGKILL)
        with pytest.raises(BrainGone):
            await asyncio.wait_for(asking, 10)
    finally:
        await brain.stop()


# A turn Claude Code opens for a background task's result, as its prompt hook names it (backgrounded_then_notified.jsonl, p2).
NOTICE = "<task-notification>\n<task-id>b74jw7b5h</task-id>\n<status>completed</status>\n</task-notification>"


def _owning() -> Brain:
    """A brain built bare, its model and its `ClaudeTurns` as `Brain` builds them, heard only through what the model hands
    over: the words each prompt hook filed (`_took`), and each telling (`_told`)."""
    brain = object.__new__(Brain)
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    _modelled(brain)
    return brain


def _typed_turn(brain: Brain, words: str) -> _Turn:  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    turn = brain._turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(), TAKE_SECONDS, PromptText(f" {words}"))  # pyright: ignore[reportPrivateUsage]
    return turn


def _took(brain: Brain, prompt: str, words: str, queued: bool = False) -> None:
    brain._took(Submitted(SessionId("b1"), PromptId(prompt), words, queued))  # pyright: ignore[reportPrivateUsage]


def _told(brain: Brain, prompt: str | None, ids: Sequence[str], waiting: tuple[str, ...] = ()) -> None:
    brain._told(Summarise(SessionId("b1"), None if prompt is None else PromptId(prompt), "Done.", waiting, ids=frozenset(map(PromptId, ids))))  # pyright: ignore[reportPrivateUsage]


async def test_a_turn_told_under_an_id_its_hook_did_not_name_ends_its_own_turn_by_the_ids_its_telling_carries() -> None:
    # A message queued into the turn flushed it onto p2: its telling goes by p2, and carries p1 too (`Summarise.ids`).
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    assert turn.running is not None and turn.running.ids == {"p1"}
    _told(brain, "p2", ("p1", "p2"))
    assert (turn.answered.result(), turn.running.matched, brain._claude_turns.open) == (BrainAnswered(PromptId("p2"), None), "id", ())  # pyright: ignore[reportPrivateUsage]


async def test_a_telling_that_names_no_prompt_ends_the_turn_its_ids_are() -> None:
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, None, ("p1",))
    assert (turn.answered.result(), brain._claude_turns.open) == (BrainAnswered(None, None), ())  # pyright: ignore[reportPrivateUsage]


async def test_a_notification_opened_before_the_last_turns_stop_is_applied_is_a_turn_of_its_own_each_ending_on_its_own_telling() -> None:
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    # Claude Code ran p1 to its end and opened p2 for a background task's result before hands applied p1's Stop.
    _took(brain, "p2", NOTICE)
    typed, notified = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert (brain.current(), typed.opener, notified.opener, notified.awaited) == (notified, "typed", "notified", False)
    _told(brain, "p1", ("p1",))
    assert (turn.answered.result(), notified.ended.done(), brain._claude_turns.opened_itself) == (BrainAnswered(PromptId("p1"), None), False, (notified,))  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p2", ("p2",))
    assert (notified.ended.result(), notified.matched, brain.current(), brain._claude_turns.open) == (BrainAnswered(PromptId("p2"), None), "id", None, ())  # pyright: ignore[reportPrivateUsage]


async def test_hands_words_filed_into_a_notification_turn_are_answered_by_that_turn() -> None:
    brain = _owning()
    _took(brain, "p1", NOTICE)
    (notified,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    turn = _typed_turn(brain, "and the tests?")
    # Typed while the notification's turn runs: Claude Code files the words under its id, and delivers them into it.
    _took(brain, "p1", "and the tests?", queued=True)
    assert (turn.running, notified.awaited) == (notified, True)
    _told(brain, "p1", ("p1",))
    assert (turn.answered.result(), notified.ended.result()) == (BrainAnswered(PromptId("p1"), None), BrainAnswered(PromptId("p1"), None))


async def test_a_notification_queued_into_a_turn_nobody_waits_on_ends_nothing_and_the_turn_after_ends_on_its_own_telling() -> None:
    # The v8 review's case: a second notification's hook comes under the running turn's id, which binds nothing new.
    brain = _owning()
    _took(brain, "p1", NOTICE)
    _took(brain, "p1", NOTICE.replace("b74jw7b5h", "c11"), queued=True)
    (first,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p1", ("p1",))
    _took(brain, "p2", NOTICE.replace("b74jw7b5h", "d22"))
    (second,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p2", ("p2",))
    assert (first.ended.result(), second.ended.result(), brain._claude_turns.open) == (BrainAnswered(PromptId("p1"), None), BrainAnswered(PromptId("p2"), None), ())  # pyright: ignore[reportPrivateUsage]


async def test_a_turn_of_a_kind_added_to_the_openers_opens_and_ends_with_no_other_change(monkeypatch: pytest.MonkeyPatch) -> None:
    # [LAW:one-type-per-behavior] a scheduled prompt is one more row: nothing that matches a turn to its end reads the kind.
    monkeypatch.setattr(process, "OPENERS", [*process.OPENERS, ("scheduled", re.compile(r"<scheduled-prompt>"))])
    brain = _owning()
    _took(brain, "p1", "<scheduled-prompt>check the build</scheduled-prompt>")
    (scheduled,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p1", ("p1",))
    assert (scheduled.opener, scheduled.ended.result(), scheduled.matched) == ("scheduled", BrainAnswered(PromptId("p1"), None), "id")


class _Keys:
    """The brain's keyboard, as fritter's typist takes keys: each typing and press, in order."""

    def __init__(self) -> None:
        self.pressed: list[str] = []

    def type(self, text: str) -> None:
        self.pressed.append(text)

    def press(self, key: str) -> None:
        self.pressed.append(key)


def _keyed(brain: Brain, recorded: list[Entry]) -> _Keys:
    """A bare brain a turn can be asked of: typed into a keyboard of the test's, with a take that never runs out."""
    loop = asyncio.get_running_loop()
    keys = _Keys()
    brain._typist, brain._record, brain._typing, brain._input = keys, recorded.append, set(), asyncio.Lock()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    brain._exit, brain._cleared = loop.create_future(), float("-inf")  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    async def never(_seconds: float) -> None:
        await loop.create_future()

    brain._pause = never  # pyright: ignore[reportPrivateUsage]
    with unit("brain.launch", lambda _entry: None):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    return keys


async def test_a_turn_asked_while_a_notification_turn_runs_is_typed_at_once_into_its_queue_and_never_escapes_it() -> None:
    recorded: list[Entry] = []
    brain = _owning()
    keys = _keyed(brain, recorded)
    reading = asyncio.ensure_future(brain._reader.keep_reading(0.01))  # pyright: ignore[reportPrivateUsage]
    # Claude Code opens a turn of its own for a background task's result, as its prompt hook says.
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt=NOTICE))
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Typed with nothing held behind the notification's turn, and nothing pressed to stop it: no Escape, no Ctrl-C.
    await until(lambda: keys.pressed == [" and now?"])
    # Claude Code queues the words under the turn it runs: they are filed into it, which now answers them.
    _took(brain, "p1", "and now?", queued=True)
    assert (brain.answering(), notified.awaited, notified.ended.done()) == (notified, True, False)
    asking.cancel()
    reading.cancel()


async def test_a_brain_stop_let_go_undecided_fails_every_turn_open_and_the_turn_it_answers_saying_the_order_was_lost() -> None:
    recorded: list[Entry] = []
    brain = _owning()
    _keyed(brain, recorded)
    brain._sessions._stop_hold = 0.05  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(1000)), 1.0))  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask("hello", unasked))
    await until(lambda: brain._turn is not None)  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="hello"))
    # A notification Claude Code opened, then a Stop under an id no record names, which no read decides in its hold.
    _took(brain, "p2", NOTICE)
    await brain.posted("Stop", _hooked("Stop", "q", last_assistant_message="Hi.", stop_hook_active=False))
    with pytest.raises(process.OrderLost):
        await asking
    assert brain._claude_turns.open == ()  # pyright: ignore[reportPrivateUsage]
    [turn] = events(recorded, "brain.turn")
    assert (turn.outcome, turn.facts["order_lost"], turn.facts["ids"]) == ("failed", "the Stop of q was let go before it was decided", ("p1",))


async def test_a_brain_prompt_hook_answered_past_its_timeout_fails_every_turn_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # Answered later than Claude Code waits on it, the turn's first request may have left before hands heard its id.
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    monkeypatch.setattr(process, "TAKE_SECONDS", -1.0)
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p2", prompt=NOTICE))
    with pytest.raises(process.OrderLost):
        turn.answered.result()
    assert (turn.order_lost, brain._claude_turns.open) == ("the UserPromptSubmit hook of p2 was answered in 0.0s, past the -1s hands has to answer it in", ())  # pyright: ignore[reportPrivateUsage]


async def test_a_permission_hook_claude_code_gave_up_on_fails_no_turn() -> None:
    # Abandoned is a permission's, not a Stop's: the order of the brain's turns does not rest on it.
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Abandoned(SessionId("b1"), RequestId("r1"), 1.0))  # pyright: ignore[reportPrivateUsage]
    assert (turn.answered.done(), len(brain._claude_turns.open)) == (False, 1)  # pyright: ignore[reportPrivateUsage]


async def test_a_task_result_queued_behind_a_turn_runs_as_its_own_turn_behind_it_never_as_the_typed_turns() -> None:
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    # Claude Code queued the notification under the running turn's id, and it still waits as that turn is told.
    _took(brain, "p1", NOTICE, queued=True)
    _told(brain, "p1", ["p1"], waiting=(NOTICE,))
    assert turn.answered.result() == BrainAnswered(PromptId("p1"), None)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert (notified.opener, notified.awaited, notified.ids) == ("notified", False, frozenset())
    # Its telling is its own: no turn of hands' is ended by it.
    _told(brain, "p9", ["p9"])
    assert (notified.ended.result(), notified.matched, brain._claude_turns.open) == (BrainAnswered(PromptId("p9"), None), "behind", ())  # pyright: ignore[reportPrivateUsage]


async def test_a_task_result_queued_behind_a_turn_already_told_runs_as_the_turn_behind_it() -> None:
    brain = _owning()
    _took(brain, "p1", NOTICE, queued=True)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert (notified.opener, notified.ids) == ("notified", frozenset())


async def test_the_answering_turn_follows_hands_words_from_the_turn_they_were_filed_into_to_the_turn_behind_it() -> None:
    brain = _owning()
    _took(brain, "n1", NOTICE)
    turn = _typed_turn(brain, "and the tests?")
    _took(brain, "n1", "and the tests?", queued=True)
    (notified,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert brain.answering() is notified
    _told(brain, "n1", ["n1"], waiting=(" and the tests?",))
    behind = brain.answering()
    assert behind is not None and behind is not notified and not turn.answered.done()
    _told(brain, "p2", ["p2"])
    assert turn.answered.result() == BrainAnswered(PromptId("p2"), None)


async def test_a_turn_claude_code_opened_itself_is_stopped_when_its_work_is_the_work_shown() -> None:
    recorded: list[Entry] = []
    brain = _owning()
    keys = _keyed(brain, recorded)
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    # No turn of hands' is in flight: the notification's is stopped all the same, so the user is never locked out by it.
    assert brain.interrupt(notified) is True
    await until(lambda: "escape" in keys.pressed)
    # A turn of no kind Claude Code opens itself, and none at all, is not the brain's to stop on its own.
    assert brain.interrupt(None) is False
    for task in brain._typing:  # pyright: ignore[reportPrivateUsage]
        task.cancel()


async def test_a_task_result_ctrl_c_cleared_from_claude_codes_queue_ends_its_turn_unran_and_holds_no_turn_after_it() -> None:
    recorded: list[Entry] = []
    brain = _owning()
    keys = _keyed(brain, recorded)
    _took(brain, "n1", NOTICE)
    # A second result waits as the first is told: it runs as the turn behind, which no hook opens.
    _told(brain, "n1", ["n1"], waiting=(NOTICE,))
    (behind,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert brain.interrupt(behind) is True
    await until(lambda: behind.ended.done())
    # Ctrl-C cleared it before it ran: no telling will end it, so the clear does, and nothing is left open to wait on.
    assert ("ctrl_c" in keys.pressed, type(behind.ended.exception()), brain._claude_turns.open) == (True, process.Unran, ())  # pyright: ignore[reportPrivateUsage]
    turn = asyncio.ensure_future(brain.ask("and now?", unasked))
    await until(lambda: " and now?" in keys.pressed)
    turn.cancel()


async def test_a_telling_of_no_turn_names_no_turn_behind() -> None:
    brain = _owning()
    _took(brain, "n1", NOTICE)
    turn = _typed_turn(brain, "hello")
    _took(brain, "n1", "hello", queued=True)
    _told(brain, "n1", ["n1"], waiting=("hello",))
    (behind,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert brain.answering() is behind
    # A telling that goes by no id ends nothing: the typed words still wait to be answered by the turn they run as.
    _told(brain, None, [])
    assert (behind.ended.done(), behind.ids, turn.answered.done()) == (False, frozenset(), False)


async def test_a_brain_prompt_hook_that_does_not_parse_fails_every_turn_open_as_the_order_lost() -> None:
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    await brain.posted("UserPromptSubmit", b"{not json")
    with pytest.raises(process.OrderLost):
        turn.answered.result()
    assert brain._claude_turns.open == ()  # pyright: ignore[reportPrivateUsage]


async def test_a_brain_stop_answered_past_its_bound_fails_every_turn_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # Decided in its hold, but answered later than hands has to: Claude Code may have gone on, so its order is not trusted.
    brain = _owning()
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(process, "TAKE_SECONDS", -1.0)
    await brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Hi.", stop_hook_active=False))
    # The turn it told is answered; the one still open is not trusted to be the one Claude Code runs next.
    assert turn.answered.result() == BrainAnswered(PromptId("p1"), None)
    with pytest.raises(process.OrderLost, match="the Stop hook of p1 was answered in"):
        notified.ended.result()


async def test_a_notification_that_opens_while_a_turn_waits_for_the_input_is_never_escaped_for_it() -> None:
    recorded: list[Entry] = []
    brain = _owning()
    keys = _keyed(brain, recorded)
    await brain._input.acquire()  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    await until(lambda: brain._turn is not None)  # pyright: ignore[reportPrivateUsage]
    # Past the ask, waiting on the input: Claude Code opens a turn of its own meanwhile.
    _took(brain, "n1", NOTICE)
    brain._input.release()  # pyright: ignore[reportPrivateUsage]
    # Typed into Claude Code's queue behind it, with nothing pressed to stop it.
    await until(lambda: keys.pressed == [" and now?"])
    _told(brain, "n1", ["n1"])
    await until(lambda: keys.pressed == [" and now?"])
    asking.cancel()


async def test_a_brain_stop_released_by_shutdown_loses_no_order() -> None:
    brain = _owning()
    _took(brain, "n1", NOTICE)
    stop = Stopped(SessionId("b1"), "done", mode=None, prompt=PromptId("n1"), again=False, heard=Stamp(0), request=RequestId("r"))
    brain._ordered(SimpleNamespace(happened=stop), "released", 0.0)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert len(brain._claude_turns.open) == 1  # pyright: ignore[reportPrivateUsage]
