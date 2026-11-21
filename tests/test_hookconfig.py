"""The plugin's hooks: one declared permission timeout, every other permission time derived from it, and a hooks.json
that is hookconfig's table and nothing else."""

import http.server
import importlib.util
import json
import socket
import threading
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

from conftest import PLUGIN, skill_command
from hands.core.effects import AllowWith, Allow, Deny, DenyAndInterrupt, Withdraw
from hands.core.reducer import UNTOLD
from hands.sessions.hookconfig import (
    BRAIN_HOOKS,
    BRAIN_REPLY_BY_SECONDS,
    HOOKS_FILE,
    PERMISSION_HOOK_TIMEOUT_SECONDS,
    PLUGIN_ID,
    POST_TIMEOUT_SECONDS,
    REPLY_MARGIN_SECONDS,
    SHIM_COMMAND,
    STOP_HOLD_SECONDS,
    SUBSCRIBED,
    plugin_hooks,
    post_timeout,
    relayed,
    rendered,
)
from hands.sessions.hooks import hook_output
import pytest

REPOSITORY = Path(__file__).resolve().parent.parent


def test_every_subscribed_hook_runs_the_shim_the_permission_hook_waits_and_tool_hooks_run_in_the_background() -> None:
    hooks = cast(dict[str, object], plugin_hooks()["hooks"])
    # MessageDisplay is posted to the daemon with no shim: tests/test_display.py.
    assert list(hooks) == [*SUBSCRIBED, "MessageDisplay"]
    command: dict[str, object] = {"type": "command", "command": SHIM_COMMAND, "args": []}
    background = {"async": True}
    for event in SUBSCRIBED:
        entries = hooks[event]
        declared = {
            "PermissionRequest": {"timeout": PERMISSION_HOOK_TIMEOUT_SECONDS},
            "PostToolUse": background,
            "PostToolUseFailure": background,
            # Only passed on, so nothing in the session waits on what hands makes of them.
            **{passed: background for passed in ("PermissionDenied", "SubagentStart", "SubagentStop", "TaskCompleted", "ConfigChange", "PreCompact")},
        }.get(event, {})
        # Claude Code's internal agents fire the subagent hooks under the empty name, which ".+" never matches.
        matched = {"matcher": ".+"} if event in ("SubagentStart", "SubagentStop") else {}
        assert entries == [{**matched, "hooks": [{**command, **declared}]}]


def test_every_command_the_plugin_runs_is_a_script_the_wheel_installs() -> None:
    # The plugin is a plain directory, so nothing renders its commands: a hook or skill naming a script pyproject does not
    # define (renamed in one place, not the other) fails in every session, and only here is that seen before a release.
    scripts = set(tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]["scripts"])
    hooks = cast(dict[str, list[dict[str, list[dict[str, str]]]]], json.loads((PLUGIN / HOOKS_FILE).read_text())["hooks"])
    run = {hook["command"] for groups in hooks.values() for group in groups for hook in group["hooks"] if hook["type"] == "command"}
    run |= {shlex.split(skill_command(skill.parent.name))[0] for skill in PLUGIN.glob("skills/*/SKILL.md")}
    assert run and run <= scripts, f"{sorted(run - scripts)} are run by the plugin and not defined in pyproject's [project.scripts]"


def test_the_checked_in_hooks_json_is_what_hookconfig_declares() -> None:
    # [LAW:one-source-of-truth] hooks.json is generated from hookconfig; a hand edit, or a change to hookconfig not
    # regenerated with `python -m hands.sessions.hookconfig > plugin/hooks/hooks.json`, fails here.
    assert (PLUGIN / HOOKS_FILE).read_text(encoding="utf-8") == rendered()


@pytest.mark.outside
def test_the_module_prints_the_hooks_json() -> None:
    printed = subprocess.run([sys.executable, "-m", "hands.sessions.hookconfig"], capture_output=True, text=True, check=True)
    assert printed.stdout == rendered()
    assert set(json.loads(printed.stdout)["hooks"]) == {*SUBSCRIBED, "MessageDisplay"}


def test_the_shim_waits_a_whole_day_as_claude_code_holds_the_hook_and_only_the_brain_replies_before_it_ends() -> None:
    assert PERMISSION_HOOK_TIMEOUT_SECONDS == 86400
    assert post_timeout("PermissionRequest") == PERMISSION_HOOK_TIMEOUT_SECONDS
    assert BRAIN_REPLY_BY_SECONDS == PERMISSION_HOOK_TIMEOUT_SECONDS - REPLY_MARGIN_SECONDS
    # A Stop's hook is held past the window the reducer waits for the record that names it, and the shim waits the
    # hold longer than for any other hook.
    assert STOP_HOLD_SECONDS > UNTOLD / 1000
    assert post_timeout("Stop") == POST_TIMEOUT_SECONDS + STOP_HOLD_SECONDS
    assert post_timeout("Notification") == POST_TIMEOUT_SECONDS


def test_a_reply_is_printed_in_the_shape_claude_code_reads() -> None:
    decided = {"hookEventName": "PermissionRequest"}
    assert hook_output(Allow()) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "allow"}}}
    assert hook_output(Deny("not on main")) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "deny", "message": "not on main"}}}
    assert hook_output(DenyAndInterrupt("hands stopped this turn")) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "deny", "message": "hands stopped this turn", "interrupt": True}}}
    assert hook_output(Withdraw()) is None
    answered: dict[str, object] = {"questions": [{"question": "Which?", "options": []}], "answers": {"Which?": "this"}}
    assert hook_output(AllowWith(answered)) == {"hookSpecificOutput": {**decided, "decision": {"behavior": "allow", "updatedInput": answered}}}


def test_the_plugin_id_is_the_plugin_s_name_at_the_marketplace_s_whose_entry_is_a_directory_of_this_repository() -> None:
    marketplace = json.loads((REPOSITORY / ".claude-plugin" / "marketplace.json").read_text())
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert f"{plugin['name']}@{marketplace['name']}" == PLUGIN_ID
    assert [entry["source"] for entry in marketplace["plugins"] if entry["name"] == plugin["name"]] == ["./plugin"]
    # A directory source is installed and updated by copying it, so Claude Code runs nothing of hands' and asks no one to accept it.


def test_the_recording_rig_takes_every_hook_the_brain_posts() -> None:
    # Imported, as nothing else in the suite does: the rig is run by hand, so a name it lost would wait for that run.
    spec = importlib.util.spec_from_file_location("record_turns", REPOSITORY / "scripts" / "record_turns.py")
    assert spec is not None and spec.loader is not None
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    assert set(BRAIN_HOOKS) <= set(rig.HOOKS)


class _Answering(http.server.BaseHTTPRequestHandler):
    """hands' listener as a relayed hook meets it: /ok echoes what it was posted, wrapped; /broken fails, saying why."""

    def do_POST(self) -> None:
        posted = self.rfile.read(int(self.headers["Content-Length"]))
        status, body = (200, b'{"posted": ' + posted + b"}") if self.path == "/ok" else (500, b"hands broke")
        # Without a Date header, which would read the machine's clock.
        self.send_response_only(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _run(hook: dict[str, object], payload: bytes) -> subprocess.CompletedProcess[bytes]:
    # As Claude Code runs a command hook in exec form: the payload on its stdin, its stdout the hook's output.
    return subprocess.run([cast(str, hook["command"]), *cast(list[str], hook["args"])], input=payload, capture_output=True)


@pytest.mark.outside
def test_a_relayed_hook_prints_hands_answer_and_fails_showing_any_other_or_none() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Answering)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # A listener that takes the connection and never answers.
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        answered = _run(relayed(f"{url}/ok", 5), b'{"hook_event_name": "SessionStart"}')
        assert (answered.returncode, json.loads(answered.stdout)) == (0, {"posted": {"hook_event_name": "SessionStart"}})
        broken = _run(relayed(f"{url}/broken", 5), b"{}")
        assert broken.returncode != 0 and broken.stdout == b"hands broke"
        unanswered = _run(relayed(f"http://127.0.0.1:{silent.getsockname()[1]}/SessionStart", 1), b"{}")
        assert unanswered.returncode != 0 and unanswered.stderr
    finally:
        server.shutdown()
        silent.close()
