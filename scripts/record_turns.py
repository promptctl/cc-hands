"""Record what the pinned Claude Code says over one scripted turn, on each channel hands reads it by: the hooks it posts,
the transcript records it writes, and the status it sets in its session file. The three are merged in time order,
scrubbed of the machine that recorded them, and written to tests/fixtures/recorded_turns/<name>.jsonl, where the
stand-in Claude Code and the reducer's tests read them (tests/turns.py). It records on the brain's pin, checked as the
brain checks it (hands.brain.pinned); docs/claude-code-behavior.md says which release the recordings are of.

Run by hand on a machine logged into Claude Code, never by a test: it spends a few haiku turns on the account.

    uv run python scripts/record_turns.py                  # every turn
    uv run python scripts/record_turns.py sent_now ...     # the turns named

Each line is one of `{"at": s, "posted": <hook body>}`, `{"at": s, "written": <transcript record>}` or
`{"at": s, "status": <session file's status>}`, `at` seconds from the first. A status's `at` is when Claude Code set
it (its statusUpdatedAt), and it holds no other time. A record's timestamp is moved the same way, so it reads as
seconds from the first in its own format (1970-01-01T00:00:14.045Z is 14.045 s in), and a player moves every time
onto its own clock by one shift.
"""

import datetime
import getpass
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from hands.brain import pinned
from hands.sessions.home import default_home

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "recorded_turns"
CONFIG = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
# The hooks a recording logs: every one the brain or a wrapped session subscribes to, and PreToolUse, which neither
# does, so a recording shows when a call started.
HOOKS = ("UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop", "StopFailure", "PermissionRequest", "Notification", "SubagentStop")
TMUX = "record-turns"
# How long a session stays idle before its recording ends: longer than any gap between one turn and the next it runs.
SETTLED = 5.0

# The records a turn is made of. The rest (a prompt's snapshot of the system prompt, hook context, skill and agent
# listings, file-history snapshots, titles, modes) carry the recording machine's setup, never the turn.
KEPT = {"user", "assistant", "system", "queue-operation"}
KEPT_ATTACHMENTS = {"queued_command"}
# The recording machine's, in every hook body and record: never read as the turn. The player puts back the session's own.
DROPPED = {"cwd", "entrypoint", "gitBranch", "userType", "version", "sessionId", "session_id", "transcript_path", "hookInfos", "hookAdditionalContext", "hookErrors"}


def slow_mcp() -> None:
    """A stdio MCP server whose `wait` returns WAITED once `seconds` have passed, each call on a thread of its own, so a
    call runs as long as an MCP tool the brain calls can."""
    lock = threading.Lock()

    def send(message: dict) -> None:
        with lock:
            sys.stdout.write(json.dumps(message) + "\n")
            sys.stdout.flush()

    def call(id: object, seconds: float) -> None:
        time.sleep(seconds)
        send({"jsonrpc": "2.0", "id": id, "result": {"content": [{"type": "text", "text": "WAITED"}]}})

    wait = {"name": "wait", "description": "Waits the given number of seconds, then returns WAITED.", "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}}
    for line in sys.stdin:
        message = json.loads(line)
        id, method = message.get("id"), message.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": id, "result": {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "slow", "version": "1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": id, "result": {"tools": [wait]}})
        elif method == "tools/call":
            threading.Thread(target=call, args=(id, float(message["params"]["arguments"]["seconds"])), daemon=True).start()
        elif id is not None:
            send({"jsonrpc": "2.0", "id": id, "result": {}})


class Rig:
    """The pinned Claude Code in a tmux session of its own, in a directory of its own, its hooks logged to the
    millisecond and its session file's status read every few milliseconds."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="record-turns-", dir="/tmp")).resolve()
        self.cwd = self.dir / "work"
        self.cwd.mkdir()
        self.log = self.dir / "hooks.log"
        self.log.touch()
        hook = self.dir / "hook.sh"
        hook.write_text(f"#!/bin/sh\n{{ printf '%s %s ' \"$(perl -MTime::HiRes=time -e 'printf \"%.3f\", time')\" \"$1\"; tr -d '\\n'; echo; }} >> {shlex.quote(str(self.log))}\n")
        hook.chmod(0o755)
        settings = {
            "hooks": {event: [{"hooks": [{"type": "command", "command": f"{hook} {event}"}]}] for event in HOOKS},
            "permissions": {"allow": ["Bash(*)", "mcp__slow__wait"]},
        }
        mcp = {"mcpServers": {"slow": {"type": "stdio", "command": sys.executable, "args": ["-I", str(Path(__file__).resolve()), "--slow-mcp"]}}}
        command = [str(pinned.verified(default_home(os.environ).brain)), "--model", "haiku", "--setting-sources", "project", "--settings", json.dumps(settings), "--mcp-config", json.dumps(mcp), "--strict-mcp-config"]
        # Background tasks on, as the brain will run: the turns recorded here are the ones they open.
        run(["tmux", "kill-session", "-t", TMUX], check=False)
        run(["tmux", "new-session", "-d", "-s", TMUX, "-x", "200", "-y", "50", "-c", str(self.cwd), "exec env -u CLAUDE_CODE_DISABLE_BACKGROUND_TASKS DISABLE_AUTOUPDATER=1 " + shlex.join(command)])
        self.pid = int(run(["tmux", "display-message", "-p", "-t", TMUX, "#{pane_pid}"]).strip())
        self.statuses: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()
        until(lambda: any(status["status"] == "idle" for _, status in self.statuses), 120, "the session never went idle")

    def _watch(self) -> None:
        file = CONFIG / "sessions" / f"{self.pid}.json"
        last = None
        while not self._stop.is_set():
            try:
                body = json.loads(file.read_bytes())
            except (FileNotFoundError, ValueError):
                body = {}
            if "status" in body:
                status = {key: body[key] for key in ("status", "waitingFor") if key in body}
                seen = (status, body["statusUpdatedAt"])
                if seen != last:
                    last = seen
                    self.statuses.append((body["statusUpdatedAt"] / 1000, status))
            time.sleep(0.005)

    def hooks(self) -> list[tuple[float, dict]]:
        out = []
        # The last line is the one a hook may be writing still: read only those it has ended.
        for line in self.log.read_text().split("\n")[:-1]:
            if line.strip():
                at, event, body = line.split(" ", 2)
                out.append((float(at), {"hook_event_name": event, **json.loads(body)}))
        return out

    def count(self, event: str) -> int:
        return sum(1 for _, body in self.hooks() if body["hook_event_name"] == event)

    def keys(self, *keys: str) -> None:
        run(["tmux", "send-keys", "-t", TMUX, *keys])

    def say(self, text: str) -> None:
        """Type `text` at the prompt and send it. A session just started ignores a Return for its first seconds, so the
        Return is sent again until its UserPromptSubmit is posted."""
        before = self.count("UserPromptSubmit")
        self.keys("-l", text)
        time.sleep(1.5)
        for _ in range(45):
            self.keys("Enter")
            try:
                until(lambda: self.count("UserPromptSubmit") > before, 4, "")
                return
            except TimeoutError:
                pass
        raise SystemExit("the prompt was never sent")

    def interject(self, text: str) -> None:
        """Type `text` while a turn runs, and send it: Claude Code takes it at once, under the running prompt id."""
        before = self.count("UserPromptSubmit")
        self.keys("-l", text)
        time.sleep(1.0)
        self.keys("Enter")
        until(lambda: self.count("UserPromptSubmit") > before, 20, "the interjection was never taken")

    def send_now(self) -> None:
        """Ctrl+X Ctrl+S: what was typed mid-turn reaches the model at once, every running call moved to the background."""
        self.keys("C-x")
        time.sleep(0.3)
        self.keys("C-s")

    def started(self, calls: int = 1) -> None:
        until(lambda: self.count("PreToolUse") >= calls, 120, "the call never started")

    def stops(self, stops: int) -> None:
        until(lambda: self.count("Stop") >= stops, 240, "the turn never stopped")

    def settled(self) -> None:
        """Wait for the session to stay idle for SETTLED seconds, so the recording holds every turn its work opens (a
        task moved to the background opens one when it ends) and the status it ends on. Claude Code sets idle for a
        moment between a turn and the next it runs, so an idle that has not held is not the end."""
        def held() -> bool:
            at, status = self.statuses[-1]
            return status["status"] == "idle" and time.time() - at >= SETTLED
        until(held, 240, "the session never settled idle")

    def close(self) -> list[tuple[float, str, dict]]:
        """End the session, and what it said on each channel, in time order."""
        time.sleep(1)
        self._stop.set()
        run(["tmux", "kill-session", "-t", TMUX], check=False)
        hooks = self.hooks()
        [session] = {body["session_id"] for _, body in hooks}
        [transcript] = (CONFIG / "projects").glob(f"*/{session}.jsonl")
        records = [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]
        lines = [(at, "posted", body) for at, body in hooks]
        lines += [(_seconds(record["timestamp"]), "written", record) for record in records if "timestamp" in record]
        lines += [(at, "status", status) for at, status in self.statuses]
        self.transcript = transcript
        self.session = session
        return sorted(lines, key=lambda line: line[0])


BASH = 'Run this exact bash command in the foreground, not in the background: python3 -c "import time; time.sleep({seconds}); print(42)" -- then tell me the number it printed. Do not call TaskStop, Monitor, Read or sleep; if it is moved to the background, just answer anything else asked and end your turn saying so.'
INTERJECTION = "Interjection: what is 2+2? Answer it right away."


def _backgrounded_then_notified(rig: Rig) -> None:
    rig.say(BASH.format(seconds=20))
    rig.started()
    time.sleep(4)
    rig.keys("C-b")
    rig.stops(2)


def _notified_mid_turn(rig: Rig) -> None:
    rig.say('First run the bash command `sleep 8; echo EARLY` with run_in_background true. Then, in a separate foreground Bash call (not in the background), run python3 -c "import time; time.sleep(25); print(42)". Then say both results. Use no other tools.')
    rig.stops(1)


def _joined_at_boundary(rig: Rig) -> None:
    rig.say(BASH.format(seconds=20))
    rig.started()
    time.sleep(4)
    rig.interject(INTERJECTION)
    rig.stops(1)


def _joined_as_next_prompt(rig: Rig) -> None:
    rig.say("Without using any tools, write a 300-word story about a fox.")
    time.sleep(2.5)
    rig.interject("Also: what is 3+3?")
    rig.stops(2)


def _sent_now(rig: Rig) -> None:
    rig.say(BASH.format(seconds=40))
    rig.started()
    time.sleep(4)
    rig.interject(INTERJECTION)
    time.sleep(1)
    rig.send_now()
    rig.stops(1)
    # The moved call ends after the Stop, and opens a turn of its own for its notification.
    rig.stops(2)


def _sent_now_mcp(rig: Rig) -> None:
    rig.say("Call the mcp__slow__wait tool once with seconds 25, then say what it returned. If it is moved to the background, answer anything else asked and end your turn; use no other tools.")
    rig.started()
    time.sleep(4)
    rig.interject(INTERJECTION)
    time.sleep(1)
    rig.send_now()
    rig.stops(2)


def _sent_now_mid_batch(rig: Rig) -> None:
    rig.say('In ONE message, make three Bash tool calls (foreground, do not use run_in_background): python3 -c "import time; time.sleep(30); print(1)" and python3 -c "import time; time.sleep(30); print(2)" and python3 -c "import time; time.sleep(30); print(3)". Do not call TaskStop, Monitor or Read. Then tell me the numbers.')
    rig.started()
    time.sleep(3)
    rig.interject("Interjection: what is 5+5? Answer it right away.")
    time.sleep(1)
    rig.send_now()
    rig.stops(1)


# Each turn by its name, the file it is written to: the scripted keys that make it happen on the pin.
TURNS: dict[str, Callable[[Rig], None]] = {
    "backgrounded_then_notified": _backgrounded_then_notified,
    "notified_mid_turn": _notified_mid_turn,
    "joined_at_boundary": _joined_at_boundary,
    "joined_as_next_prompt": _joined_as_next_prompt,
    "sent_now": _sent_now,
    "sent_now_mcp": _sent_now_mcp,
    "sent_now_mid_batch": _sent_now_mid_batch,
}


def record(name: str) -> Path:
    rig = Rig()
    try:
        TURNS[name](rig)
        rig.settled()
    finally:
        lines = rig.close()
    out = OUT / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(normalized(lines, rig))
    return out


def normalized(lines: list[tuple[float, str, dict]], rig: Rig) -> str:
    """What the run said of the turn, with nothing of the machine it ran on: refused, if anything of it is left."""
    kept = [line for line in lines if line[1] != "written" or _kept(line[2])]
    start = kept[0][0]
    text = "\n".join(json.dumps({"at": round(at - start, 3), kind: _rebased(_scrubbed(body), start)}, separators=(",", ":")) for at, kind, body in kept) + "\n"
    machine = Machine(rig)
    for real, generic in machine.paths:
        text = text.replace(real, generic)
    text = text.replace(rig.session, "SESSION")
    # Prompt ids in the order they first appear, p1 the turn typed; the API's message and tool-use ids likewise, which
    # name the recording account's requests.
    for pattern, prefix in ((r'"(?:prompt_id|promptId)":"([0-9a-f-]{36})"', "p"), (r"(msg_[0-9A-Za-z]{12,})", "msg_"), (r"(toolu_[0-9A-Za-z]{12,})", "toolu_")):
        found: dict[str, str] = {}
        for value in re.findall(pattern, text):
            found.setdefault(value, f"{prefix}{len(found) + 1}")
        for real, local in found.items():
            text = text.replace(real, local)
    for private in machine.private:
        if private in text:
            raise SystemExit(f"{private!r} of the recording machine survived the scrub")
    return text


class Machine:
    """What the run itself says of the machine that recorded it: the paths a recording's are made generic from, and
    every value a recording must not hold."""

    def __init__(self, rig: Rig) -> None:
        uid = os.getuid()
        project = rig.transcript.parent.name
        temp = os.path.realpath(os.environ.get("TMPDIR", tempfile.gettempdir())).rstrip("/")
        self.paths = [(str(rig.cwd), "/brain/cwd"), (str(rig.cwd).removeprefix("/private"), "/brain/cwd"), (project, "-brain-cwd"), (f"claude-{uid}", "claude-UID"), (temp, "/tmp/me"), (str(Path.home()), "/home/me")]
        git = [run(["git", "config", key], check=False).strip() for key in ("user.email", "user.name")]
        names = [part for value in git for part in re.split(r"[\s@.]+", value) if len(part) >= 4]
        host = socket.gethostname()
        self.private = [value for value in (str(Path.home()), getpass.getuser(), host, host.split(".")[0], temp, str(rig.dir), rig.session, f"claude-{uid}", *git, *names) if len(value) >= 3]


def _kept(record: dict) -> bool:
    return record.get("type") in KEPT or (record.get("type") == "attachment" and record["attachment"].get("type") in KEPT_ATTACHMENTS)


def _scrubbed(value: object) -> object:
    match value:
        case dict():
            return {key: "" if key == "signature" else _scrubbed(inner) for key, inner in value.items() if key not in DROPPED and not _names_a_request(key, inner)}
        case list():
            return [_scrubbed(inner) for inner in value]
        case _:
            return value


def _names_a_request(key: str, value: object) -> bool:
    """A reference to one of the recording account's API requests (requestId, requestRef, serverClassifierRequest, a
    classifier context's request, and any a later release adds): a string under a key that names a request. Counts of
    requests are numbers, and stay."""
    return isinstance(value, str) and "request" in key.lower()


def _rebased(value: object, start: float) -> object:
    """`value` with every time it holds counted from `start`."""
    match value:
        case dict():
            return {key: _moved(key, inner, start) for key, inner in value.items()}
        case list():
            return [_rebased(inner, start) for inner in value]
        case _:
            return value


def _moved(key: str, value: object, start: float) -> object:
    match key, value:
        case "timestamp", str():
            return _iso(_seconds(value) - start)
        case _:
            return _rebased(value, start)


def _seconds(iso: str) -> float:
    return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _iso(seconds: float) -> str:
    return datetime.datetime.fromtimestamp(seconds, datetime.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def run(command: list[str], check: bool = True) -> str:
    return subprocess.run(command, check=check, capture_output=True, text=True).stdout


def until(check: Callable[[], bool], seconds: float, why: str) -> None:
    end = time.monotonic() + seconds
    while not check():
        if time.monotonic() > end:
            raise TimeoutError(why)
        time.sleep(0.2)


if __name__ == "__main__":
    if sys.argv[1:] == ["--slow-mcp"]:
        slow_mcp()
    else:
        for name in sys.argv[1:] or TURNS:
            print(record(name))
