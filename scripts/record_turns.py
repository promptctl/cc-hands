"""Record what the pinned Claude Code says over one scripted turn, on each channel hands reads it by: the hooks it posts,
the transcript records it writes, and the status it sets in its session file. The three are merged in time order,
scrubbed of the machine that recorded them, and written to tests/fixtures/recorded_turns/<name>.jsonl, where the
stand-in Claude Code and the reducer's tests read them (tests/turns.py). It records on the brain's pin, checked as the
brain checks it (hands.brain.pinned); docs/claude-code-behavior.md says which release the recordings are of.

Run by hand on a machine logged into Claude Code, never by a test: it spends a few haiku turns on the account.

    uv run python scripts/record_turns.py                  # every turn
    uv run python scripts/record_turns.py sent_now ...     # the turns named

Each line is one of `{"at": s, "posted": <hook body>}`, `{"at": s, "written": <transcript record>}` or
`{"at": s, "status": <session file's status>}`, in the order they were seen, `at` seconds from the first on one clock:
when the hook arrived, when the record's line was appended, when the status was set in the file (see Rig). A status
holds no other time. A record's own timestamp, when Claude Code made it, is moved by the same start, so it reads as
seconds from the first in its own format (1970-01-01T00:00:14.045Z is 14.045 s in), and a player moves every time
onto its own clock by one shift.
"""

import datetime
import getpass
import http.server
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from hands.brain import pinned, process
from hands.sessions.hookconfig import plugin_hooks
from hands.sessions.home import default_home

TESTS = Path(__file__).resolve().parent.parent / "tests"
sys.path.insert(0, str(TESTS))
from turns import RECORDED, RECORDED_SESSION, project_name  # noqa: E402  [LAW:one-source-of-truth] the fixtures' format is tests/turns.py's

OUT = RECORDED
CONFIG = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
# The hooks a recording holds: every one a wrapped session posts (hands' plugin's own: the shim's, and MessageDisplay),
# every one the brain does, and PreToolUse, which neither subscribes to, so a recording shows when a call started.
HOOKS = tuple(dict.fromkeys((*plugin_hooks()["hooks"], *process.HOOKS, "PreToolUse")))
# The hooks Claude Code runs no HTTP hook for ("HTTP hooks are not supported for SessionStart"): a command posts each of
# these to the rig instead, as hands' shim posts a wrapped session's, so its arrival is on the same clock, a spawn later.
UNPOSTED = {"SessionStart", "Setup"}
TMUX = "record-turns"
# How often the transcript and the session file are read.
POLL = 0.001
# How long a session stays idle, after the Stops a turn's script waits on, before its recording ends: longer than the
# moment Claude Code sets idle between a turn and the next it runs. A turn opened later than that, as an MCP task's
# notification can be well after its turn's Stop, is recorded only if the script waits on its Stop.
SETTLED = 5.0

# The records a turn is made of. The rest (a prompt's snapshot of the system prompt, hook context, skill and agent
# listings, file-history snapshots, titles, modes) carry the recording machine's setup, never the turn.
KEPT = {"user", "assistant", "system", "queue-operation"}
KEPT_ATTACHMENTS = {"queued_command"}
# The recording machine's, in every hook body and record: never read as the turn. The player puts back the session's own.
DROPPED = {"cwd", "entrypoint", "gitBranch", "userType", "sessionId", "session_id", "transcript_path", "hookInfos", "hookAdditionalContext", "hookErrors"}


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
    """The pinned Claude Code in a tmux session of its own, in a directory of its own, and what it says on each channel
    stamped on this process's clock as it becomes visible, the way hands sees it: a hook as it arrives (posted over
    HTTP, as the brain's are), a record as its line is appended to the transcript, and a status as the session file
    changes. The transcript and the status file are read every POLL seconds, so a status set again within that is
    not seen, as it would not be by hands' own reader."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="record-turns-", dir="/tmp")).resolve()
        self.cwd = self.dir / "work"
        self.cwd.mkdir()
        self._lock = threading.Lock()
        self._hooks: list[tuple[float, dict]] = []
        self._records: list[tuple[float, dict]] = []
        self._seen: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._failed: BaseException | None = None
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self._server.server_address[1]}/"
        forward = {"type": "command", "command": f"curl -sS -o /dev/null -H 'Content-Type: application/json' --data-binary @- {url}"}
        settings = {
            "hooks": {event: [{"hooks": [forward if event in UNPOSTED else {"type": "http", "url": url}]}] for event in HOOKS},
            "permissions": {"allow": ["Bash(*)", "mcp__slow__wait"]},
        }
        mcp = {"mcpServers": {"slow": {"type": "stdio", "command": sys.executable, "args": ["-I", str(Path(__file__).resolve()), "--slow-mcp"]}}}
        command = [str(pinned.verified(default_home(os.environ).brain)), "--model", "haiku", "--setting-sources", "project", "--settings", json.dumps(settings), "--mcp-config", json.dumps(mcp), "--strict-mcp-config"]
        # The pane gets the tmux server's environment, not this process's: what the session must run under is named.
        # Background tasks on, as the brain will run: the turns recorded here are the ones they open.
        # Claude Code keeps a login for each config directory it is named, so the default is reached by naming none.
        config = f"CLAUDE_CONFIG_DIR={shlex.quote(os.environ['CLAUDE_CONFIG_DIR'])}" if "CLAUDE_CONFIG_DIR" in os.environ else "-u CLAUDE_CONFIG_DIR"
        environment = f"exec env -u CLAUDE_CODE_DISABLE_BACKGROUND_TASKS {config} DISABLE_AUTOUPDATER=1 "
        run(["tmux", "kill-session", "-t", TMUX], check=False)
        run(["tmux", "new-session", "-d", "-s", TMUX, "-x", "200", "-y", "50", "-c", str(self.cwd), environment + shlex.join(command)])
        self.pid = int(run(["tmux", "display-message", "-p", "-t", TMUX, "#{pane_pid}"]).strip())
        self.project = CONFIG / "projects" / project_name(str(self.cwd))
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()
        try:
            until(lambda: any(status["status"] == "idle" for _, status in self._statuses()), 120, "the session never went idle")
        except BaseException:
            self.end()
            raise

    def _handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        rig = self

        class Hook(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                at = time.time()
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with rig._lock:
                    rig._hooks.append((at, body))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                pass

        return Hook

    def _watch(self) -> None:
        try:
            self._watching()
        except BaseException as error:
            # [LAW:no-silent-failure] a watch that died leaves the channels unread from then on: whatever reads what it
            # saw raises this, so no recording is written short.
            self._failed = error

    def _watched(self) -> None:
        if self._failed is not None:
            raise RuntimeError("the transcript and session file stopped being read") from self._failed

    def _watching(self) -> None:
        """Read the session file and the transcript every POLL seconds, stamping each status and each whole record as
        it is first seen."""
        file = CONFIG / "sessions" / f"{self.pid}.json"
        last, offset, transcript = None, 0, None
        while not self._stop.is_set():
            try:
                body = json.loads(file.read_bytes())
            except (FileNotFoundError, ValueError):
                # Not written yet, or caught half rewritten: read again next time.
                body = {}
            # A file with no status set yet sets nothing to hear, as hands' reader passes over it.
            if "status" in body and "statusUpdatedAt" in body:
                status = {key: body[key] for key in ("status", "waitingFor") if key in body}
                seen = (status, body["statusUpdatedAt"])
                if seen != last:
                    last = seen
                    with self._lock:
                        self._seen.append((time.time(), status))
            transcript = transcript or next(iter(self.project.glob("*.jsonl")), None)
            if transcript is not None:
                with transcript.open("rb") as read:
                    read.seek(offset)
                    appended = read.read()
                # Only whole lines: the last may be part way through being written.
                whole = appended[: appended.rfind(b"\n") + 1]
                offset += len(whole)
                at = time.time()
                with self._lock:
                    self._records += [(at, json.loads(line)) for line in whole.splitlines() if line.strip()]
            time.sleep(POLL)

    def hooks(self) -> list[tuple[float, dict]]:
        with self._lock:
            return list(self._hooks)

    def _statuses(self) -> list[tuple[float, dict]]:
        self._watched()
        with self._lock:
            return list(self._seen)

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
        """Wait for the session to stay idle for SETTLED seconds, so the recording holds the status it ends on and any
        turn that opens within them. Claude Code sets idle for a moment between a turn and the next it runs, so an idle
        that has not held is not the end."""
        def held() -> bool:
            at, status = self._statuses()[-1]
            return status["status"] == "idle" and time.time() - at >= SETTLED
        until(held, 240, "the session never settled idle")

    def said(self) -> list[tuple[float, str, dict]]:
        """End the session, and what it said on each channel up to then, in the order it was seen. Nothing said as it
        is ended is the turn's (its SessionEnd, the status file's removal)."""
        end = time.time()
        self.end()
        self._watched()
        hooks = self.hooks()
        [session] = {body["session_id"] for _, body in hooks}
        [transcript] = self.project.glob(f"{session}.jsonl")
        self.transcript, self.session = transcript, session
        with self._lock:
            lines = [(at, "posted", body) for at, body in hooks]
            lines += [(at, "written", record) for at, record in self._records]
            lines += [(at, "status", status) for at, status in self._seen]
        # Stable: what one reading of a file found keeps the order it was written in.
        return sorted((line for line in lines if line[0] <= end), key=lambda line: line[0])

    def end(self) -> None:
        """Stop watching, once the watch in flight is done, end the session, and stop hearing its hooks."""
        self._stop.set()
        self._watcher.join()
        run(["tmux", "kill-session", "-t", TMUX], check=False)
        self._server.shutdown()
        self._server.server_close()

    def remove(self) -> None:
        """Leave nothing of the run behind: its directory, and the transcripts Claude Code kept of it."""
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.project, ignore_errors=True)


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
        try:
            TURNS[name](rig)
            rig.settled()
        except BaseException:
            # The turn's own failure is the one said: nothing of a run that went wrong is written.
            rig.end()
            raise
        text = normalized(rig.said(), rig)
    finally:
        rig.remove()
    out = OUT / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def normalized(lines: list[tuple[float, str, dict]], rig: Rig) -> str:
    """What the run said of the turn, with nothing of the machine it ran on: refused, if anything of it is left."""
    kept = [line for line in lines if line[1] != "written" or _kept(line[2])]
    start = kept[0][0]
    text = "\n".join(json.dumps({"at": round(at - start, 3), kind: _rebased(_scrubbed(body), start)}, separators=(",", ":")) for at, kind, body in kept) + "\n"
    machine = Machine(rig)
    for real, generic in machine.paths:
        text = text.replace(real, generic)
    text = text.replace(rig.session, RECORDED_SESSION)
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
        named = os.environ.get("TMPDIR", tempfile.gettempdir()).rstrip("/")
        # Each path in every form a record can name it by: as resolved (/private/...), as named (/tmp/..., /var/...),
        # and as Claude Code names a directory after one (each character but a letter or digit a dash).
        # The shared /tmp names nobody's machine; a TMPDIR of one's own does.
        temps = sorted({os.path.realpath(named), named, named.removeprefix("/private")} - {"/tmp", "/private/tmp"}, key=len, reverse=True)
        rigs = sorted({str(rig.cwd), str(rig.cwd).removeprefix("/private")}, key=len, reverse=True)
        dirs = {str(rig.dir), str(rig.dir).removeprefix("/private")}
        self.paths = [
            *[(path, "/brain/cwd") for path in rigs],
            *[(project_name(path), "-brain-cwd") for path in rigs],
            (f"claude-{uid}", "claude-UID"),
            *[(path, "/tmp/me") for path in temps],
            (str(Path.home()), "/home/me"),
        ]
        git = [run(["git", "config", key], check=False).strip() for key in ("user.email", "user.name")]
        names = [part for value in git for part in re.split(r"[\s@.]+", value) if len(part) >= 4]
        host = socket.gethostname()
        machine = (str(Path.home()), project_name(str(Path.home())), getpass.getuser(), host, host.split(".")[0], *temps, *[project_name(path) for path in temps], *dirs, *[project_name(path) for path in dirs], rig.session, f"claude-{uid}", *git, *names)
        self.private = [value for value in machine if len(value) >= 3]


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
