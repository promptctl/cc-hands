"""Fixtures more than one test module needs."""

import asyncio
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import wave
from collections.abc import AsyncGenerator, Callable, Generator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import metadata
from pathlib import Path

import huggingface_hub.constants
import mlx_whisper
import pytest
from packaging.specifiers import SpecifierSet
from hands.voice.engaged import Ears, Hears
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from hands.brain import pinned
from hands.core.front import FrontUnread, InFront
from hands.daemon import watchdog
from hands.sessions.audit import Entry, Record, SettingsEdited
from hands.sessions.delta import MARKING, READING, Deltas
from hands.sessions.home import Home
from hands.sessions.marketplace import render
from hands.sessions.wide import Fact, WideEvent
from hands.sessions.wrapper import PACKAGED
from hands.voice import engaged, streaming
from hands.voice.wake import fetched
from hands.voice.wakeword import Pretrained


def in_ci[R: pytest.TestReport | pytest.CollectReport](report: R) -> R:
    """A skip in CI as a failure, with the skip's own reason: CI provisions every tool and machine a test needs, so a skip
    there is one of them gone missing. An expected failure is not a skip."""
    # [LAW:no-silent-failure] a runner image that drops jq, or gains a Claude Code policy, would otherwise pass the run
    # with the tests that needed it never run, and nothing in the log but a count.
    if os.environ.get("CI") == "true" and report.skipped and not hasattr(report, "wasxfail"):
        assert isinstance(report.longrepr, tuple), report.longrepr
        report.outcome = "failed"
        report.longrepr = f"{report.longrepr[2]}, in CI, where every test runs"
    return report


# [LAW:single-enforcer] every skip is reported by one of these two: a test's, from its marks or its body, and a module's,
# skipped as it is collected.
@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport() -> Generator[None, pytest.TestReport, pytest.TestReport]:
    return in_ci((yield))


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report() -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    return in_ci((yield))


def events(recorded: Sequence[Entry], name: str) -> list[WideEvent]:
    """The wide events named `name`, in the order they were emitted."""
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == name]


def model_facts(load: WideEvent) -> dict[str, Fact]:
    """A model load's facts of its own: every load says how big the process was as it ended and how much it grew, which
    are taken off here once they are shown to be there (hands.voice.loads)."""
    facts = dict(load.facts)
    assert isinstance(facts.pop("footprint_mb"), float) and isinstance(facts.pop("process_grew_mb"), float), load.facts
    return facts


def spoken(text: str, at: Path) -> bytes:
    """`text` spoken by macOS's own voice, as 16 kHz mono 16-bit audio, the way the desk's microphone is heard."""
    subprocess.run(["say", "-v", "Samantha", "-o", str(at), "--data-format=LEI16@16000", text], check=True)
    with wave.open(str(at)) as clip:
        return clip.readframes(clip.getnframes())


def patient[D: Deltas](kind: type[D], record: Record, marking: float = MARKING) -> D:
    """A `kind` of Deltas whose readings are taken with as much patience as each is given to run: a test asserts what a
    reading found, never whether a loaded machine found it inside the narrator's few seconds of PATIENCE."""
    return kind(record=record, inherited=os.environ, marking=marking, patience=READING)


# How long a pipeline may take to start before a test fails on it.
STARTUP_SECS = 5.0
# How long a test's wait may look and see nothing move before it fails on a stall.
PATIENCE_SECS = 2.0
# How long a wait sleeps between two looks at what it waits on, while the loop has nothing else to run.
LOOK_SECS = 0.01


class ManualClock:
    """Time that passes only when a test moves it: `now` reads it, `pause` waits until it has moved on that far, as the
    clock and pause a unit takes in place of `time.monotonic` and `asyncio.sleep`."""

    def __init__(self) -> None:
        self.at = 0.0
        self._pauses: list[tuple[float, asyncio.Future[None]]] = []

    def now(self) -> float:
        return self.at

    async def pause(self, seconds: float) -> None:
        woken = asyncio.get_running_loop().create_future()
        self._pauses.append((self.at + seconds, woken))
        await woken

    def advance(self, seconds: float) -> None:
        self.at += seconds
        for until, woken in self._pauses:
            if until <= self.at and not woken.done():
                woken.set_result(None)
        self._pauses = [(until, woken) for until, woken in self._pauses if not woken.done()]


async def patiently(done: Callable[[], bool], moved: Callable[[], object]) -> None:
    """Until `done` holds, failing once `moved` has stood still through PATIENCE_SECS of the wait's looking.

    [LAW:no-ambient-temporal-coupling] the patience bounds a stall, not the work queued ahead of what is awaited, so a
    loaded machine that is slow but moving is waited on. It is the time the wait spent looking, LOOK_SECS a look: the
    time between two looks that the loop spent on other work is that work's, however long one step of it holds the loop,
    since a stage that takes a burst of frames in one step moves nothing anyone can see until the step ends.
    [LAW:single-enforcer] a run that never ends, moving or holding the loop, is ended by the suite's timeout."""
    loop = asyncio.get_running_loop()
    patience, seen, still, since = round(PATIENCE_SECS / LOOK_SECS), moved(), 0, loop.time()
    while not done():
        if still == patience:
            # The clock beside the looking tells a loop held by other work from one that sat free.
            raise TimeoutError(f"nothing moved through {PATIENCE_SECS}s of looking, {loop.time() - since:.1f}s on the clock, at {seen!r}, while waiting on {done!r}")
        await asyncio.sleep(LOOK_SECS)
        if (now := moved()) != seen:
            seen, still, since = now, 0, loop.time()
        else:
            still += 1


@pytest.fixture(autouse=True)
def watchdog_past_the_timeout(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    """A `hands run` in the test's own process waits on no watchdog of its own: its timer ends the whole process, so a run
    that stalls is failed by pytest-timeout, named, with its stack, and the watchdog ends only one that holds the GIL
    past it. [LAW:single-enforcer] pytest-timeout is the one owner of a test that never finishes."""
    monkeypatch.setattr(watchdog, "HUNG", timedelta(seconds=2 * float(pytestconfig.getini("timeout"))))


@pytest.fixture(autouse=True)
def no_whisper_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a Whisper transcribes nothing: its load would fetch 1.6 GB of weights, which no test is about."""

    def transcribe(_audio: object, **_options: object) -> dict[str, object]:
        return {"segments": []}

    monkeypatch.setattr(mlx_whisper, "transcribe", transcribe)


@dataclass
class Unspoken:
    """The run's streaming session of a pair that hears no speaker say a word in anything it is fed."""

    def decoded(self, samples: streaming.Samples) -> streaming.Spoken:
        return ()

    def closed(self) -> streaming.Closed:
        return streaming.Closed(frozenset(), ())


@pytest.fixture(autouse=True)
def no_streaming_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the voice hears holds with a pair that hears no word: its load would fetch 2.5 GB of weights, which no
    test is about but the ones that ask for the real models."""
    monkeypatch.setattr(streaming, "load", Unspoken)


async def unprimed() -> None:
    """Whisper's prompt where no test is about its vocabulary: none, so it transcribes unprimed."""


# Stays until it is killed, or for two minutes, as the `sleep 120` it stands in for did: a test killed before it kills
# its own leaves no process behind for good. Given --moves, a directory and a descriptor, it first moves to the
# directory and writes a "." to the descriptor, as a session moves into a worktree; given --moves any other way, it ends.
STAND_IN = r"""
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The descriptor `given` names, or -1, which no write goes to, for anything but a whole number. */
static int descriptor(const char *given) {
    char *end;
    long fd = strtol(given, &end, 10);
    return end != given && *end == '\0' && fd >= 0 && fd <= INT_MAX ? (int)fd : -1;
}

int main(int argc, char **argv) {
    alarm(120);
    if (argc > 1 && strcmp(argv[1], "--moves") == 0 && (argc != 4 || chdir(argv[2]) != 0 || write(descriptor(argv[3]), ".", 1) != 1))
        return 1;
    for (;;)
        pause();
}
"""


def executable(path: Path, text: str) -> Path:
    """A script at `path` that runs `text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def stand_in(at: Path) -> Path:
    """A program built at `at` that runs as a session does, until it is killed, signed as the linker signs any program
    not Apple's, as Claude Code and Homebrew's tmux are: macOS 26 kills a copy of one of Apple's own programs as it
    starts, and shows no other process the environment of one of them, kern.procargs2 and `ps -E` alike."""
    # The compiler's own word on a failed build, a missing Command Line Tools or an unaccepted Xcode license, is left on
    # stderr for pytest to show.
    subprocess.run(["cc", "-x", "c", "-o", str(at), "-"], input=STAND_IN, text=True, check=True)
    # Run through once, as `--moves` alone returns at once. Gatekeeper (syspolicyd) scans a program's first run, reading
    # the file at its path, and kills the run when that file is gone ("Terminating process due to Gatekeeper rejection"
    # in `log show`): on a loaded Mac a stand-in deleted as it started was killed 24 runs in 25, and none once it had run
    # through (promptctl-testing-c3u.axy).
    ran = subprocess.run([at, "--moves"])
    assert ran.returncode == 1, f"the stand-in at {at} ran through as {ran.returncode}, not as 1"
    return at


def _dead_pid() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


@pytest.fixture(scope="session")
def smart_turn() -> None:
    """Smart Turn's MLX port in the hub's cache, so a desk that loads it fetches nothing."""
    engaged.ported()


async def fetch_all(directory: Path, words: Sequence[Pretrained]) -> None:
    for word in words:
        await fetched(directory, word)


@pytest.fixture(scope="session")
def models(pytestconfig: pytest.Config, smart_turn: None) -> Path:
    """The wake word's models, Hey Jarvis's and Hey Mycroft's, fetched from openWakeWord's release once and kept in
    pytest's cache, beside Smart Turn in the hub's: every model a switch to the wake word loads. A fetch, so of the run,
    as the other fetches are, and not of a test."""
    directory = pytestconfig.cache.mkdir("wake-word") if pytestconfig.cache else pytest.fail("the cache provider is off")
    asyncio.run(fetch_all(directory, (Pretrained("Hey Jarvis"), Pretrained("Hey Mycroft"))))
    return directory


@pytest.fixture
def dead_pid() -> Callable[[], int]:
    """Makes the pid of a process that has just exited, which no process holds for now, fresh at each call."""
    return _dead_pid


@pytest.fixture
def plugin(tmp_path: Path) -> Path:
    """hands' plugin as `hands plugin` writes it for the hands under test: its launcher runs this interpreter."""
    return render(Home(tmp_path / "rendering"), sys.executable).plugin


# The system's own directories: the PATH of a child that must find nothing of the developer's. It holds no Python new
# enough for hands, since /usr/bin's python3 on macOS is 3.9, so the plugin's launcher can only be running the
# interpreter it names; and no Claude Code, so a child that looks for claude runs none, not the developer's logged-in one.
SYSTEM_PATH = "/usr/bin:/bin"


def pytest_configure() -> None:
    """[LAW:single-enforcer] SYSTEM_PATH holds what it is said to, proven once before any test gives it to a child: a
    machine whose system directories break it stops the suite, naming what they hold."""
    if claude := shutil.which("claude", path=SYSTEM_PATH):
        raise pytest.UsageError(f"{claude} is on SYSTEM_PATH, which the tests give children as a PATH with no Claude Code")
    if python := shutil.which("python3", path=SYSTEM_PATH):
        version = subprocess.run([python, "-c", "import platform; print(platform.python_version())"], capture_output=True, text=True, check=True).stdout.strip()
        if version in SpecifierSet(metadata("hands")["Requires-Python"]):
            raise pytest.UsageError(f"{python} is Python {version}, new enough for hands, on SYSTEM_PATH, which the tests give children as a PATH with no such Python")


# The names a test takes from the machine as they are: macOS's per-user temporary directory, and CI's word that the run
# is in CI, which in_ci reads as each report is made. pytest's own PYTEST_ names are its to set and clear.
PASSED = ("TMPDIR", "CI")


def without_claude(path: str, links: Path) -> str:
    """`path`'s absolute directories, each that holds a `claude` in place of one in `links` that holds a link to
    everything else in it: the developer's tools, lit and go among them, and not their logged-in Claude Code beside them.
    A relative or empty entry is the directory a test is in, which no PATH made once can say holds no `claude`."""

    def kept(index: int, directory: str) -> str:
        if shutil.which("claude", path=directory) is None:
            return directory
        copy = links / str(index)
        copy.mkdir()
        # casefold: on macOS's case-insensitive disk, a `Claude` is what a lookup for `claude` finds.
        for entry in Path(directory).iterdir():
            if entry.name.casefold() != "claude":
                (copy / entry.name).symlink_to(entry)
        return str(copy)

    return os.pathsep.join(kept(index, directory) for index, directory in enumerate(path.split(os.pathsep)) if os.path.isabs(directory))


def hermetic(machine: Mapping[str, str], path: str, hub: str, home: Path, sockets: Path) -> dict[str, str]:
    """The environment a test runs in, from the `machine`'s: the names PASSED, a `path` of its tools, the Hugging Face
    `hub` cache the weights are already in, no git config but a repository's own, and a `home` and a tmux socket
    directory of the test's own. Every other name the machine has, a credential, a Claude Code session's, hands', git's,
    tmux's or a proxy, which hands reads as a session's own, is absent, as it is on CI."""
    return {name: machine[name] for name in PASSED if name in machine} | {
        "PATH": path,
        "HF_HUB_CACHE": hub,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(home),
        "TMUX_TMPDIR": str(sockets),
    }


@pytest.fixture(scope="session")
def tools(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The PATH every test runs with, its copies made once."""
    return without_claude(os.environ["PATH"], tmp_path_factory.mktemp("path"))


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, tools: str) -> dict[str, str]:
    """Each test runs in the hermetic environment, with a HOME and a tmux socket directory of its own that hold nothing,
    as pytest's own pytester gives each of its runs, so ~/.claude, ~/.hands, ~/.gitconfig and ~/.config are all empty
    and no tmux server is found. [LAW:single-enforcer] the one owner of the environment a test runs in: a test that needs
    a name sets it on top, and builds no environment of its own to keep the developer's out. A fixture wider than a test,
    which fetches and builds tools and weights, runs before it, in the machine's environment, which monkeypatch gives
    back after each test."""
    # The hub resolved its cache from the developer's HOME as it was imported; a child that loads weights finds them
    # there too, and not the hub's token beside them.
    tested = hermetic(os.environ, tools, huggingface_hub.constants.HF_HUB_CACHE, tmp_path_factory.mktemp("home"), tmp_path_factory.mktemp("tmux"))
    for name in [name for name in os.environ if not name.startswith("PYTEST_")]:
        monkeypatch.delenv(name)
    for name, value in tested.items():
        monkeypatch.setenv(name, value)
    return tested


async def unread() -> InFront:
    """What is in front, for a run whose test never reads it."""
    return FrontUnread("not read in this test")


@dataclass
class Running:
    """A pipeline running as the daemon runs one, and the errors it has raised: what the system channel would say."""

    worker: PipelineWorker
    errors: list[ErrorFrame]


@asynccontextmanager
async def running(processors: list[FrameProcessor], observers: Sequence[BaseObserver] = ()) -> AsyncGenerator[Running]:
    """Runs the processors as one pipeline, watched by the observers, from its start until the block ends, then cancels it."""
    worker = PipelineWorker(Pipeline(processors), observers=list(observers), idle_timeout_secs=None)
    started = asyncio.Event()
    errors: list[ErrorFrame] = []

    @worker.event_handler("on_pipeline_started")
    async def _started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        started.set()

    @worker.event_handler("on_pipeline_error")
    async def _failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        errors.append(error)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), STARTUP_SECS)
    try:
        yield Running(worker, errors)
    finally:
        await worker.cancel()
        await task


@pytest.fixture(scope="session")
def fritter() -> Path:
    """The fritter this checkout's hands package carries, which its editable install built."""
    return PACKAGED


def onboard(brain: Path, settings: bytes = b'{"syncClaudeAiSkills": false, "syncClaudeAiPlugins": false}', trusted: bool = True) -> None:
    """A brain home Claude Code has been through its onboarding on, holding `settings`; the directory it runs in trusted, as
    Claude Code trusts it, by a directory above it, unless not `trusted`."""
    brain.mkdir(parents=True)
    projects = {str(brain.resolve().parent): {"hasTrustDialogAccepted": trusted}}
    (brain / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True, "projects": projects}))
    (brain / "settings.json").write_bytes(settings)


def sha256(path: Path) -> str:
    with path.open("rb") as held:
        return hashlib.file_digest(held, "sha256").hexdigest()


def located(script: Path) -> Callable[[Path], Path]:
    """A stand-in for pinned.claude that puts the brain's pinned claude at `script`, whatever directory the brain is on."""
    return lambda _config_dir: script


def pin(monkeypatch: pytest.MonkeyPatch, script: Path) -> None:
    """`script`, as its bytes are now, is the brain's pinned claude, whatever directory the brain is on: nothing on PATH
    is ever the brain's."""
    monkeypatch.setattr(pinned, "claude", located(script))
    monkeypatch.setattr(pinned, "CHECKSUMS", dict.fromkeys(pinned.CHECKSUMS, sha256(script)))


def pinned_to(script: Path) -> str:
    """The lines a hands run in a process of its own starts with, so its brain's pinned claude is `script`, as fake_claude
    pins it in this one."""
    return f"from hands.brain import pinned\npinned.claude = lambda _config_dir: __import__('pathlib').Path({str(script)!r})\npinned.CHECKSUMS = dict.fromkeys(pinned.CHECKSUMS, {sha256(script)!r})\n"


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The brain's pinned `claude`, wherever its directory, that reports its login from LOGGED_IN, or from an `auth login` or a first run (`--setting-sources user` alone, which also records its onboarding, the trust of the directory it ran in, and its approval of an API key its settings.json sets) it recorded, made by AUTH_METHOD (claude.ai unless named, api_key after an `auth login --console`, holding what Claude Code says of each) through API_PROVIDER (firstParty unless named), and as the brain is Claude Code at a keyboard: it
    reads its terminal raw, STARTS_AFTER seconds in if named, in bursts, takes a prompt when a Return that ends a burst sends it, and says each turn as Claude Code does on each channel hands reads: the hooks its --settings name, as Claude Code fills them, the records of its transcript, and the status in its sessions/<pid>.json, idle as it starts; a prompt sent within STOP_TAIL seconds of a Stop, if named, is queued as Claude Code queues one sent while its other Stop hooks still run. Everything it
    reads is written, one line each, to the file TYPED names, a side question with the session it was asked under, and
    each prompt a Return sent is a line in the file ANSWERED names once it is done with all it does with it at once; a side
    question it is started with, after `--`, is taken as if typed. A turn "wait" runs until Escape, "stuck" runs whatever is pressed, "fail" is failed by the API, "deaf"
    is never taken, "garbled" is taken as other words and runs until Escape, an Escape writing the record of the interruption and setting it idle, "write" asks permission to write notes.txt beside TYPED, writing it only if allowed, then an MCP server's input, "stray" asks one under an earlier turn's prompt id, and "die", as a turn or a side question, ends the program; a side question "stubborn" writes that it was told to end, and does not. A turn "play <name>" is the turn of
    that name tests/turns.py holds as Claude Code was recorded running it, opened by the words typed, and writes
    ["played", name] once it has said all of it."""
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"""#!{sys.executable}
import datetime, itertools, json, os, select, signal, sys, termios, time, tty, urllib.request
first_run = sys.argv[1:] == ["--setting-sources", "user"]
if sys.argv[1:2] == ["auth"] or first_run:
    login = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "login.json")
# `auth login`, or a first run, its screens answered and its login made at its keyboard; one that fails finishes neither.
if sys.argv[1:3] == ["auth", "login"] or first_run:
    os.makedirs(os.environ["CLAUDE_CONFIG_DIR"], exist_ok=True)
    if os.environ.get("LOGIN_EXIT", "0") != "0":
        sys.exit(int(os.environ["LOGIN_EXIT"]))
    with open(login, "w") as made:
        json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "settings": os.path.exists(os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "settings.json")), "credentials": sorted(set(os.environ) & {{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}})}}, made)
    # LOGIN_UNANSWERED: quit before its last screen, which Claude Code exits 0 on as on any other.
    if first_run and not os.environ.get("LOGIN_UNANSWERED"):
        with open(os.path.join(os.environ["CLAUDE_CONFIG_DIR"], ".claude.json"), "w") as state:
            settings = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "settings.json")
            key = json.load(open(settings)).get("env", {{}}).get("ANTHROPIC_API_KEY") if os.path.exists(settings) else None
            json.dump({{"hasCompletedOnboarding": True, "projects": {{os.getcwd(): {{"hasTrustDialogAccepted": True}}}}, "customApiKeyResponses": {{"approved": [key[-20:]] if key else [], "rejected": []}}}}, state)
    sys.exit(0)
if sys.argv[1:3] == ["auth", "status"]:
    console = os.path.exists(login) and "--console" in json.load(open(login))["argv"]
    method = "api_key" if console else os.environ.get("AUTH_METHOD", "claude.ai")
    holder = {{"claude.ai": {{"email": "brain@example.com"}}, "api_key": {{"email": "brain@example.com", "apiKeySource": "/login managed key"}}, "oauth_token": {{}}}}[method]
    print(json.dumps({{"loggedIn": os.environ["LOGGED_IN"] == "1" or os.path.exists(login), "authMethod": method, "apiProvider": os.environ.get("API_PROVIDER", "firstParty"), **holder}}))
    sys.exit(0)
hooks = json.loads(sys.argv[sys.argv.index("--settings") + 1])["hooks"] if "--settings" in sys.argv else {{}}
session = sys.argv[sys.argv.index("--session-id" if "--session-id" in sys.argv else "--resume") + 1]
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import turns
# Where Claude Code keeps the session's transcript, and the file it keeps its status in, under its config directory.
transcript = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "projects", turns.project_name(os.getcwd()), session + ".jsonl")
status = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "sessions", f"{{os.getpid()}}.json")
def post(event, **fields):
    [[url]] = [[hook["url"] for hook in matcher["hooks"]] for matcher in hooks[event]]
    body = json.dumps({{"session_id": session, "transcript_path": transcript, "cwd": os.getcwd(), "permission_mode": "default", "hook_event_name": event, **fields}}).encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, body, {{"Content-Type": "application/json"}}), timeout=30).read())
def typed(line):
    with open(os.environ["TYPED"], "a") as log:
        log.write(json.dumps(line) + "\\n")
def written(record):
    # Each record stamped as Claude Code stamps it, in UTC to the millisecond.
    os.makedirs(os.path.dirname(transcript), exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    with open(transcript, "a") as file:
        file.write(turns.transcribed({{"sessionId": session, "timestamp": stamp, **record}}))
def stated(name):
    # Rewritten whole, as Claude Code rewrites it each time it sets the status.
    os.makedirs(os.path.dirname(status), exist_ok=True)
    with open(status + ".tmp", "w") as file:
        json.dump({{"pid": os.getpid(), "sessionId": session, "status": name, "statusUpdatedAt": round(time.time() * 1000)}}, file)
    os.replace(status + ".tmp", status)
# As a loaded machine starts it slowly.
time.sleep(float(os.environ.get("STARTS_AFTER", "0")))
# Raw without flushing: Claude Code keeps what was typed while it started (keys typed
# 0.05s in).
tty.setraw(0, termios.TCSANOW)
# Claude Code asks its terminal for bracketed paste, and fritter pastes only into a program that asked.
os.write(1, b"\\x1b[?2004h> ")
stated("idle")
pending, box, turn, turn_text, prompts = b"", "", None, "", 0
# The last prompt to end, and until when its other Stop hooks run on after hands' own was answered: STOP_TAIL seconds.
last, tail = None, 0.0
# A prompt as Claude Code's UserPromptSubmit says it took it: a paste of four lines or more in its tags, with a
# line end added where it had none, and any other with its trailing whitespace trimmed.
def taken(text):
    if text.count("\\n") < 3:
        return text.rstrip()
    return '\\n\\n<pasted_content id="4c2e">\\n' + (text if text.endswith("\\n") else text + "\\n") + '</pasted_content id="4c2e">\\n'
# Claude Code taking a prompt, as it takes one at its prompt: its UserPromptSubmit, then busy, then the prompt's record
# under its id.
def took(prompt, kept, text):
    post("UserPromptSubmit", prompt_id=prompt, prompt=kept)
    stated("busy")
    written({{"type": "user", "promptId": prompt, "message": {{"role": "user", "content": text.strip()}}}})
# A turn's end: Claude's reply written, its Stop or StopFailure posted, and the session set idle.
def ended(event, prompt, reply, **fields):
    written({{"type": "assistant", "message": {{"role": "assistant", "content": [{{"type": "text", "text": reply if isinstance(reply, str) else "odd"}}]}}}})
    post(event, prompt_id=prompt, **fields)
    stated("idle")
# A turn as Claude Code was recorded running it: each hook posted, each record appended to the session's transcript and
# each status set in its file, in the order Claude Code said them and each stamped as it is said, the words typed the
# opening prompt on both channels that carry it. It runs the recording's prompts, the typed one first.
def play(name, text):
    global prompts, last, tail
    lines = turns.played(turns.RECORDINGS[name], subscribed=list(hooks), session=session, first=prompts, prompt=taken(text))
    for line in lines:
        line = turns.shifted(line, time.time() - line.at)
        if isinstance(line, turns.Posted):
            post(line.body["hook_event_name"], **{{key: value for key, value in line.body.items() if key != "hook_event_name"}})
        elif isinstance(line, turns.Written):
            os.makedirs(os.path.dirname(transcript), exist_ok=True)
            with open(transcript, "a") as file:
                file.write(turns.transcribed(line.record))
        else:
            with open(status + ".tmp", "w") as file:
                file.write(turns.status_file(line, pid=os.getpid(), session=session))
            os.replace(status + ".tmp", status)
    prompts += turns.prompts(turns.RECORDINGS[name]) - 1
    last, tail = turns.stops(lines)[-1], time.monotonic() + float(os.environ.get("STOP_TAIL", "0"))
    typed(["played", name])
def submit(text):
    global turn, turn_text, prompts, last, tail
    if text.startswith("/btw "):
        if text == "/btw stubborn":
            signal.signal(signal.SIGTERM, lambda *_: typed(["sigterm", text, session]))
        typed(["btw", text[len("/btw "):], session])
        if text == "/btw die":
            os.write(1, b"bye\\r\\n")
            sys.exit(3)
        return
    if time.monotonic() < tail:
        # Sent while the last prompt's other Stop hooks still run: queued, and taken as they end. Its UserPromptSubmit is
        # posted before it has a prompt id of its own, under the last prompt's, holding what was queued, which Claude Code
        # trims of its leading whitespace; Claude Code's queue records say it was queued and taken out to run, and it
        # then runs under an id of its own, which its records and its Stop name.
        time.sleep(tail - time.monotonic())
        kept = taken(text.lstrip())
        post("UserPromptSubmit", prompt_id=last, prompt=kept)
        typed(["prompt", text])
        written({{"type": "queue-operation", "operation": "enqueue", "content": kept}})
        written({{"type": "queue-operation", "operation": "dequeue"}})
        prompts += 1
        last = f"p{{prompts}}"
        stated("busy")
        written({{"type": "user", "promptId": last, "message": {{"role": "user", "content": text.strip()}}}})
        ended("Stop", last, "Two.", last_assistant_message="Two.", stop_hook_active=False)
        tail = time.monotonic() + float(os.environ["STOP_TAIL"])
        return
    typed(["prompt", text])
    prompts += 1
    prompt = f"p{{prompts}}"
    said = text.strip()
    if said.startswith("play "):
        play(said[len("play "):], text)
        return
    if said == "deaf":
        return
    if said == "garbled":
        took(prompt, taken(text) + " (as kept)", text)
        turn, turn_text = prompt, text
        return
    took(prompt, taken(text), text)
    if said == "die":
        os.write(1, b"bye\\r\\n")
        sys.exit(3)
    if said in ("wait", "stuck"):
        turn, turn_text = prompt, text
        return
    if said == "slow":
        time.sleep(0.5)
    if said == "write":
        # Its setup asks before a write, held at its hook until it is answered: the file is written only on an allow,
        # and the turn goes on either way. Then an MCP server asks for input.
        notes = os.path.join(os.path.dirname(os.environ["TYPED"]), "notes.txt")
        decision = post("PermissionRequest", prompt_id=prompt, tool_name="Write", tool_input={{"file_path": notes, "content": "hello"}})["hookSpecificOutput"]["decision"]
        if decision["behavior"] == "allow":
            with open(notes, "w") as file:
                file.write("hello")
        typed(["permission", decision["behavior"], decision.get("message", "")])
        declined = post("Elicitation", prompt_id=prompt, mcp_server_name="probe", message="Which?")
        typed(["elicitation", declined["hookSpecificOutput"]["action"]])
    if said == "stray":
        # A permission an earlier turn left behind, posted under that turn's prompt id.
        stray = post("PermissionRequest", prompt_id="p0", tool_name="Write", tool_input={{"file_path": "notes.txt", "content": "hello"}})["hookSpecificOutput"]["decision"]
        typed(["permission", stray["behavior"], stray.get("message", "")])
    if said == "fail":
        ended("StopFailure", prompt, "API Error: 400 refused", error="unknown", last_assistant_message="API Error: 400 refused")
    elif said == "fail blank":
        ended("StopFailure", prompt, "", error="authentication_failed", last_assistant_message="")
    elif said == "fail odd":
        ended("StopFailure", prompt, {{"text": "odd"}}, error="server_error", last_assistant_message={{"text": "odd"}})
    elif said == "fail unnamed":
        ended("StopFailure", prompt, "")
    else:
        ended("Stop", prompt, "Two.", last_assistant_message="Two.", stop_hook_active=False)
    last, tail = prompt, time.monotonic() + float(os.environ.get("STOP_TAIL", "0"))
# As Claude Code reads its options: --mcp-config takes every word up to the next option or `--`, each a config or a
# config's file, and one that is neither ends the program (a prompt read as a path: ENAMETOOLONG, exit 1).
end = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
if "--mcp-config" in sys.argv:
    for value in itertools.takewhile(lambda word: not word.startswith("-"), sys.argv[sys.argv.index("--mcp-config") + 1:end]):
        try:
            json.loads(value)
        except ValueError:
            if not os.path.exists(value):
                sys.stderr.write(f"Error: MCP config file not found: {{value}}\\n")
                sys.exit(1)
# The prompt it opens with is the one word after `--`.
if end < len(sys.argv):
    [opening] = sys.argv[end + 1:]
    submit(opening)
while True:
    data = os.read(0, 65536)
    if not data:
        break
    pending += data
    # Claude Code reads what reaches it close together as one burst.
    time.sleep(0.05)
    while select.select([0], [], [], 0)[0]:
        more = os.read(0, 65536)
        if not more:
            break
        pending += more
    while pending:
        if pending.startswith(b"\\x1b[200~"):
            end = pending.find(b"\\x1b[201~")
            if end < 0:
                break
            box += pending[6:end].decode()
            pending = pending[end + 6:]
        elif pending.startswith(b"\\x1b"):
            pending = pending[1:]
            typed(["escape", box])
            if turn is not None and turn_text.strip() != "stuck":
                # As Claude Code does, the stopped prompt is put back in the input, and the turn's end is the record of its
                # interruption, written under its id, and the session set idle: no Stop.
                written({{"type": "user", "promptId": turn, "message": {{"role": "user", "content": [{{"type": "text", "text": "[Request interrupted by user]"}}]}}}})
                stated("idle")
                box, turn = turn_text, None
        elif pending.startswith(b"\x03"):
            pending = pending[1:]
            typed(["ctrl_c", box])
            box = ""
        # A newline, as a Return typed before it read raw reaches it, is one more line in the input: Claude Code takes
        # Ctrl+J as multiline input (its docs), and sends on a Return alone.
        elif pending.startswith(b"\\r"):
            pending = pending[1:]
            if pending:
                # A Return with more behind it in the same burst is read as pasted, and sends nothing.
                continue
            text, box = box, ""
            submit(text)
            # Done with what this Return sent: each hook it posted was answered, so hands has heard them all.
            with open(os.environ["ANSWERED"], "a") as answered:
                answered.write("\\n")
        else:
            box += pending[:1].decode()
            pending = pending[1:]
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    pin(monkeypatch, script)
    monkeypatch.setenv("LOGGED_IN", "1")
    monkeypatch.setenv("TYPED", str(tmp_path / "typed.jsonl"))
    monkeypatch.setenv("ANSWERED", str(tmp_path / "answered"))
    return script


async def unedited() -> SettingsEdited:
    """Settings a run's start watches that are never edited."""
    await asyncio.Event().wait()
    raise AssertionError("an event nothing sets was set")


def hearing_with(ears: Ears, woken: Hears) -> Callable[[], AbstractAsyncContextManager[tuple[Ears, Hears]]]:
    """The desk driver's models, already loaded: `ears` for Silero and Smart Turn, and `woken` for the wake word."""

    @asynccontextmanager
    async def hearing() -> AsyncGenerator[tuple[Ears, Hears]]:
        yield ears, woken

    return hearing
