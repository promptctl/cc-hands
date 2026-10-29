"""The plugin's restart skill: a running daemon is started again in the same process, with the sessions it listed."""

import asyncio
import json
import ctypes
import os
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from conftest import SYSTEM_PATH, unedited
from hands.core.session import Membership, SessionId
from hands.daemon import cli
from hands.daemon.cli import launch, retire
from hands.daemon.restart import RESTART_SIGNAL
from hands.daemon.starting import Ended, QuitKey, Start, Stop
from hands.sessions import audit, heartbeat
from hands.sessions.hookconfig import LAUNCHER
from hands.sessions.home import Home
from hands.sessions.membership import write_membership
from hands.sessions.wide import WideEvent

STANDIN = Path(__file__).resolve().parent / "fixtures" / "standin_daemon.py"
NOW = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
EDITED = audit.SettingsEdited(path="/home/config.toml", refused=None)


def started(home: Home) -> list[dict[str, Any]]:
    """Each start's event in the home's audit log, oldest first."""
    return [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.start"]


def start_outcomes(recorded: list[audit.Entry]) -> list[str]:
    return [entry.outcome for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hands.start"]


def ran(recorded: list[audit.Entry]) -> list[tuple[str, object, object]]:
    """Each hands.run event's outcome, causes, and ending."""
    return [(entry.outcome, entry.facts["stopped_by"], entry.facts["ending"]) for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hands.run"]


def restart(plugin: Path, home: Home, cwd: Path) -> subprocess.CompletedProcess[str]:
    """`/hands:restart` as the skill runs it: the plugin's launcher, in the session's directory."""
    environment = os.environ | {"HANDS_HOME": str(home.root), "PATH": SYSTEM_PATH}
    return subprocess.run([plugin / LAUNCHER, "-m", "hands.daemon", "restart"], env=environment, cwd=cwd, capture_output=True, text=True, timeout=60)


def running(home: Home, deadline: float = 10.0) -> heartbeat.Status:
    """The heartbeat once it says the pipeline is running."""
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        status = heartbeat.read(home.status)
        if status is not None and status.pipeline == "running":
            return status
        time.sleep(0.02)
    raise AssertionError(f"the daemon did not say running within {deadline}s: {heartbeat.read(home.status)}")


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Membership]:
    """A session that is running: a process, started before its membership file was written."""
    process = subprocess.Popen(["sleep", "60"])
    try:
        yield Membership(SessionId("the-session"), pid=process.pid, cwd=tmp_path, transcript=tmp_path / "the-session.jsonl")
    finally:
        process.kill()
        process.wait()


def test_a_restart_asked_through_the_plugin_brings_the_daemon_back_with_its_sessions(tmp_path: Path, plugin: Path, session: Membership) -> None:
    home = Home(tmp_path / "home")
    write_membership(home, session)
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    try:
        before = running(home)
        assert before.live_sessions == 1

        done = restart(plugin, home, tmp_path)

        assert done.returncode == 0, done.stderr
        after = heartbeat.read(home.status)
        assert after is not None
        # The same process, started again: what the skill said is what the heartbeat says.
        assert daemon.poll() is None
        assert (after.pid, after.pipeline, after.live_sessions) == (daemon.pid, "running", 1)
        assert after.started_at > before.started_at
        assert done.stdout.startswith(f"hands restarted: pid {daemon.pid} is running again after ")
        assert done.stdout.endswith(", with 1 live session.\n")
        # Each start is one event: the first, and the restart's, in the same process, ready.
        starts = started(home)
        assert [(line["outcome"], line["facts"]["pid"], line["facts"]["restarted"]) for line in starts] == [("ok", daemon.pid, False), ("ok", daemon.pid, True)]
        # [LAW:nothing-unseen] and the restart asked is one command's event, which says what it saw the daemon come back as, and
        # how long that took.
        [command] = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.command"]
        outcome = command["facts"]["outcome"]
        assert (command["outcome"], command["facts"]["command"], command["facts"]["exit_code"], outcome["type"]) == ("ok", "restart", 0, "Restarted")
        assert (outcome["status"]["pid"], outcome["status"]["started_at"]) == (daemon.pid, after.started_at.isoformat(timespec="milliseconds"))
        assert outcome["took"] > 0
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)
    stopped = heartbeat.read(home.status)
    assert stopped is not None and stopped.pipeline == "stopped"


def test_an_edit_to_the_settings_brings_the_daemon_back_on_them_with_its_sessions(tmp_path: Path, session: Membership) -> None:
    home = Home(tmp_path / "home")
    write_membership(home, session)
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    try:
        before = running(home)
        home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
        after = running(home)
        until = time.monotonic() + 10
        while after.started_at == before.started_at and time.monotonic() < until:
            time.sleep(0.02)
            after = running(home)
        # Started again, in the same process, with nothing asked of it but the edit.
        assert daemon.poll() is None
        assert (after.pid, after.live_sessions) == (daemon.pid, 1) and after.started_at > before.started_at
        said = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line["type"] == "SettingsEdited" or line.get("event") == "hands.start"]
        assert [(line["type"], line.get("refused"), line.get("facts", {}).get("restarted")) for line in said] == [("WideEvent", None, False), ("SettingsEdited", None, None), ("WideEvent", None, True)]
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)


async def test_settings_edited_end_a_run_as_a_restart(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10), kick_watchdog=lambda: None)
    edit = asyncio.Event()

    async def run(stop: Stop) -> Ended:
        edit.set()
        await stop.wait()
        return Ended(None, 0)

    async def edited() -> audit.SettingsEdited:
        await edit.wait()
        return EDITED

    recorded: list[audit.Entry] = []
    assert await launch(lambda: run, heart, lambda: (), edited, recorded.append, Start(restarted=False)) == "restart"
    # Told to stop before it was ready, the start ends cancelled.
    assert recorded[0] == EDITED and start_outcomes(recorded) == ["cancelled"] and len(recorded) == 3
    assert ran(recorded) == [("ok", ("settings edited",), "restart")]


async def test_settings_edited_as_a_run_ends_are_taken_up_by_the_next_start_and_not_said(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10), kick_watchdog=lambda: None)
    edit = asyncio.Event()

    async def run(stop: Stop) -> Ended:
        # The q key, then the run winds down while the file is saved.
        stop.tell(QuitKey())
        edit.set()
        await asyncio.sleep(0.05)
        return Ended(None, 0)

    async def edited() -> audit.SettingsEdited:
        await edit.wait()
        return EDITED

    recorded: list[audit.Entry] = []
    assert await launch(lambda: run, heart, lambda: (), edited, recorded.append, Start(restarted=False)) == "quit"
    assert start_outcomes(recorded) == ["cancelled"] and len(recorded) == 2
    assert ran(recorded) == [("ok", ("q key",), "quit")]


async def test_settings_that_cannot_be_watched_stop_the_run_saying_so(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10), kick_watchdog=lambda: None)

    async def run(stop: Stop) -> Ended:
        await stop.wait()
        return Ended(None, 0)

    async def edited() -> audit.SettingsEdited:
        raise PermissionError("config.toml")

    recorded: list[audit.Entry] = []
    with pytest.raises(PermissionError, match="config.toml"):
        await launch(lambda: run, heart, lambda: (), edited, recorded.append, Start(restarted=False))
    # The run and its start, which it ended raising, are failed, with what it raised.
    failed = "PermissionError: config.toml"
    assert [(entry.event, entry.outcome, entry.error) for entry in recorded if isinstance(entry, WideEvent)] == [("hands.run", "failed", failed), ("hands.start", "failed", failed)]
    assert ran(recorded) == [("failed", ("settings unreadable",), "failed")]
    # As a run whose background task failed: its last heartbeat does not read as stopped.
    status = heartbeat.read(heart.path)
    assert status is None or status.pipeline != "stopped"


def test_a_daemon_that_is_not_running_is_not_asked_and_the_skill_says_why(tmp_path: Path, plugin: Path) -> None:
    home = Home(tmp_path / "home")
    done = restart(plugin, home, tmp_path)
    assert (done.returncode, done.stdout) == (1, "")
    assert done.stderr == f"hands was not restarted: hands has not run: there is no heartbeat at {home.status}.\n"
    # The command's event fails with what it said, not its exit code alone.
    [command] = [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.command"]
    assert (command["outcome"], command["facts"]["outcome"]["type"]) == ("failed", "NotRunning")
    assert command["error"] == f"exited 1: hands was not restarted: hands has not run: there is no heartbeat at {home.status}."


def test_a_daemon_still_starting_is_not_asked(tmp_path: Path, plugin: Path) -> None:
    # Its pid is this test's, which a restart signal would end: being refused, it is never sent.
    home = Home(tmp_path / "home")
    heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT, kick_watchdog=lambda: None).beat("starting", None, 0, listening=False, degraded=())
    done = restart(plugin, home, tmp_path)
    assert done.returncode == 1
    assert done.stderr.startswith(f"hands was not restarted: hands is up: pid {os.getpid()}, ")
    assert "pipeline starting" in done.stderr


async def test_the_restart_signal_ends_a_run_as_a_restart_whose_last_heartbeat_says_starting(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10), kick_watchdog=lambda: None)
    told: list[Stop] = []

    async def run(stop: Stop) -> Ended:
        told.append(stop)
        await stop.wait()
        return Ended(NOW, 3)

    recorded: list[audit.Entry] = []
    launched = asyncio.create_task(launch(lambda: run, heart, lambda: (), unedited, recorded.append, Start(restarted=False)))
    while not told:
        await asyncio.sleep(0.005)
    os.kill(os.getpid(), RESTART_SIGNAL)
    assert await launched == "restart"
    assert ran(recorded) == [("ok", ("SIGUSR1",), "restart")]
    last = heartbeat.read(heart.path)
    # Not stopped: the indicator reads the moment before the next run as starting, and posts nothing.
    assert last is not None and (last.pipeline, last.last_audio_out, last.live_sessions) == ("starting", NOW, 3)


async def test_a_restart_asked_while_a_quit_winds_the_run_down_does_not_start_it_again(tmp_path: Path) -> None:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=NOW, period=timedelta(milliseconds=10), kick_watchdog=lambda: None)
    winding_down = asyncio.Event()

    async def run(stop: Stop) -> Ended:
        # The q key, with no signal, and the run takes a while to close.
        stop.tell(QuitKey())
        winding_down.set()
        await asyncio.sleep(0.05)
        return Ended(None, 0)

    recorded: list[audit.Entry] = []
    launched = asyncio.create_task(launch(lambda: run, heart, lambda: (), unedited, recorded.append, Start(restarted=False)))
    await winding_down.wait()
    os.kill(os.getpid(), RESTART_SIGNAL)
    assert await launched == "quit"
    # The restart asked is kept beside the quit, which decided.
    assert ran(recorded) == [("ok", ("q key", "SIGUSR1"), "quit")]
    last = heartbeat.read(heart.path)
    assert last is not None and last.pipeline == "stopped"


def spawned(*argv: str) -> int:
    """A child of this process, as the indicator is of the run that starts it and of every run exec'd after."""
    # A session of its own, so it leads a process group, as `start_indicator` makes the indicator.
    return os.posix_spawnp(argv[0], list(argv), os.environ, setsid=True)


# The kernel's process table, as ps reads it: sysctl's CTL_KERN, KERN_PROC, KERN_PROC_PID names a process's kinfo_proc,
# whose kp_proc.p_stat is SZOMB once it has exited and is not yet reaped. Read here, not by ps: a process in a sandbox,
# as the run is, cannot exec a setuid binary.
KINFO_PROC_SIZE = 648
P_STAT = 36
SZOMB = 5
_libc = ctypes.CDLL(None, use_errno=True)
_libc.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]


def zombie(pid: int) -> bool:
    """Whether `pid` has exited and is not yet reaped. [LAW:no-silent-failure] a process the table does not hold, or a
    kinfo_proc not the size its layout is read at, fails, never reads as a process not yet exited."""
    info = ctypes.create_string_buffer(KINFO_PROC_SIZE)
    size = ctypes.c_size_t(KINFO_PROC_SIZE)
    if _libc.sysctl((ctypes.c_int * 4)(1, 14, 1, pid), 4, info, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), f"sysctl of process {pid}")
    if size.value != KINFO_PROC_SIZE:
        raise OSError(f"the process table gave {size.value} bytes for process {pid}, not a kinfo_proc of {KINFO_PROC_SIZE}")
    return info.raw[P_STAT] == SZOMB


def test_a_restart_ends_the_indicator_the_run_before_showed() -> None:
    shown = spawned("sleep", "60")
    assert retire(shown) == "ended"
    # Reaped as it was ended: no zombie is left under the run.
    with pytest.raises(ChildProcessError):
        os.waitpid(shown, os.WNOHANG)


def test_an_indicator_that_does_not_end_when_asked_is_killed_with_its_group() -> None:
    ready, said = os.pipe()
    argv = ["sh", "-c", 'trap "" TERM; sleep 60 & echo; wait']
    stubborn = os.posix_spawnp("sh", argv, os.environ, file_actions=[(os.POSIX_SPAWN_DUP2, said, 1)], setsid=True)
    os.close(said)
    with os.fdopen(ready) as lines:
        # Its line comes once SIGTERM is ignored, so the ask below cannot land before it is.
        lines.readline()
        assert retire(stubborn, grace=0.2) == "killed"
        with pytest.raises(ChildProcessError):
            os.waitpid(stubborn, os.WNOHANG)
        # The sleep it started holds the pipe too, so the pipe ends only once it went with the group.
        assert select.select([lines], [], [], 5)[0] == [lines]
        assert lines.read() == ""


def test_an_indicator_that_exited_before_the_restart_is_reaped_and_said() -> None:
    exited = spawned("false")
    # Exited and not reaped, as an indicator that exits during the exec is: no reap thread outlives it.
    while not zombie(exited):
        time.sleep(0.01)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="ERROR")
    try:
        assert retire(exited) == "exited"
    finally:
        logger.remove(sink)
    assert logged == ["the menu-bar indicator exited (1) while hands runs; hands is not shown in the menu bar"]
    # Reaped by the run before, as its reap thread does with an indicator that exits, and said there.
    assert retire(exited) == "reaped"


def test_a_restarted_run_shows_itself_through_an_indicator_it_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The run before's indicator judged the heartbeat by the code that run had; this run's is started on the code it runs.
    from hands.voice import talkkey

    home = Home(tmp_path)
    previous = spawned("sleep", "60")
    # [LAW:behavior-not-structure] the order is the contract: the one before is gone before this run's is shown.
    steps: list[str] = []

    def retired(indicator: int) -> cli.Retired:
        steps.append("retired")
        return retire(indicator)

    def start_indicator(started: Home) -> int:
        assert started == home
        steps.append("started")
        return 0

    def kept(*_: object) -> None:
        pass

    def loaded(*_: object) -> cli.Run:
        async def quits(stop: Stop) -> Ended:
            stop.tell(QuitKey())
            return Ended(None, 0)

        return quits

    monkeypatch.setattr(talkkey, "granted", lambda: True)
    monkeypatch.setattr(cli, "retire", retired)
    monkeypatch.setattr(cli, "start_indicator", start_indicator)
    monkeypatch.setattr(cli, "reap", kept)
    monkeypatch.setattr(cli, "loaded", loaded)
    assert cli.main(["--home", str(home.root), "run", "--restarted", str(previous)]) == 0
    assert steps == ["retired", "started"]
    with pytest.raises(ChildProcessError):
        os.waitpid(previous, os.WNOHANG)
    [start] = started(home)
    assert (start["facts"]["restarted"], start["facts"]["previous_indicator"]) == (True, "ended")
