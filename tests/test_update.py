"""`hands update`: hands from master, its fritter, and the brain's Claude Code, each step in order, then the restart onto
it; a step that fails stops it with hands unrestarted. The brain asks for it through its hands:update skill."""

import errno
import io
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import pytest

from hands.brain.process import PLUGIN
from hands.daemon import cli, update
from hands.daemon.restart import Outcome, Restarted
from hands.daemon.starting import invocation
from hands.daemon.update import Done, Failed, Installed, Ran, Step
from hands.sessions import audit, heartbeat
from hands.sessions.home import Home

STANDIN = Path(__file__).resolve().parent / "fixtures" / "standin_daemon.py"
SHA = "0123456789abcdef0123456789abcdef01234567"


def tools_of_this_hands(tmp_path: Path) -> Path:
    """A uv tool directory whose hands is the environment these tests run in, as `uv tool dir` would name it."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / update.TOOL).symlink_to(sys.prefix)
    return tools


class Commands:
    """The commands an update runs, each answered as it would be: each exits 0 unless it is the one told to fail."""

    def __init__(self, tools: Path, failing: Step | None = None) -> None:
        self.tools, self.failing, self.ran = tools, failing, list[tuple[tuple[str, ...], Path]]()

    def __call__(self, argv: Sequence[str], cwd: Path) -> Done:
        self.ran.append((tuple(argv), cwd))
        if self.failing is not None and len(self.ran) - 1 == list(Step).index(self.failing):
            return Done(1, "", f"{self.failing} broke\n")
        # A warning on a command's errors is no part of what the update reads from its output.
        match tuple(argv[:3]):
            case ("uv", "tool", "dir"):
                return Done(0, f"{self.tools}\n", "warning: `VIRTUAL_ENV=/elsewhere` does not match the tool's environment\n")
            case ("git", "rev-parse", "--verify"):
                return Done(0, f"{SHA}\n", "warning: refname 'HEAD' is ambiguous.\n")
            case _:
                return Done(0, "", "")


def expected(home: Home, work: Path) -> list[tuple[tuple[str, ...], Path]]:
    """Every command of an update into `work`, in order, and the directory each runs in."""
    clone, python = work / "cc-hands", f"{sys.version_info.major}.{sys.version_info.minor}"
    return [
        (("uv", "tool", "dir"), work),
        (("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", "master", "https://github.com/promptctl/cc-hands.git", str(clone)), work),
        (("git", "rev-parse", "--verify", "HEAD^{commit}"), clone),
        (("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(work / "constraints.txt")), clone),
        # The commit itself from GitHub, not the clone, so uv's record of the tool outlives the clone.
        (("uv", "tool", "install", "--reinstall-package", "hands", "--python", python, "--constraints", str(work / "constraints.txt"), f"hands @ git+https://github.com/promptctl/cc-hands.git@{SHA}"), work),
        (tuple(invocation(home, "install-fritter")), work),
        (tuple(invocation(home, "install-brain")), work),
    ]


def events(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.command"]


def test_the_steps_run_in_order_installing_the_commit_master_is_at(tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    commands = Commands(tools_of_this_hands(tmp_path))

    installed = update.install(home, Path(sys.prefix), work, commands)

    # Cloned fresh from master, installed on uv.lock's versions into this hands' own environment, and its fritter and
    # the brain's Claude Code installed by the hands just installed.
    assert commands.ran == expected(home, work)
    assert installed == Installed(SHA, tuple(Ran(step, argv, 0) for step, (argv, _cwd) in zip(Step, expected(home, work), strict=True)))


@pytest.mark.parametrize("failing", list(Step))
def test_a_step_that_fails_stops_the_update_and_hands_is_not_restarted(failing: Step, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    commands = Commands(tools_of_this_hands(tmp_path), failing)
    asked: list[Home] = []

    def restart(home: Home, *_: object) -> Outcome:
        asked.append(home)
        raise AssertionError("a failed update restarted hands")

    def detach() -> None:
        raise AssertionError("a failed update went on to its restart")

    monkeypatch.setattr(update, "run", commands)
    monkeypatch.setattr(update, "detach", detach)
    monkeypatch.setattr(cli, "restart", restart)

    assert cli.main(["--home", str(home.root), "update"]) == 1

    # Nothing after the failed step ran, and nothing restarted.
    upto = list(Step).index(failing) + 1
    assert [argv[:3] for argv, _cwd in commands.ran] == [argv[:3] for argv, _cwd in expected(home, Path("/"))[:upto]]
    assert asked == []
    out, err = capsys.readouterr()
    assert out == ""
    why = f"`{' '.join(commands.ran[-1][0])}` exited 1:\n{failing} broke\n"
    # A step after the install leaves master installed: what hands runs now is as it was, and its next start is not.
    unfinished = failing in (Step.INSTALL_FRITTER, Step.INSTALL_BRAIN)
    if unfinished:
        assert err.startswith(f"hands update: {failing} failed, so hands was not restarted. It runs as it did, but master at {SHA} is installed, and hands starts on it next time without what {failing} installs; `hands update` again finishes it: {why}")
    else:
        assert err.startswith(f"hands update: {failing} failed, so hands was not restarted and runs as it did: {why}")
    # [LAW:nothing-unseen] the event says which step failed, each step run up to it, and the commit installed, if one was.
    [event] = events(home)
    facts = event["facts"]
    assert (event["outcome"], facts["command"], facts["exit_code"], facts["failed"]) == ("failed", "update", 1, str(failing))
    assert [(step["step"], step["code"]) for step in facts["steps"]] == [(str(step), 0) for step in list(Step)[: upto - 1]] + [(str(failing), 1)]
    assert facts.get("sha") == (SHA if unfinished else None)


def test_a_hands_run_from_another_environment_is_not_updated(tmp_path: Path) -> None:
    elsewhere = tmp_path / "tools"
    (elsewhere / update.TOOL).mkdir(parents=True)
    commands = Commands(elsewhere)

    installed = update.install(Home(tmp_path / "home"), Path(sys.prefix), tmp_path / "work", commands)

    # Its restart would start it again on what it runs now, so nothing is fetched or installed.
    assert [argv for argv, _cwd in commands.ran] == [("uv", "tool", "dir")]
    assert isinstance(installed, Failed) and installed.step is Step.WHERE
    assert installed.said.startswith(f"this hands runs from {sys.prefix}, not from {elsewhere / update.TOOL}, the uv tool environment the update installs into")


# `hands update` run as the brain runs it, from a process group hands kills once the restart ends the brain. Its
# commands are stood in for, each logged to a file, so nothing is fetched or installed.
CALLER = """
import json, sys
from hands.daemon import cli, update

home, tools, sha, log = sys.argv[1:]

def run(argv, cwd):
    with open(log, "a") as file:
        file.write(json.dumps(list(argv)) + "\\n")
    if list(argv[:3]) == ["uv", "tool", "dir"]:
        return update.Done(0, tools + "\\n", "")
    if list(argv[:2]) == ["git", "rev-parse"]:
        return update.Done(0, sha + "\\n", "")
    return update.Done(0, "", "")

update.run = run
sys.exit(cli.main(["--home", home, "update"]))
"""


def test_an_update_restarts_hands_onto_master_and_says_the_commit_after_its_caller_is_killed(tmp_path: Path) -> None:
    home, log, said = Home(tmp_path / "home"), tmp_path / "commands", tmp_path / "said"
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    # The brain: a process group of its own, as hands starts it, which the update is run from.
    brain = subprocess.Popen(["sleep", "60"], process_group=0)
    try:
        before = up(home)
        with said.open("w") as out:
            caller = subprocess.Popen([sys.executable, "-c", CALLER, str(home.root), str(tools_of_this_hands(tmp_path)), SHA, str(log)], stdout=out, stderr=subprocess.STDOUT, process_group=brain.pid)
        # The caller has its answer at once, before the restart.
        assert caller.wait(timeout=30) == 0
        # The restart ends the brain, and hands kills everything in its process group.
        os.killpg(brain.pid, signal.SIGKILL)

        event = waited_for(home)

        after = heartbeat.read(home.status)
        assert after is not None
        assert (after.pid, after.pipeline) == (daemon.pid, "running") and after.started_at > before.started_at
        facts = event["facts"]
        assert (event["outcome"], facts["command"], facts["exit_code"], facts["sha"], facts["outcome"]["type"]) == ("ok", "update", 0, SHA, "Restarted")
        # The run after the restart says it runs from the environment master went into, and the caller's output had the line.
        assert (facts["runs_master"], facts["told"]) == (True, True)
        assert [step["step"] for step in facts["steps"]] == list(Step)
        assert [json.loads(line)[:3] for line in log.read_text().splitlines()] == [list(argv[:3]) for argv, _cwd in expected(home, Path("/"))]
        first, restarted = said.read_text().splitlines()
        assert first == f"installed hands, its fritter, and the brain's Claude Code from master at {SHA}; restarting hands onto it"
        assert restarted.startswith(f"hands restarted: pid {daemon.pid} is running again after ")
        assert restarted.endswith(f", with 0 live sessions. It runs master at {SHA}.")
    finally:
        brain.kill()
        brain.wait()
        daemon.terminate()
        daemon.wait(timeout=10)


def test_the_brain_is_given_the_update_skill_and_may_run_it_without_asking() -> None:
    skill = (PLUGIN / "skills" / "update" / "SKILL.md").read_text()
    assert skill.startswith("---\nname: update\n")
    # The command it runs is the one it is allowed, and the one the CLI takes.
    assert "\nallowed-tools: Bash(hands update)\n" in skill and "run `hands update`" in skill


def up(home: Home, deadline: float = 10.0) -> heartbeat.Status:
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        status = heartbeat.read(home.status)
        if status is not None and status.pipeline == "running":
            return status
        time.sleep(0.02)
    raise AssertionError(f"the daemon did not say running within {deadline}s: {heartbeat.read(home.status)}")


def waited_for(home: Home, deadline: float = 60.0) -> dict[str, Any]:
    """The update's event, once the update has written it."""
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        if found := [event for event in events(home) if event["facts"]["command"] == "update"]:
            [event] = found
            return event
        time.sleep(0.05)
    raise AssertionError(f"no update event within {deadline}s")


def test_an_update_with_no_hands_running_installs_and_tells_its_caller_hands_was_not_restarted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    commands = Commands(tools_of_this_hands(tmp_path))

    def detach() -> None:
        raise AssertionError("an update with no hands to restart left its caller before saying so")

    monkeypatch.setattr(update, "run", commands)
    monkeypatch.setattr(update, "detach", detach)

    # Nothing ends the caller, so the caller hears it, and exits saying hands was not restarted.
    assert cli.main(["--home", str(home.root), "update"]) == 1

    out, err = capsys.readouterr()
    assert out == f"installed hands, its fritter, and the brain's Claude Code from master at {SHA}; restarting hands onto it\n"
    assert err.startswith("hands was not restarted: hands has not run")
    [event] = events(home)
    facts = event["facts"]
    assert (facts["command"], facts["exit_code"], facts["sha"], facts["outcome"]["type"]) == ("update", 1, SHA, "NotRunning")


def status_from(runs_from: Path | None) -> heartbeat.Status:
    now = datetime.now(UTC)
    return heartbeat.Status(os.getpid(), now, now, heartbeat.HEARTBEAT, "running", None, 0, False, (), runs_from)


def restarting_from(monkeypatch: pytest.MonkeyPatch, runs_from: Path | None) -> None:
    """hands is running from this hands' environment, and its restart brings it back running from `runs_from`."""

    def running(_home: Home, _now: datetime) -> heartbeat.Status:
        return status_from(Path(sys.prefix))

    def restart(*_: object) -> Outcome:
        return Restarted(status_from(runs_from), timedelta(seconds=3))

    monkeypatch.setattr(cli, "running", running)
    monkeypatch.setattr(cli, "restart", restart)


@pytest.mark.parametrize("runs_from", [Path("/checkout/.venv"), None], ids=["elsewhere", "unsaid"])
def test_a_hands_that_restarts_from_another_environment_is_not_said_to_run_master(runs_from: Path | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    detached: list[bool] = []
    monkeypatch.setattr(update, "run", Commands(tools_of_this_hands(tmp_path)))
    monkeypatch.setattr(update, "detach", lambda: detached.append(True))
    # A daemon started by a checkout's `uv run hands run` starts again on the checkout, whatever the update installed.
    restarting_from(monkeypatch, runs_from)

    assert cli.main(["--home", str(home.root), "update"]) == 1

    assert detached == [True]
    _out, err = capsys.readouterr()
    assert err.endswith(f" It runs from {runs_from or 'an environment it does not name'}, not from {sys.prefix}, where master at {SHA} is installed.\n")
    [event] = events(home)
    assert (event["facts"]["exit_code"], event["facts"]["runs_master"], event["facts"]["told"]) == (1, False, True)


class HungUp(io.TextIOWrapper):
    """A terminal whose other end has hung up: every write fails, as a pty's does once its master is closed."""

    def write(self, s: str, /) -> int:
        raise OSError(errno.EIO, os.strerror(errno.EIO))


def closed_pipe() -> tuple[TextIO, str]:
    """The brain's end of the pipe its command printed to, closed by the restart that ended it."""
    read, write = os.pipe()
    os.close(read)
    return os.fdopen(write, "w"), "Broken pipe"


def hung_up_terminal() -> tuple[TextIO, str]:
    _read, write = os.pipe()
    return HungUp(os.fdopen(write, "wb")), "Input/output error"


@pytest.mark.parametrize("caller", [closed_pipe, hung_up_terminal])
def test_an_update_whose_caller_has_gone_says_its_line_to_the_event_alone(caller: Callable[[], tuple[TextIO, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path / "home")
    gone, why = caller()
    monkeypatch.setattr(update, "run", Commands(tools_of_this_hands(tmp_path)))
    monkeypatch.setattr(update, "detach", lambda: monkeypatch.setattr(sys, "stdout", gone))
    restarting_from(monkeypatch, Path(sys.prefix))

    assert cli.main(["--home", str(home.root), "update"]) == 0

    [event] = events(home)
    facts = event["facts"]
    assert (event["outcome"], facts["runs_master"], facts["told"]) == ("ok", True, False)
    assert why in facts["untold"]


def test_a_command_s_output_is_read_apart_from_its_errors(tmp_path: Path) -> None:
    # A warning uv or git prints is never taken for the path or the commit the update reads.
    assert update.run([sys.executable, "-c", "import sys; print('/tools'); print('warning: stale', file=sys.stderr)"], tmp_path) == Done(0, "/tools\n", "warning: stale\n")
