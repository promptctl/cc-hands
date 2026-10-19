"""`hands update`: hands from master, its fritter, and the brain's Claude Code, each step in order, then the restart onto
it; a step that fails stops it with hands unrestarted. The brain asks for it through its hands:update skill."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from hands.brain.process import PLUGIN
from hands.daemon import cli, update
from hands.daemon.restart import Outcome
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
            return Done(1, f"{self.failing} broke\n")
        match tuple(argv[:3]):
            case ("uv", "tool", "dir"):
                return Done(0, f"{self.tools}\n")
            case ("git", "rev-parse", "--verify"):
                return Done(0, f"{SHA}\n")
            case _:
                return Done(0, "")


def expected(home: Home, work: Path) -> list[tuple[tuple[str, ...], Path]]:
    """Every command of an update into `work`, in order, and the directory each runs in."""
    clone, python = work / "cc-hands", f"{sys.version_info.major}.{sys.version_info.minor}"
    return [
        (("uv", "tool", "dir"), work),
        (("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", "master", "https://github.com/promptctl/cc-hands.git", str(clone)), work),
        (("git", "rev-parse", "--verify", "HEAD^{commit}"), clone),
        (("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(work / "constraints.txt")), clone),
        (("uv", "tool", "install", "--reinstall-package", "hands", "--python", python, "--constraints", str(work / "constraints.txt"), str(clone)), work),
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
    assert err.startswith(f"hands update: {failing} failed, so hands was not restarted and runs as it did: `{' '.join(commands.ran[-1][0])}` exited 1:\n{failing} broke\n")
    # [LAW:nothing-unseen] the event says which step failed, and each step run up to it.
    [event] = events(home)
    facts = event["facts"]
    assert (event["outcome"], facts["command"], facts["exit_code"], facts["failed"]) == ("failed", "update", 1, str(failing))
    assert [(step["step"], step["code"]) for step in facts["steps"]] == [(str(step), 0) for step in list(Step)[: upto - 1]] + [(str(failing), 1)]
    assert "sha" not in facts


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
        return update.Done(0, tools + "\\n")
    if list(argv[:2]) == ["git", "rev-parse"]:
        return update.Done(0, sha + "\\n")
    return update.Done(0, "")

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
