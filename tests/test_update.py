"""`hands update`: hands from master, its fritter, and the brain's Claude Code, each step in order, then the restart onto
it; a step that fails stops it with hands unrestarted. The brain asks for it through its hands:update skill."""

import errno
import fnmatch
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import pytest

from hands.brain.process import PLUGIN
from hands.clock import Clock
from hands.daemon import cli, update
from hands.daemon.restart import Outcome, Restarted
from hands.daemon.starting import invocation
from hands.daemon.update import Done, Failed, Installed, Ran, Step
from hands.sessions import audit, heartbeat
from hands.sessions.home import Home
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.wrapper import Installing
from hands.voice.intermediary_instruction import brain_instruction

STANDIN = Path(__file__).resolve().parent / "fixtures" / "standin_daemon.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
# The steps of an update of the uv tool environment, in order; a checkout's update runs pull and sync in place of fetch,
# constraints, and install.
TOOL_STEPS = [step for step in Step if step not in (Step.PULL, Step.SYNC)]


def tools_of_this_hands(tmp_path: Path) -> Path:
    """A uv tool directory whose hands is the environment these tests run in, as `uv tool dir` would name it."""
    tools = tmp_path / "tools"
    tools.mkdir(parents=True)
    (tools / update.TOOL).symlink_to(sys.prefix)
    return tools


class Commands:
    """The commands an update runs, each answered as it would be: each exits 0 unless it is the one told to fail, which
    exits as `failed` says, or 1 saying it broke, as any crash does."""

    def __init__(self, tools: Path, failing: Step | None = None, failed: Done | None = None, steps: Sequence[Step] = TOOL_STEPS) -> None:
        self.tools, self.failing, self.failed, self.ran = tools, failing, failed, list[tuple[tuple[str, ...], Path]]()
        # Where among the update's commands the failing one is: its place in the steps the update runs.
        self.failing_at = None if failing is None else steps.index(failing)
        self.environments = list[Mapping[str, str]]()
        # What `claude plugin list --json` prints: hands' plugin installed for the user, as the README installs it.
        self.listed = json.dumps([{"id": PLUGIN_ID, "scope": "user", "enabled": True}])

    def __call__(self, argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> Done:
        self.ran.append((tuple(argv), cwd))
        self.environments.append(env)
        if self.failing is not None and len(self.ran) - 1 == self.failing_at:
            return self.failed or Done(1, "", f"{self.failing} broke\n")
        # A warning on a command's errors is no part of what the update reads from its output.
        match tuple(argv[:3]):
            case ("uv", "tool", "dir"):
                return Done(0, f"{self.tools}\n", "warning: `VIRTUAL_ENV=/elsewhere` does not match the tool's environment\n")
            case ("git", "rev-parse", "--verify"):
                return Done(0, f"{SHA}\n", "warning: refname 'HEAD' is ambiguous.\n")
            case ("claude", "plugin", "list"):
                return Done(0, self.listed, "")
            case _:
                return Done(0, "", "")


def expected(home: Home, work: Path) -> list[tuple[tuple[str, ...], Path]]:
    """Every command of an update into `work`, in order, and the directory each runs in."""
    clone = work / "cc-hands"
    return [
        (("uv", "tool", "dir"), work),
        (("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", "master", "https://github.com/promptctl/cc-hands.git", str(clone)), work),
        (("git", "rev-parse", "--verify", "HEAD^{commit}"), clone),
        (("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(work / "constraints.txt")), clone),
        # The commit itself from GitHub, not the clone, so uv's record of the tool outlives the clone.
        (("uv", "tool", "install", "--reinstall-package", "hands", "--python", update.made_from(), "--constraints", str(work / "constraints.txt"), f"hands @ git+https://github.com/promptctl/cc-hands.git@{SHA}"), work),
        (tuple(invocation(home, "install-fritter")), work),
        (tuple(invocation(home, "install-brain")), work),
        (("claude", "plugin", "list", "--json"), work),
        # The marketplace first: an install from before the plugin was a plain directory holds an entry whose source is
        # the removed command `hands plugin`, which updating the plugin would run.
        (("claude", "plugin", "marketplace", "update", "cc-hands"), work),
        (("claude", "plugin", "update", "hands@cc-hands", "--scope", "user"), work),
        (("rm", "-rf", str(home.root / "plugins")), work),
    ]


def events(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, audit.tail(home.audit, 10_000)[0]) if line.get("event") == "hands.command"]


def test_the_steps_run_in_order_installing_the_commit_master_is_at(tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    commands = Commands(tools_of_this_hands(tmp_path))

    installed = update.install(home, Path(sys.prefix), None, {}, work, commands)

    # Cloned fresh from master, installed on uv.lock's versions into this hands' own environment, and its fritter and
    # the brain's Claude Code installed by the hands just installed.
    assert commands.ran == expected(home, work)
    assert installed == Installed(update.Tool(), SHA, tuple(Ran(step, argv, 0) for step, (argv, _cwd) in zip(TOOL_STEPS, expected(home, work), strict=True)), ())


@pytest.mark.parametrize("failing", [step for step in TOOL_STEPS if step is not Step.RENDERED_PLUGINS])
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
    upto = TOOL_STEPS.index(failing) + 1
    assert [argv[:3] for argv, _cwd in commands.ran] == [argv[:3] for argv, _cwd in expected(home, Path("/"))[:upto]]
    assert asked == []
    out, err = capsys.readouterr()
    assert out == ""
    why = f"`{' '.join(commands.ran[-1][0])}` exited 1:\n{failing} broke\n"
    # A step after the install leaves master installed: what hands runs now is as it was, and its next start is not.
    unfinished = TOOL_STEPS.index(failing) > TOOL_STEPS.index(Step.INSTALL)
    if unfinished:
        assert err.startswith(f"hands update: {failing} failed, so hands was not restarted, but master at {SHA} is installed: what starts from hands' environment now, its hooks and its next start, is master's, short of what {failing} installs, beside a running hands that is not; `hands update` again finishes it: {why}")
    else:
        assert err.startswith(f"hands update: {failing} failed, so hands was not restarted and runs as it did: {why}")
    # [LAW:nothing-unseen] the event says which step failed, each step run up to it, and the commit installed, if one was.
    [event] = events(home)
    facts = event["facts"]
    assert (event["outcome"], facts["command"], facts["exit_code"], facts["failed"]) == ("failed", "update", 1, str(failing))
    assert [(step["step"], step["code"]) for step in facts["steps"]] == [(str(step), 0) for step in TOOL_STEPS[: upto - 1]] + [(str(failing), 1)]
    assert facts.get("sha") == (SHA if unfinished else None)
    # It fails saying why, in the words it said it.
    assert event["error"] == f"exited 1: {err.removeprefix('hands update: ').removesuffix('\n')}"


def a_checkout(tmp_path: Path, project: str = update.TOOL) -> Path:
    """A checkout's root, whose `.venv` is the environment a hands run from it has, of the project named `project`."""
    root = tmp_path / "checkout"
    (root / ".venv").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{project}"\n')
    return root


def test_where_hands_runs_from_is_the_target_its_update_brings_current(tmp_path: Path) -> None:
    tool = tmp_path / "tools" / update.TOOL
    tool.mkdir(parents=True)
    root = a_checkout(tmp_path / "hands")
    other = a_checkout(tmp_path / "other", "not-hands")
    bare = tmp_path / "bare" / ".venv"
    bare.mkdir(parents=True)
    broken = a_checkout(tmp_path / "broken")
    (broken / "pyproject.toml").write_text("[project")
    renamed = a_checkout(tmp_path / "renamed")
    (renamed / ".venv").rename(renamed / "env")

    assert update.parse_target(tool, tool) == update.Tool()
    assert update.parse_target(root / ".venv", tool) == update.Checkout(root)
    # Not a checkout of hands: another project's, one with no pyproject, one whose pyproject cannot be read, and an
    # environment that is not the checkout's own `.venv`.
    assert [update.parse_target(prefix, tool) for prefix in (other / ".venv", bare, broken / ".venv", renamed / "env")] == [update.Unsupported()] * 4


def test_a_hands_run_from_another_environment_is_not_updated(tmp_path: Path) -> None:
    commands = Commands(tmp_path / "tools")
    elsewhere = tmp_path / "elsewhere" / ".venv"
    elsewhere.mkdir(parents=True)

    installed = update.install(Home(tmp_path / "home"), elsewhere, None, {}, tmp_path / "work", commands)

    # Its restart would start it again on what it runs now, so nothing is fetched or installed.
    assert [argv for argv, _cwd in commands.ran] == [("uv", "tool", "dir")]
    assert isinstance(installed, Failed) and installed.step is Step.WHERE
    assert installed.said.startswith(f"this hands runs from {elsewhere}, which is neither {tmp_path / 'tools' / update.TOOL}, the uv tool environment")


def test_a_running_hands_that_restarts_from_another_environment_is_not_updated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    commands = Commands(tools_of_this_hands(tmp_path))
    monkeypatch.setattr(update, "run", commands)
    # A daemon started by a checkout's `uv run hands run`, as its heartbeat says.
    heartbeat.write(home.status, status_from(Path("/checkout/.venv")))

    assert cli.main(["--home", str(home.root), "update"]) == 1

    # Its restart would start it again on the checkout, so nothing is fetched, installed, or restarted.
    assert [argv for argv, _cwd in commands.ran] == [("uv", "tool", "dir")]
    _out, err = capsys.readouterr()
    assert err.startswith(f"hands update: where failed, so hands was not restarted and runs as it did: the running hands runs from /checkout/.venv, not from {sys.prefix}, ")
    [event] = events(home)
    assert (event["facts"]["failed"], event["facts"]["exit_code"], event["facts"]["restarts_from"]) == ("where", 1, "/checkout/.venv")


# What an update of a checkout runs in place of the tool environment's fetch, commit, constraints, and install.
CHECKOUT_STEPS = [Step.WHERE, Step.PULL, Step.COMMIT, Step.SYNC]


def test_a_hands_run_from_a_checkout_pulls_it_and_syncs_it(tmp_path: Path) -> None:
    home, root = Home(tmp_path / "home"), a_checkout(tmp_path)
    commands = Commands(tmp_path / "tools")

    installed = update.install(home, root / ".venv", root / ".venv", {}, tmp_path / "work", commands)

    # Pulled and synced in the checkout, with no clone of master and no uv tool install; the steps after, the fritter,
    # the brain, and the plugin, are the ones the tool environment's update runs.
    assert commands.ran[1:4] == [
        (("git", "pull", "--quiet", "--ff-only"), root),
        (("git", "rev-parse", "--verify", "HEAD^{commit}"), root),
        # hands built anew, as the tool environment's install does.
        (("uv", "sync", "--locked", "--reinstall-package", "hands"), root),
    ]
    assert commands.ran[4:] == expected(home, tmp_path / "work")[5:]
    assert isinstance(installed, Installed) and (installed.target, installed.sha) == (update.Checkout(root), SHA)
    assert [ran.step for ran in installed.ran][:4] == CHECKOUT_STEPS


@pytest.mark.parametrize("failing", CHECKOUT_STEPS[1:])
def test_a_checkout_that_cannot_be_pulled_or_synced_is_not_restarted_onto(failing: Step, tmp_path: Path) -> None:
    root = a_checkout(tmp_path)
    commands = Commands(tmp_path / "tools", failing, steps=CHECKOUT_STEPS)

    installed = update.install(Home(tmp_path / "home"), root / ".venv", None, {}, tmp_path / "work", commands)

    # Nothing after the failed step ran, and none of it is installed: hands is as it was.
    assert isinstance(installed, Failed) and installed.step is failing
    assert len(commands.ran) == CHECKOUT_STEPS.index(failing) + 1


def test_an_update_of_a_checkout_that_fails_after_its_sync_says_the_checkout_is_what_a_start_runs_now(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home, root = Home(tmp_path / "home"), a_checkout(tmp_path)
    commands = Commands(tmp_path / "tools", Step.INSTALL_FRITTER, steps=[*CHECKOUT_STEPS, Step.INSTALL_FRITTER])
    monkeypatch.setattr(update, "run", commands)
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))

    def detach(_clock: Clock) -> Callable[[int], bool]:
        raise AssertionError("a failed update went on to its restart")

    monkeypatch.setattr(update, "detach", detach)

    assert cli.main(["--home", str(home.root), "update"]) == 1

    _out, err = capsys.readouterr()
    assert f"but the checkout {root} at {SHA} is installed: what starts from hands' environment now, its hooks and its next start, is the checkout {root}'s, short of what install-fritter installs" in err


# `hands update` run as the brain runs it, from a process group hands kills once the restart ends the brain. Its
# commands are stood in for, each logged to a file, so nothing is fetched or installed.
CALLER = """
import json, sys
from hands.daemon import cli, update

home, tools, sha, log = sys.argv[1:]

def run(argv, cwd, env):
    with open(log, "a") as file:
        file.write(json.dumps(list(argv)) + "\\n")
    if list(argv[:3]) == ["uv", "tool", "dir"]:
        return update.Done(0, tools + "\\n", "")
    if list(argv[:2]) == ["git", "rev-parse"]:
        return update.Done(0, sha + "\\n", "")
    if list(argv[:3]) == ["claude", "plugin", "list"]:
        return update.Done(0, json.dumps([{"id": "hands@cc-hands", "scope": "user", "enabled": True}]), "")
    return update.Done(0, "", "")

update.run = run
sys.exit(cli.main(["--home", home, "update"]))
"""


@pytest.mark.parametrize("killed", [True, False], ids=["brain", "terminal"])
def test_an_update_restarts_hands_onto_master_and_says_the_commit_whether_or_not_its_caller_is_killed(killed: bool, tmp_path: Path) -> None:
    home, log, said = Home(tmp_path / "home"), tmp_path / "commands", tmp_path / "said"
    # The finally reads it however early this test fails.
    said.touch()
    daemon = subprocess.Popen([sys.executable, STANDIN, str(home.root)], stdin=subprocess.DEVNULL)
    # The brain: a process group of its own, as hands starts it, which the update is run from.
    brain = subprocess.Popen(["sleep", "60"], process_group=0)
    try:
        before = up(home)
        with said.open("w") as out:
            caller = subprocess.Popen([sys.executable, "-c", CALLER, str(home.root), str(tools_of_this_hands(tmp_path)), SHA, str(log)], stdout=out, stderr=subprocess.STDOUT, process_group=brain.pid)
        if killed:
            # The restart ends the brain, and hands kills everything in its process group.
            restarted_from(home, before)
            os.killpg(brain.pid, signal.SIGKILL)
            assert caller.wait(timeout=30) in (0, -signal.SIGKILL)
        else:
            # A terminal the restart does not end has the update's exit once hands is back, and both its lines.
            assert caller.wait(timeout=60) == 0

        event = waited_for(home)

        after = heartbeat.read(home.status)
        assert after is not None
        assert (after.pid, after.pipeline) == (daemon.pid, "running") and after.started_at > before.started_at
        facts = event["facts"]
        assert (event["outcome"], facts["command"], facts["exit_code"], facts["sha"], facts["outcome"]["type"]) == ("ok", "update", 0, SHA, "Restarted")
        # The run after the restart says it runs from the environment master went into, and the caller's output had the line.
        assert (facts["runs_master"], facts["told"]) == (True, True)
        if not killed:
            assert facts["exit_taken"] is True
        # The running hands named the environment its restart starts it on, the one the update installed into.
        assert facts["restarts_from"] == sys.prefix
        assert [step["step"] for step in facts["steps"]] == TOOL_STEPS
        assert [json.loads(line)[:3] for line in log.read_text().splitlines()] == [list(argv[:3]) for argv, _cwd in expected(home, Path("/"))]
        first, restarted = said.read_text().splitlines()
        assert first == f"installed hands, its fritter, and the brain's Claude Code from master at {SHA}; restarting hands onto it"
        assert restarted.startswith(f"hands restarted: pid {daemon.pid} is running again after ")
        assert restarted.endswith(f", with 0 live sessions. It runs master at {SHA}.")
    finally:
        # pytest's captured output of a failed test: the update's own words, whichever way this test failed.
        print(said.read_text())
        brain.kill()
        brain.wait()
        daemon.terminate()
        daemon.wait(timeout=10)


# An update gone on in a process of its own that ends, as a crash or a kill ends it, before it answers its caller.
UNANSWERED = """
import os
from hands.clock import SYSTEM
from hands.daemon import update

update.detach(SYSTEM)
os._exit(3)
"""


def test_a_caller_the_update_ends_without_answering_exits_1_saying_so() -> None:
    caller = subprocess.run([sys.executable, "-c", UNANSWERED], capture_output=True, text=True, timeout=30)

    assert caller.returncode == 1
    assert caller.stderr == "hands update: the update ended before it said how hands' restart went; `hands status` says whether hands is running\n"


def test_a_caller_the_restart_ended_is_not_answered() -> None:
    read, write = os.pipe()
    os.close(read)
    assert update.answered(write, 0) is False
    # The answer's end is closed either way.
    with pytest.raises(OSError):
        os.fstat(write)


@pytest.mark.parametrize("command", ["update", "restart"])
def test_the_brain_is_given_the_update_and_restart_skills_and_may_run_each_without_asking(command: str, tmp_path: Path) -> None:
    skill = (PLUGIN / "skills" / command / "SKILL.md").read_text()
    assert skill.startswith(f"---\nname: {command}\n")
    # The command it runs is the one its instructions name, hands' own interpreter and home, the one it is allowed.
    [allowed] = re.findall(r"\nallowed-tools: Bash\((.*)\)\n", skill)
    named = shlex.join(invocation(Home(tmp_path / "a home"), command))
    assert f"The {command} command is: {named}" in brain_instruction(tmp_path / "audit", tmp_path / "brain", shlex.join(invocation(Home(tmp_path / "a home"))), None)
    assert fnmatch.fnmatchcase(named, allowed) and f"run the {command} command" in skill


def up(home: Home, deadline: float = 10.0) -> heartbeat.Status:
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        status = heartbeat.read(home.status)
        if status is not None and status.pipeline == "running":
            return status
        time.sleep(0.02)
    raise AssertionError(f"the daemon did not say running within {deadline}s: {heartbeat.read(home.status)}")


def restarted_from(home: Home, before: heartbeat.Status, deadline: float = 60.0) -> None:
    """Once the run `before` was has been restarted."""
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        status = heartbeat.read(home.status)
        if status is not None and status.started_at > before.started_at:
            return
        time.sleep(0.02)
    raise AssertionError(f"hands was not restarted within {deadline}s")


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

    answers: list[int] = []
    monkeypatch.setattr(update, "run", commands)
    monkeypatch.setattr(update, "detach", detaching(lambda: answering(answers)))

    # Nothing ends the caller, so the caller hears it, and exits saying hands was not restarted.
    assert cli.main(["--home", str(home.root), "update"]) == 1
    assert answers == [1]

    out, err = capsys.readouterr()
    assert out == f"installed hands, its fritter, and the brain's Claude Code from master at {SHA}; restarting hands onto it\n"
    assert err.startswith("hands was not restarted: hands has not run")
    [event] = events(home)
    facts = event["facts"]
    assert (facts["command"], facts["exit_code"], facts["sha"], facts["outcome"]["type"]) == ("update", 1, SHA, "NotRunning")
    assert event["error"] == f"exited 1: {err.removesuffix('\n')}"


def detaching(answer: Callable[[], Callable[[int], bool]]) -> Callable[[Clock], Callable[[int], bool]]:
    """`update.detach`, standing in: what `answer` makes, on whatever clock it is given."""
    return lambda _clock: answer()


def answering(answers: list[int], *, taken: bool = True) -> Callable[[int], bool]:
    """What answers a caller, keeping each exit it is given in `answers`, and saying whether the caller took it."""

    def answer(code: int) -> bool:
        answers.append(code)
        return taken

    return answer


def status_from(runs_from: Path | None) -> heartbeat.Status:
    now = datetime.now(UTC)
    return heartbeat.Status(os.getpid(), now, now, heartbeat.HEARTBEAT, "running", None, 0, False, (), runs_from)


def restarting_from(monkeypatch: pytest.MonkeyPatch, runs_from: Path | None) -> None:
    """hands' restart brings it back running from `runs_from`."""

    def restart(*_: object) -> Outcome:
        return Restarted(status_from(runs_from), timedelta(seconds=3))

    monkeypatch.setattr(cli, "restart", restart)


@pytest.mark.parametrize("runs_from", [Path("/checkout/.venv"), None], ids=["elsewhere", "unsaid"])
def test_a_hands_that_restarts_from_another_environment_is_not_said_to_run_master(runs_from: Path | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    detached: list[bool] = []
    monkeypatch.setattr(update, "run", Commands(tools_of_this_hands(tmp_path)))
    monkeypatch.setattr(update, "detach", detaching(lambda: detached.append(True) or answering([])))
    # A daemon started by a checkout's `uv run hands run` starts again on the checkout, whatever the update installed.
    restarting_from(monkeypatch, runs_from)

    assert cli.main(["--home", str(home.root), "update"]) == 1

    assert detached == [True]
    _out, err = capsys.readouterr()
    assert err.endswith(f" It runs from {runs_from or 'an environment it does not name'}, not from {sys.prefix}, where master at {SHA} is installed.\n")
    [event] = events(home)
    assert (event["facts"]["exit_code"], event["facts"]["runs_master"], event["facts"]["told"]) == (1, False, True)
    assert event["error"] == f"exited 1: {err.removesuffix('\n')}"


def test_an_install_fritter_that_installed_and_finds_another_claude_first_on_the_caller_s_path_is_said_and_gone_past(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    # What install-fritter says from a shell whose PATH does not have hands' bin first, as the brain's Bash tool's has not.
    said = f"`claude` on this PATH is /Users/someone/.local/bin/claude, not hands' shim, so no session started from it can be typed into: export PATH=\"{home.bin}:$PATH\""
    unfound = Done(Installing.UNFOUND, f"copied fritter to {home.fritter}\nwrote {home.shim}\n", f"{said}\n")
    commands = Commands(tools_of_this_hands(tmp_path), Step.INSTALL_FRITTER, unfound)
    monkeypatch.setattr(update, "run", commands)
    monkeypatch.setattr(update, "detach", detaching(lambda: answering([])))
    restarting_from(monkeypatch, Path(sys.prefix))

    assert cli.main(["--home", str(home.root), "update"]) == 0

    # Every step ran, the brain's Claude Code installed after the fritter, and hands restarted onto master.
    assert [argv[:3] for argv, _cwd in commands.ran] == [argv[:3] for argv, _cwd in expected(home, Path("/"))]
    out, err = capsys.readouterr()
    assert out.endswith(f" It runs master at {SHA}.\n")
    assert err == f"hands update: install-fritter: {said}\n"
    # [LAW:nothing-unseen] the event says the step it went past, and what that step found.
    [event] = events(home)
    facts = event["facts"]
    assert (event["outcome"], facts["exit_code"], facts["runs_master"]) == ("ok", 0, True)
    assert [(step["step"], step["code"]) for step in facts["steps"]] == [(str(step), Installing.UNFOUND if step is Step.INSTALL_FRITTER else 0) for step in TOOL_STEPS]
    assert facts["went_past"] == [{"type": "Advised", "step": "install-fritter", "said": said}]


def test_an_install_fritter_whose_install_does_not_work_stops_the_update(tmp_path: Path) -> None:
    home = Home(tmp_path / "home")
    broken = Done(Installing.BROKEN, "", "hands install-fritter: hands' package carries no fritter\n")
    commands = Commands(tools_of_this_hands(tmp_path), Step.INSTALL_FRITTER, broken)

    work = tmp_path / "work"

    installed = update.install(home, Path(sys.prefix), None, {}, work, commands)

    # Anything but the one exit it goes past is the step failing, the brain's Claude Code left uninstalled.
    assert isinstance(installed, update.Unfinished) and (installed.sha, installed.step) == (SHA, Step.INSTALL_FRITTER)
    assert commands.ran == expected(home, work)[: TOOL_STEPS.index(Step.INSTALL_FRITTER) + 1]


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
    monkeypatch.setattr(update, "detach", detaching(lambda: monkeypatch.setattr(sys, "stdout", gone) or answering([], taken=False)))
    restarting_from(monkeypatch, Path(sys.prefix))

    assert cli.main(["--home", str(home.root), "update"]) == 0

    [event] = events(home)
    facts = event["facts"]
    assert (event["outcome"], facts["runs_master"], facts["told"], facts["exit_taken"]) == ("ok", True, False, False)
    assert why in facts["untold"]


def test_a_command_s_output_is_read_apart_from_its_errors(tmp_path: Path) -> None:
    # A warning uv or git prints is never taken for the path or the commit the update reads.
    assert update.run([sys.executable, "-c", "import sys; print('/tools'); print('warning: stale', file=sys.stderr)"], tmp_path, os.environ) == Done(0, "/tools\n", "warning: stale\n")


@pytest.mark.parametrize(
    "listed",
    [
        json.dumps([]),
        json.dumps([{"id": "other@elsewhere", "scope": "user", "enabled": True}]),
        json.dumps([{"id": PLUGIN_ID, "scope": "project", "enabled": True}]),
    ],
    ids=["none", "another-plugin", "project-scope-only"],
)
def test_a_user_who_never_installed_the_plugin_is_not_blocked_by_it(listed: str, tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    commands = Commands(tools_of_this_hands(tmp_path))
    commands.listed = listed

    installed = update.install(home, Path(sys.prefix), None, {}, work, commands)

    # The list is asked and the plugin is not touched: a project's install is that project's to update.
    assert isinstance(installed, Installed)
    assert commands.ran == [step for step in expected(home, work) if step[0][:3] not in (("claude", "plugin", "marketplace"), ("claude", "plugin", "update"))]


def test_no_claude_at_all_is_said_and_gone_past(tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    gone = Done(update.UNSTARTABLE, "", "[Errno 2] No such file or directory: 'claude'")
    commands = Commands(tools_of_this_hands(tmp_path), Step.PLUGIN_LIST, gone)

    installed = update.install(home, Path(sys.prefix), None, {}, work, commands)

    assert isinstance(installed, Installed)
    assert installed.advised == (update.Advised(Step.PLUGIN_LIST, gone.errors),)
    # The rendered trees are removed all the same.
    assert commands.ran[-1] == (("rm", "-rf", str(home.root / "plugins")), work)


def test_a_list_hands_cannot_read_leaves_master_installed_and_the_plugin_untouched(tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    commands = Commands(tools_of_this_hands(tmp_path))
    commands.listed = "not json"

    installed = update.install(home, Path(sys.prefix), None, {}, work, commands)

    assert isinstance(installed, update.Unfinished) and (installed.sha, installed.step) == (SHA, Step.PLUGIN_LIST)
    assert commands.ran == expected(home, work)[: TOOL_STEPS.index(Step.PLUGIN_LIST) + 1]


def test_a_failure_to_remove_the_rendered_trees_is_said_and_the_update_goes_on(tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    refused = Done(1, "", "rm: permission denied\n")
    commands = Commands(tools_of_this_hands(tmp_path), Step.RENDERED_PLUGINS, refused)

    installed = update.install(home, Path(sys.prefix), None, {}, work, commands)

    # Nothing reads those trees, so hands is still restarted onto master.
    assert isinstance(installed, Installed)
    assert installed.advised == (update.Advised(Step.RENDERED_PLUGINS, "rm: permission denied"),)


@pytest.mark.parametrize("name", ["brain", "users"])
def test_the_plugin_commands_run_under_the_callers_claude_config_not_the_brain_s(name: str, tmp_path: Path) -> None:
    home, work = Home(tmp_path / "home"), tmp_path / "work"
    # The brain runs `hands update` from a shell whose config is the brain's, which holds none of the user's plugins;
    # a user's own config is theirs to keep.
    config = {"brain": home.brain, "users": tmp_path / "users-claude"}[name]
    commands = Commands(tools_of_this_hands(tmp_path))

    update.install(home, Path(sys.prefix), None, {"CLAUDE_CONFIG_DIR": str(config), "KEPT": "yes"}, work, commands)

    steps = list(zip(commands.ran, commands.environments, strict=True))
    claudes = [env for (argv, _cwd), env in steps if argv[0] == "claude"]
    assert len(claudes) == 3
    # The brain's directory is removed, not rewritten; a user's is passed through untouched.
    assert all(("CLAUDE_CONFIG_DIR" in env) == (name == "users") for env in claudes)
    assert all(env["CLAUDE_CONFIG_DIR"] == str(config) for env in claudes if name == "users")
    # Only the claude commands run under the plugin environment; every other step gets the caller's own.
    others = [env for (argv, _cwd), env in steps if argv[0] != "claude"]
    assert all(env["CLAUDE_CONFIG_DIR"] == str(config) and env["KEPT"] == "yes" for env in others)
