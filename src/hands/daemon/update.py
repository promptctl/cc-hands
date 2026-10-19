"""`hands update`: install hands from the current master of promptctl/cc-hands, then restart onto it in place.

The brain's hands:update skill runs it when the user asks hands to update, reinstall, or restart itself from master.
Each step is a command, run in order, and the first that fails stops the update: nothing after it runs, and hands is
not restarted, so the running hands goes on as it was and never restarts onto a half-installed build.

1. where: `uv tool dir`. This hands must run from the uv tool environment the install replaces, since a restart
   starts the daemon again on its own interpreter (hands.daemon.starting.invocation): an update from anywhere else
   would install master and restart onto what was there before.
2. fetch: master is cloned fresh, never taken from a checkout on this machine, which may be stale or on a branch.
3. commit: the commit that clone holds, which is the one installed and the one the update reports.
4. constraints: uv.lock's versions, the ones master's tests ran on, as release.yml exports them for a release.
5. install: `uv tool install` of that commit from GitHub, on the Python this hands runs on, so the interpreter path
   stays the one the daemon restarts on. uv's record of the tool names the commit, not the clone, which is gone once
   the update ends, so `uv tool upgrade` still finds what it installed. It builds hands, fritter with it, so it needs Go.
6. install-fritter and install-brain, each run by the hands just installed, so the fritter and the brain's Claude Code
   are the ones master carries and pins. One of them failing leaves master installed without it: the running hands
   goes on as it was, and its next start is master's (Unfinished).
"""

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from hands.daemon.starting import invocation
from hands.sessions.home import Home

REPOSITORY = "https://github.com/promptctl/cc-hands.git"
BRANCH = "master"
# The name `uv tool install` gives hands' environment in its tool directory: the package's.
TOOL = "hands"
COMMIT = re.compile(r"[0-9a-f]{40}")
# How often the update, gone on in a process of its own, looks whether the one it left has exited.
LOOK_SECONDS = 0.01


class Step(StrEnum):
    WHERE = "where"
    FETCH = "fetch"
    COMMIT = "commit"
    CONSTRAINTS = "constraints"
    INSTALL = "install"
    INSTALL_FRITTER = "install-fritter"
    INSTALL_BRAIN = "install-brain"


@dataclass(frozen=True)
class Done:
    """What one command did: its exit code, what it printed to its output, and what to its errors."""

    code: int
    output: str
    errors: str


type Run = Callable[[Sequence[str], Path], Done]


@dataclass(frozen=True)
class Ran:
    """One step as the update's event records it: the step, the command it ran, and its exit code."""

    step: Step
    argv: tuple[str, ...]
    code: int


@dataclass(frozen=True)
class Installed:
    """Every step ran and exited 0: hands, its fritter, and the brain's Claude Code are master's at `sha`."""

    sha: str
    ran: tuple[Ran, ...]


@dataclass(frozen=True)
class Failed:
    """`step` failed, before hands was installed, for the reason `said` gives; `ran` is every step run, it last, and none
    after it ran."""

    step: Step
    said: str
    ran: tuple[Ran, ...]


@dataclass(frozen=True)
class Unfinished:
    """hands is installed from master at `sha`, and `step`, after it, failed for the reason `said` gives: hands' next
    start is master's, without what `step` installs. `ran` is every step run, it last, and none after it ran."""

    sha: str
    step: Step
    said: str
    ran: tuple[Ran, ...]


class _Stopped(Exception):
    def __init__(self, stopped: Failed | Unfinished) -> None:
        super().__init__(stopped.said)
        self.stopped = stopped


def install(home: Home, prefix: Path, work: Path, run: Run) -> Installed | Failed | Unfinished:
    """Install master into the uv tool environment at `prefix`, this hands', cloning it into `work`; or the step that failed."""
    ran: list[Ran] = []
    # The commit installed, once the install step has put it in hands' environment: a failure from then on is Unfinished.
    installed: str | None = None

    def step(name: Step, argv: Sequence[str], cwd: Path) -> str:
        done = run(argv, cwd)
        ran.append(Ran(name, tuple(argv), done.code))
        if done.code != 0:
            said = f"`{' '.join(argv)}` exited {done.code}:\n{done.errors}{done.output}"
            raise _Stopped(Failed(name, said, tuple(ran)) if installed is None else Unfinished(installed, name, said, tuple(ran)))
        return done.output

    def refused(name: Step, said: str) -> _Stopped:
        return _Stopped(Failed(name, said, tuple(ran)))

    clone, constraints = work / "cc-hands", work / "constraints.txt"
    try:
        environment = Path(step(Step.WHERE, ("uv", "tool", "dir"), work).strip()) / TOOL
        if environment.resolve() != prefix.resolve():
            raise refused(Step.WHERE, f"this hands runs from {prefix}, not from {environment}, the uv tool environment the update installs into, so a restart would not run what it installed: run `hands update` with the hands `uv tool install` put on PATH")
        step(Step.FETCH, ("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", BRANCH, REPOSITORY, str(clone)), work)
        # [LAW:parse-dont-validate] the commit as git names it, or the step fails saying what git printed instead.
        sha = step(Step.COMMIT, ("git", "rev-parse", "--verify", "HEAD^{commit}"), clone).strip()
        if not COMMIT.fullmatch(sha):
            raise refused(Step.COMMIT, f"git named the commit {BRANCH} is at as {sha!r}, not a commit's 40 hex digits")
        step(Step.CONSTRAINTS, ("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(constraints)), clone)
        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        step(Step.INSTALL, ("uv", "tool", "install", "--reinstall-package", TOOL, "--python", python, "--constraints", str(constraints), f"{TOOL} @ git+{REPOSITORY}@{sha}"), work)
        installed = sha
        # Run by the interpreter the install just filled, so each is master's own command.
        step(Step.INSTALL_FRITTER, invocation(home, "install-fritter"), work)
        step(Step.INSTALL_BRAIN, invocation(home, "install-brain"), work)
    except _Stopped as stopped:
        return stopped.stopped
    return Installed(sha, tuple(ran))


def run(argv: Sequence[str], cwd: Path) -> Done:
    """Run `argv` in `cwd`, with nothing to read and no terminal to ask on, and what it did."""
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            # git asks for a login on the terminal when a fetch is refused; the brain's has nobody at it.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            stdin=subprocess.DEVNULL,
            # Apart: what a command prints to its output is what the update reads, a path or a commit, and a warning
            # on its errors is no part of that.
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as error:
        # [LAW:no-silent-failure] a command that could not start, uv or git missing, is the step failing with why.
        return Done(127, "", str(error))
    return Done(completed.returncode, completed.stdout, completed.stderr)


def detach() -> None:
    """Go on in a process of its own, outside the caller's process tree, session, and process group; the caller has
    this process's exit, 0, now.

    The brain runs `hands update` from its shell, and the restart the update asks for stops the brain: Claude Code ends
    the commands it ran, the brain's terminal hangs up on its session, and hands kills the brain's process group. The
    update outlives all three only once it is in a session of its own and launchd has taken it in, so it asks for the
    restart, waits for it, and writes its event only after that. What it prints after this still goes where its
    output went.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    caller = os.getpid()
    if os.fork() != 0:
        # The process the caller started ends here, with nothing of Python's exit: the unit of work open in it is the
        # child's to end, and its event the child's to write.
        os._exit(0)
    os.setsid()
    os.dup2(os.open(os.devnull, os.O_RDONLY), sys.stdin.fileno())
    # [LAW:no-ambient-temporal-coupling] out of the caller's tree once its parent has exited, not after a guess at how long that takes.
    while os.getppid() == caller:
        time.sleep(LOOK_SECONDS)
