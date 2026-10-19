"""`hands update`: install hands from the current master of promptctl/cc-hands, then restart onto it in place.

The brain's hands:update skill runs it when the user asks hands to update, reinstall, or restart itself from master.
Each step is a command, run in order, and the first that fails stops the update: nothing after it runs, and hands is
not restarted, so the running hands goes on as it was and never restarts onto a half-installed build.

1. where: `uv tool dir`. This hands, and the running hands its heartbeat names, must run from the uv tool environment
   the install replaces, since a restart starts the daemon again on its own interpreter
   (hands.daemon.starting.invocation): an update of a hands run from anywhere else would install master and restart
   it onto what it ran before.
2. fetch: master is cloned fresh, never taken from a checkout on this machine, which may be stale or on a branch.
3. commit: the commit that clone holds, which is the one installed and the one the update reports.
4. constraints: uv.lock's versions, the ones master's tests ran on, as release.yml exports them for a release.
5. install: `uv tool install` of that commit from GitHub, on the very interpreter this hands' environment was made
   from: uv makes the environment anew on any other, even one of the same version, under the running daemon. uv's record of the tool names the commit, not the clone, which is gone once
   the update ends, so `uv tool upgrade` still finds what it installed. It builds hands, fritter with it, so it needs Go.
6. install-fritter and install-brain, each run by the hands just installed, so the fritter and the brain's Claude Code
   are the ones master carries and pins. One of them failing leaves master installed without it: the running hands
   is not restarted, while what starts from the environment from then on, its hooks and its own next start, is
   master's (Unfinished).
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
from typing import cast

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
    """hands is installed from master at `sha`, and `step`, after it, failed for the reason `said` gives: what starts
    from hands' environment from then on is master's, short of what `step` installs. `ran` is every step run, it last, and none after it ran."""

    sha: str
    step: Step
    said: str
    ran: tuple[Ran, ...]


class _Stopped(Exception):
    def __init__(self, stopped: Failed | Unfinished) -> None:
        super().__init__(stopped.said)
        self.stopped = stopped


def install(home: Home, prefix: Path, restarts_from: Path | None, work: Path, run: Run) -> Installed | Failed | Unfinished:
    """Install master into the uv tool environment at `prefix`, this hands', cloning it into `work`; or the step that
    failed. `restarts_from` is the environment the running hands starts again on, by its own word: None with none
    running, or one from before its heartbeat said."""
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
        if restarts_from is not None and restarts_from.resolve() != environment.resolve():
            raise refused(Step.WHERE, f"the running hands runs from {restarts_from}, not from {environment}, the uv tool environment the update installs into, so its restart would start it again on what it runs now: run hands with the hands `uv tool install` put on PATH")
        step(Step.FETCH, ("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", BRANCH, REPOSITORY, str(clone)), work)
        # [LAW:parse-dont-validate] the commit as git names it, or the step fails saying what git printed instead.
        sha = step(Step.COMMIT, ("git", "rev-parse", "--verify", "HEAD^{commit}"), clone).strip()
        if not COMMIT.fullmatch(sha):
            raise refused(Step.COMMIT, f"git named the commit {BRANCH} is at as {sha!r}, not a commit's 40 hex digits")
        step(Step.CONSTRAINTS, ("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(constraints)), clone)
        # The interpreter itself, not its version: uv keeps an environment in place only on the one it was made from
        # (measured, uv 0.12.15: `--python 3.12` found another 3.12 and made the environment anew).
        step(Step.INSTALL, ("uv", "tool", "install", "--reinstall-package", TOOL, "--python", made_from(), "--constraints", str(constraints), f"{TOOL} @ git+{REPOSITORY}@{sha}"), work)
        installed = sha
        # Run by the interpreter the install just filled, so each is master's own command.
        step(Step.INSTALL_FRITTER, invocation(home, "install-fritter"), work)
        step(Step.INSTALL_BRAIN, invocation(home, "install-brain"), work)
    except _Stopped as stopped:
        return stopped.stopped
    return Installed(sha, tuple(ran))


def made_from() -> str:
    """The interpreter this hands' environment was made from: Python's own name for it, which typeshed does not declare."""
    return cast(str, getattr(sys, "_base_executable"))


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


def detach() -> Callable[[int], bool]:
    """Go on in a process of its own, outside the caller's process tree, session, and process group; and what answers
    the caller, still waiting in the process it started, with the exit to end with.

    The brain runs `hands update` from its shell, and the restart the update asks for stops the brain: Claude Code ends
    the commands it ran, the brain's terminal hangs up on its session, and hands kills the brain's process group. The
    update outlives all three only once it is in a session of its own and launchd has taken it in, so it asks for the
    restart, waits for it, and writes its event only after that. What it prints after this still goes where its
    output went, and the process the caller started waits for its answer and exits with it: a caller the restart does
    not end, a terminal, has the outcome before its prompt, and its exit.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    read, write = os.pipe()
    middle = os.fork()
    if middle != 0:
        os.close(write)
        # The middle process exits at once, leaving the update to launchd.
        os.waitpid(middle, 0)
        with os.fdopen(read, "rb") as answers:
            answer = answers.read()
        if len(answer) != 1:
            # [LAW:no-silent-failure] the update ended, a crash or a kill, before it said how its restart went.
            print("hands update: the update ended before it said how hands' restart went; `hands status` says whether hands is running", file=sys.stderr, flush=True)
        # The process the caller started ends here, with nothing of Python's exit: the unit of work open in it is the
        # update's to end, and its event the update's to write.
        os._exit(answer[0] if len(answer) == 1 else 1)
    os.close(read)
    os.setsid()
    os.dup2(os.open(os.devnull, os.O_RDONLY), sys.stdin.fileno())
    middle = os.getpid()
    if os.fork() != 0:
        os._exit(0)
    # [LAW:no-ambient-temporal-coupling] out of the caller's tree once its parent has exited, not after a guess at how long that takes.
    while os.getppid() == middle:
        time.sleep(LOOK_SECONDS)
    return lambda code: answered(write, code)


def answered(write: int, code: int) -> bool:
    """Give the caller waiting at `write` the exit `code`; False when it is gone, ended by the restart."""
    try:
        os.write(write, bytes([code]))
    except BrokenPipeError:
        return False
    finally:
        os.close(write)
    return True
