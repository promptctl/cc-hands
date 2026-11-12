"""`hands update`: bring the code hands runs from up to date, then restart onto it in place.

The brain's hands:update skill runs it when the user asks hands to update or reinstall itself.
Each step is a command, run in order, and the first that fails stops the update: nothing after it runs, and hands is
not restarted, so the running hands goes on as it was and never restarts onto a half-installed build.

1. where: `uv tool dir`. A restart starts the daemon again on its own interpreter (hands.daemon.starting.invocation),
   so what the update brings current is the code of the environment this hands, and the running hands its heartbeat
   names, run from (`Target`): the uv tool environment, which gets master of promptctl/cc-hands; or a checkout's own
   `.venv`, which gets the checkout. Any other environment is refused: an update would change something else and
   restart hands onto what it ran before.
2. fetch (the tool environment): master is cloned fresh, never taken from a checkout on this machine, which may be
   stale or on a branch. pull (a checkout): `git pull --ff-only` of the branch it is on, which never loses a commit
   of the checkout's and stops the update where it cannot fast-forward.
3. commit: the commit that clone, or that checkout, holds: the one installed and the one the update reports.
4. constraints (the tool environment): uv.lock's versions, the ones master's tests ran on, as release.yml exports them
   for a release.
5. install (the tool environment): `uv tool install` of that commit from GitHub, on the very interpreter this hands'
   environment was made from: uv makes the environment anew on any other, even one of the same version, under the
   running daemon. uv's record of the tool names the commit, not the clone, which is gone once the update ends, so
   `uv tool upgrade` still finds what it installed. sync (a checkout): `uv sync --locked`, hands built anew as the
   install does. Either builds hands, fritter with it, so it needs Go.
6. install-fritter and install-brain, each run by the hands just installed, so the fritter and the brain's Claude Code
   are the ones master carries and pins. One of them failing leaves master installed without it: the running hands
   is not restarted, while what starts from the environment from then on, its hooks and its own next start, is
   master's (Unfinished). install-fritter finding the claude it installed not first on the caller's PATH is no failure:
   it is the caller's shell, the brain's never has hands' bin first, so the update says it and goes on (Advised).
7. plugin-list, marketplace, plugin: the plugin Claude Code loads is a plain directory of this repository
   (.claude-plugin/marketplace.json), so it is current only once Claude Code takes master's copy: the marketplace
   first, since an install from before the plugin was a plain directory holds an entry whose source is the removed
   command `hands plugin`, which `claude plugin update` would run and fail; then the plugin. A user who never
   installed the plugin has no user-scope entry in the list, and the two are not run for them. claude not found at
   all is said and gone past (Advised): there is no plugin to bring current.
8. rendered-plugins: the digest directories under the home's `plugins` that `hands plugin` rendered. Nothing reads
   them, so a failure to remove them is said and gone past (Advised), never a reason to leave hands unrestarted.
"""

import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from hands.clock import Clock
from hands.daemon import readiness
from hands.daemon.starting import invocation
from hands.sessions.home import Home
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.payload import Rejected
from hands.sessions.wrapper import Installing

REPOSITORY = "https://github.com/promptctl/cc-hands.git"
BRANCH = "master"
# The name `uv tool install` gives hands' environment in its tool directory: the package's.
TOOL = "hands"
COMMIT = re.compile(r"[0-9a-f]{40}")
# How often the update, gone on in a process of its own, looks whether the one it left has exited.
LOOK_SECONDS = 0.01
# The exit `run` gives a command that could not start.
UNSTARTABLE = 127
# Every exit but success, for a step that is said and gone past whichever way it fails.
ANY_FAILURE = frozenset(range(1, 256))
# The marketplace hands' plugin comes from, named by the plugin's own id.
MARKETPLACE = PLUGIN_ID.partition("@")[2]
# What `hands plugin` rendered its digest directories under, in the home: nothing writes or reads it any more, so its
# step goes with the ticket that removes it, hands-install-dye.uk8.
RENDERED = "plugins"


@dataclass(frozen=True)
class Tool:
    """hands runs from the uv tool environment, which the update replaces with master's."""


@dataclass(frozen=True)
class Checkout:
    """hands runs from the `.venv` of the checkout at `root`, which the update brings up to date."""

    root: Path


@dataclass(frozen=True)
class Unsupported:
    """hands runs from an environment that is neither the uv tool environment nor a checkout's own `.venv`."""


type Target = Tool | Checkout


def parse_target(prefix: Path, tool: Path) -> Target | Unsupported:
    """What the update brings current, from the environment hands runs from.

    [LAW:parse-dont-validate] the one place the environment is judged: everything after it matches on the Target and
    never asks where hands runs from again. A checkout is a project named hands whose `.venv` is `prefix`.
    """
    if prefix.resolve() == tool.resolve():
        return Tool()
    root = prefix.parent
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return Unsupported()
    return Checkout(root) if prefix.name == ".venv" and project.get("name") == TOOL else Unsupported()


class Step(StrEnum):
    WHERE = "where"
    FETCH = "fetch"
    PULL = "pull"
    COMMIT = "commit"
    CONSTRAINTS = "constraints"
    INSTALL = "install"
    SYNC = "sync"
    INSTALL_FRITTER = "install-fritter"
    INSTALL_BRAIN = "install-brain"
    PLUGIN_LIST = "plugin-list"
    MARKETPLACE = "marketplace"
    PLUGIN = "plugin"
    RENDERED_PLUGINS = "rendered-plugins"


@dataclass(frozen=True)
class Done:
    """What one command did: its exit code, what it printed to its output, and what to its errors."""

    code: int
    output: str
    errors: str


type Run = Callable[[Sequence[str], Path, Mapping[str, str]], Done]


@dataclass(frozen=True)
class Ran:
    """One step as the update's event records it: the step, the command it ran, and its exit code."""

    step: Step
    argv: tuple[str, ...]
    code: int


@dataclass(frozen=True)
class Advised:
    """A step that did what it is for and found something of its caller's beside it, which the update says and goes on
    past: the step, and what it said of it."""

    step: Step
    said: str


@dataclass(frozen=True)
class Installed:
    """Every step ran and did what it is for: hands, its fritter, and the brain's Claude Code are the `target`'s at `sha`;
    `advised` is what any of them found of its caller's and went on past."""

    target: Target
    sha: str
    ran: tuple[Ran, ...]
    advised: tuple[Advised, ...]


@dataclass(frozen=True)
class Failed:
    """`step` failed, before hands was installed, for the reason `said` gives; `ran` is every step run, it last, and none
    after it ran."""

    step: Step
    said: str
    ran: tuple[Ran, ...]


@dataclass(frozen=True)
class Unfinished:
    """hands is installed from the `target` at `sha`, and `step`, after it, failed for the reason `said` gives: what starts
    from hands' environment from then on is the target's, short of what `step` installs. `ran` is every step run, it last, and none after it ran."""

    target: Target
    sha: str
    step: Step
    said: str
    ran: tuple[Ran, ...]


class _Stopped(Exception):
    def __init__(self, stopped: Failed | Unfinished) -> None:
        super().__init__(stopped.said)
        self.stopped = stopped


def install(home: Home, prefix: Path, restarts_from: Path | None, caller: Mapping[str, str], work: Path, run: Run) -> Installed | Failed | Unfinished:
    """Bring the code hands runs from, at `prefix`, up to date, as `parse_target` judges it, cloning master into `work`
    for the uv tool environment; or the step that failed. `restarts_from` is the environment the running hands starts
    again on, by its own word: None with none running, or one from before its heartbeat said. `caller` is the caller's
    environment, whose Claude Code config the plugin steps run under."""
    ran: list[Ran] = []
    advised: list[Advised] = []
    # The target and commit installed, once the install or sync step has put it in hands' environment: a failure from
    # then on is Unfinished.
    installed: tuple[Target, str] | None = None

    def ending(name: Step, said: str) -> _Stopped:
        return _Stopped(Failed(name, said, tuple(ran)) if installed is None else Unfinished(*installed, name, said, tuple(ran)))

    def step(name: Step, argv: Sequence[str], cwd: Path, past: frozenset[int] = frozenset(), env: Mapping[str, str] = caller) -> Done:
        """What the step did; it stops the update on any exit but 0 and those in `past`, which it says and goes on past."""
        done = run(argv, cwd, env)
        ran.append(Ran(name, tuple(argv), done.code))
        if done.code in past:
            advised.append(Advised(name, done.errors.strip()))
        elif done.code != 0:
            raise ending(name, f"`{' '.join(argv)}` exited {done.code}:\n{done.errors}{done.output}")
        return done

    def commit(cwd: Path, of: str) -> str:
        # [LAW:parse-dont-validate] the commit as git names it, or the step fails saying what git printed instead.
        sha = step(Step.COMMIT, ("git", "rev-parse", "--verify", "HEAD^{commit}"), cwd).output.strip()
        if not COMMIT.fullmatch(sha):
            raise ending(Step.COMMIT, f"git named the commit {of} is at as {sha!r}, not a commit's 40 hex digits")
        return sha

    clone, constraints = work / "cc-hands", work / "constraints.txt"
    try:
        tool = Path(step(Step.WHERE, ("uv", "tool", "dir"), work).output.strip()) / TOOL
        target = parse_target(prefix, tool)
        if isinstance(target, Unsupported):
            raise ending(Step.WHERE, f"this hands runs from {prefix}, which is neither {tool}, the uv tool environment the update installs master into, nor the .venv of a checkout of hands, so a restart would not run what it updated: run `hands update` with the hands `uv tool install` put on PATH, or from a checkout's own .venv")
        if restarts_from is not None and restarts_from.resolve() != prefix.resolve():
            raise ending(Step.WHERE, f"the running hands runs from {restarts_from}, not from {prefix}, the environment the update brings current, so its restart would start it again on what it runs now: run hands from {prefix}")
        match target:
            case Tool():
                step(Step.FETCH, ("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", BRANCH, REPOSITORY, str(clone)), work)
                sha = commit(clone, BRANCH)
                step(Step.CONSTRAINTS, ("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(constraints)), clone)
                # The interpreter itself, not its version: uv keeps an environment in place only on the one it was made from
                # (measured, uv 0.12.15: `--python 3.12` found another 3.12 and made the environment anew).
                step(Step.INSTALL, ("uv", "tool", "install", "--reinstall-package", TOOL, "--python", made_from(), "--constraints", str(constraints), f"{TOOL} @ git+{REPOSITORY}@{sha}"), work)
            case Checkout(root=root):
                step(Step.PULL, ("git", "pull", "--quiet", "--ff-only"), root)
                sha = commit(root, "the checkout")
                step(Step.SYNC, ("uv", "sync", "--locked", "--reinstall-package", TOOL), root)
        installed = (target, sha)
        # Run by the interpreter the install just filled, so each is the target's own command.
        step(Step.INSTALL_FRITTER, invocation(home, "install-fritter"), work, frozenset({Installing.UNFOUND}))
        step(Step.INSTALL_BRAIN, invocation(home, "install-brain"), work)
        # [LAW:one-source-of-truth] hands check's own command, environment, and reading of the list: the plugin is
        # installed for the user or it is not; no claude at all is no plugin.
        under = readiness.plugin_environment(home, caller)
        listing = step(Step.PLUGIN_LIST, readiness.PLUGIN_LIST, work, frozenset({UNSTARTABLE}), under)
        try:
            held = listing.code != UNSTARTABLE and bool(readiness.installed_everywhere(listing.output))
        except Rejected as error:
            raise ending(Step.PLUGIN_LIST, f"`claude plugin list --json` printed what hands cannot read: {error}") from None
        if held:
            step(Step.MARKETPLACE, ("claude", "plugin", "marketplace", "update", MARKETPLACE), work, env=under)
            step(Step.PLUGIN, ("claude", "plugin", "update", PLUGIN_ID, "--scope", "user"), work, env=under)
        step(Step.RENDERED_PLUGINS, ("rm", "-rf", str(home.root / RENDERED)), work, ANY_FAILURE)
    except _Stopped as stopped:
        return stopped.stopped
    return Installed(target, sha, tuple(ran), tuple(advised))


def made_from() -> str:
    """The interpreter this hands' environment was made from: Python's own name for it, which typeshed does not declare."""
    return cast(str, getattr(sys, "_base_executable"))


def run(argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> Done:
    """Run `argv` in `cwd`, with nothing to read and no terminal to ask on, and what it did."""
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            # git asks for a login on the terminal when a fetch is refused; the brain's has nobody at it.
            env={**env, "GIT_TERMINAL_PROMPT": "0"},
            stdin=subprocess.DEVNULL,
            # Apart: what a command prints to its output is what the update reads, a path or a commit, and a warning
            # on its errors is no part of that.
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as error:
        # [LAW:no-silent-failure] a command that could not start, uv or git missing, is the step failing with why.
        return Done(UNSTARTABLE, "", str(error))
    return Done(completed.returncode, completed.stdout, completed.stderr)


def detach(clock: Clock) -> Callable[[int], bool]:
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
        clock.rest(LOOK_SECONDS)
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
