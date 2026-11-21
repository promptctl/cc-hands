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

import re
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import assert_never, cast

from hands.daemon import readiness
from hands.daemon.starting import invocation
from hands.sessions import child
from hands.sessions.child import Children, Exited, NeverStarted, Outcome, TimedOut, fields, summary
from hands.sessions.wide import Fact
from hands.sessions.home import Home
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.payload import Rejected
from hands.sessions.wrapper import Installing

REPOSITORY = "https://github.com/promptctl/cc-hands.git"
BRANCH = "master"
# The name `uv tool install` gives hands' environment in its tool directory: the package's.
TOOL = "hands"
COMMIT = re.compile(r"[0-9a-f]{40}")
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
    if prefix.name != ".venv":
        return Unsupported()
    root = prefix.parent
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return Unsupported()
    return Checkout(root) if project.get("name") == TOOL else Unsupported()


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


# [LAW:types-are-the-program] the ends of a step's command it says and goes on past, read off how it ended: never a code
# standing in for a signal, or for a command that could not start.
type Past = Callable[[child.Ran | NeverStarted], bool]


def _exits(*codes: int) -> Past:
    """Past these exits alone."""
    return lambda ended: isinstance(ended, child.Ran) and ended.status in [Exited(code) for code in codes]


def _unstarted(ended: child.Ran | NeverStarted) -> bool:
    """Past a command that could not start: a claude that is not there."""
    return isinstance(ended, NeverStarted)


def _any_exit(ended: child.Ran | NeverStarted) -> bool:
    """Past every exit but success, for a step said and gone past whichever code it fails with; never past a signal."""
    return isinstance(ended, child.Ran) and isinstance(ended.status, Exited) and ended.status.code != 0


type Run = Callable[[Sequence[str], Path, Mapping[str, str], float], Outcome]


def limit(step: Step) -> float:
    """How long `step` may run before it is killed and stops the update: sized for a slow link where it downloads.

    [LAW:types-are-the-program] exhaustive, so a step added without its limit is refused by pyright."""
    match step:
        case Step.WHERE | Step.COMMIT | Step.RENDERED_PLUGINS:
            return 60.0
        case Step.CONSTRAINTS | Step.PULL:
            return 600.0
        case Step.FETCH | Step.INSTALL | Step.SYNC | Step.INSTALL_BRAIN:
            return 3600.0
        case Step.INSTALL_FRITTER | Step.MARKETPLACE | Step.PLUGIN:
            return 600.0
        case Step.PLUGIN_LIST:
            # [LAW:one-source-of-truth] hands check's own bound on the same listing.
            return readiness.LIST_TIMEOUT_SECONDS
        case _:
            assert_never(step)


@dataclass(frozen=True)
class StepRun:
    """One step as the update's event records it: the step, the command it ran, and how it ended (`child.fields`)."""

    step: Step
    argv: tuple[str, ...]
    ended: Mapping[str, Fact]


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
    ran: tuple[StepRun, ...]
    advised: tuple[Advised, ...]


@dataclass(frozen=True)
class Failed:
    """`step` failed, before hands was installed, for the reason `said` gives; `ran` is every step run, it last, and none
    after it ran."""

    step: Step
    said: str
    ran: tuple[StepRun, ...]


@dataclass(frozen=True)
class Unfinished:
    """hands is installed from the `target` at `sha`, and `step`, after it, failed for the reason `said` gives: what starts
    from hands' environment from then on is the target's, short of what `step` installs. `ran` is every step run, it last, and none after it ran."""

    target: Target
    sha: str
    step: Step
    said: str
    ran: tuple[StepRun, ...]


class _Stopped(Exception):
    def __init__(self, stopped: Failed | Unfinished) -> None:
        super().__init__(stopped.said)
        self.stopped = stopped


def install(home: Home, prefix: Path, runs_from: Path | None, caller: Mapping[str, str], work: Path, run: Run) -> Installed | Failed | Unfinished:
    """Bring the code hands runs from, at `prefix`, up to date, as `parse_target` judges it, cloning master into `work`
    for the uv tool environment; or the step that failed. `runs_from` is the environment the running hands runs from,
    and its restart starts it again on, by its own word: None with none running, or one from before its heartbeat said. `caller` is the caller's
    environment, whose Claude Code config the plugin steps run under."""
    ran: list[StepRun] = []
    advised: list[Advised] = []
    # The target and commit installed, once the install step has put it in hands' environment, or the pull has put the
    # checkout's commit where hands runs from: a failure from then on is Unfinished.
    installed: tuple[Target, str] | None = None

    def ending(name: Step, said: str) -> _Stopped:
        return _Stopped(Failed(name, said, tuple(ran)) if installed is None else Unfinished(*installed, name, said, tuple(ran)))

    def took(name: Step, argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> Outcome:
        outcome = run(argv, cwd, env, limit(name))
        ran.append(StepRun(name, tuple(argv), fields(outcome)))
        return outcome

    def ended_by(name: Step, argv: Sequence[str], outcome: Outcome) -> _Stopped:
        command = " ".join(argv)
        match outcome:
            case child.Ran(out=out, err=err) | TimedOut(out=out, err=err):
                # What it printed before it ended is how far it got.
                return ending(name, f"`{command}` {summary(outcome)}:\n{err}{out.decode(errors='replace')}")
            case NeverStarted():
                return ending(name, f"`{command}` {summary(outcome)}")

    def step(name: Step, argv: Sequence[str], cwd: Path, env: Mapping[str, str] = caller) -> str:
        """What the step printed, once it exited 0; any other end stops the update."""
        match outcome := took(name, argv, cwd, env):
            case child.Ran(status=Exited(code=0), out=out):
                return out.decode(errors="replace")
            case child.Ran() | TimedOut() | NeverStarted():
                raise ended_by(name, argv, outcome)

    def advisable(name: Step, argv: Sequence[str], cwd: Path, past: Past, env: Mapping[str, str] = caller) -> str | Advised:
        """`step`, which says and goes on past the ends `past` goes past; running past its limit is never one."""
        match outcome := took(name, argv, cwd, env):
            case child.Ran(status=Exited(code=0), out=out):
                return out.decode(errors="replace")
            case child.Ran(err=err) as ended if past(ended):
                advised.append(advice := Advised(name, err.strip()))
                return advice
            case NeverStarted(error=error) as ended if past(ended):
                advised.append(advice := Advised(name, str(error)))
                return advice
            case child.Ran() | TimedOut() | NeverStarted():
                raise ended_by(name, argv, outcome)

    def commit(cwd: Path, of: str) -> str:
        # [LAW:parse-dont-validate] the commit as git names it, or the step fails saying what git printed instead.
        sha = step(Step.COMMIT, ("git", "rev-parse", "--verify", "HEAD^{commit}"), cwd).strip()
        if not COMMIT.fullmatch(sha):
            raise ending(Step.COMMIT, f"git named the commit {of} is at as {sha!r}, not a commit's 40 hex digits")
        return sha

    clone, constraints = work / "cc-hands", work / "constraints.txt"
    try:
        tool = Path(step(Step.WHERE, ("uv", "tool", "dir"), work).strip()) / TOOL
        target = parse_target(prefix, tool)
        if isinstance(target, Unsupported):
            raise ending(Step.WHERE, f"this hands runs from {prefix}, which is neither {tool}, the uv tool environment the update installs master into, nor the .venv of a checkout of hands, so a restart would not run what it updated: run `hands update` with the hands `uv tool install` put on PATH, or from a checkout's own .venv")
        if runs_from is not None and runs_from.resolve() != prefix.resolve():
            raise ending(Step.WHERE, f"the running hands runs from {runs_from}, not from {prefix}, the environment the update brings current, so the update would leave it running what it runs now: run hands from {prefix}")
        match target:
            case Tool():
                step(Step.FETCH, ("git", "clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", BRANCH, REPOSITORY, str(clone)), work)
                sha = commit(clone, BRANCH)
                step(Step.CONSTRAINTS, ("uv", "export", "--quiet", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--format", "requirements.txt", "-o", str(constraints)), clone)
                # The interpreter itself, not its version: uv keeps an environment in place only on the one it was made from
                # (measured, uv 0.12.15: `--python 3.12` found another 3.12 and made the environment anew).
                step(Step.INSTALL, ("uv", "tool", "install", "--reinstall-package", TOOL, "--python", made_from(), "--constraints", str(constraints), f"{TOOL} @ git+{REPOSITORY}@{sha}"), work)
                installed = (target, sha)
            case Checkout(root=root):
                step(Step.PULL, ("git", "pull", "--quiet", "--ff-only"), root)
                sha = commit(root, "the checkout")
                # The checkout's source is hands' own, editable in its .venv: the pull is what a start runs from now, so a
                # sync that fails after it leaves the checkout's commit in place beside an environment not brought to it.
                installed = (target, sha)
                step(Step.SYNC, ("uv", "sync", "--locked", "--reinstall-package", TOOL), root)
        # Run by the interpreter the install just filled, so each is the target's own command.
        advisable(Step.INSTALL_FRITTER, invocation(home, "install-fritter"), work, _exits(Installing.UNFOUND))
        step(Step.INSTALL_BRAIN, invocation(home, "install-brain"), work)
        # [LAW:one-source-of-truth] hands check's own command, environment, and reading of the list: the plugin is
        # installed for the user or it is not; no claude at all is no plugin.
        under = readiness.plugin_environment(home, caller)
        match advisable(Step.PLUGIN_LIST, readiness.PLUGIN_LIST, work, _unstarted, under):
            case Advised():
                held = False
            case str() as listing:
                try:
                    held = bool(readiness.installed_everywhere(listing))
                except Rejected as error:
                    raise ending(Step.PLUGIN_LIST, f"`claude plugin list --json` printed what hands cannot read: {error}") from None
        if held:
            step(Step.MARKETPLACE, ("claude", "plugin", "marketplace", "update", MARKETPLACE), work, env=under)
            step(Step.PLUGIN, ("claude", "plugin", "update", PLUGIN_ID, "--scope", "user"), work, env=under)
        advisable(Step.RENDERED_PLUGINS, ("rm", "-rf", str(home.root / RENDERED)), work, _any_exit)
    except _Stopped as stopped:
        return stopped.stopped
    return Installed(target, sha, tuple(ran), tuple(advised))


def made_from() -> str:
    """The interpreter this hands' environment was made from: Python's own name for it, which typeshed does not declare."""
    return cast(str, getattr(sys, "_base_executable"))


def run(children: Children, argv: Sequence[str], cwd: Path, env: Mapping[str, str], limit: float) -> Outcome:
    """Run `argv` in `cwd`, with nothing to read and no terminal to ask on, inside `limit` seconds; how it ended."""
    # git asks for a login on the terminal when a fetch is refused; the brain's has nobody at it.
    return children.run_blocking(*argv, limit=limit, cwd=cwd, env={**env, "GIT_TERMINAL_PROMPT": "0"})
