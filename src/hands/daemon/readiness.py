"""Whether hands is set up to work here: each step of the README's install, named, and done or missing.

    hands check     # one line a step, and the command that fixes one missing; exits 0 only when every step is done
    hands check -v  # each step said in full beneath its line

The steps, in the README's order: Claude Code from its installer; PortAudio, which the microphone opens through; the
installed `hands` on PATH, whose `hands-shim` Claude Code runs for the plugin; the claude shim that runs sessions under fritter;
the plugin that joins sessions to hands; the brain's own Claude Code, the release hands pins; the brain with its login; the Input
Monitoring grant that lets hands hear the talk key; hands running; and the running sessions themselves. `hands run` logs each step in full as it
starts, and a daemon that is up says nothing about any of them, so this is where a missing one is heard.
"""

import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import get_args

from hands.brain import configdir, pinned
from hands.brain.configdir import Unstartable
from hands.core.effects import Fritter
from hands.core.stopfailure import LOGIN_EXPIRY, LOGIN_RENEWAL
from hands.core.reach import Unwrapped, through, writer
from hands.core.tmux import Behind, Keyboard, NotInTmux, Pane, PaneUnread
from hands.core.session import Membership, Runnable
from hands.daemon.backend import backend as resolve
from hands.daemon.config import Claude, load
from hands.daemon.preflight import Refused, Startable, beside, preflight
from hands.sessions import child, heartbeat, liveness, wrapper
from hands.sessions.child import Children
from hands.sessions.hookconfig import PLUGIN_ID
from hands.sessions.files import Unread
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import Processes, TerminalProcess, TerminalProcesses, Undescribed, terminal_processes
from hands.voice import backends


# Each finding carries a `step`, the few words its checklist line names it by, beside its `detail`, what is said of it in
# full: the line is read at a glance, and the full saying is `hands check -v`'s and `hands run`'s log's.
@dataclass(frozen=True)
class Ready:
    step: str
    detail: str

    @property
    def said(self) -> str:
        return self.detail


@dataclass(frozen=True)
class Missing:
    step: str
    detail: str  # what is missing, and what that stops
    fix: str  # the command, or the one act, that puts it there

    @property
    def said(self) -> str:
        # The fix follows the detail's first line, ahead of any list that line heads, as the running sessions' does.
        head, sep, rest = self.detail.partition("\n")
        return f"{head}: {self.fix}{sep}{rest}"


@dataclass(frozen=True)
class Unknown:
    step: str
    detail: str  # why the piece could not be looked at

    @property
    def said(self) -> str:
        return self.detail


Finding = Ready | Missing | Unknown

# The tmux pane whose keys reach each of a list of processes, as `hands.sessions.panes.keyboards` reads it.
Keyboards = Callable[[Sequence[int]], list[Keyboard]]

# `claude plugin list` answers in a quarter of a second; one that has not answered in this long is not going to.
LIST_TIMEOUT_SECONDS = 20.0
# `hands --version` imports hands' CLI, a couple of seconds; one that has not answered in this long is not going to.
VERSION_TIMEOUT_SECONDS = 30.0
INSTALL_CLAUDE = "`curl -fsSL https://claude.ai/install.sh | bash`"


def check(children: Children, processes: Processes, home: Home, path: str, granted: bool, reached: Finding, running: Finding, keyboards: Keyboards) -> list[Finding]:
    """Every step, in the README's order. `path` is the PATH sessions are started from; `granted`, this terminal's grant;
    `reached`, whether the brain has its login; `running`, whether hands is up; `keyboards`, what
    reads the tmux pane in front of each running session."""
    # [LAW:dataflow-not-control-flow] every step is looked at every time: one that is missing hides none after it.
    return [claude(path), portaudio(), tmux(path), installed(children, path), shim(home, path), plugin(children, home, path), reached, grant(granted), running, sessions(processes, home, path, keyboards)]


def claude(path: str) -> Finding:
    """Whether this PATH has a Claude Code of its own, apart from any hands shim, installed as its installer puts it."""
    match claude_code(wrapper.real_claude(path)):
        case Uninstalled() as uninstalled:
            return Missing(uninstalled.step, f"no Claude Code that hands can join its sessions of: {uninstalled.detail}", uninstalled.fix)
        case executable:
            # [LAW:one-source-of-truth] its version is the part of its path that the sessions' match blanks as one. A path
            # that names none is said in full beneath the line.
            blanked = (part for part, kept in zip(reversed(executable.parts), reversed(_unversioned(executable)), strict=True) if kept is None)
            named = next((f"Claude Code {part}" for part in blanked), "Claude Code")
            return Ready(named, f"Claude Code is installed: {executable}")


def portaudio() -> Finding:
    """Whether PyAudio, which hands opens the microphone through, can load Homebrew's PortAudio."""
    try:
        import pyaudio
    except ImportError as error:
        return Missing("PortAudio missing", f"PyAudio cannot load PortAudio ({error}), so hands cannot open the microphone", "`brew install portaudio`, with Homebrew from https://brew.sh")
    return Ready("PortAudio", f"PortAudio is there for the microphone: {pyaudio.get_portaudio_version_text()}")


def tmux(path: str) -> Finding:
    """Whether this PATH has the tmux the brain runs in."""
    found = shutil.which("tmux", path=path)
    if found is None:
        return Missing("tmux missing", "this PATH has no tmux, so the brain cannot start: it runs in a tmux session of its own", "`brew install tmux`")
    return Ready("tmux", f"tmux is there for the brain to run in: {found}")


def installed(children: Children, path: str) -> Finding:
    """Whether `hands` on this PATH, whose `hands-shim` and `hands-attention` Claude Code runs for the plugin's hooks and skills, is this hands."""
    this = f"hands {version('hands')}"
    found = shutil.which("hands", path=path)
    if found is None:
        return Missing(
            "`hands` not on PATH",
            "this PATH has no `hands`, so Claude Code cannot run `hands-shim` and no session gets hands' hooks",
            "`uv tool update-shell` puts the directory `uv tool install` writes it to on PATH",
        )
    match _text(children.run_blocking(found, "--version", limit=VERSION_TIMEOUT_SECONDS)):
        case str() as said:
            pass
        case Unasked(why=why):
            return Unknown("`hands` on PATH", f"cannot ask {found} which hands it is: `{found} --version` {why}")
    # [LAW:one-source-of-truth] the plugin's hooks are whichever hands Claude Code finds, so one that is not this hands
    # is said here, where this hands' own lines would otherwise vouch for it.
    if said.strip() != this:
        return Missing(
            "another `hands` first on PATH",
            f"`hands` on this PATH, {found}, is {said.strip()}, not this {this}, so sessions run its hooks",
            "put this one first on PATH, or install it again",
        )
    # [LAW:one-source-of-truth] the plugin's hooks name `hands-shim` alone: hands installs it beside `hands`, and a hands
    # from before the plugin ran it has no such script. One found in another directory is another install's, which
    # Claude Code would run in its place.
    shim = shutil.which("hands-shim", path=path)
    if shim is None or Path(shim).parent != Path(found).parent:
        return Missing(
            "`hands-shim` not beside `hands` on PATH",
            f"`hands` on this PATH, {found}, has no `hands-shim` beside it, so Claude Code cannot run the plugin's hooks and no session gets them" + ("" if shim is None else f" (the one on PATH is {shim}, another install's)"),
            "install this hands again: `hands update`",
        )
    return Ready(f"{this} on PATH", f"`hands` on this PATH is {found}, {this}: Claude Code runs its `hands-shim` for the plugin in every session")


def configured(children: Children, home: Home, pin: pinned.Pin, machine: configdir.Machine, environment: Mapping[str, str], running: heartbeat.Verdict) -> Finding:
    """Whether hands would start on its settings and its home's files, its brain on `pin`, and the brain has the login it
    reaches its model with: as the brain of `running`, hands' heartbeat now, last said of it, where it said it expired."""
    try:
        config = load(home).config
    except Rejected as error:
        # The refusal names the file and what is wrong with it, so it is the detail, beside every other refusal of the
        # start, each naming its own fix, and the line names the file to mend.
        return Missing("settings unreadable", f"hands cannot read its settings: {'; '.join(beside(preflight(home, pin, machine, environment), error))}", f"correct {home.config}")
    # [LAW:single-enforcer] the start's own preflight, so what a start would refuse is said here in its words, every
    # refusal at once, and the login beside them wherever the brain's own config directory runs one.
    checked = preflight(home, pin, machine, environment)
    match checked.verdict:
        case Startable(brain=brain):
            refusals: tuple[str, ...] = ()
        case Refused(refusals=refusals, brain=brain):
            pass
    if brain is None:
        # The run that asks would load what the brain's own settings keep out.
        return unstartable("; ".join(refusals))
    match _login(children, config.llm, home, checked.environment, brain, running):
        case reached if not refusals:
            return reached
        case Missing(fix=fix):
            return unstartable("; ".join((*refusals, fix)))
        case Ready():
            return unstartable("; ".join(refusals))


def _login(children: Children, llm: Claude, home: Home, environment: Mapping[str, str], brain: Runnable, running: heartbeat.Verdict) -> Ready | Missing:
    """The brain's login: expired, where the brain of hands running now last said so, which no Claude Code is started to learn;
    else as its Claude Code says it. Its `claude auth status` says only that it holds a token, not that the token works,
    and a brain that has said nothing of its login has shown nothing of it."""
    if any(each.kind == heartbeat.LOGIN_EXPIRED.kind for each in heartbeat.last_said(running)):
        return Missing(heartbeat.LOGIN_EXPIRED.brief, LOGIN_EXPIRY, LOGIN_RENEWAL)
    return _reached(children, llm, home, environment, brain)


def _reached(children: Children, llm: Claude, home: Home, environment: Mapping[str, str], brain: Runnable) -> Ready | Missing:
    try:
        reached: Ready | Missing = reaching(resolve(children, llm, home, environment, brain))
    except Rejected as error:
        # The refusal is the line's fix: where there is a mend, `hands login` most often, the refusal is what names it,
        # and the few that name none (a provider other than Anthropic's, a status hands cannot read) say what was
        # refused, which is the most there is to act on.
        reached = Missing("brain cannot reach its model", "hands has no model to talk with", str(error))
    except Unstartable as error:
        # The brain's claude could not be asked, which says nothing of its login and is a start refused all the same.
        reached = unstartable(str(error))
    return reached


def reaching(reached: backends.ClaudeCodeBackend) -> Ready:
    """What a brain that has its login reaches, and the Claude Code it runs."""
    return Ready(f"brain logged in as {reached.account}", f"the brain is logged in as {reached.account}, and reaches {reached.model} on Claude Code {reached.brain.claude.version}, hands' own copy: {reached.brain.claude.path}")


def unstartable(reason: str) -> Missing:
    """What a start refuses: `reason` names every refusal and its fix, so the line claims none of them."""
    return Missing("hands cannot start here", "hands cannot start here", reason)


def daemon(verdict: heartbeat.Verdict, now: datetime) -> Finding:
    """Whether hands is running, from what its heartbeat says of it `now`."""
    said = heartbeat.describe(verdict, now)
    start = "`hands run`, in a terminal that has the Input Monitoring grant"
    match verdict:
        case heartbeat.Up():
            return Ready("hands running", said)
        case heartbeat.Unreadable():
            return Unknown("hands running", said)
        case heartbeat.NeverRan():
            return Missing("hands not running", said, start)
        case heartbeat.Down():
            return Missing("hands is down", said, start)
        case heartbeat.Unresponsive():
            return Missing("hands not responding", said, start)
        case heartbeat.Stopped():
            return Missing("hands stopped", said, start)
        case heartbeat.Refused():
            return Missing("hands refused to start", said, start)


PLUGIN_LIST = ("claude", "plugin", "list", "--json")


def plugin_environment(home: Home, caller: Mapping[str, str]) -> dict[str, str]:
    """The environment of every `claude plugin` command hands runs: the caller's, under the caller's Claude Code config.

    The brain runs `hands update` and `hands check` from a shell whose CLAUDE_CONFIG_DIR is the brain's own, which holds
    none of the user's plugins. Not a session to the shim, since nothing here is a terminal, so the shim runs the real claude.
    """
    brain = "CLAUDE_CONFIG_DIR" in caller and config_dir(caller, Path.cwd()) == Path(os.path.realpath(home.brain))
    return {**{name: value for name, value in caller.items() if not (brain and name == "CLAUDE_CONFIG_DIR")}, **wrapper.EVERY_RUN}


def plugin(children: Children, home: Home, path: str) -> Finding:
    """Whether Claude Code, as `claude` on this PATH runs it, has hands' plugin installed and enabled."""
    # [LAW:one-source-of-truth] Claude Code is asked, never its files read: where it keeps its plugins is its own.
    match _text(children.run_blocking(*PLUGIN_LIST, env={**plugin_environment(home, os.environ), "PATH": path}, limit=LIST_TIMEOUT_SECONDS)):
        case str() as listed:
            return plugin_listed(listed)
        case Unasked(why=why):
            return Unknown("plugin", f"cannot ask `claude plugin list` whether the plugin {PLUGIN_ID} is installed: it {why}")


@dataclass(frozen=True)
class Unasked:
    """A command a step asks that gave no answer it can read: why."""

    why: str


def _text(asked: child.Outcome) -> str | Unasked:
    """[LAW:parse-dont-validate] what a command a step asks printed, as text, where it exited 0; or why there is none."""
    match asked:
        case child.Ran(ending=child.Exited(code=0), out=out):
            try:
                return out.data.decode()
            except UnicodeDecodeError as error:
                return Unasked(f"printed what is not text: {error}")
        case child.Ran() | child.NeverStarted() | child.Unreaped() | child.Lost():
            return Unasked(child.said(asked))


def installed_everywhere(raw: str) -> set[bool]:
    """Whether each user-scope install of hands' plugin in `claude plugin list --json`'s `raw` is enabled; empty when there is none."""
    # Only a user-scope install joins every session; a project or local one joins the sessions of one directory,
    # and is listed as enabled or not by the directory the listing was asked from.
    hands = [entry for entry in Payload.parse_list(raw.encode(), "each plugin") if entry.fields.get("id") == PLUGIN_ID]
    return {entry.flag("enabled") for entry in hands if entry.text("scope") == "user"}


def plugin_listed(raw: str) -> Finding:
    """What `claude plugin list --json` says of hands' plugin."""
    try:
        everywhere = installed_everywhere(raw)
    except Rejected as error:
        return Unknown("plugin", f"`claude plugin list --json` printed what hands cannot read: {error}")
    if True in everywhere:
        return Ready("plugin installed", f"the plugin {PLUGIN_ID} is installed and enabled for every session: a session started since, or reloaded with /reload-plugins, joins hands")
    if everywhere:
        return Missing("plugin disabled", f"the plugin {PLUGIN_ID} is disabled, so no session joins hands", f"`claude plugin enable --scope user {PLUGIN_ID}`, then /reload-plugins in each running session")
    return Missing(
        "plugin not installed globally",
        f"the plugin {PLUGIN_ID} is not installed for every session, so only the sessions of a project it is installed in join hands",
        f"`claude plugin install {PLUGIN_ID}` (after `claude plugin marketplace add brandon-fryslie/cc-hands`), then /reload-plugins in each running session",
    )


def shim(home: Home, path: str) -> Finding:
    """Whether `claude` on this PATH is the shim this hands writes, with the fritter this hands carries, so it starts
    every interactive session under fritter."""
    # Every fix below is `hands install-fritter`'s, which a hands built without its fritter cannot run: the rebuild is said instead.
    try:
        carried = wrapper.packaged()
    except wrapper.Unpackaged as error:
        return Missing("hands built without fritter", "hands cannot write the claude shim", str(error))
    found = shutil.which("claude", path=path)
    claude = None if found is None else Path(found)
    # [LAW:dataflow-not-control-flow] what to do is read off what is there: an installed shim wants only the PATH.
    first = f'export PATH="{home.bin}:$PATH"'
    fix = first if isinstance(wrapper.shim_of(home.shim), wrapper.Shim) else f"`hands install-fritter`, then {first}"
    refritter = "`hands install-fritter`"
    match None if claude is None else wrapper.shim_of(claude):
        case None:
            return Missing("claude shim not first on PATH", f"`claude` on this PATH is {found or 'nothing'}, not hands' shim, so no session started from it can be typed into", fix)
        case Unread(reason=reason):
            return Unknown("claude shim", f"cannot tell whether `claude` on this PATH, {found}, is hands' shim: {reason}")
        case wrapper.Stale():
            return Missing("claude shim out of date", f"`claude` on this PATH is hands' shim, {found}, but not the one this hands writes", refritter)
        case wrapper.Shim(fritter=runs) if not os.access(runs, os.X_OK):
            return Missing(
                "claude shim's fritter missing",
                f"`claude` on this PATH is hands' shim, {found}, but its fritter {runs} is not there to run, so every interactive claude fails to start",
                refritter,
            )
        case wrapper.Shim(fritter=runs):
            # [LAW:one-source-of-truth] the fritter in a home's bin is a copy of the packaged one, so a copy that has
            # drifted from it, as one does when hands is upgraded or a checkout's fritter rebuilt, is said, never trusted.
            try:
                current = wrapper.carried(runs)
            except OSError as error:
                return Unknown("claude shim", f"cannot tell whether {runs} is the fritter this hands carries, {carried}: {error}")
            if not current:
                return Missing("claude shim's fritter out of date", f"`claude` on this PATH is hands' shim, {found}, but its fritter {runs} is not the one this hands carries, {carried}", refritter)
            return Ready("claude shim first on PATH", f"`claude` on this PATH is hands' shim, {found}: every interactive session started from it can be typed into")


def grant(granted: bool) -> Ready | Missing:
    """Whether this terminal's app may show hands the keys typed in other apps."""
    if granted:
        return Ready("Input Monitoring granted", "this terminal has the Input Monitoring grant, so hands run here hears the talk key (Right Shift)")
    return Missing(
        "no Input Monitoring grant",
        "this terminal has no Input Monitoring grant, so hands run here cannot hear the talk key (Right Shift)",
        "grant it to the app this terminal runs in, in System Settings > Privacy & Security > Input Monitoring, and restart that app",
    )


def sessions(processes: Processes, home: Home, path: str, keyboards: Keyboards) -> Finding:
    """Whether every running session can be typed into, those hands knows of and those it does not, each by the fritter
    that wrapped it or the tmux pane in front of it, as `keyboards` reads them. Nothing is removed or dialled."""
    try:
        running, unreadable = liveness.live(processes, home)
        # A socket that is gone is a fritter that is gone. It is never dialled here: fritter writes what it cannot read
        # into the session's own terminal.
        listening = {member.fritter for member in running if member.fritter is not None and member.fritter.is_socket()}
    except OSError as error:
        return Unknown(SESSIONS, f"cannot look at the running sessions in {home.memberships}: {error}")
    found = unrecorded(processes, home, path, {member.pid for member in running})
    # One read of the panes, for every session the check looks at.
    joining = found.sessions if isinstance(found, Unrecorded) else []
    pids = [*(member.pid for member in running), *(session.process.pid for session in joining)]
    return sessions_found(running, listening, unreadable, found, dict(zip(pids, keyboards(pids), strict=True)))


SESSIONS = "running sessions"


@dataclass(frozen=True)
class Unfindable:
    said: str  # why a running session hands has no record of cannot be told from the other programs at a terminal


@dataclass(frozen=True)
class Uninstalled:
    """Why `claude` on PATH is no Claude Code whose sessions hands can tell, and what puts one in."""

    step: str
    detail: str
    fix: str


@dataclass(frozen=True)
class Unjoined:
    """A running session hands has no record of, and the fritter that wrapped it, if one did."""

    process: TerminalProcess
    fritter: TerminalProcess | None


@dataclass(frozen=True)
class Unrecorded:
    """The sessions at a terminal hands has no record of, how many runs of claude beside them are no session, by why, and
    the processes at a terminal the kernel would not describe, any of which may be one."""

    sessions: list[Unjoined]
    others: Counter[wrapper.NotASession]
    unread: list[Undescribed]


def unrecorded(processes: Processes, home: Home, path: str, members: Collection[int]) -> Unrecorded | Unfindable:
    """The sessions running at a terminal that hands has no record of, among this user's processes now."""
    match claude_code(wrapper.real_claude(path)):
        case Uninstalled(detail=detail, fix=fix):
            return Unfindable(f"{detail}; {fix}")
        case executable:
            try:
                config = config_dir(os.environ, Path.cwd())
            except OSError as error:
                return Unfindable(f"cannot tell which Claude Code config this check runs under: {error}")
            try:
                return unjoined(home, executable, config, terminal_processes(processes), members, processes.interactive)
            except OSError as error:
                return Unfindable(f"cannot look at this user's processes at a terminal: {error}")


def claude_code(claude: Path | None) -> Path | Uninstalled:
    """The executable a session of the real `claude` on PATH runs as."""
    install = f"Claude Code's installer puts it in: {INSTALL_CLAUDE}"
    if claude is None:
        return Uninstalled("Claude Code not installed", "this PATH has no `claude` of its own, apart from any hands shim", install)
    executable = claude.resolve()
    try:
        with executable.open("rb") as start:
            script = start.read(2) == b"#!"
    except OSError as error:
        return Uninstalled("Claude Code unreadable", f"cannot read the real `claude`, {executable}: {error}", install)
    if script:
        return Uninstalled(
            "Claude Code is a script install",
            f"the real `claude`, {executable}, is a script, so its sessions run as its interpreter",
            "`claude install` puts in the native one, whose sessions hands can tell",
        )
    return executable


def config_dir(environment: Mapping[str, str], cwd: Path) -> Path:
    """The Claude Code config a process with this environment, working in `cwd`, runs under: its plugins, and so whether
    hands' is one. It is the directory itself, by whatever path or link it was named, read as that process reads it."""
    named = environment.get("CLAUDE_CONFIG_DIR") or Path(environment.get("HOME") or Path.home()) / ".claude"
    # realpath, not Path.resolve: on Python 3.12 resolve raises RuntimeError on a link loop, where realpath stops.
    return Path(os.path.realpath(cwd / named))


def unjoined(
    home: Home, claude: Path, config: Path, terminals: TerminalProcesses, members: Collection[int], interactive: Callable[[TerminalProcess], bool | Undescribed | None]
) -> Unrecorded:
    """The sessions at a terminal under `config` that no running membership names: started before the plugin, and not
    reloaded since.

    A session is a process at a terminal running `claude`, the Claude Code executable, or any other version of it, since
    an update leaves running sessions on the version they started on, that the shim would have run as a session: a
    `claude -p`, a subcommand, or a claude piped into, is none. One whose parent runs claude is that run's own helper. The
    hook records that same process, so it is matched by pid. One under another config, as the brain is, has other
    plugins, and is no session of the plugin this check looks at; one under the same directory by another path, through
    a link, is, since `config` and each process's are both as `config_dir` reads them. `interactive` says whether a process
    reads and writes its terminal; it is asked only of a run of claude, the one process it matters for, and one that has
    exited since is none. A process the kernel would not describe may be a session, unless it is one hands knows or a
    run's own helper; and a run of claude under one is that one's to say, since whether it is a helper, or fritter's,
    cannot be told.
    """
    by_pid = {process.pid: process for process in terminals.found}
    # The file each runs, its links resolved as they are now, as `claude` and the fritter are: one deleted since is named
    # as it was. realpath, as `config_dir` reads a config.
    programs = {process.pid: Path(os.path.realpath(process.executable)) for process in terminals.found}
    install = _unversioned(claude)

    def runs_claude(pid: int) -> bool:
        return pid in programs and _unversioned(programs[pid]) == install

    fritter = home.fritter.resolve()
    unread = {each.pid for each in terminals.unread}
    runs = [
        process
        for process in terminals.found
        if runs_claude(process.pid)
        and not runs_claude(process.parent)
        and process.parent not in unread
        and config_dir(process.environment, process.cwd) == config
        and process.pid not in members
    ]
    told = [(process, interactive(process)) for process in runs]
    described = [(process, reads) for process, reads in told if isinstance(reads, bool)]
    # [LAW:one-source-of-truth] a session is what the shim would run as one, by the shim's own test.
    ran: list[tuple[TerminalProcess, wrapper.Run]] = [(process, wrapper.run(process.arguments, reads)) for process, reads in described]
    return Unrecorded(
        [Unjoined(process, wrapped if (wrapped := by_pid.get(process.parent)) and programs[wrapped.pid] == fritter else None) for process, why in ran if why == "session"],
        Counter(why for _, why in ran if why != "session"),
        [
            *(each for each in terminals.unread if each.pid not in members and not runs_claude(each.parent)),
            *(reads for _, reads in told if isinstance(reads, Undescribed)),
        ],
    )


# Claude Code keeps each version under a name that is the version: a file in the native installer's versions
# directory, a directory in a Homebrew cask's. A semantic version, with any pre-release and build.
_VERSION = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?")


def _unversioned(executable: Path) -> tuple[str | None, ...]:
    """The executable's path with the part that names its version blanked, its own name or its directory's: the same for
    every version of one install. A version further up the path is some other thing's, and part of the install."""
    parts = executable.parts
    return (*parts[:-2], *(None if _VERSION.fullmatch(part) else part for part in parts[-2:]))


def sessions_found(
    running: Sequence[Membership],
    listening: Collection[Path],
    unreadable: Sequence[liveness.Unreadable],
    unrecorded: Unrecorded | Unfindable,
    keyboards: Mapping[int, Keyboard],
) -> Finding:
    """What the running sessions are to hands, given which fritter sockets are there, which files did not parse, which
    sessions at a terminal hands has no record of, and the tmux pane in front of each session, by its pid."""
    match unrecorded:
        case Unfindable(said):
            unjoined, unread, beside = [], [Unknown(SESSIONS, f"a session hands has no record of cannot be found: {said}")], ""
        case Unrecorded(sessions, others, undescribed):
            # [LAW:nothing-unseen] a claude at a terminal set aside as no session is counted by why, so none goes unseen;
            # nor does a process the kernel would not describe, which may be a session.
            counts = ", ".join(f"{why} {others[why]}" for why in get_args(wrapper.NotASession))
            unjoined, beside = [_unjoined(session, keyboards[session.process.pid]) for session in sessions], f", runs of claude at a terminal that are none: {counts}"
            said = ", ".join(f"pid {each.pid} ({each.call}: {os.strerror(each.errno)})" for each in undescribed)
            unread = [Unknown(SESSIONS, f"the kernel would not describe these processes at a terminal, so whether any is a session hands has no record of is unknown: {said}")] if undescribed else []
    reached = [*(_typable(member, listening, keyboards[member.pid]) for member in running), *unjoined]
    files = [Missing(str(file.path), f"{file.path} names no session hands can read ({file.error})", "hands run removes it") for file in unreadable]
    lines = [*reached, *files, *unread]
    # [LAW:nothing-unseen] each session is said under what it is to hands, one that is typed into with the way it is.
    headed = ((Missing, "hands cannot reach these"), (Unknown, "whether hands can reach these is unknown"), (Ready, "these are typed into"))
    grouped = [(kind, heading, [line.said for line in lines if type(line) is kind]) for kind, heading in headed]
    said = f"running sessions hands knows of: {len(running)}{beside}" + "".join(f"\n  {heading}:" + "".join(f"\n    {line}" for line in group) for _, heading, group in grouped if group)
    # [LAW:no-silent-failure] the step is the worst any session is to hands: one that could not be looked for is never taken for none.
    # Its line counts sessions, so a file that names none, or a look for sessions that could not be made, is counted apart.
    unreached = [line for line in reached if isinstance(line, Missing)]
    if unreached or files:
        named = [*([f"{_counted(len(unreached), 'session')} unreachable"] if unreached else []), *([f"{_counted(len(files), 'membership file')} unreadable"] if files else [])]
        return Missing(", ".join(named), said, _for_each([*unreached, *files]))
    unsure = [line for line in reached if isinstance(line, Unknown)]
    if unsure or unread:
        named = [*([f"{_counted(len(unsure), 'session')} unknown"] if unsure else []), *(["unrecorded sessions unknown"] if unread else [])]
        return Unknown(", ".join(named), said)
    return Ready(f"{_counted(len(reached), 'running session')} reachable" if reached else "no running sessions", said)


def _for_each(unreached: Sequence[Missing]) -> str:
    """The fix of each missing session or file, said once however many share it; which wants which is the full saying's."""
    fixes = list(dict.fromkeys(line.fix for line in unreached))
    if len(unreached) == 1:
        return fixes[0]
    return f"for each, {fixes[0]}" if len(fixes) == 1 else f"for each, one of: {'; '.join(fixes)}"


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _unjoined(session: Unjoined, pane: Keyboard) -> Missing:
    step = f"{session.process.cwd} (pid {session.process.pid})"
    where = f"{step} is a session hands has no record of"
    # [LAW:single-enforcer] whether it can be typed into once it joins is decided as a joined session's writer is.
    match through(session.fritter, pane):
        case TerminalProcess() | Pane():
            return Missing(step, f"{where}, so it cannot be reached", "/reload-plugins in it")
        case Behind(pane=behind):
            return Missing(step, f"{where}, started outside fritter, {_behind(behind)}, so it cannot be typed into", f"{_FRONT}, then /reload-plugins in it")
        case NotInTmux():
            return Missing(step, f"{where}, started outside fritter and in no tmux pane, so it cannot be typed into", _RESTART)
        case PaneUnread(reason):
            return Missing(
                step,
                f"{where}, started outside fritter, and which tmux pane it runs in could not be read ({reason})",
                "/reload-plugins in it joins it, and whether it can be typed into then is unknown",
            )


_RESTART = "restart it from a PATH whose `claude` is hands' shim"
_FRONT = "bring it back to the front of its pane"


def _behind(pane: Pane) -> str:
    return f"and another program has the keyboard of the tmux pane it runs in, {pane.id} (window {pane.window} of {pane.session})"


def _typable(member: Membership, listening: Collection[Path], pane: Keyboard) -> Finding:
    where = f"{member.cwd} (pid {member.pid})"
    # [LAW:single-enforcer] a session's reach is judged by the writer that would type into it, and by no other test.
    match writer(member, pane):
        case Fritter(socket) if socket not in listening:
            return Missing(where, f"{where} has lost its fritter, whose socket {socket} is gone, so it cannot be typed into", _RESTART)
        case Fritter():
            return Ready(where, f"{where} is typed into through its fritter")
        case Pane(id=id):
            return Ready(where, f"{where} is typed into through its tmux pane {id}")
        case Unwrapped(pane=missing):
            match missing:
                case Behind(pane=behind):
                    return Missing(where, f"{where} was started outside fritter, {_behind(behind)}, so it cannot be typed into", f"{_FRONT}, or {_RESTART}")
                case NotInTmux():
                    return Missing(where, f"{where} was started outside fritter and runs in no tmux pane, so it cannot be typed into", _RESTART)
                case PaneUnread(reason):
                    return Unknown(where, f"{where} was started outside fritter, and which tmux pane it runs in could not be read, so whether it can be typed into is unknown: {reason}")
