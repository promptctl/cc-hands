"""Starting a Claude Code session for the user, as a person at a terminal starts one: the `start_session` tool.

The session runs `claude` in a tmux window of the tmux session named for its folder, made when there is none, so it has a
terminal the user can attach to from any terminal app, and hands can find and type into as it does any session's. It is
started by the daemon, in the environment hands was started in, which is the user's: never from the brain's shell, whose
environment hands made the brain's and holds none of the user's credentials. What a session the daemon itself was started
inside gave it is left out. The start returns once the session joined hands, naming it, or says why it did not.
"""

import json
import shlex
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.clock import Clock
from hands.core.session import Membership, SessionId
from hands.core.tmux import Unanswered
from hands.sessions import audit, wide
from hands.sessions.home import HOME_VARIABLE, Home
from hands.sessions.processes import Process, Processes, ancestors
from hands.sessions.terminals import Opened, Terminals
from hands.sessions.untap import untap_script, untapped

# What a session gives each process it runs, naming itself as their parent: Claude Code's, and fritter's
# address. A `claude` started under Claude Code's is a child of that session, not one of its own: it writes its turns
# into the parent's transcript, and its own is never made, so hands has nothing to read; and one started outside hands'
# shim under fritter's would join hands as that session's.
SESSION_GIVEN = (
    "FRITTER_SOCKET", "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_TMUX_TRUECOLOR",
)

# What a terminal outside any session is not handed: SESSION_GIVEN, and hands' own Hugging Face token, which the daemon's
# environment holds for the voice library's fetches. The brain types into and reads these windows, so the token stays out.
TERMINAL_WITHHELD = (*SESSION_GIVEN, "HF_TOKEN")

# The settings a started session is launched with. Claude Code's dialog for a project's new MCP servers holds the session
# before its first hook, so it never joins: hands trusts them as the dialog's "use this and all future MCP servers in this
# project" does, and one the user turned down stays down, since that is read first.
SETTINGS = json.dumps({"enableAllProjectMcpServers": True})

# How long a started session has to join hands: Claude Code's start and its first hook. A folder it has never been
# trusted in holds it at the trust dialog, which no hook outlasts.
JOIN_SECONDS = 30.0
# How often the sessions and the session's pane are looked at while it joins.
LOOK_SECONDS = 0.25


class NotStarted(Exception):
    """The session was not started, or did not join hands. The message says why, with what its pane showed when it had one."""


@dataclass(frozen=True)
class Started:
    session: SessionId
    # The tmux session the window is in, and the pane `claude` runs in, as tmux names them (`%12`).
    tmux_session: str
    pane: str


def as_from_a_terminal(environment: Mapping[str, str]) -> dict[str, str]:
    """`environment` as a terminal outside any session has it: no tap, and nothing else of a session it may be inside."""
    return {name: value for name, value in untapped(environment).items() if name not in TERMINAL_WITHHELD}


def as_from_a_terminal_script(claude: Path) -> str:
    """as_from_a_terminal, as sh run in the environment to clean, from the same names, ending by running `claude` with the
    script's arguments."""
    # [LAW:one-source-of-truth] the same names as as_from_a_terminal.
    return f'{untap_script()}unset {shlex.join(TERMINAL_WITHHELD)}\nexec {shlex.quote(str(claude))} "$@"\n'


def joined(members: Iterable[Membership], root: int, table: Mapping[int, Process]) -> Membership | None:
    """The session among `members` whose process is `root` or one it started, by the process `table`: a session already
    running when `root` was started is never one."""
    return next((member for member in members if descends(member.pid, root, table)), None)


def descends(pid: int, root: int, table: Mapping[int, Process]) -> bool:
    """Whether process `pid` is `root` or one of the processes it started, by each process's parent in `table`."""
    return any(process.pid == root for process in ancestors(pid, table))


async def start(home: Home, clock: Clock, terminals: Terminals, processes: Processes, record: audit.Record, folder: Path, model: str | None, environment: Mapping[str, str], members: Callable[[], Iterable[Membership]]) -> Started:
    """Start hands' `claude` in `folder` on `model`, or on its own default with none, and wait until it is among the
    `members` of hands."""
    # [LAW:nothing-unseen] a start is one unit of work: where, on what, in which tmux session and pane, which session
    # joined, and whether tmux's session was made for it.
    with wide.unit("session.start", record, wide.begun(clock)):
        wide.annotate(folder=folder, model=model, settings=SETTINGS)
        where = _folder(folder)
        terminal = as_from_a_terminal(environment)
        name = tmux_name(where)
        # [LAW:single-enforcer] the window runs `claude` as from a terminal outside any session whatever the tmux server
        # it opens in holds: one already running gives a window its own environment, which a server started inside a
        # session holds that session's in. Through sh, never the user's shell, whose startup files would set a PATH of
        # their own. hands' own `claude`, its shim, whatever comes first on the window's PATH: the session runs under
        # fritter, so hands can type into it.
        claude = ["/bin/sh", "-c", as_from_a_terminal_script(home.bin / "claude"), "claude", "--settings", SETTINGS, *(() if model is None else (f"--model={model}",))]
        # The terminal's environment: a tmux server this starts takes it as every window's, so it holds nothing of
        # hands'. The home the session reports to is given its window by name.
        match await terminals.open(name, where, ("-e", f"{HOME_VARIABLE}={home.root}", "--", *claude), terminal):
            case Unanswered(reason=reason):
                raise NotStarted(f"hands starts a session in a tmux window, and none was opened: {reason}")
            case Opened() as window:
                pass
        wide.annotate(tmux_session=name, made_tmux_session=window.made, pane=window.pane, tmux_socket=window.socket)
        member = await _joined(clock, terminals, processes, members, where, window)
        wide.annotate(session=member.id, under_fritter=member.fritter is not None)
        if member.fritter is None:
            raise NotStarted(f"session {member.id} joined hands in tmux pane {window.pane}, but not under fritter, so hands cannot type into it: {home.bin / 'claude'} did not wrap it")
        return Started(member.id, name, window.pane)


def _folder(folder: Path) -> Path:
    # [LAW:parse-dont-validate] the daemon's own working directory is no folder the user named.
    if not folder.expanduser().is_absolute():
        raise NotStarted(f"{folder} is no absolute folder, nor one from ~: give the folder's whole path")
    try:
        resolved = folder.expanduser().resolve(strict=True)
    # A symlink loop is a RuntimeError before Python 3.13.
    except (OSError, RuntimeError) as error:
        raise NotStarted(f"there is no folder {folder}: {error}") from error
    if not resolved.is_dir():
        raise NotStarted(f"{folder} is not a folder")
    return resolved


def tmux_name(folder: Path) -> str:
    """The tmux session a session in `folder` is started in: named for the folder, as tmux would spell it (tmux turns a
    '.' or ':', which its targets read as separators, into '_')."""
    return folder.name.replace(".", "_").replace(":", "_")


async def _joined(clock: Clock, terminals: Terminals, processes: Processes, members: Callable[[], Iterable[Membership]], folder: Path, window: Opened) -> Membership:
    pane = window.pane
    deadline = clock.now() + JOIN_SECONDS
    while (member := joined(members(), window.pid, _table(processes))) is None:
        # Asked of the server that opened the pane, the one its id names a pane of.
        match await terminals.state(window.socket, pane):
            case Unanswered(reason=reason):
                raise NotStarted(f"whether `claude` in {folder} still runs in tmux pane {pane} could not be read: {reason}")
            case "closed":
                raise NotStarted(f"`claude` in {folder} ended before it joined hands, and its window closed with it")
            case "dead":
                raise NotStarted(f"`claude` in {folder} ended before it joined hands; tmux pane {pane} showed:\n{_read(await terminals.drawn(window.socket, pane))}")
            case "running":
                pass
        if clock.now() > deadline:
            # [LAW:one-source-of-truth] the one capture read_screen reads a session's screen by.
            raise NotStarted(f"`claude` in {folder} has not joined hands in {JOIN_SECONDS:.0f} seconds, and is still running in tmux pane {pane}, which shows:\n{_read(await terminals.shown(window.socket, pane))}")
        await clock.sleep(LOOK_SECONDS)
    return member


def _read(screen: str | Unanswered) -> str:
    match screen:
        case str():
            return screen
        case Unanswered(reason=reason):
            return f"(tmux could not show it: {reason})"


def _table(processes: Processes) -> dict[int, Process]:
    try:
        return processes.table()
    except OSError as error:
        raise NotStarted(f"the processes could not be read to tell which session is the one started: {error}") from error
