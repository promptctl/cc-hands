"""Which tmux pane each session runs in, and the pane whose keys reach it: read over the Mac's processes and every tmux
server that may hold the pane, once each time it is asked, never kept (`hands.core.tmux`)."""

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from loguru import logger

from hands.core.tmux import InPane, Keyboard, PaneUnread, Server, Unanswered, keyboard_of, pane_of
from hands.sessions.processes import Process, Processes, StartedAs, TableRefused, Undescribed, ancestor_terminals, ancestors, foreground_terminal, still_running, table_of
from hands.sessions.terminals import Terminals

# $TMUX as tmux sets it in a pane: its server's socket, the server's pid, and the session's index.
_TMUX = re.compile(r"(?P<socket>.+),(?P<server>\d+),\d+")


async def panes(terminals: Terminals, processes: Processes, pids: Sequence[int], environment: Mapping[str, str]) -> list[InPane]:
    """The tmux pane each of `pids` runs in, from one read of the processes and of each tmux server that may hold it;
    `environment` says where tmux keeps its sockets."""
    return await _read(terminals, processes, pids, environment, lambda pid, table, read: pane_of(ancestor_terminals(pid, table), read))


async def keyboards(terminals: Terminals, processes: Processes, pids: Sequence[int], environment: Mapping[str, str]) -> list[Keyboard]:
    """The tmux pane whose keys reach each of `pids`, from one read of the processes and of each tmux server that may
    hold it."""
    return await _read(terminals, processes, pids, environment, lambda pid, table, read: keyboard_of(foreground_terminal(pid, table), ancestor_terminals(pid, table), read))


async def _read[P](
    terminals: Terminals, processes: Processes, pids: Sequence[int], environment: Mapping[str, str], of: Callable[[int, Mapping[int, Process], list[Server]], P]
) -> list[P | PaneUnread]:
    try:
        table = await table_of(processes)
        if isinstance(table, TableRefused):
            return [PaneUnread(table.reason)] * len(pids)
        read = await across(terminals, processes, environment, pids, table, terminals.panes)
    except Exception as error:
        # [LAW:no-silent-failure] the pane is a fact its readers can go without: a read that broke is logged with
        # where, and each process says why its pane is missing.
        logger.opt(exception=error).error("reading which tmux pane each session runs in broke")
        return [PaneUnread(f"{type(error).__name__}: {error}")] * len(pids)
    return [of(pid, table, read) for pid in pids]


async def across[T](
    terminals: Terminals, processes: Processes, environment: Mapping[str, str], pids: Sequence[int], table: Mapping[int, Process], ask: Callable[[Path], Awaitable[T | Unanswered]]
) -> list[T | Unanswered]:
    """What `ask` answers of each tmux server that may hold a pane of one of `pids`, as `sockets` finds them in `table`,
    and why any could not be found."""
    at = await asyncio.to_thread(sockets, terminals, processes, environment, pids, table)
    unfound: list[T | Unanswered] = [socket for socket in at if isinstance(socket, Unanswered)]
    return [*unfound, *await asyncio.gather(*(ask(socket) for socket in at if isinstance(socket, Path)))]


def sockets(terminals: Terminals, processes: Processes, environment: Mapping[str, str], pids: Sequence[int], table: Mapping[int, Process]) -> list[Path | Unanswered]:
    """The socket of every tmux server that may hold a pane of one of `pids`: each in the directory `environment` names,
    and each named by $TMUX on a line of ancestors of `pids`, which is the only word of a server started with `-S`
    elsewhere, or under another $TMUX_TMPDIR; or why one process on such a line could not be read.

    [LAW:one-source-of-truth] tmux sets $TMUX to its socket by its real path (/tmp is /private/tmp), then its pid and the
    session's index, and Terminals lists a directory's sockets by theirs: one server found both ways is asked once.
    """
    # Only the user's own processes are read, and only their own can be a server of theirs.
    mine = {process.pid: process for process in table.values() if process.uid == processes.user}
    # A pane's $TMUX is on the processes at its terminal; the tmux server, and the app and launchd above it, are at none.
    lines = {process.pid: process for pid in pids for process in ancestors(pid, table) if process.pid in mine and process.terminal is not None}
    # tmux names its directory by its real uid, which hands, set-uid to no one, runs as its effective one.
    directory = socket_directory(environment, processes.user)
    return list(dict.fromkeys([*terminals.sockets(directory), *(socket for process in lines.values() for socket in _named(processes, process, mine))]))


def _named(processes: Processes, process: Process, mine: Mapping[int, Process]) -> list[Path | Unanswered]:
    """The socket $TMUX names in the environment `process` was started with, its server among the user's processes
    `mine`: none outside tmux, or once it has exited."""
    match processes.started_as(process):
        case StartedAs(environment={"TMUX": str(named)}) if tmux := _TMUX.fullmatch(named):
            return _socket(processes, process, Path(tmux["socket"]), mine.get(int(tmux["server"])))
        # [LAW:no-silent-failure] a server its environment would name may hold a pane: never read as no pane.
        case StartedAs(environment={"TMUX": str(named)}) if named:
            return [Unanswered(f"process {process.pid} was started with a $TMUX that names no tmux server: {named!r}")]
        case Undescribed() as refused:
            return [_unread(f"which tmux server process {process.pid} runs under", refused)]
        # Outside tmux, an empty $TMUX among the ways a program says so; or gone since it was listed.
        case StartedAs() | None:
            return []


def _socket(processes: Processes, process: Process, path: Path, server: Process | None) -> list[Path | Unanswered]:
    """Where the socket $TMUX names in `process`'s environment as `path` is: tmux keeps a relative -S path as it was
    given, which is relative to the directory its server, the user's process `server` under the pid $TMUX names, was
    started in and never leaves."""
    if path.is_absolute():
        return [path]
    if server is None:
        # A server that has exited holds no pane.
        return []
    try:
        starts = processes.starts([process.pid, server.pid])
    except OSError as error:
        return [Unanswered(f"when the tmux server at {path} started could not be read: {error}")]
    # The server was running when `process` was started under it: a process under its pid that started since took the
    # pid of a server that exited, and is no server of the pane's. A `process` gone since it was listed holds no pane.
    match starts.get(process.pid):
        case float(start) if still_running(server.pid, start, starts):
            return _relative(processes, server, path)
        case _:
            return []


def _relative(processes: Processes, server: Process, path: Path) -> list[Path | Unanswered]:
    """The socket at `path` relative to the directory the running tmux `server` was started in."""
    match processes.directory_of(server):
        case Path() as directory:
            return [directory / path]
        case Undescribed() as refused:
            return [_unread(f"where the tmux server at {path} runs", refused)]
        # A server that exited since it was listed holds no pane.
        case None:
            return []


def _unread(what: str, refused: Undescribed) -> Unanswered:
    return Unanswered(f"{what} could not be read: {refused.call} refused with errno {refused.errno}")


def socket_directory(environment: Mapping[str, str], uid: int) -> Path:
    """Where tmux puts the socket of a server the user of real uid `uid` starts, unless -S names one: in tmux-<uid> under
    $TMUX_TMPDIR, or /tmp when that is unset or empty, as tmux reads it."""
    return Path(environment.get("TMUX_TMPDIR") or "/tmp") / f"tmux-{uid}"
