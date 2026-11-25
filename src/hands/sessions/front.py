"""Reads what is in front on the Mac's screen: the screen for the app that is, and its front tab where it says one; tmux
for the pane each of its clients shows; and the processes for the terminals each session runs under.

hands is tied to no one terminal. An app that can say which of its tabs is in front is asked; any other shows every
terminal running under it, which names the session in front whenever it holds only one.
"""

import asyncio
from collections.abc import Iterator, Mapping, Sequence

from loguru import logger

from hands.core.front import Candidate, FrontUnread, InFront, Shown, in_front
from hands.core.session import SessionId
from hands.core.tmux import Unanswered
from hands.sessions import panes
from hands.sessions.processes import Process, Processes, ancestor_terminals
from hands.sessions.screen import FrontApp, Screen
from hands.sessions.terminals import Terminals


async def read_front(screen: Screen, terminals: Terminals, processes: Processes, sessions: Mapping[SessionId, tuple[int, str]], environment: Mapping[str, str]) -> InFront:
    """What is in front now, among `sessions`, each a session's pid and its name as it is spoken; `environment` is the
    run's, which says where tmux keeps its sockets."""
    try:
        app = await screen.front()
        if isinstance(app, FrontUnread):
            return app
        table = await asyncio.to_thread(processes.table)
        shown = _shown(app, table)
        # A tmux client is on screen only in a terminal the app in front shows.
        clients = await _clients(terminals, processes, environment, [pid for pid, _ in sessions.values()], table) if shown.shown else {}
    except Exception as error:
        # [LAW:no-silent-failure] a read broken by a fault of its own is logged with where, and the turn's record says
        # why it was left out: the screen is a fact the turn can go without, never one that ends the brain's turns.
        logger.opt(exception=error).error("reading what is in front broke")
        return FrontUnread(f"{type(error).__name__}: {error}")
    match clients:
        case Unanswered(reason=reason):
            return FrontUnread(reason)
        case _:
            candidates = [Candidate(session, name, frozenset(ancestor_terminals(pid, table))) for session, (pid, name) in sessions.items()]
            return in_front(shown, clients, candidates)


def _shown(app: FrontApp, table: Mapping[int, Process]) -> Shown:
    match app.tab:
        case int(tab):
            return Shown(app.name, frozenset({tab}))
        case None:
            return Shown(app.name, frozenset(process.terminal.device for process in _under(app.pid, table) if process.terminal is not None))


def _under(pid: int, table: Mapping[int, Process]) -> Iterator[Process]:
    """Every process descended from pid."""
    children: dict[int, list[Process]] = {}
    for process in table.values():
        children.setdefault(process.parent, []).append(process)
    stack = list(children.get(pid, ()))
    while stack:
        process = stack.pop()
        yield process
        stack.extend(children.get(process.pid, ()))


async def _clients(terminals: Terminals, processes: Processes, environment: Mapping[str, str], pids: Sequence[int], table: Mapping[int, Process]) -> dict[int, int] | Unanswered:
    """Each tmux client's terminal, and the terminal of the pane it shows, from each server that may hold a pane of one
    of `pids`; or why one of them could not be read, which might have shown the session in front."""
    answers = await panes.across(terminals, processes, environment, pids, table, terminals.clients)
    if unanswered := [answer.reason for answer in answers if isinstance(answer, Unanswered)]:
        return Unanswered("; ".join(unanswered))
    return {client: pane for answer in answers if not isinstance(answer, Unanswered) for client, pane in answer.items()}
