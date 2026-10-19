"""The backlog page: every open ticket of a project's lit backlog, under the epic it is in, each with hands' sentence
for it, dragged to set the order the work is pulled in.

It is served beside the phone's page, on its port and its addresses (`hands.voice.phonepage`), so a browser opens it
from the address `hands phone` prints with `backlog?project=<directory>` on its path. The page is open to anyone who
can reach the port; what it reads and moves is not: each request carries the phone's key, which the page takes from its
address's fragment as the phone's page does.

[LAW:one-source-of-truth] the order is lit's: every read is `lit export` run fresh, a drop is one `lit rank`, and the
page is answered with the backlog as lit holds it after. Nothing here keeps an order of its own.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

from aiohttp import web

from hands.sessions.audit import Record
from hands.sessions.backlog import Backlog, LitFailed, Move, Place, Status, Untracked, rank, read_backlog
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, count, fail
from hands.voice.keyed import keyed
from hands.voice.sentences import Sighted, SummaryStore, sighted


def parse_project(query: str | None) -> Path:
    """The directory a page names as its project, resolved, so one directory has one name however it is spelled;
    raises Rejected naming what is wrong with it."""
    match query:
        case str() if Path(query).is_absolute():
            pass
        case _:
            raise Rejected("project is the absolute path of a directory lit has a workspace in")
    try:
        resolved = Path(query).resolve()
    except (ValueError, OSError) as error:
        # A path the system cannot even look up, such as one holding a NUL.
        raise Rejected(f"project {query!r} is no path: {error}") from error
    if not resolved.is_dir():
        raise Rejected(f"project {query} is no directory on this machine")
    return resolved


def parse_asked(body: bytes) -> tuple[Path, str, Place]:
    """The move a page sent, as its request's body: the project, the ticket, and where it was dropped; raises Rejected
    naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        raise Rejected(f"a move is JSON: {error}") from error
    match sent:
        case {"above": object(), "below": object()}:
            raise Rejected("a move is dropped above a ticket or below one, never both")
        case {"project": str(project), "ticket": str(ticket), "above": str(target)}:
            return parse_project(project), ticket, Place("above", target)
        case {"project": str(project), "ticket": str(ticket), "below": str(target)}:
            return parse_project(project), ticket, Place("below", target)
        case _:
            pass
    raise Rejected("a move is {project, ticket, and one of above or below: the ticket it is dropped next to}")


@dataclass(frozen=True)
class Node:
    """A ticket as the page draws it: its sentence where one is said, never its description; the frame lit ranks it in,
    which a drop stays inside; and the unfinished tickets under it."""

    id: str
    title: str
    status: Status | None
    frame: str | None
    summary: str | None
    children: tuple["Node", ...]


def drawn(read: Sighted) -> tuple[Node, ...]:
    """Every unfinished root in rank order, each over its unfinished children."""
    backlog, said = read.backlog, read.said

    def node(id: str) -> Node:
        ticket = backlog.tickets[id]
        return Node(id, ticket.title, ticket.status, backlog.inside.get(id), said.get(id), tuple(node(child) for child in backlog.open_children(id)))

    return tuple(node(id) for id in backlog.roots())


def every(nodes: tuple[Node, ...]) -> Iterator[Node]:
    """Each node drawn, at every depth, in the order it is drawn."""
    for node in nodes:
        yield node
        yield from every(node.children)


def shown(read: Sighted | Untracked) -> dict[str, object]:
    """The backlog as the page is given it, with whether lit tracks the project, and how many tickets it draws and how
    many of them are said, on the event of the unit open here."""
    match read:
        case Untracked():
            annotate(tracked=False)
            return {"tracked": False}
        case Sighted():
            nodes = drawn(read)
            annotate(tracked=True)
            # [LAW:one-source-of-truth] counted from the nodes the page is given, so the event cannot disagree with it.
            for node in every(nodes):
                count(tickets=1, summarised=int(node.summary is not None))
            return {"tracked": True, "items": [asdict(node) for node in nodes]}


def backlog_routes(store: SummaryStore, key: str, record: Record) -> list[web.RouteDef]:
    """The page, the read of a project's backlog, and a drop that ranks one ticket next to another.

    [LAW:nothing-unseen] each read is one `backlog.read` event, how many open tickets it showed and how many of them had a
    sentence; each drop one `backlog.ranked`, which ticket was put where, whether lit ranked anything and what it said,
    and what it showed after; each refused or failed says why.
    """
    page = resources.files("hands.voice").joinpath("backlog.html").read_text()

    async def page_served(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    @keyed("backlog.read", record, key, counts=("tickets", "summarised"))
    async def read(request: web.Request) -> web.Response:
        try:
            project = parse_project(request.query.get("project"))
        except Rejected as error:
            fail(str(error))
            return web.Response(status=400, text=str(error))
        annotate(project=project)
        try:
            return web.json_response(shown(await sighted(store, project)))
        except (LitFailed, Rejected) as error:
            fail(f"cannot read the backlog: {error}")
            return web.Response(status=502, text=f"the backlog in {project} could not be read: {error}")

    @keyed("backlog.ranked", record, key, counts=("tickets", "summarised"))
    async def ranked(request: web.Request) -> web.Response:
        try:
            project, ticket, place = parse_asked(await request.read())
        except Rejected as error:
            return _refused(400, f"not moved: {error}")
        # [LAW:nothing-unseen] whether lit ranked anything, "no", "yes", or "unknown", is on the event however the drop ends.
        annotate(project=project, ticket=ticket, place=place, ranked="no")
        try:
            before = await read_backlog(project)
        except (LitFailed, Rejected) as error:
            return _refused(502, f"not moved: the backlog in {project} could not be read: {error}")
        match before:
            case Untracked():
                return _refused(404, f"not moved: lit has no workspace in {project}, so it has no backlog to rank")
            case Backlog():
                pass
        try:
            move = before.move(ticket, place)
        except Rejected as error:
            return _refused(400, f"not moved: {error}")
        try:
            said = await rank(project, move)
        except LitFailed as error:
            if error.refused:
                return _refused(502, f"not moved: {error}")
            # [LAW:no-silent-failure] lit stopped without saying why, and may have written first: neither is claimed.
            annotate(ranked="unknown")
            return _refused(502, f"whether lit ranked {ticket} {place} is not known: {error}")
        annotate(ranked="yes", lit_said=said)
        try:
            after = await sighted(store, project)
        except (LitFailed, Rejected) as error:
            return _refused(502, f"lit ranked {ticket} as it was asked to go {place}, and the backlog could not be read again to show it: {error}")
        # [LAW:no-silent-failure] the read after is the evidence: lit ranks the parent instead of a ticket another
        # writer moved out of its frame meanwhile, and exits 0 all the same, saying what it ranked.
        if not _landed(after, move):
            return _refused(409, f"the backlog lit holds after the rank does not have {ticket} {place}: it changed while it was moved. lit said: {said}")
        return web.json_response(shown(after))

    return [web.get("/backlog", page_served), web.get("/backlog/tickets", read), web.post("/backlog/rank", ranked)]


def _refused(status: int, why: str) -> web.Response:
    """A drop's answer when it did not end as asked: its event fails saying why, and the page is told the same."""
    fail(why)
    return web.Response(status=status, text=why)


def _landed(after: Sighted | Untracked, move: Move) -> bool:
    match after:
        case Untracked():
            return False
        case Sighted(backlog=backlog):
            return backlog.landed(move)
