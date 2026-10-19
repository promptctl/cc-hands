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
from importlib import resources
from pathlib import Path

from aiohttp import web

from hands.sessions.audit import Record
from hands.sessions.backlog import Above, Backlog, Below, LitFailed, Move, Place, Untracked, rank
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, count, fail, unit
from hands.voice.phoneaddress import carries_key
from hands.voice.sentences import Sighted, SummaryStore, sighted

UNKEYED = "this page's address is missing the phone's key; open it from the address `hands phone` prints"


def parse_project(query: str | None) -> Path:
    """The project a page names on its address; raises Rejected naming what is wrong with it."""
    match query:
        case str() if Path(query).is_absolute():
            return Path(query)
        case _:
            raise Rejected("project is the absolute path of a directory lit has a workspace in")


def parse_asked(body: bytes) -> tuple[Path, str, Place]:
    """The move a page sent, as its request's body: the project, the ticket, and where it was dropped; raises Rejected
    naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        raise Rejected(f"a move is JSON: {error}") from error
    match sent:
        case {"above": object(), "below": object()}:
            pass
        case {"project": str(project), "ticket": str(ticket), "above": str(target)}:
            return parse_project(project), ticket, Above(target)
        case {"project": str(project), "ticket": str(ticket), "below": str(target)}:
            return parse_project(project), ticket, Below(target)
        case _:
            pass
    raise Rejected("a move is {project, ticket, and one of above or below: the ticket it is dropped next to}")


def shown(read: Sighted | Untracked) -> dict[str, object]:
    """The backlog as the page draws it: every unfinished root in rank order, each over its unfinished children, each
    with its sentence where one is said, and the frame lit ranks it in, which a drop stays inside."""
    match read:
        case Untracked():
            return {"tracked": False}
        case Sighted(backlog=backlog, said=said):

            def node(id: str) -> dict[str, object]:
                ticket = backlog.tickets[id]
                return {
                    "id": id,
                    "title": ticket.title,
                    "kind": ticket.kind,
                    "status": ticket.status,
                    "frame": backlog.inside.get(id),
                    # hands' sentence for the ticket, and never its description: one not said yet is not here.
                    **({"summary": said[id]} if id in said else {}),
                    "children": [node(child) for child in backlog.open_children(id)],
                }

            return {"tracked": True, "items": [node(id) for id in backlog.roots()]}


def backlog_routes(store: SummaryStore, key: str, record: Record) -> list[web.RouteDef]:
    """The page, the read of a project's backlog, and a drop that ranks one ticket next to another.

    [LAW:nothing-unseen] each read is one `backlog.read` event, how many open tickets it showed and how many of them had a
    sentence; each drop one `backlog.ranked`, which ticket was put where and whether lit has it there after; each refused
    or failed says why.
    """
    page = resources.files("hands.voice").joinpath("backlog.html").read_text()

    async def drawn(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def read(request: web.Request) -> web.Response:
        with unit("backlog.read", record, counts=("tickets", "summarised")):
            if not carries_key(request.headers.get("Authorization", ""), key):
                fail("read without the phone's key")
                return web.Response(status=401, text=UNKEYED)
            try:
                project = parse_project(request.query.get("project"))
            except Rejected as error:
                fail(str(error))
                return web.Response(status=400, text=str(error))
            annotate(project=project)
            try:
                return web.json_response(_answered(await sighted(store, project)))
            except (LitFailed, Rejected) as error:
                fail(f"cannot read the backlog: {error}")
                return web.Response(status=502, text=f"the backlog in {project} could not be read: {error}")

    async def ranked(request: web.Request) -> web.Response:
        with unit("backlog.ranked", record):
            if not carries_key(request.headers.get("Authorization", ""), key):
                fail("ranked without the phone's key")
                return web.Response(status=401, text=UNKEYED)
            try:
                project, ticket, place = parse_asked(await request.read())
            except Rejected as error:
                fail(str(error))
                return web.Response(status=400, text=str(error))
            annotate(project=project, ticket=ticket, place=place)
            try:
                before = await sighted(store, project)
            except (LitFailed, Rejected) as error:
                fail(f"cannot read the backlog: {error}")
                return web.Response(status=502, text=f"the backlog in {project} could not be read: {error}")
            match before:
                case Untracked():
                    fail("lit has no workspace there")
                    return web.Response(status=404, text=f"lit has no workspace in {project}, so it has no backlog to rank")
                case Sighted(backlog=backlog):
                    pass
            try:
                move = backlog.move(ticket, place)
            except Rejected as error:
                fail(str(error))
                return web.Response(status=400, text=str(error))
            try:
                await rank(project, move)
                after = await sighted(store, project)
            except (LitFailed, Rejected) as error:
                fail(str(error))
                return web.Response(status=502, text=str(error))
            # [LAW:no-silent-failure] the read after is the evidence: lit ranks the parent instead of a ticket another
            # writer moved out of its frame meanwhile, and exits 0 all the same.
            if not _landed(after, move):
                why = f"lit did not put {ticket} {_where(place)}: the backlog changed while it was moved, so lit ranked something else"
                fail(why)
                return web.Response(status=409, text=why)
            return web.json_response(shown(after))

    return [web.get("/backlog", drawn), web.get("/backlog/tickets", read), web.post("/backlog/rank", ranked)]


def _answered(read: Sighted | Untracked) -> dict[str, object]:
    """The read as the page is given it, with whether lit tracks the project, and how many tickets are drawn and said, on the event."""
    match read:
        case Untracked():
            annotate(tracked=False)
        case Sighted(backlog=backlog, said=said):
            annotate(tracked=True)
            drawn = [id for root in backlog.roots() for id in _under(backlog, root)]
            count(tickets=len(drawn), summarised=sum(id in said for id in drawn))
    return shown(read)


def _under(backlog: Backlog, id: str) -> list[str]:
    """`id` and every unfinished ticket under it, as the page draws them."""
    return [id, *(under for child in backlog.open_children(id) for under in _under(backlog, child))]


def _landed(after: Sighted | Untracked, move: Move) -> bool:
    match after:
        case Untracked():
            return False
        case Sighted(backlog=backlog):
            return backlog.landed(move)


def _where(place: Place) -> str:
    match place:
        case Above(target=target):
            return f"above {target}"
        case Below(target=target):
            return f"below {target}"
