"""The backlog page: every open ticket of a project's lit backlog, under the epic it is in, each with hands' sentence
for it, dragged to set the order the work is pulled in, and each put at a level: moved into its topic's high, normal,
or low epic.

It is served beside the phone's page, on its port and its addresses (`hands.voice.phonepage`), so a browser opens it
from the address `hands phone` prints with `backlog?project=<directory>` on its path. The page is open to anyone who
can reach the port; what it reads and moves is not: each request carries the phone's key, which the page takes from its
address's fragment as the phone's page does.

[LAW:one-source-of-truth] the order is lit's: every read is `lit export` run fresh, a drop is one `lit rank`, a level
one `lit parent set`, and the page is answered with the backlog as lit holds it after. Nothing here keeps an order of
its own, or which epic is which level.
"""

import asyncio
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

from aiohttp import web

from hands.sessions.audit import Record
from hands.sessions.backlog import LEVELS, Backlog, Level, LitFailed, Move, NewEpic, Place, Standing, Status, Untracked, adopt, make_epic, rank, read_backlog
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, count, fail
from hands.voice.keyed import Handler, keyed
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


# Each level by the name a page sends it under.
NAMED: Mapping[str, Level] = {level: level for level in LEVELS}


def parse_leveling(body: bytes) -> tuple[Path, str, Level]:
    """The level a page put a ticket at, as its request's body: the project, the ticket, and the level; raises Rejected
    naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        raise Rejected(f"a leveling is JSON: {error}") from error
    match sent:
        case {"project": str(project), "ticket": str(ticket), "level": str(named)} if (level := NAMED.get(named)) is not None:
            return parse_project(project), ticket, level
        case _:
            raise Rejected(f"a leveling is {{project, ticket, and level: one of {', '.join(LEVELS)}}}")


@dataclass(frozen=True)
class Node:
    """A ticket as the page draws it: its sentence where one is said, never its description; the frame lit ranks it in,
    which a drop stays inside; where it stands among its topic's levels, None where it has none; and the
    unfinished tickets under it."""

    id: str
    title: str
    status: Status | None
    frame: str | None
    summary: str | None
    standing: Standing | None
    children: tuple["Node", ...]


def drawn(read: Sighted) -> tuple[Node, ...]:
    """Every unfinished root in rank order, each over its unfinished children, and no epic with no open ticket beneath
    it: leveling empties epics, and one emptied is as finished on the page as one whose every ticket is closed."""
    backlog, said = read.backlog, read.said

    def node(id: str) -> Node:
        ticket = backlog.tickets[id]
        return Node(id, ticket.title, ticket.status, backlog.inside.get(id), said.get(id), backlog.standing(id), held(backlog.open_children(id)))

    def held(ids: tuple[str, ...]) -> tuple[Node, ...]:
        # A ticket is drawn; an epic, which lit gives no status, only over what is drawn beneath it.
        return tuple(each for each in map(node, ids) if each.status is not None or each.children)

    return held(backlog.roots())


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
            # [LAW:one-source-of-truth] the levels go with the backlog, so the page's buttons are the levels hands puts tickets at.
            return {"tracked": True, "levels": LEVELS, "items": [asdict(node) for node in nodes]}


def backlog_routes(store: SummaryStore, key: str, record: Record) -> list[web.RouteDef]:
    """The page, the read of a project's backlog, a drop that ranks one ticket next to another, and a level that moves a
    ticket into its topic's epic for it.

    [LAW:nothing-unseen] each read is one `backlog.read` event, how many open tickets it showed and how many of them had a
    sentence; each drop one `backlog.ranked`, which ticket was put where, whether lit ranked anything and what it said,
    and what it showed after; each level one `backlog.leveled`, which ticket at which level, the epic it went into,
    whether lit made that epic, ranked it, and moved the ticket, and what it showed after; each refused or failed says why.
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

    # [LAW:no-ambient-temporal-coupling] the one owner of this page's lit writes: a drop and a level each run alone, from
    # the read they are checked against to the read that shows them, so two pages pressing at once never both make one
    # level's epic, or rank against an order the other is changing.
    writing = asyncio.Lock()

    def alone(handler: Handler) -> Handler:
        async def handled(request: web.Request) -> web.StreamResponse:
            async with writing:
                return await handler(request)

        return handled

    @keyed("backlog.ranked", record, key, counts=("tickets", "summarised"))
    @alone
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
        if not _holds(after, lambda backlog: backlog.landed(move)):
            return _refused(409, f"the backlog lit holds after the rank does not have {ticket} {place}: it changed while it was moved. lit said: {said}")
        return web.json_response(shown(after))

    @keyed("backlog.leveled", record, key, counts=("tickets", "summarised"))
    @alone
    async def leveled(request: web.Request) -> web.Response:
        try:
            project, ticket, level = parse_leveling(await request.read())
        except Rejected as error:
            return _refused(400, f"not leveled: {error}")
        # [LAW:nothing-unseen] whether lit made an epic, ranked it, and moved the ticket, each "no", "yes", or "unknown",
        # are on the event however the leveling ends.
        annotate(project=project, ticket=ticket, level=level, created="no", ranked="no", moved="no")
        try:
            before = await read_backlog(project)
        except (LitFailed, Rejected) as error:
            return _refused(502, f"not leveled: the backlog in {project} could not be read: {error}")
        match before:
            case Untracked():
                return _refused(404, f"not leveled: lit has no workspace in {project}, so it has no backlog to level")
            case Backlog():
                pass
        try:
            leveling = before.leveling(ticket, level)
        except Rejected as error:
            return _refused(400, f"not leveled: {error}")
        # The ranks the read after must have: the new epic's among its topic's levels, where one is made.
        ranks: tuple[Move, ...] = ()
        match leveling.epic:
            case str(epic):
                annotate(epic=epic)
            case NewEpic(topic=topic, title=title, place=place) as new:
                try:
                    await make_epic(project, new)
                except LitFailed as error:
                    return _unwritten(error, "created", "not leveled", f"whether lit made the epic {title} is not known")
                annotate(created="yes")
                try:
                    made = await read_backlog(project)
                except (LitFailed, Rejected) as error:
                    return _refused(502, f"not leveled: lit made the epic {title}, and the backlog could not be read again to rank it: {error}")
                match made:
                    case Backlog() if (found := made.level_epic(topic, level)) is not None:
                        try:
                            move = made.move(found, place)
                        except Rejected as error:
                            return _refused(409, f"not leveled: lit made {found}, {title}, at the bottom of its frame, and it cannot be ranked {place}: {error}")
                    case _:
                        return _refused(502, f"not leveled: lit made the epic {title}, and the backlog read after it does not hold it")
                epic = move.ticket
                annotate(epic=epic)
                try:
                    await rank(project, move)
                except LitFailed as error:
                    return _unwritten(
                        error,
                        "ranked",
                        f"not leveled: lit made {epic}, {title}, and left it at the bottom of its frame, so {ticket} was not moved",
                        f"not leveled: lit made {epic}, {title}, and whether it ranked it {place} or left it at the bottom of its frame is not known, so {ticket} was not moved",
                    )
                annotate(ranked="yes")
                ranks = (move,)
        try:
            await adopt(project, ticket, epic)
        except LitFailed as error:
            return _unwritten(error, "moved", "not leveled", f"whether lit moved {ticket} into {epic} is not known")
        annotate(moved="yes")
        try:
            after = await sighted(store, project)
        except (LitFailed, Rejected) as error:
            return _refused(502, f"lit moved {ticket} into {epic}, and the backlog could not be read again to show it: {error}")
        # [LAW:no-silent-failure] the read after is the evidence, as a drop's is.
        if not _holds(after, lambda backlog: backlog.parent.get(ticket) == epic and all(backlog.landed(move) for move in ranks)):
            return _refused(409, f"the backlog lit holds after the leveling does not have {ticket} in {epic} where it was put: it changed while it was leveled")
        return web.json_response(shown(after))

    return [web.get("/backlog", page_served), web.get("/backlog/tickets", read), web.post("/backlog/rank", ranked), web.post("/backlog/level", leveled)]


def _refused(status: int, why: str) -> web.Response:
    """A drop's or a leveling's answer when it did not end as asked: its event fails saying why, and the page is told the same."""
    fail(why)
    return web.Response(status=status, text=why)


def _unwritten(error: LitFailed, fact: str, refused: str, unknown: str) -> web.Response:
    """A leveling's answer when a lit write in it failed: lit's refusal proves it wrote nothing, and the answer says
    `refused`; a lit that stopped without saying why may have written first, so `fact` on the event is "unknown", and the
    answer says `unknown` [LAW:no-silent-failure]."""
    if error.refused:
        return _refused(502, f"{refused}: {error}")
    annotate(**{fact: "unknown"})
    return _refused(502, f"{unknown}: {error}")


def _holds(after: Sighted | Untracked, holds: Callable[[Backlog], bool]) -> bool:
    """Whether the backlog read after a change is one `holds` is true of; a project lit no longer tracks holds nothing."""
    match after:
        case Untracked():
            return False
        case Sighted(backlog=backlog):
            return holds(backlog)
