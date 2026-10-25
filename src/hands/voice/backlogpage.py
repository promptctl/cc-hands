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
import getpass
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

from aiohttp import web

from hands.sessions.audit import Record
from hands.sessions.backlog import LEVELS, Backlog, Level, LitFailed, Move, NewEpic, Person, Place, Standing, Status, Untracked, adopt, comment, make_epic, parse_level, rank, read_backlog
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


def parse_leveling(body: bytes) -> tuple[Path, str, Level]:
    """The level a page put a ticket at, as its request's body: the project, the ticket, and the level; raises Rejected
    naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        raise Rejected(f"a leveling is JSON: {error}") from error
    match sent:
        case {"project": str(project), "ticket": str(ticket), "level": str(named)} if (level := parse_level(named)) is not None:
            return parse_project(project), ticket, level
        case _:
            raise Rejected(f"a leveling is {{project, ticket, and level: one of {', '.join(LEVELS)}}}")


def parse_note(body: bytes) -> tuple[Path, str, str]:
    """The note a page left on a ticket, as its request's body: the project, the ticket, and what it says, trimmed;
    raises Rejected naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        raise Rejected(f"a note is JSON: {error}") from error
    match sent:
        case {"ticket": str(ticket)} if ticket.startswith("-"):
            # lit would read it as a flag: `--help` says its help and exits 0, keeping nothing.
            raise Rejected(f"a note's ticket is a ticket's id, and {ticket!r} is a flag")
        case {"project": str(project), "ticket": str(ticket), "text": str(text)} if text.strip():
            return parse_project(project), ticket, text.strip()
        case _:
            raise Rejected("a note is {project, ticket, and text: what it says, not blank}")


@dataclass(frozen=True)
class Node:
    """A ticket as the page draws it: its sentence where one is said, never its description; the frame lit ranks it in,
    which a drop stays inside; where it stands among its topic's levels, None where it has none; and the
    unfinished tickets under it."""

    id: str
    title: str
    kind: str
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
        return Node(id, ticket.title, ticket.kind, ticket.status, backlog.inside.get(id), said.get(id), backlog.standing(id), held(backlog.open_children(id)))

    def held(ids: tuple[str, ...]) -> tuple[Node, ...]:
        # A ticket is drawn; an epic only over what is drawn beneath it.
        return tuple(each for each in map(node, ids) if each.kind != "epic" or each.children)

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


def backlog_routes(store: SummaryStore, key: str, record: Record, environment: Mapping[str, str]) -> list[web.RouteDef]:
    """The page, the read of a project's backlog, a drop that ranks one ticket next to another, a level that moves a
    ticket into its topic's epic for it, and a note left on a ticket as a lit comment.

    [LAW:nothing-unseen] each read is one `backlog.read` event, how many open tickets it showed and how many of them had a
    sentence; each drop one `backlog.ranked`, which ticket was put where, whether lit ranked anything and what it said,
    and what it showed after; each level one `backlog.leveled`, which ticket at which level, the epic it went into,
    whether lit made that epic, ranked it, moved the ticket, and ranked it last there; each note one `backlog.noted`,
    which ticket, how long, and whether lit kept it; each refused or failed says why.
    """
    page = resources.files("hands.voice").joinpath("backlog.html").read_text()
    # Every write from the page is a person's at the page, and lit signs it as the one hands runs as.
    person = Person(getpass.getuser(), environment)

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

    # [LAW:no-ambient-temporal-coupling] the one owner of a project's lit writes from this page: a drop, a level, and a
    # note in it each run alone, a drop and a level from the read they are checked against to the read that shows them,
    # so two pages pressing at once never both make one level's epic, rank against an order the other is changing, or
    # find lit's store held by the other's write.
    writing: defaultdict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)

    @keyed("backlog.ranked", record, key, counts=("tickets", "summarised"))
    async def ranked(request: web.Request) -> web.Response:
        try:
            project, ticket, place = parse_asked(await request.read())
        except Rejected as error:
            return _refused(400, f"not moved: {error}")
        async with writing[project]:
            return await ranking(project, ticket, place)

    async def ranking(project: Path, ticket: str, place: Place) -> web.Response:
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
            said = await rank(project, move, person)
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

    @keyed("backlog.leveled", record, key)
    async def leveled(request: web.Request) -> web.Response:
        try:
            project, ticket, level = parse_leveling(await request.read())
        except Rejected as error:
            return _refused(400, f"not leveled: {error}")
        async with writing[project]:
            try:
                return await leveling(project, ticket, level)
            except Halted as halted:
                return _refused(halted.status, halted.why)

    async def leveling(project: Path, ticket: str, level: Level) -> web.Response:
        # [LAW:nothing-unseen] whether lit made an epic, ranked it, moved the ticket, and ranked it last in its epic, each
        # "no", "yes", or "unknown", are on the event however the leveling ends.
        annotate(project=project, ticket=ticket, level=level, created="no", ranked="no", moved="no", placed="no")
        before = await _read(project, f"not leveled: the backlog in {project} could not be read", 404, f"not leveled: lit has no workspace in {project}, so it has no backlog to level")
        try:
            asked = before.leveling(ticket, level)
        except Rejected as error:
            raise Halted(400, f"not leveled: {error}") from error
        # The ranks the read after must have: the new epic's among its topic's levels, where one is made, and the ticket's
        # last in its epic.
        ranks: tuple[Move, ...] = ()
        match asked.epic:
            case str(epic):
                annotate(epic=epic)
            case NewEpic(topic=topic, title=title, parent=frame, place=place) as new:
                try:
                    await make_epic(project, new, person)
                except LitFailed as error:
                    raise _unwritten(error, "created", "not leveled", f"whether lit made the epic {title} is not known") from error
                annotate(created="yes")
                made = await _read(project, f"not leveled: lit made the epic {title}, and the backlog could not be read again to rank it", 502, f"not leveled: lit made the epic {title}, and lit no longer tracks {project}")
                found = made.level_epics(topic, frame).get(level)
                if found is None:
                    raise Halted(502, f"not leveled: lit made the epic {title}, and the backlog read after it does not hold it")
                annotate(epic=found)
                try:
                    move = made.move(found, place)
                except Rejected as error:
                    raise Halted(409, f"not leveled: lit made {found}, {title}, at the bottom of its frame, and it cannot be ranked {place}: {error}") from error
                epic = found
                try:
                    await rank(project, move, person)
                except LitFailed as error:
                    raise _unwritten(
                        error,
                        "ranked",
                        f"not leveled: lit made {epic}, {title}, and left it at the bottom of its frame, so {ticket} was not moved",
                        f"not leveled: lit made {epic}, {title}, and whether it ranked it {place} or left it at the bottom of its frame is not known, so {ticket} was not moved",
                    ) from error
                annotate(ranked="yes")
                ranks = (move,)
        try:
            await adopt(project, ticket, epic, person)
        except LitFailed as error:
            raise _unwritten(error, "moved", "not leveled", f"whether lit moved {ticket} into {epic} is not known") from error
        annotate(moved="yes")
        # lit keeps a ticket's rank when it moves it, which falls anywhere among its new epic's: it is ranked below the
        # last of the others, or left where it is when it is alone there.
        moved = await _read(project, f"lit moved {ticket} into {epic}, and the backlog could not be read again to rank it last there", 502, f"lit moved {ticket} into {epic}, and lit no longer tracks {project}")
        others = [child for child in moved.open_children(epic) if child != ticket]
        try:
            lasts = tuple(moved.move(ticket, Place("below", other)) for other in others[-1:])
        except Rejected as error:
            raise Halted(409, f"lit moved {ticket} into {epic}, and it cannot be ranked last there: {error}") from error
        for last in lasts:
            try:
                await rank(project, last, person)
            except LitFailed as error:
                raise _unwritten(error, "placed", f"lit moved {ticket} into {epic}, and did not rank it last there", f"lit moved {ticket} into {epic}, and whether it ranked it last there is not known") from error
            annotate(placed="yes")
        ranks = (*ranks, *lasts)
        after = await _read(project, f"lit moved {ticket} into {epic}, and the backlog could not be read again to check it", 502, f"lit moved {ticket} into {epic}, and lit no longer tracks {project}")
        # [LAW:no-silent-failure] the read after is the evidence, as a drop's is. The page drew the press when it was made
        # and redraws nothing, so a leveling that holds is answered with nothing more.
        if not (after.parent.get(ticket) == epic and after.open_children(epic)[-1:] == (ticket,) and all(after.landed(move) for move in ranks)):
            raise Halted(409, f"the backlog lit holds after the leveling does not have {ticket} last in {epic} where it was put: it changed while it was leveled")
        return web.Response(status=204)

    @keyed("backlog.noted", record, key)
    async def noted(request: web.Request) -> web.Response:
        try:
            project, ticket, text = parse_note(await request.read())
        except Rejected as error:
            return _refused(400, f"not noted: {error}")
        async with writing[project]:
            return await noting(project, ticket, text)

    async def noting(project: Path, ticket: str, text: str) -> web.Response:
        # [LAW:nothing-unseen] whether lit kept the note, "no", "yes", or "unknown", is on the event however it ends.
        annotate(project=project, ticket=ticket, chars=len(text), noted="no")
        try:
            await comment(project, ticket, text, person)
        except LitFailed as error:
            if error.refused:
                return _refused(502, f"not noted: {error}")
            annotate(noted="unknown")
            return _refused(502, f"whether lit kept the note on {ticket} is not known: {error}")
        annotate(noted="yes")
        return web.Response(status=204)

    return [web.get("/backlog", page_served), web.get("/backlog/tickets", read), web.post("/backlog/rank", ranked), web.post("/backlog/level", leveled), web.post("/backlog/note", noted)]


def _refused(status: int, why: str) -> web.Response:
    """A drop's, a leveling's, or a note's answer when it did not end as asked: its event fails saying why, and the page is
    told the same."""
    fail(why)
    return web.Response(status=status, text=why)


class Halted(Exception):
    """A leveling that stopped short of what was asked: the status and the words its answer and its event carry."""

    def __init__(self, status: int, why: str) -> None:
        super().__init__(why)
        self.status = status
        self.why = why


async def _read(project: Path, unread: str, status: int, untracked: str) -> Backlog:
    """The backlog in `project` read fresh partway through a leveling; raises Halted saying `unread` and lit's error when
    it cannot be read, and `untracked` with `status` when lit has no workspace there."""
    try:
        read = await read_backlog(project)
    except (LitFailed, Rejected) as error:
        raise Halted(502, f"{unread}: {error}") from error
    match read:
        case Untracked():
            raise Halted(status, untracked)
        case Backlog():
            return read


def _unwritten(error: LitFailed, fact: str, refused: str, unknown: str) -> Halted:
    """A leveling's stop when a lit write in it failed: lit's refusal proves it wrote nothing, and the answer says
    `refused`; a lit that stopped without saying why may have written first, so `fact` on the event is "unknown", and the
    answer says `unknown` [LAW:no-silent-failure]."""
    if error.refused:
        return Halted(502, f"{refused}: {error}")
    annotate(**{fact: "unknown"})
    return Halted(502, f"{unknown}: {error}")


def _holds(after: Sighted | Untracked, holds: Callable[[Backlog], bool]) -> bool:
    """Whether the backlog read after a change is one `holds` is true of; a project lit no longer tracks holds nothing."""
    match after:
        case Untracked():
            return False
        case Sighted(backlog=backlog):
            return holds(backlog)
