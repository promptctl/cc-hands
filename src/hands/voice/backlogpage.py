"""The backlog page: every open ticket of a project's lit backlog, under the epic it is in, each with hands' summary
for it, dragged to set the order the work is pulled in, and each put at a level: moved into its topic's high, normal,
or low epic.

It is served beside the phone's page, on its port and its addresses (`hands.voice.phonepage`), so a browser opens it
from the address `hands phone` prints with `backlog?project=<directory>` on its path. The page is open to anyone who
can reach the port; what it reads and moves is not: each request carries the phone's key, which the page takes from its
address's fragment as the phone's page does.

[LAW:one-source-of-truth] the order is lit's: every read is `lit export` run fresh, a drop is one `lit rank`, a level a
`lit parent set` and the `lit rank` that puts the ticket last in its epic, after a `lit new` and its `lit rank` where the
epic is made, and the page is answered with the backlog as lit holds it after. Nothing here keeps an order of its own,
or which epic is which level.
"""

import asyncio
import getpass
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from aiohttp import web

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.backlog import LEVELS, Backlog, Level, LitFailed, LitNotStarted, NewEpic, Person, Place, Standing, Status, Untracked, adopt, comment, make_epic, parse_level, rank, read_backlog
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, count, fail
from hands.voice.keyed import keyed
from hands.voice.summaries import Sighted, SummaryStore, sighted, sighting


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
    """A ticket as the page draws it: its summary where one is said, never its description; the frame lit ranks it in,
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
        case Untracked(gone=gone):
            annotate(tracked=False, gone=gone)
            return {"tracked": False, "why": read.why}
        case Sighted():
            nodes = drawn(read)
            annotate(tracked=True)
            # [LAW:one-source-of-truth] counted from the nodes the page is given, so the event cannot disagree with it.
            for node in every(nodes):
                count(tickets=1, summarised=int(node.summary is not None))
            # [LAW:one-source-of-truth] the levels go with the backlog, so the page's buttons are the levels hands puts tickets at.
            return {"tracked": True, "levels": LEVELS, "items": [asdict(node) for node in nodes]}


def backlog_routes(clock: Clock, store: SummaryStore, key: str, record: Record, environment: Mapping[str, str]) -> list[web.RouteDef]:
    """The page, the read of a project's backlog, a drop that ranks one ticket next to another, a level that moves a
    ticket into its topic's epic for it, and a note left on a ticket as a lit comment.

    [LAW:nothing-unseen] each read is one `backlog.read` event, how many open tickets it showed and how many of them had a
    summary; each drop one `backlog.ranked`, which ticket was put where, whether lit ranked anything, and what it showed
    after; each level one `backlog.leveled`, which ticket at which level, the epic it went into, whether lit made that
    epic, ranked it, moved the ticket, and ranked it last there; each note one `backlog.noted`, which ticket, how long,
    and whether lit kept it. Each write's fact has what lit said of it beside it, as `<fact>_said`; each refused or
    failed says why.
    """
    page = resources.files("hands.voice").joinpath("backlog.html").read_text()
    # Every write from the page is a person's at the page, and lit signs it as the one hands runs as.
    person = Person(getpass.getuser(), environment)

    async def page_served(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    @keyed(clock, "backlog.read", record, key, counts=("tickets", "summarised"))
    async def read(request: web.Request) -> web.Response:
        try:
            project = parse_project(request.query.get("project"))
        except Rejected as error:
            fail(str(error))
            return web.Response(status=400, text=str(error))
        annotate(project=project)
        try:
            return web.json_response(shown(await sighted(clock, store, project)))
        except (LitFailed, Rejected) as error:
            fail(f"cannot read the backlog: {error}")
            return web.Response(status=502, text=f"the backlog in {project} could not be read: {error}")

    # [LAW:no-ambient-temporal-coupling] the one owner of a project's lit writes from this page: a drop, a level, and a
    # note in it each run alone, a drop and a level from the read they are checked against to the read that shows them,
    # so two pages pressing at once never both make one level's epic, rank against an order the other is changing, or
    # find lit's store held by the other's write.
    writing: defaultdict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)

    @keyed(clock, "backlog.ranked", record, key, counts=("tickets", "summarised"))
    async def ranked(request: web.Request) -> web.Response:
        try:
            project, ticket, place = parse_asked(await request.read())
        except Rejected as error:
            return _refused(400, f"not moved: {error}")
        async with writing[project]:
            return await _answered(ranking(project, ticket, place))

    async def ranking(project: Path, ticket: str, place: Place) -> web.Response:
        # [LAW:nothing-unseen] whether lit ranked anything, "no", "yes", or "unknown", is on the event however the drop ends.
        annotate(project=project, ticket=ticket, place=place, ranked="no")
        before = await _read(clock, project, f"not moved: the backlog in {project} could not be read", 404, f"not moved: {project} has no backlog to rank")
        try:
            move = before.move(ticket, place)
        except Rejected as error:
            raise Halted(400, f"not moved: {error}") from error
        after = await _settled(
            clock,
            project,
            rank(clock, project, move, person),
            Write("ranked", lambda backlog: backlog.landed(move), f"{ticket} {place}", "not moved", f"whether lit ranked {ticket} {place} is not known"),
        )
        return web.json_response(shown(sighting(store, project, after)))

    @keyed(clock, "backlog.leveled", record, key)
    async def leveled(request: web.Request) -> web.Response:
        try:
            project, ticket, level = parse_leveling(await request.read())
        except Rejected as error:
            return _refused(400, f"not leveled: {error}")
        async with writing[project]:
            return await _answered(leveling(project, ticket, level))

    async def leveling(project: Path, ticket: str, level: Level) -> web.Response:
        # [LAW:nothing-unseen] whether lit made an epic, ranked it, moved the ticket, and ranked it last in its epic, each
        # "no", "yes", "unknown", or "unneeded" where the backlog already had it, are on the event however the leveling ends.
        annotate(project=project, ticket=ticket, level=level, created="no", ranked="no", moved="no", placed="no")
        now = await _read(clock, project, f"not leveled: the backlog in {project} could not be read", 404, f"not leveled: {project} has no backlog to level")
        try:
            asked = now.leveling(ticket, level)
        except Rejected as error:
            raise Halted(400, f"not leveled: {error}") from error
        match asked.epic:
            case str(epic):
                annotate(epic=epic, created="unneeded", ranked="unneeded")
            case NewEpic(topic=topic, title=title, parent=frame, place=place) as new:
                now = await _settled(
                    clock,
                    project,
                    make_epic(clock, project, new, person),
                    Write("created", lambda backlog: level in backlog.level_epics(topic, frame), f"the epic {title}", "not leveled", f"not leveled: whether lit made the epic {title} is not known"),
                )
                epic = now.level_epics(topic, frame)[level]
                annotate(epic=epic)
                try:
                    move = now.move(epic, place)
                except Rejected as error:
                    raise Halted(409, f"not leveled: lit made {epic}, {title}, at the bottom of its frame, and it cannot be ranked {place}: {error}") from error
                now = await _settled(
                    clock,
                    project,
                    rank(clock, project, move, person),
                    Write(
                        "ranked",
                        lambda backlog: backlog.landed(move),
                        f"{epic} {place}",
                        f"not leveled: lit made {epic}, {title}, and did not move {ticket} into it",
                        f"not leveled: lit made {epic}, {title}, and whether it ranked it {place} is not known, so {ticket} was not moved",
                    ),
                )
        now = await _settled(
            clock,
            project,
            adopt(clock, project, ticket, epic, person),
            Write("moved", lambda backlog: backlog.parent.get(ticket) == epic, f"{ticket} in {epic}", "not leveled", f"whether lit moved {ticket} into {epic} is not known"),
        )
        # lit keeps a ticket's rank when it moves it, which falls anywhere among its new epic's: it is ranked below the
        # last of the others, and needs no rank where it is alone there.
        match next((child for child in reversed(now.open_children(epic)) if child != ticket), None):
            case None:
                annotate(placed="unneeded")
            case last:
                try:
                    below = now.move(ticket, Place("below", last))
                except Rejected as error:
                    raise Halted(409, f"lit moved {ticket} into {epic}, and it cannot be ranked last there: {error}") from error
                await _settled(
                    clock,
                    project,
                    rank(clock, project, below, person),
                    Write("placed", lambda backlog: backlog.landed(below), f"{ticket} {below.place}", f"lit moved {ticket} into {epic}, and did not rank it last there", f"lit moved {ticket} into {epic}, and whether it ranked it last there is not known"),
                )
        # The page drew the press when it was made and redraws nothing, so a leveling that holds is answered with nothing more.
        return web.Response(status=204)

    @keyed(clock, "backlog.noted", record, key)
    async def noted(request: web.Request) -> web.Response:
        try:
            project, ticket, text = parse_note(await request.read())
        except Rejected as error:
            return _refused(400, f"not noted: {error}")
        async with writing[project]:
            return await _answered(noting(project, ticket, text))

    async def noting(project: Path, ticket: str, text: str) -> web.Response:
        # [LAW:nothing-unseen] whether lit kept the note, "no", "yes", or "unknown", is on the event however it ends.
        annotate(project=project, ticket=ticket, chars=len(text), noted="no")
        before = await _read(clock, project, f"not noted: the backlog in {project} could not be read", 404, f"not noted: {project} has no backlog to note in")
        if ticket not in before.kept:
            raise Halted(404, f"not noted: lit holds no ticket {ticket} in {project}")
        # The note is kept where the backlog after it holds one more of these words by the page's signer than before.
        held = _notes(before, ticket, text, person.name)
        await _settled(
            clock,
            project,
            comment(clock, project, ticket, text, person),
            Write("noted", lambda backlog: _notes(backlog, ticket, text, person.name) > held, f"the note on {ticket}", "not noted", f"whether lit kept the note on {ticket} is not known"),
        )
        return web.Response(status=204)

    return [web.get("/backlog", page_served), web.get("/backlog/tickets", read), web.post("/backlog/rank", ranked), web.post("/backlog/level", leveled), web.post("/backlog/note", noted)]


def _refused(status: int, why: str) -> web.Response:
    """A drop's, a leveling's, or a note's answer when it did not end as asked: its event fails saying why, and the page is
    told the same."""
    fail(why)
    return web.Response(status=status, text=why)


async def _answered(write: Awaitable[web.Response]) -> web.Response:
    """A drop's, a leveling's, or a note's answer: what `write` answers, or, where it halted, its status and words."""
    try:
        return await write
    except Halted as halted:
        return _refused(halted.status, halted.why)


class Halted(Exception):
    """A drop, a leveling, or a note that stopped short of what was asked: the status and the words its answer and its
    event carry."""

    def __init__(self, status: int, why: str) -> None:
        super().__init__(why)
        self.status = status
        self.why = why


async def _read(clock: Clock, project: Path, unread: str, status: int, untracked: str) -> Backlog:
    """The backlog in `project` read fresh before or partway through a write; raises Halted saying `unread` and lit's error when
    it cannot be read, and `untracked` with `status` when it has no backlog, saying why."""
    try:
        read = await read_backlog(clock, project)
    except (LitFailed, Rejected) as error:
        raise Halted(502, f"{unread}: {error}") from error
    match read:
        case Untracked():
            raise Halted(status, f"{untracked}: {read.why}")
        case Backlog():
            return read


# The lit writes a drop, a leveling, and a note make, each named by its fact on the event.
Fact = Literal["ranked", "created", "moved", "placed", "noted"]


@dataclass(frozen=True)
class Write:
    """One lit write in a drop, a leveling, or a note: its fact on the event, what the backlog read after it `holds`
    where it landed, said as what was `asked`, and what the answer says where it is `undone`, and where whether it
    landed is `unknown`."""

    fact: Fact
    holds: Callable[[Backlog], bool]
    asked: str
    undone: str
    unknown: str


async def _settled(clock: Clock, project: Path, lit: Awaitable[str], write: Write) -> Backlog:
    """The backlog lit holds after `lit` made `write`, where it holds what was asked, and the fact is "yes"; raises
    Halted saying `write.undone` where it does not, and the fact stays "no", and saying `write.unknown` where it cannot
    be read again to see, and the fact is "unknown" [LAW:no-silent-failure].

    [LAW:one-source-of-truth] whether a write landed is read off the one backlog read after it, whatever lit's exit, and
    the request goes on from that read. lit's reference (docs/cli-reference.md in links, "Every reason") says of no
    failure that it comes before any write, its code 6 is a write that already holds, and lit exits 0 having ranked the
    epic of a ticket another writer moved meanwhile. What lit said is on the event, and decides nothing.
    """
    try:
        said = f"lit exited 0: {await lit}"
    except LitNotStarted as error:
        # hands' own record, not lit's: no lit ran, so nothing was written, and no lit can run to read the backlog either.
        annotate(**{f"{write.fact}_said": str(error)})
        raise Halted(502, f"{write.undone}: {error}") from error
    except LitFailed as error:
        said = str(error)
    annotate(**{f"{write.fact}_said": said})
    try:
        after = await read_backlog(clock, project)
    except (LitFailed, Rejected) as unread:
        annotate(**{write.fact: "unknown"})
        raise Halted(502, f"{write.unknown}; {said}; and the backlog could not be read again to see: {unread}") from unread
    match after:
        case Untracked(why=why):
            annotate(**{write.fact: "unknown"})
            raise Halted(502, f"{write.unknown}; {said}; and lit no longer tracks {project}: {why}")
        case Backlog() if write.holds(after):
            annotate(**{write.fact: "yes"})
            return after
        case Backlog():
            raise Halted(409, f"{write.undone}: the backlog lit holds after does not have {write.asked}; {said}")


def _notes(backlog: Backlog, ticket: str, text: str, by: str) -> int:
    """How many comments on `ticket` in `backlog` say exactly `text`, signed `by`."""
    return sum(note.body == text and note.by == by for note in backlog.comments.get(ticket, ()))
