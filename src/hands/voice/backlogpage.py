"""The backlog page: every open ticket of a project's lit backlog, under the epic it is in, each with hands' summary
for it, dragged to set the order the work is pulled in, and each put at a level: moved into its topic's high, normal,
or low epic.

It is served beside the phone's page, on its port and its addresses (`hands.voice.phonepage`), so a browser opens it
from the address `hands phone` prints with `backlog?project=<directory>` on its path. The page is open to anyone who
can reach the port; what it reads and moves is not: each request carries the phone's key, which the page takes from its
address's fragment as the phone's page does.

[LAW:one-source-of-truth] the order is lit's: every read is `lit export` run fresh; a drop is one `lit rank`; a level a
`lit parent set` and the `lit rank` that puts the ticket last among its epic's open tickets, after a `lit new` and its
`lit rank` where the epic is made; a note one `lit comment add`. Each write is settled by the backlog lit holds after
it, and a drop is answered with that backlog. Nothing here keeps an order of its own, or which epic is which level.
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

from hands.sessions.audit import Record
from hands.sessions.backlog import LEVELS, Backlog, Level, LitFailed, LitNotStarted, Move, NewEpic, Person, Place, Standing, Status, Untracked, adopt, comment, make_epic, parse_level, rank, read_backlog
from hands.sessions.child import Children
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, count, fail
from hands.voice.keyed import keyed
from hands.voice.summaries import Sighted, SummaryStore, sighting


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
        case {"project": str(project), "ticket": str(ticket), "text": str(text)} if text.strip() and not any("\0" in part for part in (project, ticket, text)):
            return parse_project(project), ticket, text.strip()
        case _:
            raise Rejected("a note is {project, ticket, and text: what it says, not blank}, none of them holding a NUL")


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


def backlog_routes(children: Children, store: SummaryStore, key: str, record: Record, environment: Mapping[str, str]) -> list[web.RouteDef]:
    """The page, the read of a project's backlog, a drop that ranks one ticket next to another, a level that moves a
    ticket into its topic's epic for it, and a note left on a ticket as a lit comment.

    [LAW:nothing-unseen] each read is one `backlog.read` event, how many open tickets it showed and how many of them had a
    summary; each drop one `backlog.ranked`, which ticket was put where, whether the backlog lit holds after has it
    there, and what it showed after; each level one `backlog.leveled`, which ticket at which level, the epic it went
    into, and whether the backlog lit holds after each write has the epic made, the epic ranked, the ticket in it, and
    the ticket last there; each note one `backlog.noted`, which ticket, how long, and whether the backlog after has the
    note. A write's fact is on the event once lit is asked for it, "unknown" until the read after settles it, and a
    step the request never reached is not on it; each has what lit said of it beside it, as `<fact>_said`. Each refused
    or failed says why.
    """
    page = resources.files("hands.voice").joinpath("backlog.html").read_text()
    # Every write from the page is a person's at the page, and lit signs it as the one hands runs as.
    person = Person(getpass.getuser(), environment)

    async def page_served(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    @keyed(children.clock, "backlog.read", record, key, counts=("tickets", "summarised"))
    async def read(request: web.Request) -> web.Response:
        try:
            project = parse_project(request.query.get("project"))
        except Rejected as error:
            fail(str(error))
            return web.Response(status=400, text=str(error))
        annotate(project=project)
        match await _read(children, project):
            case Backlog() as backlog:
                return web.json_response(shown(sighting(store, project, backlog)))
            case Untracked() as untracked:
                return web.json_response(shown(untracked))
            case Unread(why=why):
                fail(why)
                return web.Response(status=502, text=why)

    # [LAW:no-ambient-temporal-coupling] the one owner of a project's lit writes from this page: a drop, a level, and a
    # note in it each run alone, a drop and a level from the read they are checked against to the read that shows them,
    # so two pages pressing at once never both make one level's epic, rank against an order the other is changing, or
    # find lit's store held by the other's write.
    writing: defaultdict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)

    @keyed(children.clock, "backlog.ranked", record, key, counts=("tickets", "summarised"))
    async def ranked(request: web.Request) -> web.Response:
        writes = Writes("not moved")
        return await writes.answered(ranking(writes, await request.read()))

    async def ranking(writes: "Writes", body: bytes) -> web.Response:
        project, ticket, place = parse_asked(body)
        async with writing[project]:
            annotate(project=project, ticket=ticket, place=place)
            move = (await _before(children, project)).move(ticket, place)
            after, _ = await writes.settled(children, project, rank(children, project, move, person), _beside("ranked", move))
            return web.json_response(shown(sighting(store, project, after)))

    @keyed(children.clock, "backlog.leveled", record, key)
    async def leveled(request: web.Request) -> web.Response:
        writes = Writes("not leveled")
        return await writes.answered(leveling(writes, await request.read()))

    async def leveling(writes: "Writes", body: bytes) -> web.Response:
        project, ticket, level = parse_leveling(body)
        async with writing[project]:
            annotate(project=project, ticket=ticket, level=level)
            now = await _before(children, project)
            match now.leveling(ticket, level).epic:
                case str(epic):
                    annotate(epic=epic)
                    _landed("unneeded", "created", "ranked")
                case NewEpic(topic=topic, title=title, parent=frame, place=place) as new:
                    now, epic = await writes.settled(children, project, make_epic(children, project, new, person), Write("created", lambda backlog: backlog.level_epics(topic, frame).get(level), f"the epic {title}"))
                    annotate(epic=epic)
                    try:
                        move = now.move(epic, place)
                    except Rejected as error:
                        raise Halted(409, f"lit made {epic}, {title}, at the bottom of its frame, and it cannot be ranked {place}: {error}") from error
                    await writes.settled(children, project, rank(children, project, move, person), _beside("ranked", move))
            now, _ = await writes.settled(children, project, adopt(children, project, ticket, epic, person), Write("moved", lambda backlog: epic if backlog.parent.get(ticket) == epic else None, f"{ticket} in {epic}"))
            # lit keeps a ticket's rank when it moves it, which falls anywhere among its new epic's: it is ranked below
            # the last of the others, unless it is the last there already, alone there included.
            match now.open_children(epic):
                case (*_, last) if last != ticket:
                    below = now.move(ticket, Place("below", last))
                    await writes.settled(children, project, rank(children, project, below, person), Write("placed", lambda backlog: ticket if backlog.open_children(epic)[-1:] == (ticket,) else None, f"{ticket} last in {epic}"))
                case _:
                    _landed("unneeded", "placed")
            # The page drew the press when it was made and redraws nothing, so a leveling that holds is answered with nothing more.
            return web.Response(status=204)

    @keyed(children.clock, "backlog.noted", record, key)
    async def noted(request: web.Request) -> web.Response:
        writes = Writes("not noted")
        return await writes.answered(noting(writes, await request.read()))

    async def noting(writes: "Writes", body: bytes) -> web.Response:
        project, ticket, text = parse_note(body)
        async with writing[project]:
            annotate(project=project, ticket=ticket, chars=len(text))
            before = await _before(children, project)
            # [LAW:single-enforcer] also what keeps an id lit would read as its own flag, `--help`, from `lit comment add`,
            # which takes no `--`: no ticket lit keeps is one.
            if ticket not in before.kept:
                raise Halted(404, f"lit holds no ticket {ticket} in {project}")
            # The note is kept where the backlog after it holds one more of these words by the page's signer than before.
            held = _notes(before, ticket, text, person.name)
            await writes.settled(children, project, comment(children, project, ticket, text, person), Write("noted", lambda backlog: text if _notes(backlog, ticket, text, person.name) > held else None, f"the note on {ticket}"))
            return web.Response(status=204)

    return [web.get("/backlog", page_served), web.get("/backlog/tickets", read), web.post("/backlog/rank", ranked), web.post("/backlog/level", leveled), web.post("/backlog/note", noted)]


class Halted(Exception):
    """A drop, a leveling, or a note that did not end as asked: the status its answer carries, and why."""

    def __init__(self, status: int, why: str) -> None:
        super().__init__(why)
        self.status = status
        self.why = why


@dataclass(frozen=True)
class Unread:
    """A backlog lit could not give, and why: lit failed, or what it said does not parse."""

    why: str


async def _read(children: Children, project: Path) -> Backlog | Untracked | Unread:
    """The backlog in `project` read fresh, before a write or after one; each caller says what each answer is for it."""
    try:
        return await read_backlog(children, project)
    except (LitFailed, Rejected) as error:
        return Unread(f"the backlog in {project} could not be read: {error}")


async def _before(children: Children, project: Path) -> Backlog:
    """The backlog in `project` a write is checked against; raises Halted where lit holds none there or cannot give it."""
    match await _read(children, project):
        case Backlog() as backlog:
            return backlog
        case Untracked(why=why):
            raise Halted(404, f"{project} has no backlog: {why}")
        case Unread(why=why):
            raise Halted(502, why)


# The lit writes a drop, a leveling, and a note make, each by the name of its fact on the event.
Step = Literal["ranked", "created", "moved", "placed", "noted"]
# Whether what a write asked for holds in the backlog read after it: "unknown" from when lit is asked until that read
# settles it, and where it cannot; "unneeded" where the read before already had it.
Landed = Literal["yes", "no", "unknown", "unneeded"]


def _landed(landed: Landed, *steps: Step) -> None:
    """Each of `steps` as `landed`, on the event of the unit of work open here."""
    annotate(**dict.fromkeys(steps, landed))


@dataclass(frozen=True)
class Write[T]:
    """One lit write in a drop, a leveling, or a note: its step, what `holds` proves of the backlog read after it where
    what was asked holds there (None where it does not), and what was `asked`, as the answer says it."""

    step: Step
    holds: Callable[[Backlog], T | None]
    asked: str


def _beside(step: Step, move: Move) -> Write[Move]:
    """A `lit rank`'s write: the backlog after it has the ticket directly beside its target."""
    return Write(step, lambda backlog: move if backlog.landed(move) else None, f"{move.ticket} {move.place}")


@dataclass
class Writes:
    """One drop's, leveling's, or note's lit writes, as each is asked and settled.

    [LAW:single-enforcer] the one owner of what the request's event says of each write and of how its answer is worded:
    a step is on the event once lit is asked for it, and not before, and its answer opens with `verb`, what was not
    done, only while no lit has run for it, since a lit that ran may have written.
    """

    verb: str
    ran: bool = False

    async def answered(self, request: Awaitable[web.Response]) -> web.Response:
        """What `request` answers, or, where it did not end as asked, why, on its event and to the page."""
        try:
            return await request
        except Rejected as error:
            status, why = 400, str(error)
        except Halted as halted:
            status, why = halted.status, halted.why
        said = why if self.ran else f"{self.verb}: {why}"
        fail(said)
        return web.Response(status=status, text=said)

    async def settled[T](self, children: Children, project: Path, lit: Awaitable[str], write: Write[T]) -> tuple[Backlog, T]:
        """The backlog lit holds after `lit` made `write`, and what `write.holds` proves of it, and the step is "yes";
        raises Halted where it does not hold there, and the step is "no", and where the backlog cannot be read again to
        see, and the step stays "unknown" [LAW:no-silent-failure].

        [LAW:one-source-of-truth] whether a write landed is read off the one backlog read after it, whatever lit's exit,
        and the request goes on from that read. lit's reference (docs/cli-reference.md in links, "Every reason") says of
        no failure that it comes before any write, its code 6 is a write that already holds, and lit exits 0 having
        ranked the epic of a ticket another writer moved meanwhile. What lit said is on the event, and decides nothing.
        """
        # Asked, and so unknown until the read after settles it, however the request ends meanwhile: cancelled included.
        _landed("unknown", write.step)
        try:
            said = f"lit exited 0: {await lit}"
        except LitNotStarted as error:
            # `_lit` raises it only where child.run could not spawn lit: no lit ran, so nothing was written, and no lit
            # can run to read the backlog either. hands' own record, not lit's.
            annotate(**{f"{write.step}_said": str(error)})
            _landed("no", write.step)
            raise Halted(502, str(error)) from error
        except LitFailed as error:
            said = str(error)
        self.ran = True
        annotate(**{f"{write.step}_said": said})
        match await _read(children, project):
            case Backlog() as after if (proof := write.holds(after)) is not None:
                _landed("yes", write.step)
                return after, proof
            case Backlog():
                _landed("no", write.step)
                raise Halted(409, f"the backlog lit holds after does not have {write.asked}; {said}")
            case Untracked(why=why):
                unseen = f"lit no longer tracks {project}: {why}"
            case Unread(why=why):
                unseen = why
        raise Halted(502, f"whether lit holds {write.asked} is not known; {said}; and {unseen}")


def _notes(backlog: Backlog, ticket: str, text: str, by: str) -> int:
    """How many comments on `ticket` in `backlog` say exactly `text`, signed `by`."""
    return sum(note.body == text and note.by == by for note in backlog.comments.get(ticket, ()))
