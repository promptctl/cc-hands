"""A project's lit backlog, read whole from `lit export` and parsed once into the tickets, their tree, and their comments.

`lit export` is lit's versioned data-export primitive, and the one command that hands over descriptions in bulk.
It is read fresh at every call and kept nowhere [LAW:one-source-of-truth]: lit is where the backlog is, and its rank
order is changed by `lit rank` alone.
"""

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, get_args

from hands.core.sentences import Thing
from hands.sessions.child import run
from hands.sessions.payload import Payload, Rejected

# The largest backlog on this machine (links, 8.6 MB of export) takes 5 s to export; past this lit is stuck, not slow.
LIT_TIMEOUT_SECONDS = 30.0

# The export versions whose every field read here means what this parser reads it as; another is refused rather than
# half read. 5 adds each ticket's plan and its law reviews to 4 (lit 0.21), and neither is read here; an issue's other
# fields are written as before (links internal/model/model.go, issueJSON).
EXPORT_VERSIONS = (4, 5)

Status = Literal["open", "in_progress", "closed"]

# The id of the whole backlog, as a thing to be said, beside the ids lit gives its tickets.
BACKLOG = "backlog"

# [LAW:one-source-of-truth] the levels a ticket is put at, highest first: each topic has an epic for each, and the
# order Brandon ranks those epics in is the order their tickets are pulled in. lit's own priority is not touched.
Level = Literal["high", "normal", "low"]
LEVELS: tuple[Level, ...] = get_args(Level)


def parse_level(named: str) -> Level | None:
    """The level `named` names, None where it names none."""
    return next((level for level in LEVELS if level == named), None)


def level_title(topic: str, level: Level) -> str:
    """The title of `topic`'s epic for `level`: the slug's words, then the level, as in `wake word - high`."""
    return f"{topic.replace('-', ' ')} - {level}"


class LitFailed(Exception):
    """lit did not do what it was asked in a directory: it could not run, did not answer, or failed or refused, and the
    message says why; `reason` is the token lit's refusal names, None where lit named none."""

    def __init__(self, message: str, reason: str | None) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def refused(self) -> bool:
        """Whether lit refused, saying why on its reason line: the one failure that proves it changed nothing. A lit that
        did not answer, could not be run, or ended without a reason may have written before it stopped."""
        return self.reason is not None


@dataclass(frozen=True)
class Untracked:
    """A directory lit was never set up in, or that is in no git repository: it has no backlog, which is an answer, not
    a failure."""

    project: Path


# A failed lit command's first stderr line names its reason as a token beside the exit code, lit's published contract
# (links docs/cli-reference.md) for telling apart the failures one code covers. Two reasons say it has no workspace
# here: `lit init` never ran in this repository, or this is no git repository.
ERROR_LINE = re.compile(r"error \(code=\d+, reason=(?P<reason>\w+)\): ")
NO_WORKSPACE = ("workspace_not_initialized", "outside_git_workspace")


@dataclass(frozen=True)
class Ticket:
    id: str
    title: str
    description: str
    kind: str
    topic: str
    # None for an epic: lit exports no status for one, since an epic's state is its children's.
    status: Status | None

    @property
    def text(self) -> str:
        """What the ticket says, and so what its sentence is made of and keyed by."""
        return f"{self.title}\n\n{self.description}"


@dataclass(frozen=True)
class Comment:
    by: str
    at: str
    body: str


# The side of another ticket a ticket is ranked on, as lit's own flag for it names it.
Side = Literal["above", "below"]


@dataclass(frozen=True)
class Place:
    """Where a ticket is ranked: directly above or directly below another of its frame."""

    side: Side
    target: str

    def __str__(self) -> str:
        return f"{self.side} {self.target}"


@dataclass(frozen=True)
class Move:
    """A rank lit carries out as asked: a ticket put next to another ticket lit ranks it among. Made by `Backlog.move`."""

    ticket: str
    place: Place


@dataclass(frozen=True)
class Standing:
    """Where a ticket stands among its topic's levels: the epic it is inside, and the level that epic is its topic's
    epic for, None where it is none of them."""

    epic: str
    level: Level | None


@dataclass(frozen=True)
class NewEpic:
    """A topic's epic for a level that lit holds none of yet: made inside the parent of the ticket's epic, and put at
    `place`, so the topic's levels stay in their order."""

    topic: str
    title: str
    parent: str | None
    place: Place


@dataclass(frozen=True)
class Leveling:
    """A ticket moved into its topic's epic for a level: one lit holds, by its id, or one it is to make first. Made by
    `Backlog.leveling`."""

    ticket: str
    level: Level
    epic: str | NewEpic


@dataclass(frozen=True)
class Backlog:
    """Every ticket lit holds for one project, in rank order, with each one's children in rank order."""

    tickets: Mapping[str, Ticket]
    # [LAW:one-source-of-truth] the one parent relation lit's export gives: each ticket's parent where lit lists it or
    # archived it. lit ranks a ticket among the others inside the same parent, an archived one included; one whose
    # parent was deleted, or is not in the export, it ranks among the roots, and so it has none here.
    inside: Mapping[str, str]
    comments: Mapping[str, tuple[Comment, ...]]

    @cached_property
    def parent(self) -> Mapping[str, str]:
        """Each ticket's parent where lit lists it: the tree is made of these, and a ticket inside an archived parent is a root of it."""
        return {child: parent for child, parent in self.inside.items() if parent in self.tickets}

    @cached_property
    def children(self) -> Mapping[str, tuple[str, ...]]:
        """Each listed parent's children, in rank order."""
        children: dict[str, tuple[str, ...]] = {}
        for id in self.tickets:  # rank order, so each parent's children are too
            if id in self.parent:
                children[self.parent[id]] = (*children.get(self.parent[id], ()), id)
        return children

    def frame(self, id: str) -> tuple[str, ...]:
        """Every ticket lit ranks `id` among, itself included, in rank order: those inside the same parent, or the roots."""
        return tuple(other for other in self.tickets if self.inside.get(other) == self.inside.get(id))

    def move(self, ticket: str, place: Place) -> Move:
        """`ticket` put at `place`, as a move lit carries out as asked; raises Rejected naming why it is not one.

        [LAW:single-enforcer] the one check of a move: lit refuses no rank across frames, it ranks the parent instead and
        exits 0, so a move it would carry out otherwise is never asked of it.
        """
        for id in (ticket, place.target):
            if id not in self.tickets:
                raise Rejected(f"the backlog has no ticket {id}")
        if ticket == place.target:
            raise Rejected(f"{ticket} cannot be ranked next to itself")
        if place.target not in self.frame(ticket):
            raise Rejected(f"{ticket} and {place.target} are not inside the same parent, and lit ranks a ticket only among those inside its own")
        return Move(ticket, place)

    def landed(self, move: Move) -> bool:
        """Whether this backlog has `move` as lit was asked to make it: the ticket directly above or below its target, in
        the frame they share."""
        frame = self.frame(move.ticket)
        if move.place.target not in frame:
            return False
        at = frame.index(move.place.target)
        match move.place.side:
            case "above":
                return frame[at - 1 : at] == (move.ticket,)
            case "below":
                return frame[at + 1 : at + 2] == (move.ticket,)

    def level_of(self, epic: str) -> Level | None:
        """The level an epic is its topic's epic for, read off its exact title; None for an epic of no level."""
        return next((level for level in LEVELS if self.tickets[epic].title == level_title(self.tickets[epic].topic, level)), None)

    def level_epics(self, topic: str, frame: str | None) -> Mapping[Level, str]:
        """`topic`'s epic for each level it has one for inside `frame` (None for the roots): the epic lit lists and ranks
        there of that topic titled exactly so, finished or not, the first ranked where there are more.

        [LAW:one-source-of-truth] the one way a level's epic is found: by its frame, title, and topic in lit, read fresh.
        A topic has one epic for a level in a frame, ever: one whose tickets are all closed is used again, never made a
        second time.
        """
        epics: dict[Level, str] = {}
        for id, ticket in self.tickets.items():  # rank order, so the first ranked is kept
            if ticket.kind == "epic" and ticket.topic == topic and self.inside.get(id) == frame and (level := self.level_of(id)) is not None:
                epics.setdefault(level, id)
        return epics

    def standing(self, id: str) -> Standing | None:
        """Where a ticket stands among its topic's levels, read off the title and topic of the epic it is inside; None
        for one no level holds: an epic, a ticket inside no epic lit lists, or one whose epic lit ranks inside an epic it
        archived, which the tree does not draw, so no level's epic can be made beside it."""
        match self.parent.get(id):
            case str(epic) if self.tickets[id].kind != "epic" and self.tickets[epic].kind == "epic" and self.inside.get(epic) == self.parent.get(epic):
                return Standing(epic, self.level_of(epic))
            case _:
                return None

    def leveling(self, ticket: str, level: Level) -> Leveling:
        """`ticket` moved into its epic's topic's epic for `level`, in the frame of its epic; raises Rejected naming why
        it cannot be.

        [LAW:single-enforcer] the one check of a leveling, as `move` is of a rank. Where lit holds no epic for the level
        there, one is made, and ranked so the topic's levels stay in their order, high over normal over low: directly
        above the first of its epics for a lower level, or else directly below the last for a higher one, or else beside
        the ticket's epic itself, above it for high and below it otherwise.
        """
        if ticket not in self.tickets:
            raise Rejected(f"the backlog has no ticket {ticket}")
        match self.standing(ticket):
            case None:
                raise Rejected(f"{ticket} is an epic, inside no epic lit lists, or inside an epic lit ranks under one it archived, so it has no level")
            case Standing(epic=epic, level=at) if at == level:
                raise Rejected(f"{ticket} is already in {epic}, its topic's {level} epic")
            case Standing(epic=epic):
                pass
        topic, parent, frame = self.tickets[epic].topic, self.parent.get(epic), self.frame(epic)
        epics = self.level_epics(topic, parent)
        # The topic's level epics beside the ticket's epic, in rank order, each by how far down the levels it is.
        ladder = sorted(((id, LEVELS.index(at)) for at, id in epics.items()), key=lambda rung: frame.index(rung[0]))
        rung = LEVELS.index(level)
        place = (
            next((Place("above", id) for id, at in ladder if at > rung), None)
            or next((Place("below", id) for id, at in reversed(ladder) if at < rung), None)
            or Place("above" if level == "high" else "below", epic)
        )
        found = epics.get(level)
        return Leveling(ticket, level, NewEpic(topic, level_title(topic, level), parent, place) if found is None else found)

    def done(self, id: str) -> bool:
        """Whether a ticket is finished: closed, or for an epic, every child done."""
        match self.tickets[id].status:
            case None:
                children = self.children.get(id, ())
                return bool(children) and all(self.done(child) for child in children)
            case status:
                return status == "closed"

    def open_children(self, id: str) -> tuple[str, ...]:
        return tuple(child for child in self.children.get(id, ()) if not self.done(child))

    def roots(self) -> tuple[str, ...]:
        """What the backlog is made of: every unfinished ticket under no unfinished parent, in rank order.

        Epics and loose tickets, and a follow-up filed under a ticket already closed, which stands on its own rather
        than vanishing with the parent the tree no longer shows.
        """
        return tuple(id for id in self.tickets if not self.done(id) and (id not in self.parent or self.done(self.parent[id])))

    def thing(self) -> Thing:
        """The backlog as a thing to be said: its unfinished roots, each over its unfinished children."""

        def of(id: str) -> Thing:
            return Thing(id, self.tickets[id].text, tuple(of(child) for child in self.open_children(id)))

        # The backlog has no text of its own: it is what it holds.
        return Thing(BACKLOG, "", tuple(of(id) for id in self.roots()))


async def read_backlog(project: Path) -> Backlog | Untracked:
    """The backlog lit holds for `project`, or that lit has no workspace there; raises LitFailed when lit cannot say, and
    Rejected when what it said does not parse."""
    try:
        out = await _lit(project, "export")
    except LitFailed as failed:
        if failed.reason in NO_WORKSPACE:
            return Untracked(project)
        raise
    # Off the loop: the largest export takes tens of milliseconds to parse, which the voice pipeline would hear.
    return await asyncio.to_thread(parse_export, out)


async def rank(project: Path, move: Move) -> str:
    """Have lit in `project` make `move`, and what lit said of what it ranked; raises LitFailed when it fails.

    lit's words are the one account of a rank it made otherwise than asked: it says it ranked the parent instead.
    """
    return (await _lit(project, "rank", move.ticket, f"--{move.place.side}", move.place.target)).decode(errors="replace").strip()


async def make_epic(project: Path, new: NewEpic) -> None:
    """Have lit in `project` make `new`, at the bottom of its frame; raises LitFailed when it fails. It is found after by
    its title and topic, as every level's epic is (`Backlog.level_epic`)."""
    await _lit(project, "new", "--type", "epic", "--topic", new.topic, "--title", new.title, *(() if new.parent is None else ("--parent", new.parent)))


async def adopt(project: Path, ticket: str, epic: str) -> None:
    """Have lit in `project` move `ticket` inside `epic`, keeping its id; raises LitFailed when it fails."""
    await _lit(project, "parent", "set", "--child", ticket, "--parent", epic)


async def _lit(project: Path, *arguments: str) -> bytes:
    """What lit run in `project` with `arguments` wrote out; raises LitFailed when it could not run, did not answer, or exited non-zero.

    [LAW:single-enforcer] the one place hands runs lit, under one bound, and reads its failure.
    """
    command = f"lit {arguments[0]}"
    try:
        ran = await run("lit", *arguments, timeout=LIT_TIMEOUT_SECONDS, cwd=project)
    except TimeoutError:
        raise LitFailed(f"{command} in {project} did not answer in {LIT_TIMEOUT_SECONDS:.0f}s", None) from None
    except OSError as error:
        raise LitFailed(f"cannot run lit in {project}: {error}", None) from error
    if ran.returncode != 0:
        err = ran.err.decode(errors="replace")
        line = ERROR_LINE.match(err)
        # The head of what lit said: its first line is its verdict, and the remediation after it is for a reader.
        raise LitFailed(f"{command} in {project} exited {ran.returncode}: {err.strip()[:300]}", None if line is None else line["reason"])
    return ran.out


def parse_export(raw: bytes) -> Backlog:
    """`lit export`'s JSON, parsed once [LAW:parse-dont-validate]: each ticket, its place in the tree, and what was said on it."""
    export = Payload.parse(raw)
    if (version := export.integer("version")) not in EXPORT_VERSIONS:
        raise Rejected(f"lit export is version {version}, and hands reads versions {', '.join(map(str, EXPORT_VERSIONS))}")
    every = [Payload.of(entry, "an issue") for entry in export.items("issues")]
    issues = sorted((issue for issue in every if _listed(issue)), key=lambda issue: issue.text("rank"))
    tickets = {issue.text("id"): _ticket(issue) for issue in issues}
    # An archived parent still frames its children's rank; a deleted or purged one frames nothing.
    framing = tickets.keys() | {issue.text("id") for issue in every if issue.optional_text("deleted_at") is None and issue.optional_text("archived_at") is not None}
    inside: dict[str, str] = {}
    for relation in (Payload.of(entry, "a relation") for entry in export.items("relations")):
        if relation.text("type") == "parent-child" and relation.text("src_id") in tickets and relation.text("dst_id") in framing:
            inside[relation.text("src_id")] = relation.text("dst_id")
    comments: dict[str, tuple[Comment, ...]] = {}
    # A comment on a ticket that is none here is read by nothing, and kept by nothing.
    kept = (comment for comment in (Payload.of(entry, "a comment") for entry in export.items("comments")) if comment.text("issue_id") in tickets)
    for comment in sorted(kept, key=lambda comment: comment.text("created_at")):
        issue = comment.text("issue_id")
        comments[issue] = (*comments.get(issue, ()), Comment(comment.text("created_by"), comment.text("created_at"), comment.text("body")))
    return Backlog(tickets, inside, comments)


def _listed(issue: Payload) -> bool:
    """Whether lit lists the ticket: one it deleted or archived is in the export under a timestamp, and in no listing."""
    return issue.optional_text("deleted_at") is None and issue.optional_text("archived_at") is None


def _ticket(issue: Payload) -> Ticket:
    return Ticket(issue.text("id"), issue.text("title"), issue.text("description"), issue.text("issue_type"), issue.text("topic"), _status(issue.optional_text("status")))


def _status(status: str | None) -> Status | None:
    match status:
        case None | "open" | "in_progress" | "closed":
            return status
        case other:
            raise Rejected(f"a ticket's status should be open, in_progress, or closed, got {other!r}")
