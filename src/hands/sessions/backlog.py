"""A project's lit backlog, read whole from `lit export` and parsed once into the tickets, their tree, and their comments.

`lit export` is lit's versioned data-export primitive, and the one command that hands over descriptions in bulk.
It is read fresh at every call and kept nowhere [LAW:one-source-of-truth]: lit is where the backlog is, and its rank
order is changed only by the lit writes here: `lit rank`, and the `lit new` and `lit parent set` that place a ticket.
"""

import asyncio
import errno
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

from hands.core.summaries import Thing
from hands.core.trace import Span
from hands.sessions import child, wide
from hands.sessions.child import Children
from hands.sessions.lines import newest_first
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


@dataclass(frozen=True)
class Lit:
    """lit as hands runs it: started by `children`, in a copy of `environment` taken as it is made, so a later change to
    the process's own environment reaches no lit run. [LAW:one-source-of-truth] every lit run for hands is run in it,
    as its signer has it (`Signer.runs_in`)."""

    children: Children
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True)
class Person:
    """A person a write is lit's for: the name lit records the write under, on a command that takes `--by`, and on
    every command, the environment it runs in for them."""

    name: str

    @property
    def signature(self) -> tuple[str, ...]:
        return ("--by", self.name)

    def runs_in(self, environment: Mapping[str, str]) -> Mapping[str, str]:
        """`environment` without the Claude Code session hands may have been started inside, which lit would take for
        who acts before any `--by`."""
        return {name: value for name, value in environment.items() if name != "CLAUDE_CODE_SESSION_ID"}


@dataclass(frozen=True)
class Inherited:
    """Whoever the environment hands runs in says acts: inside a Claude Code session, that session. A read runs as it,
    and so does a write hands makes of its own accord."""

    @property
    def signature(self) -> tuple[str, ...]:
        return ()

    def runs_in(self, environment: Mapping[str, str]) -> Mapping[str, str]:
        return environment


# Who a lit run is made as.
Signer = Person | Inherited


@dataclass(frozen=True)
class Reasoned:
    """lit refused, on `reason`, the token its line in its events file names for the refusal (links docs/cli-reference.md,
    "Every reason")."""

    reason: str


@dataclass(frozen=True)
class Unexplained:
    """lit exited non-zero, and hands could not read a reason for it off lit's line, and why: lit keeps the line only as
    it can (links docs/events.md), so a lit that could not write it, or a file hands cannot read, leaves a refusal of no
    reason."""

    why: str


@dataclass(frozen=True)
class Unexited:
    """lit did not end on an exit of its own: it was killed past its limit or by a signal, wrote past what hands reads,
    or was never reaped."""


@dataclass(frozen=True)
class NoTicketId:
    """lit exited 0 on a `lit new` and said no ticket's id."""


@dataclass(frozen=True)
class ProjectGone:
    """The directory lit was to run in no longer exists (a worktree cleaned up): lit never started there."""


@dataclass(frozen=True)
class Unlaunched:
    """The OS could not start lit's program in the directory; the message says what the OS said."""


# [LAW:types-are-the-program] how lit failed, closed, and [LAW:one-source-of-truth] the one discriminator of it: a
# caller tells each failure from the others by matching it, never by reading the message. The last two are lit never
# run, so nothing was written.
Failure = Reasoned | Unexplained | Unexited | NoTicketId | ProjectGone | Unlaunched


class LitFailed(Exception):
    """lit did not do what it was asked in a directory: the message says why, and `failure` is how."""

    def __init__(self, message: str, failure: Failure) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True)
class Untracked:
    """A directory lit was never set up in, or that is in no git repository: it has no backlog, which is an answer, not
    a failure. `gone` says the directory itself was removed (a worktree cleaned up), so there is nothing to run lit in."""

    project: Path
    gone: bool = False

    @property
    def why(self) -> str:
        """[LAW:one-source-of-truth] why there is no backlog, as every reader says it."""
        return "the directory no longer exists" if self.gone else "lit has no workspace there"


# A failed lit command's reason is the token lit's published contract (links docs/cli-reference.md) tells apart the
# failures one exit code covers by. Two say it has no workspace here: `lit init` never ran in this repository, or this
# is no git repository.
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
    labels: frozenset[str]
    # The ticket lit closed this one into, as a duplicate of it or superseded by it; None for one it closed into none.
    redirect_target: str | None
    # When lit closed it; None for a ticket open or in progress, and for an epic.
    closed_at: datetime | None

    @property
    def text(self) -> str:
        """What the ticket says, and so what its summary is made of and keyed by."""
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
    # Every ticket lit keeps, listed or archived: an archived ticket still says what it said, and what it was closed into.
    kept: Mapping[str, Ticket]
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
        A press never makes a second epic for a level lit lists in the frame: one whose tickets are all closed is used
        again. One lit archived is put away and not listed, so a press makes another epic for the level.
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
        # A closed ticket is in no epic's open tickets, so it could never be ranked last among them.
        if self.done(ticket):
            raise Rejected(f"{ticket} is closed, and a closed ticket is put at no level")
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


async def read_backlog(lit: Lit, project: Path) -> Backlog | Untracked:
    """The backlog `lit` holds for `project`, or that it has none (no lit workspace, or the directory is gone); raises
    LitFailed when lit cannot say, and Rejected when what it said does not parse."""
    try:
        out = await _lit(lit, project, "export", runs_as=Inherited())
    except LitFailed as failed:
        match failed.failure:
            case ProjectGone():
                return Untracked(project, gone=True)
            case Reasoned(reason=reason) if reason in NO_WORKSPACE:
                return Untracked(project)
            case _:
                raise
    # Off the loop: the largest export takes tens of milliseconds to parse, which the voice pipeline would hear.
    return await asyncio.to_thread(parse_export, out)


async def rank(lit: Lit, project: Path, move: Move, runs_as: Signer) -> str:
    """Have lit in `project` make `move` in the environment it runs in for `runs_as`, unsigned, and what lit said of what
    it ranked; raises LitFailed when it fails. lit records no one for a rank, and takes no `--by` for it.

    lit's words are the one account of a rank it made otherwise than asked: it says it ranked the parent instead.
    """
    return await _said(lit, project, "rank", move.ticket, f"--{move.place.side}", move.place.target, runs_as=runs_as)


async def make_epic(lit: Lit, project: Path, new: NewEpic, by: Signer) -> str:
    """Have lit in `project` make `new`, at the bottom of its frame, signed `by`, and what lit said; raises LitFailed when it fails. It is found after by its frame, title, and topic, as every level's epic is (`Backlog.level_epics`)."""
    return await _signed(lit, project, "new", "--type", "epic", "--topic", new.topic, "--title", new.title, *(() if new.parent is None else ("--parent", new.parent)), by=by)


async def adopt(lit: Lit, project: Path, ticket: str, epic: str, by: Signer) -> str:
    """Have lit in `project` move `ticket` inside `epic`, keeping its id, signed `by`, and what lit said; raises LitFailed when it fails."""
    return await _signed(lit, project, "parent", "set", "--child", ticket, "--parent", epic, by=by)


async def file_ticket(lit: Lit, project: Path, epic: str, topic: str, title: str, description: str, label: str, by: Signer) -> str:
    """Have lit in `project` file a bug under `epic`, at the top of it, carrying `label`, signed `by`; the id of the
    ticket it filed, the first word lit says. Raises LitFailed when it fails, or says no id."""
    filed = await _signed(lit, project, "new", "--type", "bug", "--parent", epic, "--topic", topic, "--title", title, "--labels", label, "--top", "--description", description, by=by)
    said = filed.split()
    if not said:
        raise LitFailed(f"lit new in {project} exited 0 and said no ticket's id", NoTicketId())
    return said[0]


async def comment(lit: Lit, project: Path, ticket: str, body: str, by: Signer) -> str:
    """Have lit in `project` add a comment to `ticket`, signed `by`, and what lit said; raises LitFailed when it fails."""
    return await _signed(lit, project, "comment", "add", ticket, "--body", body, by=by)


async def _signed(lit: Lit, project: Path, *arguments: str, by: Signer) -> str:
    """What lit said, as `_said`, of a write it records as `by`'s: signed with `by`'s name, in their environment."""
    return await _said(lit, project, *arguments, *by.signature, runs_as=by)


async def _said(lit: Lit, project: Path, *arguments: str, runs_as: Signer) -> str:
    """What lit run as `_lit` runs it said, as text: a write's own account of what it wrote."""
    return (await _lit(lit, project, *arguments, runs_as=runs_as)).decode(errors="replace").strip()


async def _lit(lit: Lit, project: Path, *arguments: str, runs_as: Signer) -> bytes:
    """What `lit` run in `project` with `arguments`, in the environment it runs in for `runs_as`, wrote out; raises
    LitFailed when it never ran, did not answer, exited non-zero, or failed once it was running.

    [LAW:single-enforcer] the one place hands runs lit, under one bound, and reads its failure.
    """
    command = f"lit {arguments[0]}"
    # lit runs as the span of its own `child.run` event, inside the work it is run for (links docs/events.md,
    # TRACEPARENT): its line in its events file, which holds a refusal's reason, names that span as its parent.
    traced = child.Traced(runs_as.runs_in(lit.environment), wide.begun(lit.children.clock))
    ran = await lit.children.run("lit", *arguments, limit=LIT_TIMEOUT_SECONDS, cwd=project, env=traced)
    match ran:
        case child.Ran(ending=child.Exited(code=0), out=out):
            return out.data
        case child.Ran(ending=child.Exited()):
            # Off the loop: the events file grows to 8 MiB before lit rotates it.
            match await asyncio.to_thread(_recorded, traced.began.span, events_file(traced.environment, project)):
                case Reasoned(reason=reason) as reasoned:
                    raise LitFailed(f"{command} in {project} {child.said(ran)} ({reason})", reasoned)
                case Unexplained(why=why) as unexplained:
                    raise LitFailed(f"{command} in {project} {child.said(ran)}; {why}", unexplained)
        case child.Ran() | child.Unreaped() | child.Lost():
            raise LitFailed(f"{command} in {project} {child.said(ran)}", Unexited())
        case child.NeverStarted(error=error):
            # The start names the path it could not use: the project directory as given, where the chdir failed, not lit.
            if error.errno == errno.ENOENT and str(error.filename) == str(project):
                raise LitFailed(f"{project} no longer exists", ProjectGone()) from error
            raise LitFailed(f"lit could not be started in {project}: {error}", Unlaunched()) from error


def events_file(environment: Mapping[str, str], cwd: Path) -> Path | Unexplained:
    """lit's events file for a run in `cwd` in `environment`, where lit keeps it (links internal/config/config.go
    StateDir): under a non-empty XDG_STATE_HOME, else under a non-empty HOME's .local/state, each taken from the
    directory lit runs in where it is relative. With neither, lit keeps no line, and says so on its stderr.

    [LAW:one-source-of-truth] hands' one copy of lit's rule, which the fakes of lit write by too, and the contract in
    tests/litcontract.py holds to lit."""
    match environment.get("XDG_STATE_HOME", ""), environment.get("HOME", ""):
        case "", "":
            return Unexplained("lit had no XDG_STATE_HOME or HOME to keep its events line under")
        case "", str(home):
            state = cwd / home / ".local" / "state"
        case str(xdg), _:
            state = cwd / xdg
    return state / "links-issue-tracker" / "events.jsonl"


def rotated(events: Path) -> Path:
    """Where lit moves its events file past 8 MiB (links docs/events.md), so a run's line may be found there since."""
    return events.with_name(f"{events.name}.1")


def _recorded(span: Span, events: Path | Unexplained) -> Reasoned | Unexplained:
    """The reason lit recorded for its run as `span`, on that run's line of its events file, or why hands has none.

    lit rotates the file to `.1` past 8 MiB, which a run since may have done, so both are searched, the newest line
    first, and a file that cannot be read stops no search of the other. A worker lit spawns has a line in the trace
    too, under the run's own span, so the run's line is the one whose parent is `span`. A line that does not parse,
    or whose fields are not lit's, cannot be shown to be the run's, so it is passed over, and counted.
    """
    if isinstance(events, Unexplained):
        return events
    looked: list[str] = []
    for path in (events, rotated(events)):
        torn = 0
        try:
            for line in newest_first(path):
                if span.span_id not in line:
                    continue
                try:
                    event = Payload.parse(line.encode())
                    run_s = (event.optional_text("trace_id"), event.optional_text("parent_span_id")) == (span.trace_id, span.span_id)
                    reason = event.optional_text("reason") if run_s else None
                except Rejected:
                    torn += 1
                    continue
                if run_s:
                    return Reasoned(reason) if reason else Unexplained(f"lit's line for {span.traceparent} in {path} names no reason")
        except OSError as error:
            looked.append(f"{path} could not be read: {error}")
            continue
        looked.append(f"{path} holds no line for it" + (f", past {torn} it could not read as lit's" if torn else ""))
    return Unexplained(f"lit wrote no line hands could read for {span.traceparent}: {'; '.join(looked)}")


def parse_export(raw: bytes) -> Backlog:
    """`lit export`'s JSON, parsed once [LAW:parse-dont-validate]: each ticket, its place in the tree, and what was said on it."""
    export = Payload.parse(raw)
    if (version := export.integer("version")) not in EXPORT_VERSIONS:
        raise Rejected(f"lit export is version {version}, and hands reads versions {', '.join(map(str, EXPORT_VERSIONS))}")
    every = [Payload.of(entry, "an issue") for entry in export.items("issues")]
    labels: dict[str, frozenset[str]] = {}
    for label in (Payload.of(entry, "a label") for entry in export.items("labels")):
        labels[label.text("issue_id")] = labels.get(label.text("issue_id"), frozenset()) | {label.text("name")}
    issues = sorted((issue for issue in every if issue.optional_text("deleted_at") is None), key=lambda issue: issue.text("rank"))
    kept = {issue.text("id"): _ticket(issue, labels.get(issue.text("id"), frozenset())) for issue in issues}
    tickets = {issue.text("id"): kept[issue.text("id")] for issue in issues if _listed(issue)}
    # An archived parent still frames its children's rank; a deleted or purged one frames nothing.
    framing = tickets.keys() | {issue.text("id") for issue in every if issue.optional_text("deleted_at") is None and issue.optional_text("archived_at") is not None}
    inside: dict[str, str] = {}
    for relation in (Payload.of(entry, "a relation") for entry in export.items("relations")):
        if relation.text("type") == "parent-child" and relation.text("src_id") in tickets and relation.text("dst_id") in framing:
            inside[relation.text("src_id")] = relation.text("dst_id")
    comments: dict[str, tuple[Comment, ...]] = {}
    # A comment on a ticket lit no longer keeps is read by nothing, and kept by nothing.
    said = (comment for comment in (Payload.of(entry, "a comment") for entry in export.items("comments")) if comment.text("issue_id") in kept)
    for comment in sorted(said, key=lambda comment: comment.text("created_at")):
        issue = comment.text("issue_id")
        comments[issue] = (*comments.get(issue, ()), Comment(comment.text("created_by"), comment.text("created_at"), comment.text("body")))
    return Backlog(tickets, kept, inside, comments)


def _listed(issue: Payload) -> bool:
    """Whether lit lists the ticket: one it deleted or archived is in the export under a timestamp, and in no listing."""
    return issue.optional_text("deleted_at") is None and issue.optional_text("archived_at") is None


def _ticket(issue: Payload, labels: frozenset[str]) -> Ticket:
    status = _status(issue.optional_text("status"))
    return Ticket(
        issue.text("id"), issue.text("title"), issue.text("description"), issue.text("issue_type"), issue.text("topic"),
        status, labels, issue.optional_text("redirect_target"), _closed_at(issue) if status == "closed" else None,
    )


def _closed_at(issue: Payload) -> datetime:
    """[LAW:parse-dont-validate] when lit closed a closed ticket, which lit exports for every one."""
    try:
        return datetime.fromisoformat(issue.text("closed_at"))
    except ValueError as error:
        raise Rejected(f"a closed ticket's closed_at should be a time: {error}") from None


def _status(status: str | None) -> Status | None:
    match status:
        case None | "open" | "in_progress" | "closed":
            return status
        case other:
            raise Rejected(f"a ticket's status should be open, in_progress, or closed, got {other!r}")
