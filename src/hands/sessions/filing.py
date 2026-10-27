"""What `hands file-tickets` files for each group of warnings and errors: a new ticket, a comment on the ticket that
already has it, or nothing, read off what the backlog already holds.

[LAW:one-source-of-truth] a group's tickets are where it stands: each carries its label, and each body the pass writes
on one, its description or a comment, ends on the line `Last seen: <at> (<label>)`. The newest of those lines, or of the
times those tickets were closed, is how far the backlog has been told of the group: a ticket closed answers for every
sighting before it. Only what the log holds after that is new. Nothing else keeps a cursor, so a
run over the same log files nothing twice, and a ticket a person closes into another, as a duplicate or superseded,
hands the group on to the ticket it was closed into.

Nothing here runs lit: it decides, and cli.py's file-tickets carries out what it decided.
"""

import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from hands.sessions.audit import SEGMENT_GLOB
from hands.sessions.backlog import Backlog, Ticket
from hands.sessions.grouping import Group

# How long a ticket's title is: where its group came from and its cause, cut.
TITLE_CHARS = 120

# [LAW:parse-dont-validate] the line exactly as _last_seen writes it, a Sighting's stamp to the millisecond with its
# offset, so a line a person wrote otherwise is no line of the pass's, and every one read is a time with a zone.
_LAST_SEEN = re.compile(
    r"^Last seen: (?P<at>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}) \((?P<label>audit-[0-9a-f]{12})\)$", re.MULTILINE
)


@dataclass(frozen=True)
class New:
    """A group no open ticket has: a ticket for it, holding the sightings the backlog has not been told of."""

    group: Group


@dataclass(frozen=True)
class Again:
    """A group an open ticket has, seen again: a comment on that ticket, of the sightings since it was last told."""

    ticket: str
    group: Group


Filing = New | Again


def filings(groups: Iterable[Group], backlog: Backlog) -> tuple[Filing, ...]:
    """What to file for each group, in the order the groups are given; a group with nothing new files nothing."""
    return tuple(filing for group in groups if (filing := _filing(group, backlog)) is not None)


def _filing(group: Group, backlog: Backlog) -> Filing | None:
    tickets = _standing(group, backlog)
    told = max((seen for ticket in tickets for seen in _told(group, ticket, backlog)), default=None)
    fresh = tuple(sighting for sighting in group.sightings if told is None or sighting.at > told)
    if not fresh:
        return None
    since = replace(group, sightings=fresh)
    working = next((ticket.id for ticket in tickets if ticket.status in ("open", "in_progress")), None)
    return New(since) if working is None else Again(working, since)


def _standing(group: Group, backlog: Backlog) -> tuple[Ticket, ...]:
    """The tickets that have the group, in rank order: those carrying its label, and each ticket one was closed into,
    as far down the chain of closings as it goes. A chain that comes back on itself ends at the first ticket met twice."""
    standing: set[str] = set()
    for ticket in (ticket for ticket in backlog.kept.values() if group.label in ticket.labels):
        while ticket.id not in standing:
            standing.add(ticket.id)
            if ticket.redirect_target is None or ticket.redirect_target not in backlog.kept:
                break
            ticket = backlog.kept[ticket.redirect_target]
    # The backlog keeps its tickets in rank order.
    return tuple(ticket for ticket in backlog.kept.values() if ticket.id in standing)


def _told(group: Group, ticket: Ticket, backlog: Backlog) -> Iterable[datetime]:
    """Each time `ticket` says the backlog was told of the group up to: when it was closed, and each time a body the
    pass wrote on it says the group was last seen."""
    if ticket.closed_at is not None:
        yield ticket.closed_at
    bodies = (ticket.description, *(comment.body for comment in backlog.comments.get(ticket.id, ())))
    for body in bodies:
        for found in _LAST_SEEN.finditer(body):
            if found["label"] == group.label:
                yield datetime.fromisoformat(found["at"])


def title(group: Group) -> str:
    """Where the group came from, as a ticket names it, and its cause, on one line, cut to TITLE_CHARS."""
    whole = " ".join(f"{group.named}: {group.cause}".split())
    return whole if len(whole) <= TITLE_CHARS else f"{whole[: TITLE_CHARS - 1]}…"


def where(directory: Path, home: Path) -> str:
    """The audit log's segments as a ticket's shell command names them: under ~ when they are in the user's `home`,
    whose path is theirs, and quoted so a space in it is no second word."""
    under = f"~/{shlex.quote(str(directory.relative_to(home)))}" if directory.is_relative_to(home) else shlex.quote(str(directory))
    return f"{under}/{SEGMENT_GLOB}"


def description(group: Group, logs: str) -> str:
    """A new ticket's description: what the group is, how often and when it was seen, and how to read its lines.

    [LAW:effects-at-boundaries] the lines stay in the audit log, on this machine: lit's tickets are pushed with the
    repository, and the log's lines hold what the user said and what each session did, which is the user's alone
    (audit.AuditLog keeps it so). A ticket carries only where the group came from, no finer than a file, and its cause,
    whose ids, paths, and quoted text are normalised away, and the command that reads its lines where they are, in
    `logs`.
    """
    return f"""hands logged this {_times(group)}, from {group.first.stamp} to {group.last.stamp}, in its audit log. Filed by \
`hands file-tickets`, which groups the log's warnings and errors by cause: where each came from and what it said, with \
ids, paths, times, numbers, and quoted text normalised.

Origin: {group.named}
Cause: {group.cause}

{_lines(group, logs)}

{_last_seen(group)}"""


def comment(group: Group, logs: str) -> str:
    """A comment on the ticket that has the group: how often and when it was seen since, and how to read its lines."""
    return f"""Seen again {_times(group)}, from {group.first.stamp} to {group.last.stamp}.

{_lines(group, logs)}

{_last_seen(group)}"""


def _lines(group: Group, logs: str) -> str:
    """The command that prints the first and newest of the group's lines from the audit log, which they never leave."""
    return f"""The first and newest of these lines, with their facts and traces, on the machine that logged them:

```sh
grep -h -e '"at": "{group.first.stamp}"' -e '"at": "{group.last.stamp}"' {logs}
```"""


def _times(group: Group) -> str:
    return "once" if len(group.sightings) == 1 else f"{len(group.sightings)} times"


def _last_seen(group: Group) -> str:
    return f"Last seen: {group.last.stamp} ({group.label})"
