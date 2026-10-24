"""What `hands file-tickets` files for each group of warnings and errors: a new ticket, a comment on the ticket that
already has it, or nothing, read off what the backlog already holds.

[LAW:one-source-of-truth] a group's tickets are where it stands: each carries its label, and each body the pass writes
on one, its description or a comment, ends on the line `Last seen: <at> (<label>)`. The newest of those lines is how far
the backlog has been told of the group, and only what the log holds after it is new. Nothing else keeps a cursor, so a
run over the same log files nothing twice, and a ticket a person closes into another, as a duplicate or superseded,
hands the group on to the ticket it was closed into.

Nothing here runs lit: it decides, and cli.py's file-tickets carries out what it decided.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime

from hands.sessions.backlog import Backlog, Ticket
from hands.sessions.grouping import Group

# How long a ticket's title is: its origin and cause, cut.
TITLE_CHARS = 120

_LAST_SEEN = re.compile(r"^Last seen: (?P<at>\S+) \((?P<label>audit-[0-9a-f]+)\)$", re.MULTILINE)


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
    told = max((seen for ticket in tickets for seen in _seen(group, ticket, backlog)), default=None)
    fresh = tuple(sighting for sighting in group.sightings if told is None or sighting.at > told)
    if not fresh:
        return None
    since = replace(group, sightings=fresh)
    working = next((ticket.id for ticket in tickets if ticket.status in ("open", "in_progress")), None)
    return New(since) if working is None else Again(working, since)


def _standing(group: Group, backlog: Backlog) -> tuple[Ticket, ...]:
    """The tickets that have the group, in rank order: those carrying its label, and each ticket one was closed into,
    as far down the chain of closings as it goes. A chain that comes back on itself ends at the first ticket met twice."""
    standing: dict[str, Ticket] = {}
    for ticket in (ticket for ticket in backlog.kept.values() if group.label in ticket.labels):
        while ticket.id not in standing:
            standing[ticket.id] = ticket
            if ticket.redirect_target is None or ticket.redirect_target not in backlog.kept:
                break
            ticket = backlog.kept[ticket.redirect_target]
    rank = {id: at for at, id in enumerate(backlog.kept)}
    return tuple(sorted(standing.values(), key=lambda ticket: rank[ticket.id]))


def _seen(group: Group, ticket: Ticket, backlog: Backlog) -> Iterable[datetime]:
    """Each time a body the pass wrote on `ticket` says the group was last seen."""
    bodies = (ticket.description, *(comment.body for comment in backlog.comments.get(ticket.id, ())))
    for body in bodies:
        for found in _LAST_SEEN.finditer(body):
            if found["label"] == group.label:
                yield datetime.fromisoformat(found["at"])


def title(group: Group) -> str:
    whole = f"{group.origin}: {group.cause}"
    return whole if len(whole) <= TITLE_CHARS else f"{whole[: TITLE_CHARS - 1]}…"


def description(group: Group) -> str:
    """A new ticket's description: what the group is, how often and when it was seen, and how to read its lines.

    [LAW:effects-at-boundaries] the lines stay in the audit log, on this machine: lit's tickets are pushed with the
    repository, and the log's lines hold what the user said and what each session did, which is the user's alone
    (audit.AuditLog keeps it so). A ticket carries only the group's origin and cause, whose ids, paths, and quoted
    text are normalised away, and the command that reads its lines where they are.
    """
    return f"""hands logged this {_times(group)}, from {group.first.stamp} to {group.last.stamp}, in its audit log. Filed by \
`hands file-tickets`, which groups the log's warnings and errors by cause: where each came from and what it said, with \
ids, paths, times, numbers, and quoted text normalised.

Origin: {group.origin}
Cause: {group.cause}

{_lines(group)}

{_last_seen(group)}"""


def comment(group: Group) -> str:
    """A comment on the ticket that has the group: how often and when it was seen since, and how to read its lines."""
    return f"""Seen again {_times(group)}, from {group.first.stamp} to {group.last.stamp}.

{_lines(group)}

{_last_seen(group)}"""


def _lines(group: Group) -> str:
    """The command that prints the first and newest of the group's lines from the audit log, which they never leave."""
    return f"""The first and newest of these lines, with their facts and traces, on the machine that logged them:

```sh
grep -h -e '"at": "{group.first.stamp}"' -e '"at": "{group.last.stamp}"' ~/.hands/audit/*.jsonl
```"""


def _times(group: Group) -> str:
    return "once" if len(group.sightings) == 1 else f"{len(group.sightings)} times"


def _last_seen(group: Group) -> str:
    return f"Last seen: {group.last.stamp} ({group.label})"
