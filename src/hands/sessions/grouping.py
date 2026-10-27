"""The warnings and errors in the audit log, grouped by cause: for `hands file-tickets` to file a ticket for each.

[LAW:domain-language] in error tracking's terms, as Sentry gave them: occurrences of one cause are one group, known by
its fingerprint. An occurrence is a sighting here, since the codebase spends that word already (hands.core.occurrences).
The cause is where the line came from (the unit of work, the module and function that logged it, the request) and what
it said, with what differs between sightings of one cause (ids, paths, times, numbers, quoted text) normalised away.

Nothing here reads a file or runs lit: it folds the log's lines, oldest first, into groups.
"""

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import get_args

from hands.core.wire import Exchanged
from hands.sessions.audit import Exported, Logged, Refocused, SettingsEdited, Severity, TypingFailed
from hands.sessions.payload import Payload, Rejected
from hands.sessions.wide import WideEvent, exited

# The levels of the lines a group is made of: every other line is the log telling what went right.
_FILED = get_args(Severity)

# Text in quotes, a byte literal's among them, opened where no word runs into the quote, so "session's" opens none; a
# quote never closed, as a cut excerpt's is, runs to the end.
_QUOTED = re.compile(r"""(?<!\w)b?'(?:[^'\\]|\\.)*(?:'|$)|(?<!\w)b?"(?:[^"\\]|\\.)*(?:"|$)""")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
# A path of two parts or more, so a slash command said in a message (/login) is a word and not a path.
_PATH = re.compile(r"""(?<![\w.])(?:~|/)[^\s'",:;()/]+(?:/[^\s'",:;()]*)+""")
# A word of eight characters or more holding a digit and a letter: a uuid, a hex digest, a cse_ or toolu_ id.
_ID = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[\w-]{8,}\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalised(text: str) -> str:
    """text with what differs between sightings of one cause put as what it is: <text>, <time>, <path>, <id>, <n>."""
    return _NUMBER.sub("<n>", _ID.sub("<id>", _PATH.sub("<path>", _TIME.sub("<time>", _QUOTED.sub("<text>", text)))))


def _ids(path: str) -> str:
    """A request's path with its ids normalised, and nothing else, so the endpoint it asked stays named."""
    return _NUMBER.sub("<n>", _ID.sub("<id>", path))


@dataclass(frozen=True)
class Sighting:
    """One warning or error line: where it came from, what it said, normalised, and when it was written."""

    origin: str
    cause: str
    at: datetime

    @property
    def stamp(self) -> str:
        """When it was written, as the log writes it, so its line is found by it."""
        return self.at.isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Group:
    """Every sighting of one cause, oldest first."""

    origin: str
    cause: str
    sightings: tuple[Sighting, ...]

    @property
    def fingerprint(self) -> str:
        """What the group is known by from one run to the next: a digest of its origin and cause."""
        return hashlib.sha256(f"{self.origin}\n{self.cause}".encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        """The lit label its tickets carry. lit keeps labels lowercase, which a hex digest is."""
        return f"audit-{self.fingerprint}"

    @property
    def first(self) -> Sighting:
        return self.sightings[0]

    @property
    def last(self) -> Sighting:
        return self.sightings[-1]


@dataclass(frozen=True)
class Read:
    """The log as groups, and what reading it took: its lines, those that were not JSON objects, the warning and error
    lines whose fields this hands cannot read, the failed units that told again a failure of one inside them, and the
    first and last time read, None for a log with no line."""

    groups: tuple[Group, ...]
    lines: int
    unreadable: int
    unparsed: int
    retold: int
    oldest: datetime | None
    newest: datetime | None


Cause = Callable[[Payload], tuple[str, str]]


def _logged(line: Payload) -> tuple[str, str]:
    return line.text("source"), line.text("message")


def _event(line: Payload) -> tuple[str, str]:
    return line.text("event"), line.optional_text("error") or "failed, and said no error"


def _exchanged(line: Payload) -> tuple[str, str]:
    reply = Payload.of(line.mapping("reply"), "a reply")
    origin = f"{line.text('method')} {_ids(line.text('path'))}"
    match reply.text("type"):
        case "Unreached":
            return origin, f"unreached: {reply.text('error')}"
        case "Uncopied" | "Unfinished" as kind:
            return origin, f"{kind.lower()}: {reply.text('reason')}"
        case "Reached":
            body = Payload.of(reply.mapping("body"), "a reply's body")
            garbled = f": {body.text('reason')}" if body.text("type") == "Garbled" else ""
            return origin, f"answered {reply.integer('status')}, {body.text('type').lower()}{garbled}"
        case other:
            raise Rejected(f"a reply of a kind this hands does not write: {other!r}")


def _exported(line: Payload) -> tuple[str, str]:
    return f"{line.text('collector')} {line.text('signal')}", line.optional_text("error") or line.text("warning")


def _said(field: str) -> Cause:
    """The cause of a line that names no origin of its own but its kind, and says what went wrong in `field`."""
    return lambda line: (line.text("type"), line.text(field))


# [LAW:one-source-of-truth] each kind of line the log judges a warning or an error, by its type's own name, and where it
# says what went wrong; a test holds this to every such kind in audit.Entry.
CAUSES: Mapping[str, Cause] = {
    Logged.__name__: _logged,
    # A Logged line's name before it held warnings, kept while the log can still hold one: promptctl-warnings-2d80.9i6
    # deletes it once none is left.
    "Failure": _logged,
    WideEvent.__name__: _event,
    Exchanged.__name__: _exchanged,
    Exported.__name__: _exported,
    TypingFailed.__name__: _said("reason"),
    Refocused.__name__: _said("failed"),
    SettingsEdited.__name__: _said("refused"),
}


def _cause(kind: str, line: Payload) -> tuple[str, str, bool]:
    """Where a warning or error line came from and what it said, normalised, and whether it read as its kind is written.

    [LAW:parse-dont-validate] a line this hands cannot read, of a kind it no longer writes or written otherwise by the
    build before, is still a sighting, grouped by its kind and why it could not be read: never one that stops the
    pass, and never one lost.
    """
    try:
        origin, said = CAUSES[kind](line)
    except KeyError:
        return kind, f"a {kind} line, a kind this hands does not write", False
    except Rejected as rejected:
        return kind, f"a {kind} line this hands cannot read: {normalised(str(rejected))}", False
    return origin, normalised(said), True


def _retold(kind: str, line: Payload, failed_inside: dict[tuple[str, str], set[str]]) -> bool:
    """Whether a unit's line tells again how a unit inside it failed, which that unit's own line told first: one cause,
    whose sighting is the innermost unit's. It tells it again when its error is the inner one's, as an exception raised
    on through both is, or is a command's exit for it, as `wide.exited` says one; any other error is a sighting of its
    own. A unit's line is written as it ends, so a unit inside another is read first, and notes its error in
    `failed_inside` under the unit around it, by trace and span.
    """
    if kind != WideEvent.__name__:
        return False
    try:
        trace, span, parent, error = line.text("trace_id"), line.text("span_id"), line.optional_text("parent_id"), line.optional_text("error")
        code = line.mapping("facts").get("exit_code")
    except Rejected:
        # A span this hands cannot read places the line inside nothing: it is a sighting, read by its cause as any is.
        return False
    if error is None:
        return False
    if parent is not None:
        failed_inside.setdefault((trace, parent), set()).add(error)
    inner = failed_inside.get((trace, span), set())
    commanded: set[str] = {exited(code, why) for why in inner} if isinstance(code, int) else set()
    return error in inner | commanded


def read(lines: Iterable[str]) -> Read:
    """The warning and error lines among `lines`, oldest first, grouped by cause, the groups in the order each began."""
    sightings: dict[tuple[str, str], list[Sighting]] = {}
    failed_inside: dict[tuple[str, str], set[str]] = {}
    count = unreadable = unparsed = retold = 0
    oldest: datetime | None = None
    newest: datetime | None = None
    for text in lines:
        count += 1
        try:
            line = Payload.parse(text.encode())
            at = datetime.fromisoformat(line.text("at"))
            level, kind = line.text("level"), line.text("type")
        except (Rejected, ValueError):
            # A torn line, as a write cut short leaves one, or one with no time, level, or kind to place it by.
            unreadable += 1
            continue
        oldest = at if oldest is None else oldest
        newest = at
        if level not in _FILED:
            continue
        if _retold(kind, line, failed_inside):
            retold += 1
            continue
        origin, cause, parsed = _cause(kind, line)
        unparsed += 0 if parsed else 1
        sightings.setdefault((origin, cause), []).append(Sighting(origin, cause, at))
    groups = tuple(Group(origin, cause, tuple(seen)) for (origin, cause), seen in sightings.items())
    return Read(groups, count, unreadable, unparsed, retold, oldest, newest)

