"""The membership file: written by the shim at SessionStart, or at the first hook of a session that never fired one, and
its transcript rewritten by the hook that names where Claude Code moved it, as the session enters a worktree;
read by the daemon.

The file's name is the session id, so the id is not repeated inside it.
"""

import json
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from hands.core.session import Membership, SessionId
from hands.sessions.home import Home
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import parse_pid, process_starts, still_running


def write_membership(home: Home, membership: Membership) -> None:
    path = home.membership(membership.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "pid": membership.pid,
            "cwd": str(membership.cwd),
            "transcript_path": str(membership.transcript),
            # Written as null rather than left out, so a file from a session nobody
            # wrapped and a file written before this field existed read the same way.
            "fritter_socket": None if membership.fritter is None else str(membership.fritter),
        }
    )
    # [LAW:no-ambient-temporal-coupling] written beside and renamed into place,
    # so the daemon reading it never sees half a file. Named for the writing shim:
    # a session's first hooks can run at once, parallel tool calls each firing one.
    staging = path.with_suffix(f".{os.getpid()}.tmp")
    staging.write_text(body)
    staging.replace(path)


def remove_membership(home: Home, session: SessionId) -> None:
    # A session that started before the hooks were installed never had a file to remove.
    home.membership(session).unlink(missing_ok=True)


def remove_ended_membership(home: Home, ended: Membership) -> None:
    """Remove the file of a session the sweep found over, unless a new process has since started the session again."""
    try:
        current = read_membership(home, ended.id)
    except Rejected:
        # Already removed by the session's end, or unreadable, which the next sweep reports.
        return
    if current.pid == ended.pid:
        home.membership(ended.id).unlink(missing_ok=True)


@dataclass(frozen=True)
class Recorded:
    membership: Membership
    written_at: float  # wall-clock seconds: the file's modification time


def holders(records: Collection[Recorded], started: Mapping[int, float]) -> dict[int, Membership]:
    """The session each running process holds, by pid: of the files naming it, the newest.

    [LAW:single-enforcer] the one rule for which session a process holds, which the sweep judges by and the shim writes by.
    """
    alive = [record for record in records if running(record, started)]
    return {record.membership.pid: record.membership for record in sorted(alive, key=lambda record: record.written_at)}


def running(record: Recorded, started: Mapping[int, float]) -> bool:
    """Whether the file's own process runs: a running pid is not enough, it must have been running when the file was written."""
    return still_running(record.membership.pid, record.written_at, started)


def holder(home: Home, pid: int) -> Membership | None:
    """The session the process holds, as its files say; None when no file written while it ran names it."""
    return holders(*naming(home, pid)).get(pid)


def naming(home: Home, pid: int) -> tuple[list[Recorded], Mapping[int, float]]:
    """The files naming the process, and when it started: all that says which session it holds."""
    # An unreadable file, which the daemon's sweep reports and removes, names no process.
    records, _ = recorded(home)
    return [record for record in records if record.membership.pid == pid], process_starts({pid})


@dataclass(frozen=True)
class Unreadable:
    """A membership file that does not parse, as it was read: it names no session hands can reach."""

    path: Path
    raw: bytes
    error: Rejected


def recorded(home: Home) -> tuple[list[Recorded], list[Unreadable]]:
    """Every membership file that parses, and every one that does not; nothing is removed here."""
    records: list[Recorded] = []
    unreadable: list[Unreadable] = []
    # A shim's staging file ends in .tmp until it is renamed into place.
    for path in sorted(home.memberships.glob("*.json")):
        try:
            written_at = path.stat().st_mtime
            raw = path.read_bytes()
        except FileNotFoundError:
            # The session ended between the listing and the read.
            continue
        try:
            records.append(Recorded(parse_membership(SessionId(path.stem), raw), written_at))
        except Rejected as error:
            unreadable.append(Unreadable(path, raw, error))
    return records, unreadable


def recorded_membership(home: Home, session: SessionId) -> Membership | None:
    """The session's membership, or None when it has no file: it ended, or its process holds another session now."""
    try:
        raw = home.membership(session).read_bytes()
    except FileNotFoundError:
        return None
    return parse_membership(session, raw)


def read_membership(home: Home, session: SessionId) -> Membership:
    match recorded_membership(home, session):
        case None:
            raise Rejected(f"no membership file for session {session} at {home.membership(session)}")
        case membership:
            return membership


def parse_membership(session: SessionId, raw: bytes) -> Membership:
    record = Payload.parse(raw)
    # [LAW:parse-dont-validate] Missing and empty both mean this session was not wrapped.
    # Path("") is PosixPath("."), and a Typist made from that would dial a directory.
    address = record.optional_text("fritter_socket")
    return Membership(
        id=session,
        pid=parse_pid(record.integer("pid")),
        cwd=Path(record.text("cwd")),
        transcript=Path(record.text("transcript_path")),
        fritter=Path(address) if address else None,
    )
