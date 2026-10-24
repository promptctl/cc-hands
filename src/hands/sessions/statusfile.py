"""Claude Code's status of each live session, read from the file it keeps for the session, and heard each time it is set.

[LAW:parse-dont-validate] the one reader of that file: past parse_report, a status is a core.status.Report, and nothing
asks again whether the file said something this version knows.
"""

import asyncio
from collections.abc import Awaitable, Callable, Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from loguru import logger

from hands.core.events import StatusReported
from hands.core.session import Instant, Membership, Session, SessionId, status_stamp
from hands.core.status import Busy, Idle, Reason, Report, Shell, Stamp, Status, Unknown, UnknownReason, Waiting
from hands.sessions.payload import Payload, Rejected

# The waitingFor reasons this version knows, by the name the file gives them.
_REASONS: dict[str, Reason] = {reason: reason for reason in get_args(Reason)}


def status_file(membership: Membership) -> Path:
    """Where Claude Code keeps the session's status: `sessions/<pid>.json` in the config directory the session runs
    under, which holds its transcript at `projects/<project>/<session>.jsonl` (2.1.282). Found from the transcript, not
    from this process's CLAUDE_CONFIG_DIR, because each session may have been started under its own."""
    if len(membership.transcript.parents) < 3:
        raise Rejected(f"the transcript {membership.transcript} is not in a config directory's projects")
    return membership.transcript.parents[2] / "sessions" / f"{membership.pid}.json"


def parse_report(membership: Membership, raw: bytes) -> Report | None:
    """The status the file sets for the member's session, None while it sets none yet, refused if the file speaks for
    another process or session. Claude Code writes the file as the session starts and sets a status in it a moment later
    (2.1.299), and its own reader passes over a session with none: until then there is nothing set to hear."""
    record = Payload.parse(raw)
    # A pid is handed out again, and a process that /clears or resumes holds a new session: the file is only this
    # session's while it names both.
    if (pid := record.integer("pid")) != membership.pid:
        raise Rejected(f"the status file names pid {pid}, not the session's {membership.pid}")
    if (held := record.text("sessionId")) != membership.id:
        raise Rejected(f"the status file of pid {pid} is for session {held} now")
    match record.optional_text("status"):
        case None:
            return None
        case name:
            return Report(_status(record, name), Stamp(record.integer("statusUpdatedAt")))


def _status(record: Payload, name: str) -> Status:
    match name:
        case "idle":
            return Idle()
        case "busy":
            return Busy()
        case "shell":
            return Shell()
        case "waiting":
            reason = record.text("waitingFor")
            return Waiting(_REASONS.get(reason) or UnknownReason(reason))
        case other:
            return Unknown(other)


@dataclass(frozen=True)
class _Unread:
    """Why a session's status could not be read, and how many reads in a row have found it so."""

    why: str
    reads: int


class Statuses:
    """Why each live session's status could not be read, so a reason is said once it outlives a read, not every read."""

    def __init__(self, clock: Callable[[], Instant]) -> None:
        # [LAW:effects-at-boundaries] the registry's one clock, as a hook is stamped from when it arrives.
        self._clock = clock
        self._unread: dict[SessionId, _Unread] = {}

    def read(self, live: Collection[SessionId], session: Callable[[SessionId], Session | None]) -> Iterator[StatusReported]:
        """A report for each live session whose status was set since the one the registry holds.

        [LAW:no-ambient-temporal-coupling] each file is read only as its report is asked for, from the registry as it
        stands then, so one applied at once is the status the session has now: no hook can be applied between the read
        and the report, as one could while an earlier session's report is applied.
        """
        self._unread = {id: unread for id in live if (unread := self._unread.get(id)) is not None}
        for id in live:
            match session(id):
                case Session() as now if (reported := self._read(now)) is not None:
                    yield reported
                case _:
                    # Ended while the one before was applied, or with nothing set since.
                    pass

    def _read(self, session: Session) -> StatusReported | None:
        member = session.membership
        try:
            path = status_file(member)
            report = parse_report(member, path.read_bytes())
        except FileNotFoundError:
            if status_stamp(session.state) is not None:
                # Claude Code removes the file it kept as its process exits, then runs the end hook, which lands 330 to
                # 640 ms later (2.1.283); a process killed before its hook runs is found dead by the sweep. Only a file
                # never read is a status hands cannot have.
                return None
            return self._unreadable(member, f"Claude Code keeps no status for it at {status_file(member)}")
        except (Rejected, OSError) as error:
            return self._unreadable(member, f"its status file is refused: {error}")
        self._unread.pop(member.id, None)
        match report:
            case None:
                return None
            # [LAW:one-source-of-truth] edge-triggered on the stamp the registry holds, not a copy kept here: a status the
            # registry let go of is heard again. [LAW:no-ambient-temporal-coupling] the stamp, not the status, so an idle,
            # busy, idle between two reads is still a status set, and still heard.
            case Report(stamp=stamp) if stamp == status_stamp(session.state):
                return None
            case Report():
                _unknown(member, report.status)
                return StatusReported(member.id, report, self._clock())

    def _unreadable(self, member: Membership, why: str) -> None:
        # [LAW:no-silent-failure] said as an error, since the status is what ends a turn no Stop ends: without it a turn
        # stopped at the keyboard runs on until one does. Said on the second read in a row to find it, not the first: a
        # file caught half rewritten is whole by the next read, and costs nothing. Said once until the reason changes.
        held = self._unread.get(member.id)
        unread = _Unread(why, held.reads + 1 if held is not None and held.why == why else 1)
        self._unread[member.id] = unread
        if unread.reads == 2:
            logger.error(f"no status for session {member.id}: {why}, so its transcript is read only when a Stop tells a turn, and a turn stopped with Escape or gone on under a queued message is not heard")


def _unknown(member: Membership, status: Status) -> None:
    # [LAW:no-silent-failure] a status this version does not know is passed on as unknown and said, never taken for another.
    match status:
        case Unknown(name=name) | Waiting(reason=UnknownReason(name=name)):
            logger.warning(f"session {member.id} reports a status hands does not know: {name!r}")
        case Idle() | Busy() | Waiting() | Shell():
            pass


async def keep_reading_statuses(
    sessions: Callable[[], Collection[SessionId]],
    session: Callable[[SessionId], Session | None],
    clock: Callable[[], Instant],
    period: float,
    apply: Callable[[StatusReported], Awaitable[None]],
) -> None:
    """Read every live session's status once a period, and apply each one set since, until cancelled.

    The period is how late a session going idle is heard.
    """
    statuses = Statuses(clock)
    while True:
        for reported in statuses.read(sessions(), session):
            await apply(reported)
        await asyncio.sleep(period)
