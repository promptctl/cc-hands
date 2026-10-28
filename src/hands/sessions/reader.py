"""What Claude Code says of every live session outside its hooks, read for the one model of a Claude Code turn: its
status, then its transcript, once a period, and through now for whoever must decide on what it has said by then."""

import asyncio
from collections.abc import Awaitable, Callable

from hands.core.events import StatusReported, Transcribed
from hands.sessions.registry import Sessions
from hands.sessions.statusfile import Statuses
from hands.sessions.tail import Tails

# How late Claude Code setting a session's status is heard, and how late a record it has written becomes a step of the
# turn it belongs to: what a turn narrated while it runs waits on, and a turn stopped at the keyboard. A Stop reads the
# rest of its own transcript before telling the turn, so a Stop waits on neither. The status file is a few hundred bytes a
# session.
READ_SECONDS = 0.1


class Reader:
    """[LAW:single-enforcer] the one reader of Claude Code's statuses and transcripts for a registry: every read of them,
    on the period or through now, is this one's, so no two are under way at once and each is applied in the order read."""

    def __init__(self, sessions: Sessions, tails: Tails, apply: Callable[[StatusReported | Transcribed], Awaitable[None]]) -> None:
        self._sessions = sessions
        self._tails = tails
        self._apply = apply
        self._statuses = Statuses(sessions.now)
        self._reading = asyncio.Lock()

    async def read(self) -> None:
        """Every live session's status, then its transcript, read through now, and what each says applied as it is read.

        [LAW:no-ambient-temporal-coupling] the status first: a transcript is read only once a status says whether a turn
        runs (Tails.catch_up), so the turn it is in when hands begins following it lands on a status that decides that.
        """
        async with self._reading:
            for reported in self._statuses.read(self._sessions.live_ids(), self._sessions.live_session):
                await self._apply(reported)
            for transcribed in await self._tails.catch_up():
                await self._apply(transcribed)

    async def keep_reading(self, period: float) -> None:
        """Read once a period, until cancelled."""
        while True:
            await self.read()
            await asyncio.sleep(period)
