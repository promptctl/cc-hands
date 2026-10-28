"""What Claude Code says of every live session, read for the one model of a Claude Code turn: its status, then its
transcript, once a period, and through now before a turn's hook is applied, so the hook is decided on what Claude Code
had said by then on every channel."""

import asyncio
from collections.abc import Awaitable, Callable

from hands.core.events import Event, PermissionRequested, Prompted, Stopped
from hands.sessions.hooks import Hook
from hands.sessions.registry import Sessions, StopHeld
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

    def __init__(self, sessions: Sessions, tails: Tails, apply: Callable[[Event], Awaitable[None]]) -> None:
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

    async def hooked(self, hook: Hook) -> StopHeld | None:
        """A session's hook applied: where its transcript is, then what it says, a turn's prompt or Stop once Claude Code
        is read through now; how a Stop was held. [LAW:single-enforcer] the one way a hook reaches the model, a working
        session's and the brain's alike.

        [LAW:no-ambient-temporal-coupling] with only the period, a Stop that comes before any status was read is held for
        no record (reducer `_stopping`) and ends nothing, and a Stop naming an id only the transcript says yet is decided
        before it is read: any turn Claude Code writes faster than a period would be.
        """
        match hook.joining:
            case None:
                pass
            case joining:
                await self._apply(joining)
        match hook.happened:
            case Stopped() as stopped:
                await self.read()
                return await self._sessions.stop(stopped)
            case Prompted() as prompted:
                await self.read()
                await self._apply(prompted)
            case PermissionRequested():
                # Applied only by Sessions.ask, which waits on the reply it is answered with: a hook answered anywhere else
                # leaves the model holding no dialog for it.
                pass
            case happened:
                await self._apply(happened)
        return None

    async def keep_reading(self, period: float) -> None:
        """Read once a period, until cancelled."""
        while True:
            await self.read()
            await asyncio.sleep(period)
