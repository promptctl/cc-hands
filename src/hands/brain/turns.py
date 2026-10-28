"""The brain's Claude Code session, read by the one model of a Claude Code turn (hands.core.reducer) that reads every
working session: from its hooks, its transcript, and the status Claude Code keeps for it, each read as hands reads a
working session's.

[LAW:one-source-of-truth] the brain keeps no account of Claude Code's turns of its own: which turn runs, under which ids,
what waits in Claude Code's queue behind it, and when it is over are this model's to say. Only what is told of a turn
is handed on, as it is decided.
"""

import asyncio
import time
from collections import Counter
from collections.abc import Callable

from loguru import logger

from hands.core.effects import Audit, Compare, Effect, Narrate, Progress, Reply, SessionGone, Snapshot, Summarise, Tell
from hands.core.events import Event
from hands.core.reducer import reduce
from hands.core.session import Instant, Membership, Registry, Session, SessionId, all_told, status_stamp
from hands.core.status import Stamp
from hands.sessions.audit import Record
from hands.sessions.clock import stamp_now
from hands.sessions.registry import EFFECT_KINDS, audit_line
from hands.sessions.statusfile import Statuses
from hands.sessions.tail import TAIL_SECONDS, Tails
from hands.sessions.wide import annotate, count, unit


class Turns:
    """The brain's session as the model holds it, applied one event at a time, and each turn it tells handed to `told`
    as it is decided, before the next event is applied."""

    def __init__(self, session: SessionId, record: Record, told: Callable[[Summarise], None], clock: Callable[[], Instant] = time.monotonic, stamp: Callable[[], Stamp] = stamp_now) -> None:
        self.session = session
        # [LAW:no-shared-mutable-globals] replaced only by `apply`, one event at a time.
        self._registry = Registry(sessions={}, drafts={})
        self._record = record
        self._told = told
        # [LAW:effects-at-boundaries] the clocks a working session's registry is read on: a hook is stamped from these,
        # and a transcript read through on the second, which a Stop is heard on.
        self._clock = clock
        self._stamp = stamp
        self._tails = Tails(self)
        self._statuses = Statuses(clock)

    def apply(self, event: Event) -> None:
        """The event reduced, and what it calls for done at once, in the order decided."""
        before = self._registry
        self._registry, effects = reduce(before, event)
        if not effects and self._registry == before:
            # A status set again, a transcript read with nothing in it: no unit of work, as the registry's own.
            return
        # [LAW:nothing-unseen] one event for each event the model applies, as for a working session's: what was applied,
        # what it called for, and the effects counted by kind.
        with unit("brain.applied", self._record, counts=EFFECT_KINDS):
            annotate(applied=event, effects=tuple(effects))
            count(**Counter(type(effect).__name__ for effect in effects))
            for effect in effects:
                self._perform(effect)

    def _perform(self, effect: Effect) -> None:
        match effect:
            case Audit(record=record):
                logger.log(*audit_line(record))
            case Summarise() as summarise:
                self._told(summarise)
            case Reply():
                # The brain answers each hook as it comes: Claude Code waits on none of them for the model, since nothing
                # the model decides of the brain's turns has to be done before Claude Code goes on.
                pass
            case Snapshot() | Compare():
                # The brain works in an empty directory, never a repository: there is nothing to mark or read.
                pass
            case Narrate() | Progress() | Tell() | SessionGone():
                # Of dialogs, ticks, occurrences, and a session's end, none of which the brain's session is fed: its
                # dialogs are the brain's own, and what it says is read from the wire.
                pass

    async def read(self) -> None:
        """Claude Code's status of the brain and its transcript, read through now and applied: the transcript after the
        status, as it is only read once a status says whether a turn runs (Tails.catch_up)."""
        for reported in self._statuses.read(self.live_ids(), self.live_session):
            self.apply(reported)
        for transcribed in await self._tails.catch_up():
            self.apply(transcribed)

    async def keep_reading(self) -> None:
        """Read the brain's status and transcript once a period, until cancelled."""
        while True:
            await self.read()
            await asyncio.sleep(TAIL_SECONDS)

    def all_told(self) -> bool:
        """Whether every turn Claude Code ran in the brain is told, and nothing waits in its queue: true before any is
        heard, since nothing runs in a Claude Code nobody typed into."""
        held = self.live_session(self.session)
        return held is None or all_told(held)

    def held(self) -> Session | None:
        """The brain's session as the model holds it now, or None before its first hook."""
        return self.live_session(self.session)

    # What the tail and the status reader need of a registry (tail.Known, statusfile.Statuses.read).
    def live_members(self) -> list[Membership]:
        return [session.membership for session in self._registry.live()]

    def live_ids(self) -> list[SessionId]:
        return [session.membership.id for session in self._registry.live()]

    def live_session(self, session: SessionId) -> Session | None:
        match self._registry.sessions.get(session):
            case Session() as live:
                return live
            case _:
                return None

    def status_read(self, session: SessionId) -> bool:
        live = self.live_session(session)
        return live is not None and status_stamp(live.state) is not None

    def membership(self, session: SessionId) -> Membership | None:
        known = self._registry.sessions.get(session)
        return None if known is None else known.membership

    def now(self) -> Instant:
        return self._clock()

    def stamp(self) -> Stamp:
        return self._stamp()
