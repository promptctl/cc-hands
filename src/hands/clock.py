"""The one clock hands reads: monotonic time a unit reads its deadlines off and waits them out on, the wall clock Claude
Code stamps its statuses and records with, the calls scheduled on that time, and the ids and randomness minted beside it.

[LAW:effects-at-boundaries] `SystemClock` is the only reader of the machine's time, ids, and randomness in hands
(tests/test_clock_boundary.py holds every other module to it), and `hands.virtualclock.VirtualClock` is the same
interface on a time a test moves. A budget's deadline and the wait that enforces it are one clock's: read off one and
enforced on another, a test that stops time still loses to a loaded machine's real seconds.

asyncio is imported where a pause, a call, or a wait needs it, never with the module: the hook shim reads only the wall
clock, and it runs in Claude Code's critical path, where asyncio's import is ten milliseconds every hook pays.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Collection
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from hands.core.status import Stamp

if TYPE_CHECKING:
    import asyncio


class Scheduled(Protocol):
    """A call the clock will make, until it is cancelled."""

    def cancel(self) -> None: ...


class Timeout(AbstractAsyncContextManager["Timeout"], Protocol):
    """`asyncio.Timeout` on a clock: the block it guards is cancelled and raises `TimeoutError` once `now()` reaches its
    deadline, and a deadline of None never comes."""

    def reschedule(self, when: float | None) -> None: ...


class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds, on the timeline `pause` and `call_at` count."""
        ...

    def wall(self) -> datetime:
        """The wall clock, in UTC."""
        ...

    def pause(self, seconds: float) -> Awaitable[None]:
        """Ends once `seconds` have passed since it was asked for."""
        ...

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        """Calls `callback` on the running loop once `now()` reaches `when`."""
        ...

    def timeout(self, seconds: float | None) -> Timeout:
        """Guards a block with a deadline `seconds` from now."""
        ...

    def token(self, nbytes: int) -> str:
        """`nbytes` random bytes, as 2 * `nbytes` hex digits."""
        ...

    def uuid(self) -> str:
        """A fresh id, as 32 hex digits."""
        ...


def stamp(clock: Clock) -> Stamp:
    """The wall clock in the milliseconds Claude Code stamps its statuses with."""
    return Stamp(int(clock.wall().timestamp() * 1000))


class SystemClock:
    """The machine's clocks, a pause on the loop's timer, and the operating system's randomness."""

    def now(self) -> float:
        return time.monotonic()

    def wall(self) -> datetime:
        return datetime.now(UTC)

    def pause(self, seconds: float) -> asyncio.Future[None]:
        import asyncio

        # Armed as it is asked for, as `asyncio.wait(timeout=)` arms its own: an `asyncio.sleep` is armed only at its first
        # step, a step after the deadline it was measured from.
        loop = asyncio.get_running_loop()
        woken = loop.create_future()
        armed = loop.call_later(seconds, _wake, woken)
        woken.add_done_callback(lambda _: armed.cancel())
        return woken

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        import asyncio

        # The loop's time is `time.monotonic`, the timeline `now` reads.
        return asyncio.get_running_loop().call_at(when, callback)

    def timeout(self, seconds: float | None) -> Timeout:
        import asyncio

        return asyncio.timeout(seconds)

    def token(self, nbytes: int) -> str:
        return secrets.token_hex(nbytes)

    def uuid(self) -> str:
        return uuid.uuid4().hex


SYSTEM = SystemClock()


async def wait[T](clock: Clock, futures: Collection[asyncio.Future[T]], timeout: float) -> None:
    """`asyncio.wait(futures, timeout=timeout)` on `clock`: until every one of `futures` is done or `timeout` has passed,
    cancelling none of them, so what is still running when the time is up runs on for whoever reads it next."""
    import asyncio

    every = asyncio.ensure_future(asyncio.wait(futures))
    timer = asyncio.ensure_future(clock.pause(timeout))
    try:
        await asyncio.wait((every, timer), return_when=asyncio.FIRST_COMPLETED)
    finally:
        every.cancel()
        timer.cancel()
        # Both over before this returns, so nothing of the wait outlives it.
        await asyncio.wait((every, timer))
    # [LAW:no-silent-failure] a pause that raised is its clock failing, never the time running out, and a wait that raised
    # (on no futures at all, as `asyncio.wait` does) is never everything done.
    for side in (every, timer):
        if not side.cancelled():
            side.result()


def _wake(woken: asyncio.Future[None]) -> None:
    # A pause cancelled in the step its timer fires in is over already; its timer is not cancelled until the next.
    if not woken.cancelled():
        woken.set_result(None)
