"""The clocks hands reads: the wall clock Claude Code stamps its statuses and records with, read on this side in the same
milliseconds, and the monotonic time a unit reads its deadlines off and waits them out on.

[LAW:effects-at-boundaries] a budget's deadline and the wait that enforces it are one clock's: read off one and enforced
on another, a test that stops time still loses to a loaded machine's real seconds.
"""

import asyncio
import time
from collections.abc import Awaitable, Collection
from typing import Protocol

from hands.core.status import Stamp


def stamp_now() -> Stamp:
    return Stamp(time.time_ns() // 1_000_000)


class Clock(Protocol):
    """Monotonic seconds, and a pause that ends once that many of them have passed."""

    def now(self) -> float: ...

    def pause(self, seconds: float) -> Awaitable[None]: ...


class SystemClock:
    """The machine's monotonic clock and asyncio's sleep."""

    def now(self) -> float:
        return time.monotonic()

    async def pause(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


SYSTEM = SystemClock()


async def wait[T](clock: Clock, futures: Collection[asyncio.Future[T]], timeout: float) -> None:
    """`asyncio.wait(futures, timeout=timeout)` on `clock`: until every one of `futures` is done or `timeout` has passed,
    cancelling none of them, so what is still running when the time is up runs on for whoever reads it next."""
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
