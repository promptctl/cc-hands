"""The one clock hands reads: monotonic time a unit reads its deadlines off and waits them out on, the wall clock Claude
Code stamps its statuses and records with, the calls scheduled on that time, and the ids and randomness minted beside it.

[LAW:effects-at-boundaries] `SystemClock` is the only reader of the machine's time, ids, and randomness in hands
(tests/test_clock_boundary.py holds every other module to it), and `VirtualClock` is the same interface on a time a test
moves. A budget's deadline and the wait that enforces it are one clock's: read off one and enforced on another, a test
that stops time still loses to a loaded machine's real seconds.
"""

import asyncio
import random
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Collection, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from typing import Protocol

from hands.core.status import Stamp


class Scheduled(Protocol):
    """A call the clock will make, until it is cancelled."""

    def cancel(self) -> None: ...


class Timeout(AbstractAsyncContextManager["Timeout"], Protocol):
    """`asyncio.Timeout` on a clock: the block it guards is cancelled and raises `TimeoutError` once `now()` reaches its
    deadline, and a deadline of None never comes."""

    def reschedule(self, when: float | None) -> None: ...


class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds, on the timeline `pause`, `sleep`, and `call_at` count."""
        ...

    def wall(self) -> datetime:
        """The wall clock, in UTC."""
        ...

    def pause(self, seconds: float) -> Awaitable[None]:
        """Ends once `seconds` have passed since it was asked for."""
        ...

    def sleep(self, seconds: float) -> None:
        """Blocks a thread off the loop for `seconds`."""
        ...

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        """Calls `callback` on the running loop once `now()` reaches `when`."""
        ...

    def timeout(self, seconds: float | None) -> Timeout:
        """Guards a block with a deadline `seconds` from now."""
        ...

    def token(self, nbytes: int) -> str:
        """A URL-safe secret of `nbytes` random bytes."""
        ...

    def choice[T](self, options: Sequence[T]) -> T: ...

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
        # Armed as it is asked for, as `asyncio.wait(timeout=)` arms its own: an `asyncio.sleep` is armed only at its first
        # step, a step after the deadline it was measured from.
        loop = asyncio.get_running_loop()
        woken = loop.create_future()
        armed = loop.call_later(seconds, _wake, woken)
        woken.add_done_callback(lambda _: armed.cancel())
        return woken

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        # The loop's time is `time.monotonic`, the timeline `now` reads.
        return asyncio.get_running_loop().call_at(when, callback)

    def timeout(self, seconds: float | None) -> Timeout:
        return asyncio.timeout(seconds)

    def token(self, nbytes: int) -> str:
        return secrets.token_urlsafe(nbytes)

    def choice[T](self, options: Sequence[T]) -> T:
        return random.SystemRandom().choice(options)

    def uuid(self) -> str:
        return uuid.uuid4().hex


SYSTEM = SystemClock()


class VirtualClock:
    """Time that passes only when a test moves it, on a wall clock that starts at `epoch` and moves with it: `pause`,
    `sleep`, and `call_at` come due as `advance` reaches them, and ids and randomness come from `seed`, the same on every
    run."""

    def __init__(self, epoch: datetime = datetime(2026, 1, 1, tzinfo=UTC), seed: int = 0) -> None:
        self.at = 0.0
        self._epoch = epoch
        self._random = random.Random(seed)
        self._pauses: list[tuple[float, asyncio.Future[None]]] = []
        self._watched: list[tuple[float, asyncio.Future[None]]] = []
        self._calls: list[_Call] = []

    def now(self) -> float:
        return self.at

    def wall(self) -> datetime:
        return self._epoch + timedelta(seconds=self.at)

    def pause(self, seconds: float) -> asyncio.Future[None]:
        woken = asyncio.get_running_loop().create_future()
        self._pauses.append((self.at + seconds, woken))
        self._seen()
        # A pause already due wakes at once, as `asyncio.sleep(0)` does.
        self._wake()
        return woken

    def sleep(self, seconds: float) -> None:
        # No thread waits on a time only the test moves: the sleep is the time passing.
        self.advance(seconds)

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        call = _Call(when, callback)
        self._calls.append(call)
        self._wake()
        return call

    def timeout(self, seconds: float | None) -> Timeout:
        return _VirtualTimeout(self, None if seconds is None else self.at + seconds)

    def token(self, nbytes: int) -> str:
        return self._random.randbytes(nbytes).hex()

    def choice[T](self, options: Sequence[T]) -> T:
        return self._random.choice(options)

    def uuid(self) -> str:
        return uuid.UUID(int=self._random.getrandbits(128), version=4).hex

    def advance(self, seconds: float) -> None:
        self.at += seconds
        self._wake()

    async def paused(self, until: float) -> None:
        """Once a pause that wakes at `until` is waiting on this clock, however long the unit takes to come to it: a test
        moves the clock only once the deadline it spends is being waited out [LAW:no-ambient-temporal-coupling], and a
        unit that never comes to it is ended by the suite's timeout."""
        seen = asyncio.get_running_loop().create_future()
        self._watched.append((until, seen))
        self._seen()
        await seen

    @property
    def waking(self) -> tuple[float, ...]:
        """When each pause still waiting wakes, on this clock."""
        return tuple(until for until, woken in self._pauses if not woken.done())

    def _wake(self) -> None:
        for until, woken in self._pauses:
            if until <= self.at and not woken.done():
                woken.set_result(None)
        self._pauses = [(until, woken) for until, woken in self._pauses if not woken.done()]
        due = [call for call in self._calls if call.when <= self.at]
        self._calls = [call for call in self._calls if call.when > self.at]
        for call in sorted(due, key=lambda call: call.when):
            call.fire()

    def _seen(self) -> None:
        waking = self.waking
        for until, seen in self._watched:
            if until in waking and not seen.done():
                seen.set_result(None)
        self._watched = [(until, seen) for until, seen in self._watched if not seen.done()]


class _Call:
    def __init__(self, when: float, callback: Callable[[], object]) -> None:
        self.when = when
        self._callback: Callable[[], object] | None = callback

    def cancel(self) -> None:
        self._callback = None

    def fire(self) -> None:
        # Called on the loop's next step, as the loop's own timers are, never inside the `advance` that brought it due.
        if self._callback is not None:
            asyncio.get_running_loop().call_soon(self._callback)


class _VirtualTimeout:
    """`asyncio.Timeout`'s contract on a `VirtualClock`: the task is cancelled when its deadline comes due, and that one
    cancellation leaves the block as `TimeoutError`."""

    def __init__(self, clock: VirtualClock, when: float | None) -> None:
        self._clock = clock
        self._when = when
        self._task: asyncio.Task[object] | None = None
        self._due: Scheduled | None = None
        self._expired = False

    async def __aenter__(self) -> "_VirtualTimeout":
        self._task = asyncio.current_task()
        self.reschedule(self._when)
        return self

    async def __aexit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: object) -> None:
        self.reschedule(None)
        if self._expired and kind is asyncio.CancelledError and self._task is not None and self._task.uncancel() == 0:
            raise TimeoutError from error

    def reschedule(self, when: float | None) -> None:
        if self._due is not None:
            self._due.cancel()
        self._when = when
        self._due = None if when is None else self._clock.call_at(when, self._expire)

    def _expire(self) -> None:
        assert self._task is not None, "a timeout comes due only inside the block it guards"
        self._expired = True
        self._task.cancel()


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


def _wake(woken: asyncio.Future[None]) -> None:
    # A pause cancelled in the step its timer fires in is over already; its timer is not cancelled until the next.
    if not woken.cancelled():
        woken.set_result(None)
