"""`Clock` on a time a test moves, and ids and randomness from a seed, the same on every run."""

import asyncio
import queue
import random
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from hands.clock import Scheduled, Timeout


class VirtualClock:
    """Time that passes only when a test moves it, on a wall clock that starts at `epoch` and moves with it: `pause` and
    `call_at` come due as `advance` reaches them, and ids and randomness come from `seed`."""

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

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        call = _Call(when, callback)
        self._calls.append(call)
        self._wake()
        return call

    def timeout(self, seconds: float | None) -> Timeout:
        return _VirtualTimeout(self, None if seconds is None else self.at + seconds)

    def token(self, nbytes: int) -> str:
        return self._random.randbytes(nbytes).hex()

    def uuid(self) -> str:
        return uuid.UUID(int=self._random.getrandbits(128), version=4).hex

    def rest(self, seconds: float) -> None:
        # Off any loop, nothing waits on this clock's pauses to be woken: the time moves, and no real second passes.
        self.at += seconds

    def take[T](self, source: queue.Queue[T] | queue.SimpleQueue[T], seconds: float) -> T:
        """What `source` holds already, else `queue.Empty` at once: the taker is another thread than the loop that moves
        this clock, so its wait moves nothing, and a linger on this clock waits no real seconds."""
        return source.get_nowait()

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
        if not any(call.when <= self.at for call in self._calls):
            return
        # Read before a due call is taken off, so a clock moved off the loop raises with every call still scheduled.
        loop = asyncio.get_running_loop()
        due = [call for call in self._calls if call.when <= self.at]
        self._calls = [call for call in self._calls if call.when > self.at]
        for call in sorted(due, key=lambda call: call.when):
            call.fire(loop)

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

    def fire(self, loop: asyncio.AbstractEventLoop) -> None:
        # Run on the loop's next step, as the loop's own timers are, never inside the `advance` that brought it due; a
        # call cancelled before that step never runs, as a loop timer's handle cancelled after it came due does not.
        loop.call_soon(self._run)

    def _run(self) -> None:
        if self._callback is not None:
            self._callback()


class _VirtualTimeout:
    """`asyncio.Timeout`'s contract on a `VirtualClock`: the task is cancelled when its deadline comes due, and that one
    cancellation leaves the block as `TimeoutError`, however the block itself ended."""

    def __init__(self, clock: VirtualClock, when: float | None) -> None:
        self._clock = clock
        self._when = when
        self._task: asyncio.Task[object] | None = None
        self._cancelling = 0
        self._due: Scheduled | None = None
        self._expired = False
        self._exited = False

    async def __aenter__(self) -> "_VirtualTimeout":
        self._task = asyncio.current_task()
        assert self._task is not None, "a timeout guards a block inside a task"
        self._cancelling = self._task.cancelling()
        self.reschedule(self._when)
        return self

    async def __aexit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: object) -> None:
        self._release()
        self._exited = True
        # As `asyncio.Timeout` does: the cancel it sent is taken back whenever it expired, and is the block's own
        # TimeoutError only when no other cancellation, older than the block or sent within it, is still pending.
        if self._expired and self._task is not None and self._task.uncancel() <= self._cancelling and kind is asyncio.CancelledError:
            raise TimeoutError from error

    def reschedule(self, when: float | None) -> None:
        # As `asyncio.Timeout` refuses: a deadline moves only while the block it guards runs, before it came due.
        if self._task is None or self._exited:
            raise RuntimeError("a timeout is rescheduled only inside the block it guards")
        if self._expired:
            raise RuntimeError("a timeout that came due cannot be rescheduled")
        self._release()
        self._when = when
        self._due = None if when is None else self._clock.call_at(when, self._expire)

    def expired(self) -> bool:
        return self._expired

    def _release(self) -> None:
        if self._due is not None:
            self._due.cancel()
        self._due = None

    def _expire(self) -> None:
        assert self._task is not None, "a timeout comes due only inside the block it guards"
        self._expired = True
        self._task.cancel()
