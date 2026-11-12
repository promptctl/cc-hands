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
        self._watched: list[tuple[Callable[[], bool], asyncio.Future[None]]] = []
        self._calls: list[_Call] = []
        self._begun: list[float] = []

    def now(self) -> float:
        return self.at

    def wall(self) -> datetime:
        return self._epoch + timedelta(seconds=self.at)

    def pause(self, seconds: float) -> asyncio.Future[None]:
        woken = asyncio.get_running_loop().create_future()
        self._pauses.append((self.at + seconds, woken))
        self._begun.append(seconds)
        # A pause ends by waking or by its waiter's cancel, and a watch for fewer pauses waiting reads the ending.
        woken.add_done_callback(lambda _: self._seen())
        self._seen()
        # A pause already due wakes at once, as `asyncio.sleep(0)` does.
        self._wake()
        return woken

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        call = _Call(when, callback)
        self._calls.append(call)
        self._seen()
        self._wake()
        return call

    def timeout(self, seconds: float | None) -> Timeout:
        return _VirtualTimeout(self, None if seconds is None else self.at + seconds)

    def token(self, nbytes: int) -> str:
        return self._random.randbytes(nbytes).hex()

    def uuid(self) -> str:
        return uuid.UUID(int=self._random.getrandbits(128), version=4).hex

    def rest(self, seconds: float) -> None:
        # Off any loop, nothing waits on this clock's pauses to be woken: the time moves, and no real second passes. A
        # pause or call a loop waits on would come due unwoken, so a clock shared that way is refused, not run.
        if self.waking or any(call.live for call in self._calls):
            raise RuntimeError("rest moves a clock a loop is waiting on; that loop's test moves it with advance")
        self._move(self.at + seconds)

    def take[T](self, source: queue.Queue[T] | queue.SimpleQueue[T], seconds: float) -> T:
        """What `source` holds already, else `queue.Empty` at once: the taker is another thread than the loop that moves
        this clock, so its wait moves nothing, and a linger on this clock waits no real seconds."""
        return source.get_nowait()

    def advance(self, seconds: float) -> None:
        self.advance_to(self.at + seconds)

    def advance_to(self, when: float) -> None:
        """The clock at `when` exactly, waking what comes due by then."""
        self._move(when)
        self._wake()

    def _move(self, when: float) -> None:
        # Monotonic, as the time it stands in for: a rewind would stamp what follows before what came first, and a NaN
        # would leave every pause and call never due.
        if not when >= self.at:
            raise ValueError(f"a clock moves only forward; {when} is not on or after {self.at}")
        self.at = when

    async def paused(self, until: float, count: int = 1) -> None:
        """Once `count` pauses that wake at `until` are waiting on this clock, however long the unit takes to come to them:
        a test moves the clock only once the deadline it spends is being waited out [LAW:no-ambient-temporal-coupling],
        and a unit that never comes to it is ended by the suite's timeout."""
        await self._watch(lambda: self.waking.count(until) >= count)

    async def settled(self, until: float, count: int) -> None:
        """Once exactly `count` pauses that wake at `until` are waiting on this clock: as `paused`, but also for the ones
        that end first, as a git that fails ends its own wait. Await `paused` for the most that will come first, so a count
        not yet reached is not read as settled."""
        await self._watch(lambda: self.waking.count(until) == count)

    async def scheduled(self, until: float, count: int = 1) -> None:
        """Once `count` calls that come due at `until`, as a timeout's deadline does, are scheduled on this clock: as
        `paused`, for what `call_at` waits out."""
        await self._watch(lambda: [call.when for call in self._calls if call.live].count(until) >= count)

    async def _watch(self, ready: Callable[[], bool]) -> None:
        seen = asyncio.get_running_loop().create_future()
        self._watched.append((ready, seen))
        self._seen()
        await seen

    @property
    def begun(self) -> tuple[float, ...]:
        """How long each pause ever begun on this clock was to last, done or not: what a test reads to say one never
        began."""
        return tuple(self._begun)

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
        for ready, seen in self._watched:
            if not seen.done() and ready():
                seen.set_result(None)
        self._watched = [watch for watch in self._watched if not watch[1].done()]


class _Call:
    def __init__(self, when: float, callback: Callable[[], object]) -> None:
        self.when = when
        self._callback: Callable[[], object] | None = callback

    def cancel(self) -> None:
        self._callback = None

    @property
    def live(self) -> bool:
        return self._callback is not None

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
