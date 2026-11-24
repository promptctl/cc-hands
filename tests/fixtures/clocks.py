"""Clocks a test and a child process it starts alike run a command on."""

import queue
from collections.abc import Awaitable, Callable
from datetime import datetime

from hands.clock import Clock, Scheduled, Timeout


class Interrupting:
    """The clock it is handed, but for its first wait, of any kind a Clock has, which is a Ctrl-C, as `hands log` is
    stopped while it follows: a command run on it that waits is stopped where it first would, one that never waits is
    not touched, and every wait after the Ctrl-C is the handed clock's own, as Ctrl-C is pressed once."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._pressed = False

    def _wait(self) -> None:
        if not self._pressed:
            self._pressed = True
            raise KeyboardInterrupt

    def now(self) -> float:
        return self._clock.now()

    def wall(self) -> datetime:
        return self._clock.wall()

    def sleep(self, seconds: float) -> Awaitable[None]:
        self._wait()
        return self._clock.sleep(seconds)

    def call_at(self, when: float, callback: Callable[[], object]) -> Scheduled:
        self._wait()
        return self._clock.call_at(when, callback)

    def timeout(self, seconds: float | None) -> Timeout:
        # No deadline bounds nothing, and is no wait.
        if seconds is not None:
            self._wait()
        return self._clock.timeout(seconds)

    def token(self, nbytes: int) -> str:
        return self._clock.token(nbytes)

    def uuid(self) -> str:
        return self._clock.uuid()

    def sleep_blocking(self, seconds: float) -> None:
        self._wait()
        self._clock.sleep_blocking(seconds)

    def take[T](self, source: queue.Queue[T] | queue.SimpleQueue[T], seconds: float) -> T:
        self._wait()
        return self._clock.take(source, seconds)
