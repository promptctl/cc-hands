"""Waiting on a clock a test moves: what is awaited is never cancelled, and the time is up only when the clock says so."""

import asyncio

import pytest

from conftest import ManualClock, patiently
from hands.sessions.clock import wait


async def test_a_wait_ends_once_everything_it_waits_on_is_done_however_little_time_has_passed() -> None:
    clock = ManualClock()
    first, second = asyncio.get_running_loop().create_future(), asyncio.get_running_loop().create_future()
    waiting = asyncio.create_task(wait(clock, (first, second), 5.0))
    first.set_result("one")
    await patiently(lambda: clock.waking == (5.0,), lambda: clock.waking)
    assert not waiting.done()
    second.set_result("two")
    await waiting
    # Its timer is over with it.
    assert clock.waking == ()


async def test_a_wait_whose_time_is_up_leaves_what_it_waited_on_running() -> None:
    clock = ManualClock()
    reading = asyncio.get_running_loop().create_future()
    waiting = asyncio.create_task(wait(clock, (reading,), 5.0))
    await patiently(lambda: clock.waking == (5.0,), lambda: clock.waking)
    clock.advance(4.9)
    assert clock.waking == (5.0,)
    clock.advance(0.1)
    await waiting
    assert not reading.done()
    reading.set_result("still read, for whoever comes for it")


class Broken:
    """A clock whose pause fails."""

    def now(self) -> float:
        return 0.0

    async def pause(self, seconds: float) -> None:
        raise RuntimeError("no timer")


async def test_a_clock_that_fails_is_a_failure_and_not_the_time_running_out() -> None:
    with pytest.raises(RuntimeError, match="no timer"):
        await wait(Broken(), (asyncio.get_running_loop().create_future(),), 5.0)
