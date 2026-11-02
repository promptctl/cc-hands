"""Waiting on a clock a test moves: what is awaited is never cancelled, and the time is up only when the clock says so."""

import asyncio

import pytest

from conftest import BrokenClock, VirtualClock
from hands.clock import SYSTEM, wait


async def test_a_wait_ends_once_everything_it_waits_on_is_done_however_little_time_has_passed() -> None:
    clock = VirtualClock()
    first, second = asyncio.get_running_loop().create_future(), asyncio.get_running_loop().create_future()
    waiting = asyncio.create_task(wait(clock, (first, second), 5.0))
    first.set_result("one")
    await clock.paused(5.0)
    assert not waiting.done()
    second.set_result("two")
    await waiting
    # Its timer is over with it.
    assert clock.waking == ()


async def test_a_wait_whose_time_is_up_leaves_what_it_waited_on_running() -> None:
    clock = VirtualClock()
    reading = asyncio.get_running_loop().create_future()
    waiting = asyncio.create_task(wait(clock, (reading,), 5.0))
    await clock.paused(5.0)
    clock.advance(4.9)
    assert clock.waking == (5.0,)
    clock.advance(0.1)
    await waiting
    assert not reading.done()
    reading.set_result("still read, for whoever comes for it")


async def test_a_clock_that_fails_is_a_failure_and_not_the_time_running_out() -> None:
    with pytest.raises(RuntimeError, match="no timer"):
        await wait(BrokenClock(), (asyncio.get_running_loop().create_future(),), 5.0)


async def test_a_wait_on_nothing_is_a_mistake_and_not_everything_done() -> None:
    with pytest.raises(ValueError):
        await wait(VirtualClock(), (), 5.0)


async def test_a_pause_already_due_wakes_without_the_clock_moving() -> None:
    clock = VirtualClock()
    await asyncio.wait_for(clock.pause(0.0), timeout=1.0)
    assert clock.waking == ()


async def test_a_system_pause_cancelled_as_its_timer_fires_is_over_with_nothing_raised() -> None:
    loop = asyncio.get_running_loop()
    raised: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: raised.append(context))
    pause = SYSTEM.pause(0)
    # The timer is due this step, and the cancellation reaches its callbacks only in the next.
    loop.call_soon(pause.cancel)
    await asyncio.sleep(0.01)
    assert pause.cancelled() and raised == []
    loop.set_exception_handler(None)


async def test_a_virtual_timeout_left_before_its_due_cancel_runs_leaves_no_cancel_behind() -> None:
    clock = VirtualClock()
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    async with clock.timeout(0.0):
        await done
    await asyncio.sleep(0)
    assert asyncio.current_task().cancelling() == 0  # type: ignore[union-attr]


async def test_a_virtual_timeout_that_expired_takes_its_cancel_back_however_the_block_ended() -> None:
    clock = VirtualClock()
    async with clock.timeout(1.0):
        clock.advance(1.0)
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
    assert asyncio.current_task().cancelling() == 0  # type: ignore[union-attr]


async def test_a_virtual_timeout_inside_a_task_already_cancelling_expires_as_asyncio_s_own_does() -> None:
    # `asyncio.Timeout` counts the task's cancellations from where they stood as the block began: its own taken back, the
    # count is back there, and the block leaves as its TimeoutError with the older cancel still pending.
    clock = VirtualClock()
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    with pytest.raises(TimeoutError):
        async with clock.timeout(1.0):
            clock.advance(1.0)
            await asyncio.sleep(0)
    assert task.uncancel() == 0
