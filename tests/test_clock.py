"""Waiting on a clock a test moves: what is awaited is never cancelled, and the time is up only when the clock says so."""

import asyncio
import queue
import uuid

import pytest

from conftest import BrokenClock, VirtualClock
from hands.clock import SYSTEM, dashed, wait


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


async def test_a_virtual_timeout_that_came_due_refuses_a_new_deadline_as_asyncio_s_own_does() -> None:
    clock = VirtualClock()
    with pytest.raises(RuntimeError, match="came due"):
        async with clock.timeout(1.0) as limit:
            clock.advance(1.0)
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                limit.reschedule(5.0)


def test_a_rest_on_a_virtual_clock_moves_its_time() -> None:
    clock = VirtualClock()
    clock.rest(30.0)
    assert clock.now() == 30.0


def test_a_take_on_a_virtual_clock_gives_what_is_queued_and_is_empty_at_once_moving_no_time() -> None:
    clock = VirtualClock()
    waiting: queue.SimpleQueue[str] = queue.SimpleQueue()
    waiting.put("queued")
    assert clock.take(waiting, 5.0) == "queued"
    with pytest.raises(queue.Empty):
        clock.take(waiting, 5.0)
    assert clock.now() == 0.0


def test_a_take_on_the_system_clock_waits_for_what_is_queued() -> None:
    waiting: queue.SimpleQueue[str] = queue.SimpleQueue()
    waiting.put("queued")
    assert SYSTEM.take(waiting, 5.0) == "queued"


def test_a_dashed_id_is_the_uuid_its_clock_minted() -> None:
    assert uuid.UUID(dashed(VirtualClock(seed=1))) == uuid.UUID(VirtualClock(seed=1).uuid())


def test_a_rest_after_every_wait_on_the_clock_ended_moves_it() -> None:
    clock = VirtualClock()

    async def waited() -> None:
        async with clock.timeout(5):
            pass

    asyncio.run(waited())
    clock.rest(1)
    assert clock.now() == 1


def test_a_rest_while_a_loop_waits_on_the_clock_is_refused() -> None:
    clock = VirtualClock()

    async def waiting() -> None:
        paused = clock.pause(5)
        with pytest.raises(RuntimeError):
            clock.rest(1)
        paused.cancel()

    asyncio.run(waiting())


def test_a_virtual_clock_moves_only_forward() -> None:
    clock = VirtualClock()
    clock.advance(2.0)
    with pytest.raises(ValueError, match="only forward"):
        clock.advance(-1.0)
    assert clock.now() == 2.0


async def test_a_deadline_is_waited_on_as_many_times_as_it_is_scheduled_and_never_stands_in_for_a_pause() -> None:
    clock = VirtualClock()
    paused = asyncio.create_task(clock.paused(3.0))
    scheduled = asyncio.create_task(clock.scheduled(3.0, count=2))
    clock.call_at(3.0, lambda: None)
    await asyncio.sleep(0)
    assert not scheduled.done() and not paused.done()
    clock.call_at(3.0, lambda: None)
    await scheduled
    assert not paused.done()
    clock.pause(3.0)
    await paused
    assert clock.begun == (3.0,)


async def test_a_clock_is_settled_once_exactly_that_many_pauses_remain_waiting_however_the_others_ended() -> None:
    clock = VirtualClock()
    first, second, third = clock.pause(3.0), clock.pause(3.0), clock.pause(3.0)
    settled = asyncio.create_task(clock.settled(3.0, count=1))
    await asyncio.sleep(0)
    assert not settled.done()
    first.cancel()
    await asyncio.sleep(0)
    assert not settled.done()
    second.cancel()
    await settled
    assert clock.waking == (3.0,)
    third.cancel()


def test_a_virtual_clock_rests_only_forward_and_never_onto_nan() -> None:
    clock = VirtualClock()
    with pytest.raises(ValueError, match="only forward"):
        clock.rest(-1.0)
    with pytest.raises(ValueError, match="only forward"):
        clock.advance(float("nan"))
    assert clock.now() == 0.0
