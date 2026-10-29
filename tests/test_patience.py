import asyncio
import itertools
import threading
import time

import conftest
import pytest
from conftest import patiently


async def test_a_step_that_holds_the_loop_past_the_patience_is_not_counted_against_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.1)
    finished: list[None] = []

    def step() -> None:
        time.sleep(0.3)
        finished.append(None)

    asyncio.get_running_loop().call_later(0.05, step)
    await patiently(lambda: bool(finished), lambda: 0)


async def test_a_wait_whose_end_comes_on_the_look_that_spends_its_patience_ends_there(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.05)
    looks = itertools.count()
    await patiently(lambda: next(looks) == round(conftest.PATIENCE_SECS / conftest.LOOK_SECS), lambda: 0)


async def test_a_wait_on_a_free_loop_where_nothing_moves_fails_once_its_patience_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.1)
    with pytest.raises(TimeoutError, match=r"nothing moved through 0\.1s of looking, \d+\.\ds on the clock, at 0"):
        await patiently(lambda: False, lambda: 0)


async def test_a_stall_says_where_each_thread_was_and_where_in_its_await_chain_each_task_waited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "PATIENCE_SECS", 0.05)
    released, opened = threading.Event(), asyncio.Event()

    def held_on_its_thread() -> None:
        released.wait()

    # The wait is two awaits below the task's own coroutine, where the desk's stalled wait is below serve_desk's.
    async def deepest() -> None:
        await opened.wait()

    async def between() -> None:
        await deepest()

    async def outermost() -> None:
        await between()

    thread = threading.Thread(target=held_on_its_thread, name="the held thread")
    task = asyncio.create_task(outermost())
    thread.start()
    try:
        with pytest.raises(TimeoutError) as stall:
            await patiently(lambda: False, lambda: 0)
    finally:
        released.set()
        thread.join()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    [where] = stall.value.__notes__
    assert "thread the held thread:" in where and "in held_on_its_thread" in where
    assert "in deepest" in where
