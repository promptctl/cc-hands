"""The run's memory: MLX keeps no more of the GPU memory its arrays free than the limit, and a sample of the process's
memory is one event saying what it holds."""

import asyncio
import contextlib
import threading
from collections.abc import Generator, Mapping

import mlx.core as mx
import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import memory, mlxthread
from hands.voice.memory import CACHE_LIMIT, SAMPLE_SECONDS, SAMPLED, SPLIT_SECONDS, limit_cache, sample
from hands.voice.mlxthread import OnMLX

MEGABYTE = 2**20


def emptied(on: OnMLX) -> None:
    mx.clear_cache()


@pytest.fixture
def limited() -> Generator[None]:
    # The limit and the cache are the process's: the cache starts empty, so no buffer an earlier test freed is handed to
    # this one's arrays, and the limit is put back after, so no other test runs under this one's.
    mlxthread.call(emptied)
    was = mlxthread.call(limit_cache)
    yield
    mlxthread.call(lambda _: mx.set_cache_limit(was))
    mlxthread.call(emptied)


def test_arrays_of_many_sizes_freed_one_after_another_leave_no_more_cached_than_the_limit(limited: None) -> None:
    # As the streaming pair's are: each a size no buffer freed before it fits, so each is a buffer of its own, freed to the
    # cache, which without the limit holds every one of them: more than the limit and the largest of them together.
    sizes = [(64 + each) * MEGABYTE for each in range(20)]

    def cached(on: OnMLX) -> int:
        for size in sizes:
            held = mx.zeros((size // 4,), mx.float32)
            mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
            del held
        mx.synchronize()
        return mx.get_cache_memory()

    # MLX trims the cache to the limit as the next buffer is allocated, so it holds at most one freed buffer past it.
    assert 0 < mlxthread.call(cached) <= CACHE_LIMIT + max(sizes)


def test_mlx_says_in_megabytes_what_its_live_arrays_hold_and_what_its_cache_holds_once_they_are_freed(limited: None) -> None:
    def split(on: OnMLX) -> tuple[Mapping[str, float], ...]:
        before = memory.mlx_held(on)
        held = mx.zeros((256 * MEGABYTE // 4,), mx.float32)
        mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
        live = memory.mlx_held(on)
        del held
        # The GPU holds the array's buffer until the work that wrote it is done, and only then is it freed to the cache.
        mx.synchronize()
        return before, live, memory.mlx_held(on)

    before, live, freed = mlxthread.call(split)
    assert (round(live["mlx_active_mb"] - before["mlx_active_mb"]), round(freed["mlx_cache_mb"] - live["mlx_cache_mb"])) == (256, 256)


def sampled(kept: list[Entry]) -> list[WideEvent]:
    return [entry for entry in kept if isinstance(entry, WideEvent) and entry.event == SAMPLED]


async def test_a_sample_is_one_event_saying_the_footprint_the_mlx_work_waiting_and_mlx_split() -> None:
    kept = list[Entry]()
    held = {"mlx_active_mb": 2587.0, "mlx_cache_mb": 1024.0}
    await sample(VirtualClock(), kept.append, read=lambda: 6354 * MEGABYTE, held=lambda _: held, waiting=lambda: 2)
    assert [(event.outcome, event.facts) for event in sampled(kept)] == [("ok", {"footprint_mb": 6354.0, "mlx_waiting": 2, **held})]


async def test_a_footprint_the_kernel_refuses_is_said_on_the_sample() -> None:
    kept = list[Entry]()

    def refused() -> int:
        raise PermissionError(1, "Operation not permitted")

    await sample(VirtualClock(), kept.append, read=refused, held=lambda _: {}, waiting=lambda: 0)
    assert [(event.outcome, event.facts) for event in sampled(kept)] == [("ok", {"footprint_error": "[Errno 1] Operation not permitted", "mlx_waiting": 0})]


async def test_an_mlx_thread_that_does_not_answer_in_time_is_said_on_the_sample_with_the_footprint() -> None:
    clock = VirtualClock()
    kept = list[Entry]()
    go = threading.Event()
    # The MLX thread, held up by work given before the sample's.
    holding = mlxthread.settles(lambda _: go.wait())
    try:
        sampling = asyncio.create_task(sample(clock, kept.append, read=lambda: 6354 * MEGABYTE, held=memory.mlx_held, waiting=lambda: 1))
        await clock.scheduled(SPLIT_SECONDS)
        clock.advance(SPLIT_SECONDS)
        await sampling
    finally:
        go.set()
    await holding
    assert [(event.outcome, event.facts) for event in sampled(kept)] == [("ok", {"footprint_mb": 6354.0, "mlx_waiting": 1, "mlx_split_error": "the MLX thread did not answer in 10 s"})]


async def test_the_memory_is_sampled_at_once_and_again_a_period_after_each_sample() -> None:
    clock = VirtualClock()
    kept = list[Entry]()
    sampling = asyncio.create_task(memory.keep_sampling(clock, kept.append))
    try:
        await clock.sleeping(SAMPLE_SECONDS)
        assert len(sampled(kept)) == 1
        clock.advance(SAMPLE_SECONDS)
        await clock.sleeping(2 * SAMPLE_SECONDS)
        assert len(sampled(kept)) == 2
    finally:
        sampling.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sampling
