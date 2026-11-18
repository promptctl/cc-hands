"""The run's memory: MLX keeps no more of the GPU memory its arrays free than the limit, and the process's memory is
sampled once a period, each sample one event saying what it holds."""

import asyncio
from collections.abc import Generator

import mlx.core as mx
import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import memory
from hands.voice.memory import CACHE_LIMIT, SAMPLE_SECONDS, SAMPLED, keep_sampling, limit_cache

MEGABYTE = 2**20


@pytest.fixture
def limited() -> Generator[None]:
    # The limit is the process's: MLX's own default, the GPU's whole recommended working set, until limit_cache sets it,
    # and what it was is put back after, so no other test runs under this one's.
    was = mx.set_cache_limit(int(mx.device_info()["max_recommended_working_set_size"]))  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    limit_cache()
    yield
    mx.set_cache_limit(was)
    mx.clear_cache()


def test_arrays_of_many_sizes_freed_one_after_another_leave_no_more_cached_than_the_limit(limited: None) -> None:
    # As the streaming pair's are: each a size no buffer freed before it fits, so each is a buffer of its own, freed to the
    # cache, which without the limit holds every one of them.
    sizes = [(64 + each) * MEGABYTE for each in range(40)]
    for size in sizes:
        held = mx.zeros((size // 4,), mx.float32)
        mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
        del held
    # MLX trims the cache to the limit as the next buffer is allocated, so it holds at most one freed buffer past it.
    assert 0 < mx.get_cache_memory() <= CACHE_LIMIT + max(sizes)


async def test_the_memory_is_sampled_at_once_and_once_a_period_after_each_sample_saying_the_footprint_and_mlx_split() -> None:
    clock, kept = VirtualClock(), list[Entry]()
    held = {"mlx_active_mb": 2587.0, "mlx_cache_mb": 1024.0, "mlx_peak_mb": 2964.0}
    sampling = asyncio.create_task(keep_sampling(clock, kept.append, read=lambda: 6354 * MEGABYTE, held=lambda: held))
    await clock.sleeping(SAMPLE_SECONDS)
    clock.advance(SAMPLE_SECONDS)
    await clock.sleeping(2 * SAMPLE_SECONDS)
    sampling.cancel()
    samples = [entry for entry in kept if isinstance(entry, WideEvent)]
    assert [(sample.event, sample.outcome, sample.facts) for sample in samples] == [(SAMPLED, "ok", {"footprint_mb": 6354.0, **held})] * 2


async def test_a_footprint_the_kernel_refuses_is_said_on_the_sample_and_the_sampling_goes_on() -> None:
    clock, kept = VirtualClock(), list[Entry]()

    def refused() -> int:
        raise PermissionError(1, "Operation not permitted")

    sampling = asyncio.create_task(keep_sampling(clock, kept.append, read=refused, held=dict))
    await clock.sleeping(SAMPLE_SECONDS)
    clock.advance(SAMPLE_SECONDS)
    await clock.sleeping(2 * SAMPLE_SECONDS)
    sampling.cancel()
    samples = [entry for entry in kept if isinstance(entry, WideEvent)]
    assert [(sample.outcome, sample.facts) for sample in samples] == [("ok", {"footprint_error": "[Errno 1] Operation not permitted"})] * 2


def test_mlx_says_its_live_cached_and_peak_memory_in_megabytes() -> None:
    assert memory.mlx_held().keys() == {"mlx_active_mb", "mlx_cache_mb", "mlx_peak_mb"}
