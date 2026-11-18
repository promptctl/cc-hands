"""The run's memory: MLX keeps no more of the GPU memory its arrays free than the limit, and a sample of the process's
memory is one event saying what it holds."""

from collections.abc import Generator

import mlx.core as mx
import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import memory
from hands.voice.memory import CACHE_LIMIT, SAMPLED, limit_cache, sample

MEGABYTE = 2**20


@pytest.fixture
def limited() -> Generator[None]:
    # The limit and the cache are the process's: the cache starts empty, so no buffer an earlier test freed is handed to
    # this one's arrays, and the limit is put back after, so no other test runs under this one's.
    mx.clear_cache()
    was = limit_cache()
    yield
    mx.set_cache_limit(was)
    mx.clear_cache()


def test_arrays_of_many_sizes_freed_one_after_another_leave_no_more_cached_than_the_limit(limited: None) -> None:
    # As the streaming pair's are: each a size no buffer freed before it fits, so each is a buffer of its own, freed to the
    # cache, which without the limit holds every one of them: more than the limit and the largest of them together.
    sizes = [(64 + each) * MEGABYTE for each in range(20)]
    for size in sizes:
        held = mx.zeros((size // 4,), mx.float32)
        mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
        del held
    mx.synchronize()
    # MLX trims the cache to the limit as the next buffer is allocated, so it holds at most one freed buffer past it.
    assert 0 < mx.get_cache_memory() <= CACHE_LIMIT + max(sizes)


def test_mlx_says_in_megabytes_what_its_live_arrays_hold_and_what_its_cache_holds_once_they_are_freed(limited: None) -> None:
    before = memory.mlx_held()
    held = mx.zeros((256 * MEGABYTE // 4,), mx.float32)
    mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
    live = memory.mlx_held()
    del held
    # The GPU holds the array's buffer until the work that wrote it is done, and only then is it freed to the cache.
    mx.synchronize()
    freed = memory.mlx_held()
    assert (round(live["mlx_active_mb"] - before["mlx_active_mb"]), round(freed["mlx_cache_mb"] - live["mlx_cache_mb"])) == (256, 256)


def test_a_sample_is_one_event_saying_the_footprint_and_mlx_split() -> None:
    kept = list[Entry]()
    held = {"mlx_active_mb": 2587.0, "mlx_cache_mb": 1024.0}
    sample(VirtualClock(), kept.append, read=lambda: 6354 * MEGABYTE, held=lambda: held)
    samples = [entry for entry in kept if isinstance(entry, WideEvent)]
    assert [(event.event, event.outcome, event.facts) for event in samples] == [(SAMPLED, "ok", {"footprint_mb": 6354.0, **held})]


def test_a_footprint_the_kernel_refuses_is_said_on_the_sample() -> None:
    kept = list[Entry]()

    def refused() -> int:
        raise PermissionError(1, "Operation not permitted")

    sample(VirtualClock(), kept.append, read=refused, held=dict)
    samples = [entry for entry in kept if isinstance(entry, WideEvent)]
    assert [(event.outcome, event.facts) for event in samples] == [("ok", {"footprint_error": "[Errno 1] Operation not permitted"})]
