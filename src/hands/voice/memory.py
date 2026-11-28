"""The run's memory: the bound on what MLX keeps of the GPU memory its models free, and a sample of what the process
holds, once a period, as the run goes.

MLX keeps each Metal buffer an array frees in a cache, to hand to the next array of a size it fits, and by default lets
that cache grow to the GPU's whole recommended working set, 107 GB on this Mac. The streaming pair allocates arrays of a
new size with each chunk it takes, which a cached buffer seldom fits, so the cache only grows: fed 60 clips, the pair's
live arrays stayed at 2.6 GB and peaked at 3.0 GB while the cache held 11.3 GB more, all of it in the process's footprint
as IOAccelerator memory, and in the daemon the footprint went from 6.4 GB after its loads to 19 GB in eleven minutes
(promptctl-voice-i15t.uib). CACHE_LIMIT bounds it: the same 60 clips under limits of 0, 256 MB and 1 GB each held the
cache at its limit, with the live arrays and their peak unchanged, and decoded within the run-to-run spread of a GPU the
running daemon shared.

[LAW:nothing-unseen] the sample is a `memory.sampled` wide event: the footprint, and MLX's own split of the GPU memory it
holds, live and cached. It goes where every wide event goes, to the audit log and to the collector where one is set, so a
footprint that grows past its size after the loads shows in hands' own telemetry, and which of the two grew with it. The
split is read on the MLX thread, so the sample is that thread's pulse too: the work waiting there as it was taken, and
how long the thread took to answer, the event's own time; a thread that does not answer within SPLIT_SECONDS is said to
have not, so one that hangs is seen within a minute, and the footprint is never held up behind it.
"""

from collections.abc import Callable, Mapping

from hands import footprint
from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.wide import Fact, annotate, begun, unit
from hands.voice import mlxthread
from hands.voice.mlxthread import OnMLX

# The most MLX keeps cached of the Metal buffers its arrays free, in bytes.
CACHE_LIMIT = 2**30
# How often `hands run` samples the memory.
SAMPLE_SECONDS = 60.0
# How long a sample waits for MLX's split behind the work the MLX thread has: one model's piece is a third of a second, a
# load a few seconds.
SPLIT_SECONDS = 10.0
SAMPLED = "memory.sampled"


def limit_cache(on: OnMLX) -> int:
    """Bound MLX's cache of freed buffers at CACHE_LIMIT, for every model the process runs on MLX, and say the bound
    it had before."""
    import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]

    return mx.set_cache_limit(CACHE_LIMIT)


def mlx_held(on: OnMLX) -> Mapping[str, float]:
    """MLX's split of the GPU memory it holds: its live arrays, and its cache of freed buffers."""
    import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]

    return {"mlx_active_mb": footprint.megabytes(mx.get_active_memory()), "mlx_cache_mb": footprint.megabytes(mx.get_cache_memory())}


async def keep_sampling(clock: Clock, record: Record) -> None:
    """Sample now and SAMPLE_SECONDS after each sample ends, until cancelled: a sample waiting on the MLX thread never has
    another queued behind it."""
    while True:
        await sample(clock, record)
        await clock.sleep(SAMPLE_SECONDS)


async def sample(clock: Clock, record: Record, read: Callable[[], int] = footprint.footprint, held: Callable[[OnMLX], Mapping[str, Fact]] = mlx_held, waiting: Callable[[], int] = mlxthread.waiting) -> None:
    """One sample of the process's memory, as one event: the footprint and the MLX work waiting, now, and MLX's split, as
    the MLX thread reads it, or that it did not within SPLIT_SECONDS. A footprint the kernel refuses is on the event as
    `footprint_error`."""
    with unit(SAMPLED, record, begun(clock)):
        annotate(**footprint.size(footprint.measured(read)), mlx_waiting=waiting())
        bound = clock.timeout(SPLIT_SECONDS)
        try:
            async with bound:
                split = await mlxthread.settles(held)
        except TimeoutError:
            annotate(mlx_split_error=f"the MLX thread did not answer in {SPLIT_SECONDS:g} s")
        else:
            annotate(**split)
