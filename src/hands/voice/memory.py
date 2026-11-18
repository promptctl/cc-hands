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

[LAW:nothing-unseen] the sample is a `memory.sampled` event: the footprint, and MLX's own split of the GPU memory it
holds, live, cached, and the most live at once since the process began. Its sizes are the metrics' hands_memory_bytes
(hands.sessions.metrics), so a footprint that grows past its size after the loads shows in hands' own telemetry.
"""

from collections.abc import Callable, Mapping

from hands import footprint
from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.metrics import MEMORY
from hands.sessions.wide import Fact, annotate, begun, unit

# The most MLX keeps cached of the Metal buffers its arrays free, in bytes.
CACHE_LIMIT = 2**30
# How often the memory is sampled, and so how often the metrics are pushed.
SAMPLE_SECONDS = 60.0


def limit_cache() -> None:
    """Bound MLX's cache of freed buffers at CACHE_LIMIT, for every model the process runs on MLX."""
    import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]

    mx.set_cache_limit(CACHE_LIMIT)


def mlx_held() -> Mapping[str, Fact]:
    """MLX's split of the GPU memory it holds: its live arrays, its cache of freed buffers, and the most live at once."""
    import mlx.core as mx  # pyright: ignore[reportMissingTypeStubs]

    return {
        "mlx_active_mb": footprint.megabytes(mx.get_active_memory()),
        "mlx_cache_mb": footprint.megabytes(mx.get_cache_memory()),
        "mlx_peak_mb": footprint.megabytes(mx.get_peak_memory()),
    }


async def keep_sampling(clock: Clock, record: Record, period: float = SAMPLE_SECONDS, read: Callable[[], int] = footprint.footprint, held: Callable[[], Mapping[str, Fact]] = mlx_held) -> None:
    """Sample the process's memory now and once a period after, until cancelled. A footprint the kernel refuses is on
    the event as `footprint_error`, and the sampling goes on."""
    while True:
        with unit(MEMORY, record, begun(clock)):
            annotate(**footprint.size(footprint.measured(read)), **held())
        await clock.sleep(period)
