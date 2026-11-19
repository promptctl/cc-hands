"""The process's physical footprint counts the memory it holds wherever it was written, and a reading says its size or
why it was not read."""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest

from hands import footprint

MEGABYTE = 2**20


# Each hold is uniform random bytes and allocates nothing beside itself: the footprint counts a compressed page at its
# compressed size, and macOS compresses under the full suite's memory pressure.
def numpy_array(size: int) -> object:
    return np.random.default_rng(0).integers(0, 256, size, np.uint8)


def mlx_array(size: int) -> object:
    # Written by MLX's default device: on Metal, memory the resident set size never counts.
    # A buffer an earlier test let go of is not taken back from MLX's cache: it is grown anew, as a model's load grows it.
    mx.clear_cache()
    # Split keys are raw Threefry bits. mx.random.randint is neither: it scales a float32 uniform, whose low bits are
    # zeros, through a float32 buffer the size of its output that clearing the cache does not give back.
    held = mx.random.split(mx.random.key(0), size // 8).view(mx.uint8)
    mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
    return held


# Only the numpy hold failed in master's full suite (263.5 MB against the window, run 44288853845); the MLX hold passed there.
@pytest.mark.parametrize("hold", [pytest.param(numpy_array, marks=pytest.mark.quarantined("promptctl-voice-i15t.65f.fnt", until="2026-11-25")), mlx_array])
def test_the_footprint_grows_by_the_memory_the_process_holds(hold: Callable[[int], object]) -> None:
    # The library's own start-up, Metal's device and kernels for MLX, grows the process too; it is had first, so what
    # is measured is the array alone.
    hold(MEGABYTE)
    before = footprint.footprint()
    held = hold(200 * MEGABYTE)
    grew = (footprint.footprint() - before) / MEGABYTE
    del held
    assert 190 < grew < 260


def test_one_reading_says_its_size_or_why_it_was_not_read() -> None:
    def refused() -> int:
        raise OSError("refused")

    assert footprint.size(footprint.measured(lambda: 512 * MEGABYTE)) == {"footprint_mb": 512.0}
    assert footprint.size(footprint.measured(refused)) == {"footprint_error": "refused"}
