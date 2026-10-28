"""The process's physical footprint counts the memory it holds wherever it was written, and a reading says its size or
why it was not read."""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest

from hands import footprint

MEGABYTE = 2**20


def numpy_array(size: int) -> object:
    held = np.ones(size, np.uint8)
    assert held.sum() == size
    return held


def mlx_array(size: int) -> object:
    # Written by MLX's default device: on Metal, memory the resident set size never counts.
    # A buffer an earlier test let go of is not taken back from MLX's cache: it is grown anew, as a model's load grows it.
    mx.clear_cache()
    held = mx.ones((size,), dtype=mx.uint8)
    mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
    return held


@pytest.mark.parametrize("hold", [numpy_array, mlx_array])
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
