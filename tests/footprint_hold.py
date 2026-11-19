"""A process that holds HELD_MB as numpy or MLX writes it and prints how much its footprint grew, in megabytes.

tests/test_footprint.py runs it: a fresh process holds nothing but the array, so no earlier test's memory is in the
reading."""

import sys
from collections.abc import Callable

import mlx.core as mx
import numpy as np

from hands import footprint

HELD_MB = 200
_WARMUP = 2**20


# Each hold is uniform random bytes and allocates nothing beside itself: the footprint counts a compressed page at its
# compressed size, and macOS compresses under memory pressure, which a fresh process does not escape.
def numpy_array(size: int) -> object:
    return np.random.default_rng(0).integers(0, 256, size, np.uint8)


def mlx_array(size: int) -> object:
    # Written by MLX's default device: on Metal, memory the resident set size never counts, which the footprint does.
    # Split keys are raw Threefry bits. mx.random.randint is neither: it scales a float32 uniform, whose low bits are
    # zeros, through a float32 buffer the size of its output.
    held = mx.random.split(mx.random.key(0), size // 8).view(mx.uint8)
    mx.eval(held)  # pyright: ignore[reportUnknownMemberType]
    return held


HOLDS: dict[str, Callable[[int], object]] = {"numpy": numpy_array, "mlx": mlx_array}


def main(name: str) -> None:
    hold = HOLDS[name]
    # The library's own start-up, Metal's device and kernels for MLX, grows the process by about 160 MB too: it is had
    # first, so that what is measured is the array alone.
    hold(_WARMUP)
    before = footprint.footprint()
    _held = hold(HELD_MB * 2**20)  # held to the reading, which counts what is alive
    print(footprint.megabytes(footprint.footprint() - before))


if __name__ == "__main__":
    main(sys.argv[1])
