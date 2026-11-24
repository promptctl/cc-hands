"""A file of lines, read back from its end: complete lines only, newest first, at the cost of what is read and not the
file's size. Each line ends at the newline its writer ended it with."""

import io
import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def newest_first(path: Path) -> Iterator[str]:
    """The complete lines of path, newest first, read back from its end a block at a time; a path that is gone holds
    none. A line still being written is not yet among them."""
    # [LAW:no-ambient-temporal-coupling] the end is found in the file the lines are read back from: opened once, it is
    # the same file for both, whatever its path comes to name.
    with held(path) as log:
        yield from before(log, complete(log))


def complete(log: BinaryIO) -> int:
    """How many bytes of log are complete lines: a line still being written is not among them."""
    size = log.seek(0, os.SEEK_END)
    return next((start + block.rfind(b"\n") + 1 for start, block in _back(log, size) if b"\n" in block), 0)


def before(log: BinaryIO, end: int) -> Iterator[str]:
    """The line of log that ends at byte end, where one does, and each before it, newest first."""
    if end == 0:
        return
    # The blocks of the line being read that lie past the block in hand, the nearest first.
    later: list[bytes] = []
    # The newline at end - 1 ends the newest line: every newline before it ends an older one.
    for _, block in _back(log, end - 1):
        head, *rest = block.split(b"\n")
        if rest:
            yield text(rest.pop() + b"".join(reversed(later)))
            yield from map(text, reversed(rest))
            later = []
        later.append(head)
    yield text(b"".join(reversed(later)))


# How much of a file is read at a time, going back from its end.
_BLOCK = 64 * 1024


def _back(log: BinaryIO, end: int) -> Iterator[tuple[int, bytes]]:
    """The bytes of log before end a block at a time, the block nearest end first, each with the offset it begins at:
    reading back from the end costs what is read, and not the file's size."""
    for past in range(end, 0, -_BLOCK):
        start = max(0, past - _BLOCK)
        log.seek(start)
        yield start, log.read(past - start)


@contextmanager
def held(path: Path) -> Generator[BinaryIO]:
    """path, open to read. A path that is gone holds nothing: a file its keeper deleted or rotated away."""
    try:
        log: BinaryIO = path.open("rb")
    except FileNotFoundError:
        log = io.BytesIO()
    with log:
        yield log


def text(line: bytes) -> str:
    # [LAW:one-source-of-truth] a line ends at the newline its writer ended it with, and nowhere else: U+2028 and U+0085
    # are written raw inside a line's JSON. A write that failed partway can cut a character in two; its torn line reads
    # with U+FFFD in place of the half, and is then no JSON, as every torn line is.
    return line.decode("utf-8", errors="replace")
