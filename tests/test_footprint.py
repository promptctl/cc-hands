"""The process's physical footprint counts the memory it holds wherever it was written, and a reading says its size or
why it was not read."""

import subprocess
import sys

import pytest

import footprint_hold
from footprint_hold import HELD_MB, HOLDS
from hands import footprint

MEGABYTE = 2**20


# The reading is of a whole process, so it is taken in one where nothing runs beside the array. In the full suite's own
# process one reading was 63 MB over the hold, and the cause was not found. No timeout is given here: pytest's is the
# one that ends a hang, and a child's stderr is not captured, so a child that fails says why in the report.
@pytest.mark.outside
@pytest.mark.parametrize("hold", HOLDS)
def test_the_footprint_grows_by_the_memory_the_process_holds(hold: str) -> None:
    ran = subprocess.run([sys.executable, footprint_hold.__file__, hold], stdout=subprocess.PIPE, text=True, check=True)
    grew = float(ran.stdout)
    assert HELD_MB - 10 < grew < HELD_MB * 1.3, f"a {hold} hold of {HELD_MB} MB grew the footprint {grew:.1f} MB"


def test_the_kernel_reads_this_process_as_holding_something() -> None:
    assert footprint.footprint() > 10 * MEGABYTE


def test_one_reading_says_its_size_or_why_it_was_not_read() -> None:
    def refused() -> int:
        raise OSError("refused")

    assert footprint.size(footprint.measured(lambda: 512 * MEGABYTE)) == {"footprint_mb": 512.0}
    assert footprint.size(footprint.measured(refused)) == {"footprint_error": "refused"}
