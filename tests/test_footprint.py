"""The process's physical footprint counts the memory it holds wherever it was written, and a reading says its size or
why it was not read."""

import subprocess
import sys
from pathlib import Path

import pytest

from hands import footprint

MEGABYTE = 2**20


# The reading is of a whole process, so it is taken in one that holds nothing but the array. In the full suite's own
# process it read 63 MB high on CI and passed alone, and what else grew that process was never found (promptctl-voice-i15t.65f.fnt).
@pytest.mark.outside
@pytest.mark.parametrize("hold", ["numpy", "mlx"])
def test_the_footprint_grows_by_the_memory_the_process_holds(hold: str) -> None:
    script = Path(__file__).with_name("footprint_hold.py")
    ran = subprocess.run([sys.executable, str(script), hold], capture_output=True, text=True, check=True, timeout=120)
    grew = float(ran.stdout)
    assert 190 < grew < 260, f"a {hold} hold of 200 MB grew the footprint {grew:.1f} MB"


def test_one_reading_says_its_size_or_why_it_was_not_read() -> None:
    def refused() -> int:
        raise OSError("refused")

    assert footprint.size(footprint.measured(lambda: 512 * MEGABYTE)) == {"footprint_mb": 512.0}
    assert footprint.size(footprint.measured(refused)) == {"footprint_error": "refused"}
