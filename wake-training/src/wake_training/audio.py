"""Audio as openWakeWord's trainer takes it: 16 kHz, mono, 16-bit."""

from math import gcd
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly  # pyright: ignore[reportUnknownVariableType]  (untyped in scipy)

RATE = 16_000  # what openWakeWord hears, and the only rate its trainer takes


def write(path: Path, samples: NDArray[np.float64], rate: int) -> None:
    """`samples`, mono in [-1, 1) at `rate`, as a 16 kHz 16-bit WAV at `path`."""
    common = gcd(RATE, rate)
    resampled = cast(NDArray[np.float64], resample_poly(samples, RATE // common, rate // common))
    sf.write(path, np.clip(np.round(resampled * 32768), -32768, 32767).astype(np.int16), RATE, subtype="PCM_16")  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
