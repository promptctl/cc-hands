"""What hands' measurement scripts share: public speech fetched once, heard as hands hears it, and AMI cut into turns.

[LAW:one-source-of-truth] every script measures on these, so their numbers come from one feed and one turn rule, and the
feed is the real-pair tests' own (tests/listening.py), so what a test asserts is what a script measured.
"""

import sys
import urllib.request
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from hands.voice import streaming

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from listening import SHAPES, Live, turn  # noqa: E402

RATE = streaming.RATE
AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
# A turn: one person's speech with no gap of GAP or more inside it, at least SHORTEST long.
GAP = 0.5
SHORTEST = 0.8


def fetched(directory: Path, url: str) -> Path:
    path = directory / url.rsplit("/", 1)[1]
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def mono(path: Path) -> np.ndarray:
    """Mono audio at RATE."""
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    common = gcd(rate, RATE)
    return resample_poly(audio.mean(axis=1), RATE // common, rate // common).astype(np.float32)


@dataclass(frozen=True)
class Stretch:
    """A stretch of a meeting, in seconds, one person is heard in, with the words said in it where the annotation has
    them."""

    start: float
    end: float
    person: str
    words: tuple[str, ...] = ()


def turns(spoken: list[Stretch], sounds: list[Stretch], recorded: float, before: float = 0.0, after: float = 0.0) -> list[Stretch]:
    """Each person's spoken stretches, merged across gaps under GAP, that are at least SHORTEST long, that nobody else
    sounds in, and that nobody sounds in the `before` ahead of or the `after` past, the recording, `recorded` seconds
    long, holding both silences: a turn that ends in `after` of silence."""
    merged: list[Stretch] = []
    for stretch in sorted(spoken, key=lambda stretch: (stretch.start, stretch.end)):
        if merged and merged[-1].person == stretch.person and stretch.start - merged[-1].end < GAP:
            last = merged[-1]
            merged[-1] = Stretch(last.start, max(last.end, stretch.end), last.person, last.words + stretch.words)
        else:
            merged.append(stretch)

    def heard_over(turn: Stretch, sound: Stretch) -> bool:
        # The turn's own person, heard within it, is the turn; anyone heard around it, or them before or after it, a laugh
        # that runs on past its last word included, is not.
        inside = sound.person == turn.person and turn.start <= sound.start and sound.end <= turn.end
        return sound.start < turn.end + after and sound.end > turn.start - before and not inside

    # A transcript says nothing outside the recording, so a turn at either end of it is not known to be set in silence.
    held = [turn for turn in merged if turn.end - turn.start >= SHORTEST and 0 <= turn.start - before and turn.end + after <= recorded]
    return [turn for turn in held if not any(heard_over(turn, sound) for sound in sounds)]
