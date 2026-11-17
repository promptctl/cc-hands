"""What hands' measurement scripts share: public speech fetched once, heard as hands hears it, and AMI cut into turns.

[LAW:one-source-of-truth] every script measures on these, so their numbers come from one feed and one turn rule, and the
feed is the real-pair tests' own (tests/listening.py), so what a test asserts is what a script measured.
"""

import sys
import urllib.request
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
from scipy.signal import resample_poly

from hands.voice import streaming

# Last on the path, so nothing in tests/ stands in for a module of the same name installed.
sys.path.append(str(Path(__file__).resolve().parent.parent / "tests"))
from listening import LEAD, SHAPES, Live, turn  # noqa: E402

RATE = streaming.RATE
AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
# Who sounds when in each AMI meeting, per pyannote's AMI-diarization-setup references, marking each person's words alone
# or their words and every laugh, cough and other sound they make.
AMI_REFERENCES = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/{0}/rttms/test/{1}.rttm"
type Marked = Literal["only_words", "word_and_vocalsounds"]
# AMI's sixteen test meetings, which the models were not trained on.
AMI_TEST = tuple(f"{series}{part}" for series in ("EN2002", "ES2004", "IS1009", "TS3003") for part in "abcd")
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


def referenced(directory: Path, name: str, marked: Marked) -> list[Stretch]:
    """Who sounds when in AMI meeting `name`, as pyannote's references mark it, fetched into `directory` once."""
    (directory / marked).mkdir(exist_ok=True)
    path = fetched(directory / marked, AMI_REFERENCES.format(marked, name))
    return [Stretch(float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (line.split() for line in path.read_text().splitlines())]


def heard_in(sound: Stretch, start: float, end: float) -> bool:
    """Whether `sound` is heard for any time between `start` and `end`, none when they are the same."""
    return min(sound.end, end) > max(sound.start, start)


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
        # Anyone else heard in the turn or the silences around it; the turn's own person only in those silences, so a
        # laugh of theirs that runs on past their last word into `after` is, and the part of it in the turn is the turn.
        if sound.person == turn.person:
            return heard_in(sound, turn.start - before, turn.start) or heard_in(sound, turn.end, turn.end + after)
        return heard_in(sound, turn.start - before, turn.end + after)

    # A transcript says nothing outside the recording, so a turn at either end of it is not known to be set in silence.
    held = [turn for turn in merged if turn.end - turn.start >= SHORTEST and 0 <= turn.start - before and turn.end + after <= recorded]
    return [turn for turn in held if not any(heard_over(turn, sound) for sound in sounds)]


def meeting_turns(directory: Path, name: str, recorded: float, before: float = 0.0, after: float = 0.0) -> list[Stretch]:
    """`turns` of AMI meeting `name`, `recorded` seconds long: what each person says is their words, and what is heard is
    every word, laugh, cough and other sound, per pyannote's references."""
    spoken, sounds = referenced(directory, name, "only_words"), referenced(directory, name, "word_and_vocalsounds")
    return turns(spoken, sounds, recorded, before, after)
