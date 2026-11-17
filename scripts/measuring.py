"""What hands' measurement scripts share: public speech fetched once, heard as hands hears it, and AMI cut into turns.

[LAW:one-source-of-truth] every script measures on these, so their numbers come from one feed and one turn rule, and the
feed is the real-pair tests' own (tests/listening.py), so what a test asserts is what a script measured.
"""

import sys
import urllib.request
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
from turntaking import Stretch, turns  # noqa: E402

RATE = streaming.RATE
AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
# Who sounds when in each AMI meeting, per pyannote's AMI-diarization-setup references, marking each person's words alone
# or their words and every laugh, cough and other sound they make.
AMI_REFERENCES = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/{0}/rttms/test/{1}.rttm"
type Marked = Literal["only_words", "word_and_vocalsounds"]
# AMI's sixteen test meetings, which the models were not trained on.
AMI_TEST = tuple(f"{series}{part}" for series in ("EN2002", "ES2004", "IS1009", "TS3003") for part in "abcd")


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


def referenced(directory: Path, name: str, marked: Marked) -> list[Stretch]:
    """Who sounds when in AMI meeting `name`, as pyannote's references mark it, fetched into `directory` once."""
    (directory / marked).mkdir(exist_ok=True)
    path = fetched(directory / marked, AMI_REFERENCES.format(marked, name))
    return [Stretch(float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (line.split() for line in path.read_text().splitlines())]


def meeting_turns(directory: Path, name: str, recorded: float, before: float = 0.0, after: float = 0.0) -> list[Stretch]:
    """`turns` of AMI meeting `name`, `recorded` seconds long: what each person says is their words, and what is heard is
    every word, laugh, cough and other sound, per pyannote's references."""
    spoken, sounds = referenced(directory, name, "only_words"), referenced(directory, name, "word_and_vocalsounds")
    return turns(spoken, sounds, recorded, before, after)
