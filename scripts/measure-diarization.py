"""Measure hands' own diarization (hands.voice.streaming.Stream, the paired session) on public speech, fed in holds as
the listener feeds it.

    uv run scripts/measure-diarization.py DIR    # fetches what it needs into DIR once, then measures

A hold's speaker is the speaker who said most of its words; a hold with no words is heard as nobody.

- One person for an hour: The Invisible Man, chapters 1-3, read solo for LibriVox (public domain), cut into holds at
  its pauses. One speaker, nothing of it given to anyone else, is the pass.
- Two meetings of four people: AMI Meeting Corpus ES2004a and IS1009a (CC BY 4.0), test meetings, which the model was
  not trained on, cut into holds at each speaker's own turns where nobody else speaks, per pyannote's AMI-diarization-setup
  references. People and speakers are paired one to one, as many holds as possible on their pair (the optimal
  assignment); a hold is right where its speaker is its own person's pair, so two people heard as one speaker count one of
  them wrong.
"""

import sys
import urllib.request
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import soundfile
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

from hands.voice import streaming

RATE = streaming.RATE
LIBRIVOX = "https://archive.org/download/invisibleman_1209_librivox_cb/invisibleman_{:02d}_wells_64kb.mp3"
AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
AMI_TURNS = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/{}.rttm"
MEETINGS = ("ES2004a", "IS1009a")


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


class Run:
    """One stream of the pair for the run, fed holds 20 ms at a time, each followed by the silence `through` says."""

    def __init__(self) -> None:
        self.stream, self.fed = streaming.Stream(streaming.load()), 0

    def spoken(self, samples: np.ndarray) -> Counter[int]:
        """A hold: how many words each speaker said in it."""
        self.stream.opened(np.zeros(0, np.float32))
        for start in range(0, len(samples), 320):
            self.stream.heard(samples[start : start + 320].astype(np.float32))
        self.fed += len(samples)
        silence = streaming.through(self.fed) - self.fed
        self.stream.silence(silence)
        self.fed += silence
        return Counter({said.speaker: len(said.text.split()) for said in self.stream.ended().words})


def pauses(audio: np.ndarray) -> list[tuple[int, int]]:
    """Spans of `audio` louder than 35 dB below its loudest, in 1024-sample frames centred every 160 samples, as
    librosa.effects.split cuts them."""
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, 512), 1024)[::160]
    level = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)
    loud = np.concatenate([[False], level > level.max() - 35, [False]])
    edges = np.flatnonzero(np.diff(loud.astype(np.int8)))
    return [(start * 160, min(end * 160, len(audio))) for start, end in zip(edges[0::2], edges[1::2])]


def one_person(directory: Path) -> None:
    audio = np.concatenate([mono(fetched(directory, LIBRIVOX.format(n))) for n in (1, 2, 3)])
    # A hold is speech between pauses of a quarter second or more, as a person holding the key would cut it.
    holds: list[tuple[int, int]] = []
    for start, end in pauses(audio):
        if holds and start - holds[-1][1] < RATE // 4:
            holds[-1] = (holds[-1][0], end)
        else:
            holds.append((start, end))
    run, words, wordless = Run(), Counter[int](), 0
    for start, end in holds:
        said = run.spoken(audio[start:end])
        words.update(said)
        wordless += not said
    first, *others = [speaker for speaker, _ in words.most_common()] or [None]
    print(
        f"one person, {len(audio) / RATE / 60:.1f} min in {len(holds)} holds: words by speaker {dict(words.most_common())}; "
        f"{sum(words[other] for other in others)} words given to anyone but speaker {first}; {wordless} holds with no words"
    )


def meeting(directory: Path, name: str) -> None:
    audio = mono(fetched(directory, AMI_AUDIO.format(name)))
    segments = sorted((float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (line.split() for line in fetched(directory, AMI_TURNS.format(name)).read_text().splitlines()))
    turns: list[tuple[float, float, str]] = []
    for start, end, who in segments:
        if turns and turns[-1][2] == who and start - turns[-1][1] < 0.5:
            turns[-1] = (turns[-1][0], max(end, turns[-1][1]), who)
        else:
            turns.append((start, end, who))
    alone = [turn for turn in turns if turn[1] - turn[0] >= 0.8 and not any(o[2] != turn[2] and o[0] < turn[1] and o[1] > turn[0] for o in segments)]
    run, heard = Run(), list[tuple[str, int | None, float]]()
    for start, end, who in alone:
        said = run.spoken(audio[int(start * RATE) : int(end * RATE)])
        heard.append((who, said.most_common(1)[0][0] if said else None, end - start))
    people = sorted({who for who, _, _ in heard})
    speakers = sorted({first for _, first, _ in heard if first is not None})
    together = np.array([[sum(1 for w, first, _ in heard if (w, first) == (who, speaker)) for speaker in speakers] for who in people])
    paired = dict(zip(*linear_sum_assignment(-together)))
    mapped = {who: speakers[paired[row]] if row in paired else None for row, who in enumerate(people)}
    right = [who for who, first, _ in heard if first is not None and first == mapped[who]]
    short = [(who, first) for who, first, held in heard if held < 2]
    print(
        f"{name}: {len(people)} people, {len(heard)} holds; speakers found {len(speakers)}; "
        f"holds given their own person's speaker {len(right)}/{len(heard)}, under 2 s {sum(1 for who, first in short if first is not None and first == mapped[who])}/{len(short)}; "
        f"heard as nobody {sum(1 for _, first, _ in heard if first is None)}"
    )
    for who in people:
        print(f"  {who}: {Counter(first for w, first, _ in heard if w == who).most_common()}")


def main() -> None:
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    one_person(directory)
    for name in MEETINGS:
        meeting(directory, name)


if __name__ == "__main__":
    main()
