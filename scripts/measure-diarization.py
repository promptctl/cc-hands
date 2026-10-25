"""Measure hands' own diarization (hands.voice.diarization.Stream) on public speech, fed in holds as Whisper feeds it.

    uv run scripts/measure-diarization.py DIR    # fetches what it needs into DIR once, then measures

- One person for an hour: The Invisible Man, chapters 1-3, read solo for LibriVox (public domain), cut into holds at
  its pauses. One speaker, nothing of it given to anyone else, is the pass.
- Two meetings of four people: AMI Meeting Corpus ES2004a and IS1009a (CC BY 4.0), test meetings, which the model was
  not trained on, cut into holds at each speaker's own turns where nobody else speaks, per pyannote's AMI-diarization-setup
  references. People and speakers are paired one to one, as many holds as possible on their pair (the optimal
  assignment); a hold is right where its first speaker is its own person's pair, so two people heard as one speaker
  count one of them wrong.
"""

import sys
import urllib.request
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
from scipy.optimize import linear_sum_assignment

from hands.voice import diarization

RATE = diarization.RATE
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


def spoken(stream: diarization.Stream, samples: np.ndarray) -> diarization.Diarized | diarization.Undiarized:
    """A hold spoken into the stream 20 ms at a time, as Whisper gives it, and ended."""
    for start in range(0, len(samples), 2 * diarization.FRAME):
        stream.heard(samples[start : start + 2 * diarization.FRAME].astype(np.float32))
    return stream.ended()


def one_person(directory: Path, model: diarization.Model) -> None:
    audio = np.concatenate([librosa.load(fetched(directory, LIBRIVOX.format(n)), sr=RATE)[0] for n in (1, 2, 3)])
    # A hold is speech between pauses of a quarter second or more, as a person holding the key would cut it.
    pauses = librosa.effects.split(audio, top_db=35, frame_length=1024, hop_length=diarization.FRAME)
    holds: list[tuple[int, int]] = []
    for start, end in pauses:
        if holds and start - holds[-1][1] < RATE // 4:
            holds[-1] = (holds[-1][0], end)
        else:
            holds.append((start, end))
    stream, seconds, undiarized = diarization.Stream(model), Counter[int](), 0
    for start, end in holds:
        match spoken(stream, audio[start:end]):
            case diarization.Diarized(speakers=speakers):
                seconds.update({speech.speaker: speech.seconds for speech in speakers})
            case diarization.Undiarized():
                undiarized += 1
    first, *others = [speaker for speaker, _ in seconds.most_common()] or [None]
    print(
        f"one person, {len(audio) / RATE / 60:.1f} min in {len(holds)} holds: speakers {dict(seconds.most_common())} s; "
        f"{sum(seconds[other] for other in others):.2f} s given to anyone but speaker {first}; {undiarized} holds undiarized"
    )


def meeting(directory: Path, model: diarization.Model, name: str) -> None:
    audio = librosa.load(fetched(directory, AMI_AUDIO.format(name)), sr=RATE)[0]
    segments = sorted((float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (line.split() for line in fetched(directory, AMI_TURNS.format(name)).read_text().splitlines()))
    turns: list[tuple[float, float, str]] = []
    for start, end, who in segments:
        if turns and turns[-1][2] == who and start - turns[-1][1] < 0.5:
            turns[-1] = (turns[-1][0], max(end, turns[-1][1]), who)
        else:
            turns.append((start, end, who))
    alone = [turn for turn in turns if turn[1] - turn[0] >= 0.8 and not any(o[2] != turn[2] and o[0] < turn[1] and o[1] > turn[0] for o in segments)]
    stream, heard = diarization.Stream(model), list[tuple[str, int | None, float]]()
    for start, end, who in alone:
        diarized = spoken(stream, audio[int(start * RATE) : int(end * RATE)])
        first = diarized.speakers[0].speaker if isinstance(diarized, diarization.Diarized) and diarized.speakers else None
        heard.append((who, first, end - start))
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
    model = diarization.load()
    one_person(directory, model)
    for name in MEETINGS:
        meeting(directory, model, name)


if __name__ == "__main__":
    main()
