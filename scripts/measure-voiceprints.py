"""Measure how hands' voiceprint tells its own echo from a person, by how much of a speaker is scored: the numbers
hands.voice.voiceprints.FLOORS is fitted on.

    uv run scripts/measure-voiceprints.py CAPTURE [PERSON.wav|.flac ...]

CAPTURE is a directory recorded as promptctl-voice-i15t.99s.zoj's desk captures were:
  print.wav               hands' own TTS output, the print's source, as hands makes its print
  captured.wav            the room's microphone while hands played its sentences, no canceller, play starting PLAY_STARTS in
  spans.json              each sentence's start and end, in seconds, on the played timeline
  mic2.wav                the same sentences with a second voice played over them
  other_spans.json        {"other_spans": [...]}, the second voice's lines on its own timeline, which reaches the
                          microphone OTHER_LAG later
  other.wav               the second voice's own track, a person alone
Each PERSON is more of people alone, scored from each speech onset (the fixtures one_reader_quiet, two_speakers_ami and
turns_ending_in_silence were the plan's). Every window starts at an onset and runs WINDOWS seconds, so a row says what a
speaker's score is once that much of them is heard; a sentence, line or file that ends sooner is left out of that row,
never scored short in it. Each row of FLOORS is checked against its own length.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile
import soxr

from hands.voice import voiceprints
from hands.voice.holds import RATE, Samples

PLAY_STARTS = 0.5
OTHER_LAG = 1.0
WINDOWS = (0.5, 0.8, 1.0, 1.5, 3.0)


def mono(path: Path) -> Samples:
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return soxr.resample(samples.mean(axis=1), rate, RATE).astype(np.float32)


def onsets(samples: Samples, most: int = 40) -> list[float]:
    """Where speech starts after at least 0.3 s under 30 dB below the loudest 10 ms, in seconds; a file's start counts as
    that quiet, so speech from its first 10 ms is an onset."""
    hop = RATE // 100
    level = 20 * np.log10(np.sqrt(np.mean(samples[: len(samples) // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-9)
    quiet, found = 30, list[float]()
    for at, db in enumerate(level):
        if db < level.max() - 30:
            quiet += 1
            continue
        found += [at / 100] if quiet >= 30 else []
        quiet = 0
    return found[:most]


def summary(name: str, scores: list[float], worst: str) -> str:
    """One kind of speaker's scores in a row: how many, the worst of them for the floor, and the median."""
    match scores:
        case []:
            return f"{name} n=0"
        case _:
            extreme = min(scores) if worst == "min" else max(scores)
            return f"{name} n={len(scores)} {worst} {extreme:.2f} median {np.median(scores):.2f}"


def main(capture: Path, people: list[Path]) -> None:
    embed = voiceprints.load()
    hands = embed(mono(capture / "print.wav"))

    def scored(samples: Samples, start: float, seconds: float) -> float:
        return voiceprints.score(embed(samples[round(start * RATE) : round((start + seconds) * RATE)]), hands)

    captured, mic2 = mono(capture / "captured.wav"), mono(capture / "mic2.wav")
    spans = json.loads((capture / "spans.json").read_text())
    others = json.loads((capture / "other_spans.json").read_text())["other_spans"]
    alone = {path.stem: mono(path) for path in [capture / "other.wav", *people]}
    floors = {floor.seconds: floor.score for floor in voiceprints.FLOORS}
    for window in WINDOWS:
        # Only what holds the whole window: each row is scored at exactly its own length.
        echo = [scored(captured, span["start"] + PLAY_STARTS, window) for span in spans if span["end"] - span["start"] >= window]
        over = [scored(mic2, span["start"] + OTHER_LAG, window) for span in others if span["end"] - span["start"] >= window]
        lone = [scored(samples, at, window) for samples in alone.values() for at in onsets(samples) if at + window <= len(samples) / RATE]
        line = f"{window:.1f} s: {summary('echo', echo, 'min')} | {summary('over hands', over, 'max')} | {summary('alone', lone, 'max')}"
        floor = floors.get(window)
        # A row holds where every echo scores at or over it: no echo is ever a person.
        line += "" if floor is None else f" | floor {floor:.2f}: echo {'over' if all(score >= floor for score in echo) else 'UNDER'}, people under {sum(score < floor for score in over + lone)}/{len(over + lone)}"
        print(line)

if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(each) for each in sys.argv[2:]])
