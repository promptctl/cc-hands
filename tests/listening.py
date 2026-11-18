"""Clips fed to the pair as hands' listener feeds them, for the real-pair tests and the measurements behind them
(scripts/measuring.py): [LAW:one-source-of-truth] what a test asserts and what a script measured are cut and heard alike."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hands.voice import engaged, streaming
from hands.voice.streaming import RATE, Ended, Samples, Session, Spoken, through

# A turn's clip opens LEAD before its first word and ends `after` past its last, in one of the two shapes a clip takes:
# the shortcut, let go RELEASE after it, or engaged conversation, closed on the STOP_SECS of silence after it.
LEAD = 0.1
RELEASE = 0.2
SHAPES = {"shortcut": RELEASE, "engaged": engaged.STOP_SECS}


def turn(audio: Samples, start: float, end: float, after: float) -> Samples:
    """The clip of a turn said from `start` to `end`, in seconds of `audio`, ending `after` past its last word."""
    first, last = int((start - LEAD) * RATE), int((end + after) * RATE)
    # [LAW:no-silent-failure] a slice past either end of the audio would be a shorter or empty clip, quietly; a sample
    # past the end is rounding.
    if first < 0 or last > len(audio) + 1:
        raise ValueError(f"a clip from {LEAD} s before its first word to {after} s past its last runs outside the audio's {len(audio) / RATE:.3f} s")
    return audio[first:last]


class Live:
    """One stream of the real pair, fed clips as the listener feeds it: 20 ms at a time, each followed by the silence
    `through` says, and closed."""

    def __init__(self, paired: Session) -> None:
        self.stream = streaming.Stream(paired)
        self.fed = 0

    def spoken(self, samples: Samples) -> tuple[Spoken, Ended]:
        """A clip of `samples`: each speaker's words heard while it was open, and the clip as it ended."""
        self.stream.opened(np.zeros(0, np.float32), self.fed)
        for start in range(0, len(samples), 320):
            self.stream.heard(samples[start : start + 320])
        self.fed += len(samples)
        live = self.stream.said().added
        silence = through(self.fed) - self.fed
        self.stream.silence(silence)
        self.fed += silence
        return live, self.stream.ended()


@dataclass
class Timed:
    """The pair's mlx-audio session, keeping each token it decodes as it is fed, with its speaker and when it starts, in
    seconds of the run: mlx-audio's own alignment, which `Said` leaves out."""

    session: Any
    tokens: list[tuple[int, float, str]] = field(default_factory=list[tuple[int, float, str]])

    def feed(self, samples: Samples) -> Any:
        decoded = self.session.feed(samples)
        self.tokens += [(streaming.speaker(delta.speaker), token.start, token.text) for delta in decoded for token in delta.tokens]
        return decoded

    def __getattr__(self, name: str) -> Any:
        # Everything else the pair's close reaches is the session's own.
        return getattr(self.session, name)


@contextmanager
def tapped(pair: Any) -> Generator[Timed]:
    """The pair's own session tapped in place for the length of the block, and put back after: a copy of the pair
    (`dataclasses.replace`) wraps the session's step again, and only its first decode would carry the diarizer's frames."""
    tap = Timed(pair.session)
    pair.session = tap
    try:
        yield tap
    finally:
        pair.session = tap.session
