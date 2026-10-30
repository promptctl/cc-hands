"""Holds fed to the pair as hands' listener feeds them, for the real-pair tests and the measurements behind them
(scripts/measuring.py): [LAW:one-source-of-truth] what a test asserts and what a script measured are cut and heard alike."""

from dataclasses import dataclass

import numpy as np

from hands.voice import engaged, streaming
from hands.voice.holds import Hold
from hands.voice.voiceprints import Voiced
from hands.voice.streaming import RATE, Closed, Decoded, Ended, Said, Samples, Session, Spoken, steps, through

# A turn's hold opens LEAD before its first word and ends `after` past its last, in one of the two shapes a hold takes:
# the held key, let go RELEASE after it, or engaged conversation, closed on the STOP_SECS of silence after it.
LEAD = 0.1
RELEASE = 0.2
SHAPES = {"held": RELEASE, "engaged": engaged.STOP_SECS}


def turn(audio: Samples, start: float, end: float, after: float) -> Samples:
    """The hold of a turn said from `start` to `end`, in seconds of `audio`, ending `after` past its last word."""
    first, last = int((start - LEAD) * RATE), int((end + after) * RATE)
    # [LAW:no-silent-failure] a slice past either end of the audio would be a shorter or empty hold, quietly; a sample
    # past the end is rounding.
    if first < 0 or last > len(audio) + 1:
        raise ValueError(f"a hold from {LEAD} s before its first word to {after} s past its last runs outside the audio's {len(audio) / RATE:.3f} s")
    return audio[first:last]


@dataclass
class Gridded:
    """The pair's session, counting each feed it decoded a word on that `steps` says completed no step of the
    diarizer's."""

    session: Session
    fed: int = 0
    off_grid: int = 0

    def decoded(self, samples: Samples) -> Decoded:
        decoded = self.session.decoded(samples)
        self.off_grid += bool(decoded.words) and steps(self.fed + len(samples)) == steps(self.fed)
        self.fed += len(samples)
        return decoded

    def closed(self) -> Closed:
        return self.session.closed()


def unidentified(_hold: Hold, _voiced: tuple[Voiced, ...]) -> None:
    """Nobody identified: what a test of the words alone measures is the words."""


class Live:
    """One stream of the real pair, fed holds as the listener feeds it: 20 ms at a time, each followed by the silence
    `through` says, and closed."""

    def __init__(self, paired: Session) -> None:
        self.session = Gridded(paired)
        self.stream = streaming.Stream(self.session)

    def spoken(self, samples: Samples) -> tuple[Spoken, Ended]:
        """A hold of `samples`: each speaker's words heard while it was open, and the hold as it ended."""
        self.stream.opened(np.zeros(0, np.float32), self.session.fed)
        for start in range(0, len(samples), 320):
            self.stream.heard(samples[start : start + 320])
        live = tuple(Said(voiced.speaker, voiced.words) for voiced in self.stream.said() if voiced.words)
        self.stream.silence(through(self.session.fed) - self.session.fed)
        return live, self.stream.ended()
