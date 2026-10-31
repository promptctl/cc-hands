"""Holds fed to the pair as hands' listener feeds them, for the real-pair tests and the measurements behind them
(scripts/measuring.py): [LAW:one-source-of-truth] what a test asserts and what a script measured are cut and heard alike."""

import numpy as np

from hands.voice import engaged, streaming
from hands.voice.holds import Hold
from hands.voice.voiceprints import Voiced
from hands.voice.streaming import RATE, Ended, Samples, Session, Spoken, through

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


class Unidentifying:
    """Nobody identified: what a test of the words alone measures is the words."""

    def identify(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None:
        pass

    def ended(self, hold: Hold, voiced: tuple[Voiced, ...]) -> None:
        pass


unidentified = Unidentifying()


class Live:
    """One stream of the real pair, fed holds as the listener feeds it: 20 ms at a time, each followed by the silence
    `through` says, and closed."""

    def __init__(self, paired: Session) -> None:
        self.stream = streaming.Stream(paired)
        self.fed = 0

    def spoken(self, samples: Samples) -> tuple[Spoken, Ended]:
        """A hold of `samples`: each speaker's words heard while it was open, and the hold as it ended."""
        self.stream.opened(np.zeros(0, np.float32), self.fed, ())
        for start in range(0, len(samples), 320):
            self.stream.heard(samples[start : start + 320], ())
        self.fed += len(samples)
        live = self.stream.said().added
        silence = through(self.fed) - self.fed
        self.stream.silence(silence)
        self.fed += silence
        return live, self.stream.ended()
