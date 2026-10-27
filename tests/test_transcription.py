"""Parakeet v3 on MLX: a hold heard in windows and the windows merged into one transcript, with a stand-in model, and the
real model on real speech."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile  # pyright: ignore[reportMissingTypeStubs]
from parakeet_mlx.alignment import AlignedToken

from hands.voice import transcription
from hands.voice.transcription import RATE, Model, Samples, load as parakeet, transcribed

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Counting:
    """A model that hears, in each second of a window, the word that second's first sample numbers, timed where it falls
    in the window: what the hold holds is known to the second, and so is what each window heard of it."""

    windows: list[float] = field(default_factory=list[float])

    def window(self, samples: Samples) -> list[AlignedToken]:
        self.windows.append(len(samples) / RATE)
        numbered = (round(float(samples[second * RATE]) * 1000) for second in range(len(samples) // RATE))
        return [AlignedToken(number, f" w{number}", float(second), 0.5) for second, number in enumerate(numbered)]


def numbered(seconds: int) -> Samples:
    """A hold of `seconds`, each second's samples numbering it."""
    return np.repeat(np.arange(seconds) / 1000, RATE).astype(np.float32)


def test_a_short_hold_is_heard_whole_in_one_window() -> None:
    counting = Counting()
    assert transcribed(Model(counting.window), numbered(3)) == "w0 w1 w2"
    assert counting.windows == [3.0]


def test_a_long_hold_is_heard_in_windows_no_longer_than_one_and_every_word_in_it_once() -> None:
    counting = Counting()
    seconds = 2 * transcription.WINDOW_SECONDS + 10

    assert transcribed(Model(counting.window), numbered(seconds)) == " ".join(f"w{second}" for second in range(seconds))
    # [LAW:behavior-not-structure] the window bounds what the model holds at once: ten minutes heard whole took 15 GB.
    assert len(counting.windows) > 1 and max(counting.windows) == transcription.WINDOW_SECONDS


# The real model, on real speech: the weights are fetched into the Hugging Face cache the first time, and CI keeps that
# cache.


@pytest.fixture(scope="module")
def model() -> Model:
    return parakeet()


def test_the_real_model_hears_what_was_read_aloud(model: Model) -> None:
    # One LibriVox reader, 20 to 30 dB down (The Invisible Man, read solo; public domain).
    audio, rate = cast(tuple[Samples, int], soundfile.read(FIXTURES / "one_reader_quiet.flac", dtype="float32"))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
    assert rate == RATE
    heard = transcribed(model, audio)
    assert heard.startswith("Door was ajar.") and heard.endswith("His big slouch hat, even, was cocked jauntily over the bed.")


def test_the_real_model_hears_nothing_in_silence(model: Model) -> None:
    assert transcribed(model, np.zeros(2 * RATE, dtype=np.float32)) == ""
