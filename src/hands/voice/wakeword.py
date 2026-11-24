"""Which wake word the desk listens for at its wake word level (`hands.voice.wake`): one of openWakeWord's own, or
one the user trained with openWakeWord. config.toml names it (`hands.daemon.config`), which reads it without loading the
models, so nothing here does either.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# openWakeWord's own wake words, by how each is said.
type Phrase = Literal["Hey Jarvis", "Hey Mycroft", "Hey Rhasspy", "Alexa"]


# openWakeWord's own default: its pretrained models are tuned to score a wake word above it and little else.
THRESHOLD = 0.5


@dataclass(frozen=True)
class Pretrained:
    """One of openWakeWord's own wake words, its model fetched from openWakeWord's release."""

    phrase: Phrase = "Hey Jarvis"


@dataclass(frozen=True)
class Trained:
    """A wake word of the user's own: the ONNX model they trained with openWakeWord, and the phrase it was trained on,
    which is what hands tells them to say."""

    phrase: str
    model: Path


# [LAW:one-type-per-behavior] a word whose model hands fetches, or one whose model the user keeps.
type Word = Pretrained | Trained

