"""The wake word being trained: the phrase as the user says it, and the names derived from it."""

import re
from dataclasses import dataclass

# Words said before another, in speech and in other wake words: "okay hands" and "my hands" are not "hey hands".
LEADS = ("hey", "okay", "my", "the", "your", "these")

# Words of letters, an apostrophe inside one allowed ("what's"), one space between words: what TTS can say and a file
# name can carry.
_SAID = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?(?: [A-Za-z]+(?:'[A-Za-z]+)?)*")


class Rejected(ValueError):
    """A phrase no wake word can be trained on."""


@dataclass(frozen=True)
class Word:
    """A wake word: `phrase` as hands shows it ("Hey Hands"). Every Word is one that can be trained."""

    phrase: str

    def __post_init__(self) -> None:
        if not _SAID.fullmatch(self.phrase):
            raise Rejected(f"{self.phrase!r} is not a phrase to train: words of letters, one space between them, such as 'Hey Hands'")

    @property
    def spoken(self) -> str:
        """What the synthetic voices say, and the trainer's target phrase."""
        return self.phrase.lower()

    @property
    def near_misses(self) -> tuple[str, ...]:
        """What is said that is not the word but shares its sound: each part of it said alone ("hey", "hands"), and
        each of its endings led by another word ("okay hands", "my hands"). A word of one word has neither."""
        said = self.spoken.split()
        parts = {" ".join(said[start:end]) for start in range(len(said)) for end in range(start + 1, len(said) + 1)}
        led = {" ".join((lead, *said[start:])) for lead in LEADS for start in range(1, len(said))}
        return tuple(sorted((parts | led) - {self.spoken}))

    @property
    def name(self) -> str:
        """The model's name, and the stem of its file: "hey_hands"."""
        return self.spoken.replace("'", "").replace(" ", "_")


def word(said: str) -> Word:
    """The wake word in `said`, however it is spaced."""
    return Word(" ".join(said.split()))
