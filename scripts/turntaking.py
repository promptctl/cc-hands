"""One person's turns in a recording, cut from who is heard when: the rule every AMI measurement is cut by."""

from dataclasses import dataclass

# A turn: one person's speech with no gap of GAP or more inside it, at least SHORTEST long.
GAP = 0.5
SHORTEST = 0.8


@dataclass(frozen=True)
class Stretch:
    """A stretch of a meeting, in seconds, one person is heard in, with the words said in it where the annotation has
    them."""

    start: float
    end: float
    person: str
    words: tuple[str, ...] = ()


def heard_in(sound: Stretch, start: float, end: float) -> bool:
    """Whether `sound`, a mark of no length included, is heard between `start` and `end`, nothing being heard in no time."""
    return start < end and sound.start < end and sound.end > start


def turns(spoken: list[Stretch], sounds: list[Stretch], recorded: float, before: float = 0.0, after: float = 0.0) -> list[Stretch]:
    """Each person's spoken stretches, merged across gaps under GAP, that are at least SHORTEST long, that nobody else
    sounds in, and that nobody sounds in the `before` ahead of or the `after` past, the recording, `recorded` seconds
    long, holding both silences: a turn that ends in `after` of silence."""
    merged: list[Stretch] = []
    for stretch in sorted(spoken, key=lambda stretch: (stretch.start, stretch.end)):
        if merged and merged[-1].person == stretch.person and stretch.start - merged[-1].end < GAP:
            last = merged[-1]
            merged[-1] = Stretch(last.start, max(last.end, stretch.end), last.person, last.words + stretch.words)
        else:
            merged.append(stretch)

    def heard_over(turn: Stretch, sound: Stretch) -> bool:
        # Anyone else counts if heard in the turn or the silences around it. The turn's own person counts only in those
        # silences: their laugh that runs on past their last word is heard over the turn where `after` asks for silence.
        if sound.person == turn.person:
            return heard_in(sound, turn.start - before, turn.start) or heard_in(sound, turn.end, turn.end + after)
        return heard_in(sound, turn.start - before, turn.end + after)

    # A transcript says nothing outside the recording, so a turn at either end of it is not known to be set in silence.
    held = [turn for turn in merged if turn.end - turn.start >= SHORTEST and 0 <= turn.start - before and turn.end + after <= recorded]
    return [turn for turn in held if not any(heard_over(turn, sound) for sound in sounds)]
