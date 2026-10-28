"""Triple taps: Right Shift or Right Command pressed three times by itself, with nothing else between, sets how the desk
listens (`hands.voice.engaged`). A pure machine from what the keyboard did to the tap it completed, if any.

A tap is a press of the modifier alone; what it is held for does not matter, short of Right Shift held long enough to
mean talk (`hands.voice.hold`), which is a hold and no tap. Typing never makes one: a capital or a shortcut puts another
key between the presses, and any other key starts the count again.
"""

from dataclasses import dataclass
from typing import Literal

from hands.voice.hold import CommandPressed, Instant, KeyEvent, Pressed, Typed

# From the first press to the third: three deliberate taps take about half of it.
TAP_SECONDS = 1.0
# How many presses a tap takes.
TAPS = 3

TapKey = Literal["right shift", "right command"]


@dataclass(frozen=True)
class Taps:
    """The presses counted so far: of which key, how many, from the first press at `first`."""

    key: TapKey | None = None
    count: int = 0
    first: Instant = 0.0


def counted(taps: Taps, event: KeyEvent) -> tuple[Taps, TapKey | None]:
    """The count after `event`, and the key whose triple tap it completed, if it did. Right Shift held to a turn is no
    tap: the desk, which takes the hold's moves, starts the count again as the turn opens (`Taps()`)."""
    match event:
        case Pressed(at=at):
            return _pressed(taps, "right shift", at)
        case CommandPressed(at=at):
            return _pressed(taps, "right command", at)
        case Typed():
            return Taps(), None
        case _:
            return taps, None


def _pressed(taps: Taps, key: TapKey, at: Instant) -> tuple[Taps, TapKey | None]:
    if taps.key == key and at - taps.first <= TAP_SECONDS:
        count, first = taps.count + 1, taps.first
    else:
        count, first = 1, at
    if count == TAPS:
        return Taps(), key
    return Taps(key, count, first), None
