"""Triple taps: Right Shift or Right Command pressed three times by itself, with nothing else between, sets the level the
desk listens at (`hands.voice.desk`). A pure machine from what the keyboard did to the tap it completed, if any.

A tap is a press of the modifier alone. Right Shift held long enough to mean talk (`hands.voice.hold`) is a hold and no
tap, so its third press completes the tap only as it is let go, once it is known not to be one. Typing never makes a
tap: a capital or a shortcut puts another key between the presses, and any other key starts the count again.
"""

from dataclasses import dataclass
from typing import Literal

from hands.voice.hold import CommandPressed, Instant, KeyEvent, Pressed, Released, Typed

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
            return _pressed(taps, "right shift", at), None
        case Released() if taps == Taps("right shift", TAPS, taps.first):
            return Taps(), "right shift"
        case CommandPressed(at=at):
            pressed = _pressed(taps, "right command", at)
            return (Taps(), "right command") if pressed.count == TAPS else (pressed, None)
        case Typed():
            return Taps(), None
        case _:
            return taps, None


def _pressed(taps: Taps, key: TapKey, at: Instant) -> Taps:
    if taps.key == key and at - taps.first <= TAP_SECONDS:
        return Taps(key, taps.count + 1, taps.first)
    return Taps(key, 1, at)
