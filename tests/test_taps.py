"""Triple taps: three presses of Right Shift or Right Command alone, inside a second, with nothing else between."""

from collections.abc import Sequence

import pytest

from hands.voice.hold import CommandPressed, CommandReleased, KeyEvent, Pressed, Released, Ripe, Typed
from hands.voice.taps import TAP_SECONDS, TapKey, Taps, counted


def tapped(events: Sequence[KeyEvent]) -> list[TapKey]:
    taps = Taps()
    made: list[TapKey] = []
    for event in events:
        taps, done = counted(taps, event)
        made.extend([done] if done else [])
    return made


def shift(at: float) -> list[KeyEvent]:
    return [Pressed(at), Released()]


def command(at: float) -> list[KeyEvent]:
    return [CommandPressed(at), CommandReleased()]


@pytest.mark.parametrize(
    ("events", "made"),
    [
        # Three presses alone, inside the second: one tap, of Right Command on its third press, and of Right Shift as
        # its third press is let go, known then to be no hold.
        ([*shift(1.0), *shift(1.2), *shift(1.4)], ["right shift"]),
        ([*command(1.0), *command(1.2), *command(1.4)], ["right command"]),
        ([*command(1.0), *command(1.2), CommandPressed(1.4)], ["right command"]),
        ([*shift(1.0), *shift(1.2), Pressed(1.4)], []),
        # Two are nothing.
        ([*shift(1.0), *shift(1.2)], []),
        # Releases and the hold's own timings break no count: a hold that opens a turn is the desk's to reset.
        ([*shift(1.0), Ripe(1.0), *shift(1.2), *shift(1.4)], ["right shift"]),
        # Any other key between, as a capital typed or a shortcut, starts the count again.
        ([*shift(1.0), Typed(), *shift(1.2), *shift(1.4)], []),
        ([*command(1.0), *command(1.2), Typed(), *command(1.4)], []),
        # The other modifier between is no tap of the first.
        ([*shift(1.0), *command(1.1), *shift(1.2), *shift(1.4)], []),
        # A third press past the second from the first starts a new count, which the next two complete.
        ([*shift(1.0), *shift(1.5), *shift(1.0 + TAP_SECONDS + 0.1), *shift(2.2), *shift(2.3)], ["right shift"]),
        # Six presses are two taps, not one held open.
        ([*command(1.0), *command(1.1), *command(1.2), *command(1.3), *command(1.4), *command(1.5)], ["right command", "right command"]),
    ],
)
def test_a_triple_tap_is_three_presses_alone_inside_a_second(events: list[KeyEvent], made: list[TapKey]) -> None:
    assert tapped(events) == made
