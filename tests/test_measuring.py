"""The turn rule every AMI measurement is cut by (scripts/measuring.py): a person's turn is kept only when nobody else is
heard in it or the silences asked for around it, and they themselves are not heard in those silences."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from measuring import Stretch, turns  # noqa: E402

TURN = Stretch(10.0, 12.0, "A")


def kept(*sounds: Stretch, before: float = 0.0, after: float = 0.0) -> bool:
    return turns([TURN], [TURN, *sounds], 60.0, before, after) == [TURN]


def test_a_turn_nobody_else_is_heard_in_is_kept() -> None:
    assert kept(Stretch(5.0, 9.0, "B"), Stretch(13.0, 14.0, "B"))


def test_anyone_else_heard_in_a_turn_drops_it_a_mark_of_no_length_included() -> None:
    assert not kept(Stretch(11.9, 13.0, "B"))
    assert not kept(Stretch(11.0, 11.0, "B"))


def test_anyone_else_heard_in_the_silence_asked_for_around_a_turn_drops_it() -> None:
    assert not kept(Stretch(12.2, 13.0, "B"), after=0.5)
    assert not kept(Stretch(9.0, 9.8, "B"), before=0.5)


def test_the_speakers_own_laugh_past_their_last_word_drops_the_turn_only_where_silence_is_asked_for_after_it() -> None:
    laugh = Stretch(11.5, 12.4, "A")
    assert kept(laugh)
    assert not kept(laugh, after=0.5)


def test_the_speakers_own_sound_before_their_first_word_drops_the_turn_only_where_silence_is_asked_for_before_it() -> None:
    breath = Stretch(9.6, 10.3, "A")
    assert kept(breath)
    assert not kept(breath, before=0.5)
