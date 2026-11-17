"""The turn rule every AMI measurement is cut by (scripts/turntaking.py): a person's turn is kept only when nobody else is
heard in it or the silences asked for around it, and they themselves are not heard in those silences."""

from turntaking import GAP, SHORTEST, Stretch, turns

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


def test_one_persons_words_join_into_a_turn_across_a_gap_shorter_than_GAP_and_split_at_a_longer_one() -> None:
    first, near, far = Stretch(10.0, 11.0, "A", ("one",)), Stretch(11.0 + GAP / 2, 12.0, "A", ("two",)), Stretch(13.0, 14.0, "A")
    assert turns([first, near, far], [], 60.0) == [Stretch(10.0, 12.0, "A", ("one", "two")), far]


def test_a_turn_shorter_than_SHORTEST_is_not_kept() -> None:
    assert turns([Stretch(10.0, 10.0 + SHORTEST / 2, "A")], [], 60.0) == []


def test_a_turn_whose_silences_run_past_either_end_of_the_recording_is_not_kept() -> None:
    assert turns([TURN], [], 60.0, before=TURN.start + 0.1) == []
    assert turns([TURN], [], TURN.end + 0.4, after=0.5) == []
    assert turns([TURN], [], TURN.end + 0.5, before=TURN.start, after=0.5) == [TURN]
