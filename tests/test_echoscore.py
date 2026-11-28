"""The harness that scores an echo-cancelling chain (tests/echoscore.py): what it counts as hands' words leaked and a
person's words lost, on takes built here, so the numbers a later ticket picks a chain by mean what they say."""

from pathlib import Path

import numpy as np
import pytest

import echoes
from conftest import PAIRS
from echoscore import CHAINS, LATE, TAKES, Line, Take, Word, erle, heard_words, judged, normal, recorded, scored, summary, words_of
from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Playhead, Synthesis
from hands.voice import mlxthread, streaming
from hands.voice.streaming import Said

HANDS = (Playhead(Synthesis("charles", "Hands here.", begun(VirtualClock())), 0.0),)


def built(person: tuple[Line, ...], playing: tuple[float, float], seconds: float = 12.0, captured: float = 0.1) -> Take:
    """A take of `seconds` where hands plays from `playing[0]` to `playing[1]` and the person says `person`."""
    slots = round(seconds * RATE / SLOT)
    heads = tuple(HANDS if playing[0] <= slot * SLOT / RATE < playing[1] else () for slot in range(slots))
    return Take("built", np.full(slots * SLOT, round(captured * 32768), np.int16), np.zeros(slots * SLOT, np.int16), heads, person)


def spoken(*pairs: tuple[str, float]) -> tuple[Word, ...]:
    return tuple(Word(text, start) for text, start in pairs)


def test_pieces_of_a_word_are_one_word_dated_by_its_first_and_punctuation_is_none() -> None:
    tokens = [(0, 9.12, " No"), (0, 9.44, ","), (0, 9.44, " the"), (0, 9.6, " bu"), (0, 9.6, "ild"), (0, 9.92, " bro"), (0, 9.92, "k"), (0, 9.92, "en"), (0, 11.36, ".")]
    assert words_of(tokens, (), 0.0) == spoken(("no", 9.12), ("the", 9.44), ("build", 9.6), ("broken", 9.92))


def test_two_speakers_pieces_are_not_spelled_into_each_others_words() -> None:
    tokens = [(0, 1.0, " a"), (1, 1.1, " on"), (0, 1.2, "gain"), (1, 1.3, "ly")]
    assert words_of(tokens, (), 0.0) == spoken(("again", 1.0), ("only", 1.1))


def test_the_words_decoded_as_the_clip_closed_are_dated_where_it_closed() -> None:
    assert words_of([(0, 2.0, " say")], [Said(0, " the next, step.")], 5.0) == spoken(("say", 2.0), ("the", 5.0), ("next", 5.0), ("step", 5.0))


def test_a_word_hands_said_before_the_person_spoke_leaked_and_the_persons_own_words_are_kept() -> None:
    take = built((Line("No, the build is broken.", 5.0),), (0.0, 12.0))
    heard = spoken(("hands", 1.0), ("here", 1.5), ("no", 6.0), ("the", 6.2), ("build", 6.4), ("is", 6.6), ("broken", 6.8))
    assert judged(take, heard) == ((), spoken(("hands", 1.0), ("here", 1.5)), ())


def test_a_word_of_the_person_that_comes_out_as_another_is_lost_and_not_hands() -> None:
    take = built((Line("Say the next step again.", 5.0),), (0.0, 12.0))
    heard = spoken(("say", 6.0), ("the", 6.2), ("next", 6.4), ("end", 6.6))
    assert judged(take, heard) == (("step", "again"), (), ())


def test_a_word_past_the_persons_lines_where_hands_plays_leaked() -> None:
    take = built((Line("Say the next step.", 5.0),), (0.0, 12.0))
    heard = spoken(("say", 6.0), ("the", 6.2), ("next", 6.4), ("step", 6.6), ("when", 8.0), ("ready", 8.4))
    assert judged(take, heard) == ((), spoken(("when", 8.0), ("ready", 8.4)), ())


def test_a_word_where_hands_played_nothing_is_the_rooms() -> None:
    take = built((), (0.0, 2.0))
    heard = spoken(("hands", 1.0), ("hm", 9.0))
    assert judged(take, heard) == ((), spoken(("hands", 1.0)), spoken(("hm", 9.0)))


def test_a_word_decoded_late_leaked_if_hands_played_in_the_chunk_it_was_spoken_in() -> None:
    take = built((), (0.0, 2.0))
    assert judged(take, spoken(("hands", 2.0 + LATE - 0.1)))[1] != ()
    assert judged(take, spoken(("hands", 2.0 + LATE + 0.5)))[1] == ()


def test_a_person_whose_words_are_all_missing_lost_every_one_and_leaked_none() -> None:
    take = built((Line("Yes.", 5.0),), (0.0, 12.0))
    assert judged(take, ()) == (("yes",), (), ())


def test_a_word_said_by_both_is_credited_to_the_person() -> None:
    take = built((Line("the build is", 5.0),), (0.0, 12.0))
    assert judged(take, spoken(("the", 6.0), ("build", 6.2), ("is", 6.4))) == ((), (), ())


def test_words_heard_in_place_of_more_said_are_lost_and_the_one_left_over_is_not_the_persons() -> None:
    take = built((Line("Say step.", 5.0),), (0.0, 12.0))
    assert judged(take, spoken(("sue", 6.0), ("stop", 6.2), ("ready", 6.4))) == (("say", "step"), spoken(("ready", 6.4)), ())


def test_a_word_among_the_persons_that_they_did_not_say_is_not_theirs() -> None:
    take = built((Line("Say the next step.", 5.0),), (0.0, 12.0))
    heard = spoken(("say", 6.0), ("the", 6.2), ("hands", 6.3), ("next", 6.4), ("step", 6.6))
    assert judged(take, heard) == ((), spoken(("hands", 6.3)), ())


def test_a_typographic_apostrophe_spells_a_word_as_the_plain_one_does() -> None:
    assert normal("Don\u2019t say don't.") == ["don't", "say", "don't"]


def test_erle_is_the_captured_energy_over_the_cleaned_before_the_person_speaks_where_hands_played() -> None:
    take = built((Line("Yes.", 8.0),), (1.0, 12.0), captured=0.1)
    cleaned = np.full(len(take.captured), 0.01, np.float32)
    assert erle(take, cleaned) == pytest.approx(20.0, abs=0.01)
    # After the person speaks, the chain's job is to keep them: that is lost words, not echo returned.
    cleaned[round(8.0 * RATE) :] = 0.1
    assert erle(take, cleaned) == pytest.approx(20.0, abs=0.01)


def test_erle_is_zero_where_the_microphone_and_the_cleaned_sound_are_both_silent() -> None:
    take = built((Line("Yes.", 8.0),), (1.0, 12.0), captured=0.0)
    assert erle(take, np.zeros(len(take.captured), np.float32)) == 0.0


def test_a_take_scored_has_what_was_said_lost_leaked_and_its_erle_and_a_line_says_them() -> None:
    take = built((Line("Say the next step.", 5.0),), (0.0, 12.0), captured=0.1)
    score = scored(take, np.full(len(take.captured), 0.01, np.float32), spoken(("hands", 1.0), ("say", 6.0), ("the", 6.2)))
    assert (score.said, score.lost, score.leaked, score.stray) == (4, ("next", "step"), spoken(("hands", 1.0)), ())
    assert score.erle == pytest.approx(20.0, abs=0.01)
    line = summary(score)
    assert "leaked  1  lost  2 of  4  stray  0  ERLE  20.0 dB" in line and "leaked: hands;  lost: next step" in line


def test_erle_is_none_where_hands_played_nothing_before_the_person_spoke() -> None:
    assert erle(built((Line("Yes.", 1.0),), (5.0, 8.0)), np.zeros(12 * RATE, np.float32)) is None


@pytest.mark.parametrize("fixture", TAKES, ids=lambda fixture: fixture.name)
def test_each_take_recorded_has_what_a_chain_is_run_over(fixture: Path) -> None:
    take = recorded(fixture)
    assert take.person or fixture == echoes.LONG_AND_SLOW, "a person speaks, but in the one take of hands alone"
    assert all(line.start > 0 and normal(line.text) for line in take.person)
    before = take.playheads[: round(take.cut_in * RATE / SLOT)] if take.person else take.playheads
    assert any(before), "hands plays before the person speaks"


def test_the_takes_hold_what_a_person_and_hands_do_at_the_desk() -> None:
    def said(fixture: Path) -> dict[str, str]:
        """Each of hands' sentences the take plays, with its voice."""
        return {playhead.synthesis.text: playhead.synthesis.voice for slot in recorded(fixture).playheads for playhead in slot}

    def over_hands(fixture: Path) -> list[Line]:
        take = recorded(fixture)
        return [line for line in take.person if take.played(line.start, line.start + 0.5)]

    long_and_slow = recorded(echoes.LONG_AND_SLOW)
    assert not long_and_slow.person and max(len(normal(text)) for text in said(echoes.LONG_AND_SLOW)) >= 30, "long sentences of hands', with no one over them"
    assert len(set(said(echoes.TWO_VOICES).values())) > 1, "two of hands' voices"
    assert {normal(line.text)[0] for line in over_hands(echoes.SHORT_WORDS) if len(normal(line.text)) == 1} & {"yes", "no", "stop"}, "a short word over hands"
    assert any(f" {' '.join(normal(line.text))} " in f" {' '.join(normal(text))} " for line in over_hands(echoes.SAME_WORDS) for text in said(echoes.SAME_WORDS)), "a person saying what hands says"
    alone = recorded(echoes.PERSON_ALONE)
    assert alone.person and not over_hands(echoes.PERSON_ALONE), "a person alone"


def test_a_take_whose_samples_and_slots_are_not_one_stretch_is_none() -> None:
    with pytest.raises(ValueError, match="not one stretch"):
        Take("short", np.zeros(SLOT, np.int16), np.zeros(2 * SLOT, np.int16), ((), ()), ())


@pytest.fixture(scope="module")
def pair() -> streaming.Session:
    return mlxthread.call(PAIRS.fetched(streaming.GATING))


@pytest.mark.outside
@pytest.mark.parametrize("fixture", TAKES, ids=lambda fixture: fixture.name)
def test_the_desks_canceller_lets_none_of_hands_words_through_and_keeps_a_persons_words_on_every_take(fixture: Path, pair: streaming.Session) -> None:
    """The epic's numbers, held: over the twelve takes, 0 of hands' words leaked and 1 of 101 of a person's lost. That
    one, "wait" in hands_no_echo_in_two_voices.flac, is lost with no canceller as well (the `raw` chain), the recognizer's own."""
    take = recorded(fixture)
    cleaned = CHAINS["desk"](take)
    score = scored(take, cleaned, heard_words(pair, cleaned))
    assert score.leaked == () and score.stray == (), summary(score)
    assert score.lost == (("wait",) if fixture == echoes.NO_ECHO_TWO_VOICES else ()), summary(score)
