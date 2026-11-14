"""A recording serves the real pair's answers to the run it was made from, and names itself when the run is another."""

import numpy as np
import pytest

import recordings
from hands.voice.streaming import Active, Closed, Decoded, Said, Samples


class Pair:
    """A pair that decodes a word on the feed that takes the run to 300 samples and another to 600, and closes on the last."""

    def __init__(self) -> None:
        self.total = 0

    def decoded(self, samples: Samples) -> Decoded:
        self.total += len(samples)
        match self.total:
            case 300:
                return Decoded((Said(0, "hello"),), (Active(0, 3),))
            case 600:
                return Decoded((Said(1, " there"),), ())
            case _:
                return Decoded((), ())

    def closed(self) -> Closed:
        return Closed(frozenset({0, 1}), (Said(1, "."),))


def audio(length: int, level: float = 0.5) -> Samples:
    return np.full(length, level, np.float32)


def made() -> recordings.Recording:
    """The recording of a run fed 600 samples in 100s, then 100 more, then closed."""
    recorder = recordings.Recorder(Pair())
    for _ in range(6):
        recorder.decoded(audio(100))
    recorder.decoded(audio(100))
    recorder.closed()
    return recorder.recording()


def test_a_recording_serves_the_run_it_was_made_from_as_the_pair_answered_it() -> None:
    replay = recordings.Replay("run", made())
    answers = [replay.decoded(audio(100)) for _ in range(7)]
    assert [(a.words, a.active) for a in answers if a.words or a.active] == [((Said(0, "hello"),), (Active(0, 3),)), ((Said(1, " there"),), ())]
    assert replay.closed() == Closed(frozenset({0, 1}), (Said(1, "."),))


def test_a_recording_kept_as_text_is_served_as_it_was_made() -> None:
    kept = made()
    assert recordings.Recording.parsed(kept.text()) == kept


def test_a_run_fed_up_to_a_point_is_served_the_recording_up_to_it() -> None:
    replay = recordings.Replay("run", made())
    assert replay.decoded(audio(300)).words == (Said(0, "hello"),)


def test_a_run_fed_other_audio_than_the_recording_was_made_from_fails_at_the_close_naming_it() -> None:
    replay = recordings.Replay("the-run", made())
    for _ in range(7):
        replay.decoded(audio(100, level=0.25))
    with pytest.raises(AssertionError, match=r"'the-run'.*audio fed up to the close"):
        replay.closed()


def test_a_run_fed_in_other_feeds_than_the_recording_was_made_from_fails_naming_it() -> None:
    replay = recordings.Replay("the-run", made())
    with pytest.raises(AssertionError, match=r"'the-run'.*decode recorded at 300 before it"):
        replay.decoded(audio(350))


def test_a_run_fed_past_the_end_of_the_recording_fails_naming_it() -> None:
    replay = recordings.Replay("the-run", made())
    for _ in range(7):
        replay.decoded(audio(100))
    with pytest.raises(AssertionError, match=r"'the-run'.*recording ends at 700"):
        replay.decoded(audio(1))


def test_a_run_closed_elsewhere_than_the_recording_was_fails_naming_it() -> None:
    replay = recordings.Replay("the-run", made())
    replay.decoded(audio(100))
    with pytest.raises(AssertionError, match=r"'the-run'.*closes at 100 samples, the recording at 700"):
        replay.closed()
