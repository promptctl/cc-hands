"""Voiceprints: each speaker decided by the one table of score floors by length, and told with the window scored;
hands' own prints made from its TTS on threads of their own, once ever for each voice and kept in the home, of the voice
it speaks in now and the one before; every speaker Unidentified, saying why, where there is no model or no print of a
voice to score against; a person in a hold one for that hold alone; and a model that does not load failing loudly,
naming its file. The real model on recorded echo is test_streaming's."""

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from conftest import model_facts
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import voiceprints
from hands.voice.holds import RATE, Hold, Samples
from hands.voice.voiceprints import (
    ALONE, HANDS, PERSON, Embed, Hands, Person, Print, Speaker, Synthesize, Unheard, Unidentified, Unsure, Voiced, Voiceprints, Who, Window,
    decided, scored,
)

NONE = Window(1.0, 0.0)


def test_a_speaker_is_decided_by_the_longest_row_of_the_table_they_have_reached() -> None:
    # From 1 s, a person alone: under what hands' echo ever scored at that length.
    assert decided(ALONE.score - 0.01, 1.0, NONE) == Person(ALONE.score - 0.01, 1.0, NONE)
    assert decided(ALONE.score, 1.0, NONE) == Unsure(ALONE.score, 1.0, NONE)
    # From 1.5 s, a person over hands too.
    assert decided(PERSON.score - 0.01, 1.49, NONE) == Unsure(PERSON.score - 0.01, 1.49, NONE)
    assert decided(PERSON.score - 0.01, 1.5, NONE) == Person(PERSON.score - 0.01, 1.5, NONE)
    assert decided(PERSON.score, 2.99, NONE) == Unsure(PERSON.score, 2.99, NONE)
    # From 3 s, hands or a person, nothing between.
    assert decided(HANDS.score - 0.01, 3.0, NONE) == Person(HANDS.score - 0.01, 3.0, NONE)
    assert decided(HANDS.score, 3.0, NONE) == Hands(HANDS.score, 3.0, NONE)


def unit(*values: float) -> Print:
    vector = np.array(values, np.float32)
    return vector / np.linalg.norm(vector)


def by_length(prints: dict[int, Print]) -> Embed:
    """A model that prints each stretch by its length."""
    return lambda voice: prints[len(voice)]


def voiced(length: int, seconds: float, speaker: int = 0, shared: float = 0.0, played: tuple[str, ...] = ()) -> Voiced:
    return Voiced(speaker, np.zeros(length, np.float32), seconds, shared, played)


def rounded(who: Who) -> Who:
    """`who` with its score to three places, as an event's reader compares it."""
    match who:
        case Person(score=score, seconds=seconds, window=window):
            return Person(round(score, 3) + 0.0, seconds, window)
        case Hands(score=score, seconds=seconds, window=window):
            return Hands(round(score, 3) + 0.0, seconds, window)
        case Unsure(score=score, seconds=seconds, window=window):
            return Unsure(round(score, 3) + 0.0, seconds, window)
        case Unheard() | Unidentified():
            return who


def test_a_speaker_under_a_second_is_unheard_and_never_scored_and_one_heard_longer_scores_highest_against_any_print() -> None:
    embedded: list[int] = []

    def embed(voice: Samples) -> Print:
        embedded.append(len(voice))
        return unit(1, 1)

    assert scored(embed, (unit(1, 0),), voiced(5, 0.99)) == Unheard(0.99)
    assert embedded == []
    # [LAW:nothing-unseen] the window scored, and how much of it another speaker shared: what a refit is fitted on.
    assert rounded(scored(embed, (unit(1, 0), unit(1, 1)), voiced(RATE, 3.0, shared=0.25))) == Hands(1.0, 3.0, Window(1.0, 0.25))


def entries(recorded: list[Entry], event: str) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == event]


async def until(recorded: list[Entry], event: str, count: int) -> list[WideEvent]:
    async with asyncio.timeout(10):
        while len(entries(recorded, event)) < count:
            await asyncio.sleep(0.01)
    return entries(recorded, event)


def said_in(lengths: dict[str, int], made: list[str]) -> Callable[[], Synthesize]:
    """hands' TTS, each load of it noted in `made`, saying anything in each voice as that many samples."""

    def speech() -> Synthesize:
        made.append("loaded")

        def say(voice: str, text: str) -> Samples:
            assert text == voiceprints.ENROLMENT
            return np.zeros(lengths[voice], np.float32)

        return say

    return speech


def whos(event: WideEvent) -> list[Who]:
    speakers = event.facts["speakers"]
    assert isinstance(speakers, tuple)
    return [rounded(speaker.who) for speaker in speakers if isinstance(speaker, Speaker)]


HOLD = Hold(1, "engaged conversation")
SECOND = Hold(2, "engaged conversation")


async def test_hands_keeps_the_print_of_the_voice_it_speaks_in_and_the_one_before(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    # charles prints as (1, 0), alba as (0, 1), mary as (-1, 0); the speaker's voice, 7 samples, as charles's.
    model = by_length({1: unit(1, 0), 2: unit(0, 1), 3: unit(-1, 0), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1, "alba": 2, "mary": 3}, []), tmp_path, recorded.append)
    prints.speaking_in("charles")
    prints.speaking_in("alba")
    await until(recorded, "voiceprints.enrolled", 2)
    prints.identify(HOLD, (voiced(7, 3.0),))
    # Against charles's print, kept as the voice before the last change: hands.
    [first] = await until(recorded, "voiceprints.identified", 1)
    assert whos(first) == [Hands(1.0, 3.0, Window(0.0, 0.0))]
    prints.speaking_in("mary")
    await until(recorded, "voiceprints.enrolled", 3)
    prints.identify(SECOND, (voiced(7, 3.0),))
    # charles's print let go of: alba's and mary's say nothing like this voice is hands.
    second = (await until(recorded, "voiceprints.identified", 2))[1]
    assert whos(second) == [Person(0.0, 3.0, Window(0.0, 0.0))]


async def test_each_voice_is_enrolled_once_ever_its_tts_loaded_for_that_alone_and_its_print_kept_in_the_home(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    made: list[str] = []
    model = by_length({1: unit(1, 0), 2: unit(0, 1), 7: unit(1, 0)})
    speech = said_in({"charles": 1, "alba": 2}, made)
    prints = Voiceprints(lambda: model, speech, tmp_path, recorded.append)
    prints.speaking_in("charles")
    prints.speaking_in("charles")  # chosen again while its print is being made: nothing more
    await until(recorded, "voiceprints.enrolled", 1)
    prints.speaking_in("alba")
    prints.speaking_in("charles")  # chosen again with its print held: nothing
    await until(recorded, "voiceprints.enrolled", 2)
    # [LAW:nothing-unseen] each print made is a unit, saying which voice, how long the speech, and where it is kept.
    assert [(event.outcome, event.facts) for event in entries(recorded, "voiceprints.enrolled")] == [
        ("ok", {"voice": voice, "seconds": round(length / RATE, 3), "file": tmp_path / voiceprints.REVISION / f"{voice}.npy"})
        for voice, length in (("charles", 1), ("alba", 2))
    ]
    # A Pocket TTS model for each print made, none kept between.
    assert made == ["loaded", "loaded"]
    assert [event.facts["voice"] for event in entries(recorded, "enrolment.loaded")] == ["charles", "alba"]
    # The next run reads both back, and makes neither.
    later: list[Entry] = []
    again = Voiceprints(lambda: model, speech, tmp_path, later.append)
    again.speaking_in("charles")
    again.speaking_in("alba")
    await until(later, "voiceprints.recalled", 2)
    again.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(later, "voiceprints.identified", 1)
    assert whos(event) == [Hands(1.0, 3.0, Window(0.0, 0.0))]
    assert entries(later, "voiceprints.enrolled") == [] and made == ["loaded", "loaded"]


async def test_until_its_print_is_made_every_speaker_is_unidentified_and_nothing_waits_on_the_print(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    speaking = threading.Event()

    def speech() -> Synthesize:
        def say(voice: str, text: str) -> Samples:
            assert speaking.wait(timeout=10)
            return np.zeros(1, np.float32)

        return say

    model = by_length({1: unit(1, 0), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, speech, tmp_path, recorded.append)
    # Returns while the print is still being said.
    prints.speaking_in("charles")
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "no print yet of charles")]
    speaking.set()
    await until(recorded, "voiceprints.enrolled", 1)
    prints.identify(HOLD, (voiced(7, 3.0),))
    assert whos((await until(recorded, "voiceprints.identified", 2))[1]) == [Hands(1.0, 3.0, Window(0.0, 0.0))]


async def test_a_speaker_heard_over_a_voice_hands_holds_no_print_of_is_unidentified_never_a_person(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    # The speaker's voice, 7 samples, prints nothing like charles: against charles's print alone, a person.
    model = by_length({1: unit(1, 0), 7: unit(0, 1)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1}, []), tmp_path, recorded.append)
    prints.speaking_in("charles")
    await until(recorded, "voiceprints.enrolled", 1)
    prints.identify(HOLD, (voiced(7, 3.0, played=("charles", "fantine")),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "hands played fantine over them and holds no print of it")]


async def test_a_print_that_cannot_be_made_fails_its_unit_and_every_speaker_says_why(tmp_path: Path) -> None:
    recorded: list[Entry] = []

    def speech() -> Synthesize:
        def say(voice: str, text: str) -> Samples:
            raise RuntimeError("the voice's embedding is not in the cache")

        return say

    prints = Voiceprints(lambda: by_length({}), speech, tmp_path, recorded.append)
    prints.speaking_in("charles")
    [enrolled] = await until(recorded, "voiceprints.enrolled", 1)
    assert (enrolled.outcome, enrolled.error, enrolled.facts["voice"], enrolled.facts["seconds"]) == (
        "failed", "RuntimeError: the voice's embedding is not in the cache", "charles", None,
    )
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "hands' voiceprint of charles could not be made: RuntimeError: the voice's embedding is not in the cache")]


async def test_a_print_that_cannot_be_kept_fails_its_unit_and_is_still_scored_against(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("in the way")
    recorded: list[Entry] = []
    model = by_length({1: unit(1, 0), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1}, []), blocked, recorded.append)
    prints.speaking_in("charles")
    [enrolled] = await until(recorded, "voiceprints.enrolled", 1)
    assert (enrolled.outcome, enrolled.error) == ("failed", f"NotADirectoryError: [Errno 20] Not a directory: '{blocked / voiceprints.REVISION}'")
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Hands(1.0, 3.0, Window(0.0, 0.0))]


async def test_a_model_that_does_not_load_fails_its_unit_and_every_speaker_says_why(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    made: list[str] = []

    def model() -> Embed:
        raise RuntimeError("the voiceprint model x.onnx did not load")

    prints = Voiceprints(model, said_in({"charles": 1}, made), tmp_path, recorded.append)
    prints.speaking_in("charles")
    [loaded] = await until(recorded, "voiceprints.loaded", 1)
    assert (loaded.outcome, loaded.error) == ("failed", "RuntimeError: the voiceprint model x.onnx did not load")
    [enrolled] = await until(recorded, "voiceprints.enrolled", 1)
    # No TTS loaded for a print nothing could make.
    assert enrolled.outcome == "failed" and made == []
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "the voiceprint model did not load: RuntimeError: the voiceprint model x.onnx did not load")]


async def test_a_person_in_a_hold_stays_one_for_that_hold_alone_even_where_its_end_comes_after_the_next_holds_start(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    # Speaker 0 is a person's voice at 5 samples and hands' at 7.
    model = by_length({1: unit(1, 0), 5: unit(0, 1), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1}, []), tmp_path, recorded.append)
    prints.speaking_in("charles")
    await until(recorded, "voiceprints.enrolled", 1)
    prints.identify(HOLD, (voiced(5, 1.5),))
    prints.identify(SECOND, (voiced(7, 3.0),))
    # Hold 1's end, behind hold 2's first identification: still its person.
    prints.ended(HOLD, (voiced(7, 3.0),))
    # And hold 2's speaker is not hold 1's person.
    prints.identify(SECOND, (voiced(7, 3.0),))
    told = [whos(event) for event in await until(recorded, "voiceprints.identified", 4)]
    window = Window(0.0, 0.0)
    assert told == [[Person(0.0, 1.5, window)], [Hands(1.0, 3.0, window)], [Person(0.0, 1.5, window)], [Hands(1.0, 3.0, window)]]
    assert [event.facts["hold"] for event in entries(recorded, "voiceprints.identified")] == [1, 2, 1, 2]


async def test_an_echo_span_is_identified_as_one_voice_on_an_event_of_its_own(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    model = by_length({1: unit(1, 0), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1}, []), tmp_path, recorded.append)
    prints.speaking_in("charles")
    await until(recorded, "voiceprints.enrolled", 1)
    prints.spanned(3, voiced(7, 3.0, played=("charles",)))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert event.facts["span"] == 3 and "hold" not in event.facts
    assert whos(event) == [Hands(1.0, 3.0, Window(0.0, 0.0))]


def test_the_model_loads_and_prints_are_of_unit_length() -> None:
    embed = voiceprints.load()
    assert np.linalg.norm(embed(np.random.default_rng(0).normal(0, 0.1, RATE).astype(np.float32))) == pytest.approx(1.0)


async def test_the_models_load_is_a_unit_saying_what_it_loaded(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    Voiceprints(voiceprints.load, said_in({}, []), tmp_path, recorded.append)
    [load] = await until(recorded, "voiceprints.loaded", 1)
    assert (load.outcome, model_facts(load)) == ("ok", {"model": voiceprints.MODEL, "revision": voiceprints.REVISION, "file": voiceprints.FILE})


def test_a_model_that_does_not_load_fails_loudly_naming_its_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voiceprints, "FILE", "no_such_model.onnx")
    with pytest.raises(RuntimeError, match="the voiceprint model no_such_model.onnx from csukuangfj/speaker-embedding-models at 0743f30"):
        voiceprints.load()
