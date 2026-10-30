"""Voiceprints: each speaker decided by the one table of score floors by length; hands' own prints made from its TTS on
threads of their own, of the voice it speaks in now and the one before; every speaker Unidentified, saying why, where
there is no model or no print to score against; and a model that does not load failing loudly, naming its file. The
real model on recorded echo is test_streaming's."""

import asyncio
import threading
from collections.abc import Callable

import numpy as np
import pytest

from conftest import model_facts
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import voiceprints
from hands.voice.holds import RATE, Hold, Samples
from hands.voice.voiceprints import (
    ALONE, HANDS, PERSON, Embed, Hands, Person, Print, Speaker, Synthesize, Unheard, Unidentified, Unsure, Voiced, Voiceprints, Who, decided, scored,
)


def test_a_speaker_is_decided_by_the_longest_row_of_the_table_they_have_reached() -> None:
    # From 1 s, a person alone: under what hands' echo ever scored at that length.
    assert decided(ALONE.score - 0.01, 1.0) == Person(ALONE.score - 0.01, 1.0)
    assert decided(ALONE.score, 1.0) == Unsure(ALONE.score, 1.0)
    # From 1.5 s, a person over hands too.
    assert decided(PERSON.score - 0.01, 1.49) == Unsure(PERSON.score - 0.01, 1.49)
    assert decided(PERSON.score - 0.01, 1.5) == Person(PERSON.score - 0.01, 1.5)
    assert decided(PERSON.score, 2.99) == Unsure(PERSON.score, 2.99)
    # From 3 s, hands or a person, nothing between.
    assert decided(HANDS.score - 0.01, 3.0) == Person(HANDS.score - 0.01, 3.0)
    assert decided(HANDS.score, 3.0) == Hands(HANDS.score, 3.0)


def unit(*values: float) -> Print:
    vector = np.array(values, np.float32)
    return vector / np.linalg.norm(vector)


def by_length(prints: dict[int, Print]) -> Embed:
    """A model that prints each stretch by its length."""
    return lambda voice: prints[len(voice)]


def voiced(length: int, seconds: float, speaker: int = 0) -> Voiced:
    return Voiced(speaker, "words", np.zeros(length, np.float32), seconds)


def rounded(who: Who) -> Who:
    """`who` with its score to three places, as an event's reader compares it."""
    match who:
        case Person(score=score, seconds=seconds):
            return Person(round(score, 3) + 0.0, seconds)
        case Hands(score=score, seconds=seconds):
            return Hands(round(score, 3) + 0.0, seconds)
        case Unsure(score=score, seconds=seconds):
            return Unsure(round(score, 3) + 0.0, seconds)
        case Unheard() | Unidentified():
            return who


def test_a_speaker_under_a_second_is_unheard_and_never_scored_and_one_heard_longer_scores_highest_against_any_print() -> None:
    embedded: list[int] = []

    def embed(voice: Samples) -> Print:
        embedded.append(len(voice))
        return unit(1, 1)

    assert scored(embed, (unit(1, 0),), voiced(5, 0.99)) == Unheard(0.99)
    assert embedded == []
    assert rounded(scored(embed, (unit(1, 0), unit(1, 1)), voiced(5, 3.0))) == Hands(1.0, 3.0)


def entries(recorded: list[Entry], event: str) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == event]


async def until(recorded: list[Entry], event: str, count: int) -> list[WideEvent]:
    async with asyncio.timeout(10):
        while len(entries(recorded, event)) < count:
            await asyncio.sleep(0.01)
    return entries(recorded, event)


def said_in(lengths: dict[str, int], made: list[str]) -> Callable[[], Synthesize]:
    """hands' TTS, loaded once into `made`, saying anything in each voice as that many samples."""

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


async def test_hands_keeps_the_print_of_the_voice_it_speaks_in_and_the_one_before_and_loads_its_tts_once() -> None:
    recorded: list[Entry] = []
    made: list[str] = []
    # charles prints as (1, 0), alba as (0, 1), mary as (-1, 0); the speaker's voice, 7 samples, as charles's.
    model = by_length({1: unit(1, 0), 2: unit(0, 1), 3: unit(-1, 0), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1, "alba": 2, "mary": 3}, made), recorded.append)
    prints.speaking_in("charles")
    prints.speaking_in("alba")
    await until(recorded, "voiceprints.enrolled", 2)
    prints.identify(HOLD, (voiced(7, 3.0),))
    # Against charles's print, kept as the voice before the last change: hands.
    [first] = await until(recorded, "voiceprints.identified", 1)
    assert whos(first) == [Hands(1.0, 3.0)]
    prints.speaking_in("mary")
    await until(recorded, "voiceprints.enrolled", 3)
    prints.identify(Hold(2, "engaged conversation"), (voiced(7, 3.0),))
    # charles's print let go of: alba's and mary's say nothing like this voice is hands.
    second = (await until(recorded, "voiceprints.identified", 2))[1]
    assert whos(second) == [Person(0.0, 3.0)]
    # [LAW:nothing-unseen] each print made is a unit, saying which voice and how long the speech it was made from.
    assert [(event.outcome, event.facts) for event in entries(recorded, "voiceprints.enrolled")] == [
        ("ok", {"voice": voice, "seconds": round(length / RATE, 3)}) for voice, length in (("charles", 1), ("alba", 2), ("mary", 3))
    ]
    assert made == ["loaded"]


async def test_until_its_print_is_made_every_speaker_is_unidentified_and_nothing_waits_on_the_print() -> None:
    recorded: list[Entry] = []
    speaking = threading.Event()

    def speech() -> Synthesize:
        def say(voice: str, text: str) -> Samples:
            assert speaking.wait(timeout=10)
            return np.zeros(1, np.float32)

        return say

    model = by_length({1: unit(1, 0), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, speech, recorded.append)
    # Returns while the print is still being said.
    prints.speaking_in("charles")
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "no print yet of charles")]
    speaking.set()
    await until(recorded, "voiceprints.enrolled", 1)
    prints.identify(HOLD, (voiced(7, 3.0),))
    assert whos((await until(recorded, "voiceprints.identified", 2))[1]) == [Hands(1.0, 3.0)]


async def test_a_print_that_cannot_be_made_fails_its_unit_and_every_speaker_says_why() -> None:
    recorded: list[Entry] = []

    def speech() -> Synthesize:
        def say(voice: str, text: str) -> Samples:
            raise RuntimeError("the voice's embedding is not in the cache")

        return say

    model = by_length({})
    prints = Voiceprints(lambda: model, speech, recorded.append)
    prints.speaking_in("charles")
    [enrolled] = await until(recorded, "voiceprints.enrolled", 1)
    assert (enrolled.outcome, enrolled.error, enrolled.facts) == ("failed", "RuntimeError: the voice's embedding is not in the cache", {"voice": "charles", "seconds": None})
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "hands' voiceprint of charles could not be made: RuntimeError: the voice's embedding is not in the cache")]


async def test_a_model_that_does_not_load_fails_its_unit_and_every_speaker_says_why() -> None:
    recorded: list[Entry] = []

    def model() -> Embed:
        raise RuntimeError("the voiceprint model x.onnx did not load")

    prints = Voiceprints(model, said_in({"charles": 1}, []), recorded.append)
    prints.speaking_in("charles")
    [loaded] = await until(recorded, "voiceprints.loaded", 1)
    assert (loaded.outcome, loaded.error) == ("failed", "RuntimeError: the voiceprint model x.onnx did not load")
    [enrolled] = await until(recorded, "voiceprints.enrolled", 1)
    assert enrolled.outcome == "failed"
    prints.identify(HOLD, (voiced(7, 3.0),))
    [event] = await until(recorded, "voiceprints.identified", 1)
    assert whos(event) == [Unidentified(3.0, "the voiceprint model did not load: RuntimeError: the voiceprint model x.onnx did not load")]


async def test_a_person_in_a_hold_stays_one_for_it_and_the_next_hold_identifies_its_speakers_afresh() -> None:
    recorded: list[Entry] = []
    # Speaker 0 is first a person's voice, 5 samples, then hands', 7.
    model = by_length({1: unit(1, 0), 5: unit(0, 1), 7: unit(1, 0)})
    prints = Voiceprints(lambda: model, said_in({"charles": 1}, []), recorded.append)
    prints.speaking_in("charles")
    await until(recorded, "voiceprints.enrolled", 1)
    prints.identify(HOLD, (voiced(5, 1.5),))
    prints.identify(HOLD, (voiced(7, 3.0),))
    prints.identify(Hold(2, "engaged conversation"), (voiced(7, 3.0),))
    told = [whos(event) for event in await until(recorded, "voiceprints.identified", 3)]
    assert told == [[Person(0.0, 1.5)], [Person(0.0, 1.5)], [Hands(1.0, 3.0)]]


def test_the_model_loads_and_prints_are_of_unit_length() -> None:
    embed = voiceprints.load()
    assert np.linalg.norm(embed(np.random.default_rng(0).normal(0, 0.1, RATE).astype(np.float32))) == pytest.approx(1.0)


async def test_the_models_load_is_a_unit_saying_what_it_loaded() -> None:
    recorded: list[Entry] = []
    Voiceprints(voiceprints.load, said_in({}, []), recorded.append)
    [load] = await until(recorded, "voiceprints.loaded", 1)
    assert (load.outcome, model_facts(load)) == ("ok", {"model": voiceprints.MODEL, "revision": voiceprints.REVISION, "file": voiceprints.FILE})


def test_a_model_that_does_not_load_fails_loudly_naming_its_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voiceprints, "FILE", "no_such_model.onnx")
    with pytest.raises(RuntimeError, match="the voiceprint model no_such_model.onnx from csukuangfj/speaker-embedding-models at 0743f30"):
        voiceprints.load()
