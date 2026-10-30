"""Voiceprints: each speaker decided by the one table of score floors by length, hands' own print made from its TTS in
each voice it speaks in before anything is said in that voice, and a model that does not load failing loudly, naming its
file. The real model on recorded echo is test_streaming's."""

import asyncio
import threading
from collections.abc import Callable

import numpy as np
import pytest
from pipecat.frames.frames import Frame, TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings

from conftest import model_facts, running
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import voiceprints
from hands.voice.holds import RATE, Hands, Person, Samples, Unheard, Unsure
from hands.voice.voiceprints import ALONE, HANDS, PERSON, HandsVoice, Print, decided
from hands.voice.voices import Enrolling


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


def entries(recorded: list[Entry], event: str) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == event]


class Prints:
    """A model scripted by the length of what it is given: each length's print, and every stretch it embedded."""

    def __init__(self, prints: dict[int, Print]) -> None:
        self.prints = prints
        self.embedded: list[Samples] = []

    def __call__(self, voice: Samples) -> Print:
        self.embedded.append(voice)
        return self.prints[len(voice)]


def said_in(lengths: dict[str, int]) -> tuple[list[str], Callable[[str, str], Samples]]:
    """A TTS saying anything in each voice as that many samples, and the voices it was asked for."""
    asked: list[str] = []

    def synthesize(voice: str, text: str) -> Samples:
        assert text == voiceprints.ENROLMENT
        asked.append(voice)
        return np.zeros(lengths[voice], np.float32)

    return asked, synthesize


def test_hands_is_scored_against_every_voice_it_has_spoken_in_and_each_voice_is_enrolled_once() -> None:
    recorded: list[Entry] = []
    model = Prints({1: unit(1, 0), 2: unit(0, 1), 3: unit(1, 1)})
    asked, synthesize = said_in({"charles": 1, "alba": 2})
    voice = HandsVoice("charles", model, synthesize, recorded.append)
    assert voice.score(unit(0, 1)) == pytest.approx(0.0, abs=1e-6)
    voice.enrol("alba")
    voice.enrol("charles")
    # Whichever voice hands speaks in now, and the one before, whose echo may still be in the room.
    assert voice.score(unit(0, 1)) == pytest.approx(1.0)
    assert voice.score(unit(1, 1)) == pytest.approx(np.sqrt(0.5))
    assert asked == ["charles", "alba"]
    # [LAW:nothing-unseen] each print made is a unit, saying which voice and how long the speech it was made from.
    assert [(event.outcome, event.facts) for event in entries(recorded, "voiceprints.enrolled")] == [
        ("ok", {"voice": "charles", "seconds": round(1 / RATE, 3)}), ("ok", {"voice": "alba", "seconds": round(2 / RATE, 3)}),
    ]


def test_a_speaker_under_a_second_is_unheard_and_unscored_and_one_heard_longer_is_scored_on_their_latest_window() -> None:
    window = round(voiceprints.WINDOW * RATE)
    model = Prints({1: unit(1, 0), window: unit(1, 0)})
    voice = HandsVoice("charles", model, said_in({"charles": 1})[1], lambda _: None)
    assert voice.identify(np.zeros(RATE - 1, np.float32), (RATE - 1) / RATE) == Unheard((RATE - 1) / RATE)
    latest = np.arange(2 * window, dtype=np.float32)
    match voice.identify(latest, 7.5):
        case Hands(score=score, seconds=7.5):
            assert score == pytest.approx(1.0)
        case other:
            raise AssertionError(other)
    # The print's own enrolment, then the speaker's latest window alone.
    assert len(model.embedded) == 2 and np.array_equal(model.embedded[1], latest[-window:])


class Seen(FrameProcessor):
    """What reached the TTS service, in order."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames += [frame] if isinstance(frame, TTSUpdateSettingsFrame | TTSSpeakFrame) else []
        await self.push_frame(frame, direction)


async def test_a_change_of_voice_reaches_the_tts_only_once_hands_print_of_it_is_made() -> None:
    asked, synthesize = said_in({"charles": 1, "alba": 2})
    speaking = threading.Event()

    def slow(voice: str, text: str) -> Samples:
        # alba's print waits on the test, so what reaches the TTS meanwhile is seen.
        assert voice == "charles" or speaking.wait(timeout=5)
        return synthesize(voice, text)

    voice = HandsVoice("charles", Prints({1: unit(1, 0), 2: unit(0, 1)}), slow, lambda _: None)
    seen = Seen()
    async with running([Enrolling(voice), seen]) as run:
        change, line = TTSUpdateSettingsFrame(delta=PocketTTSSettings(voice="alba")), TTSSpeakFrame("hello", append_to_context=False)
        await run.worker.queue_frames([change, line])
        await asyncio.sleep(0.2)
        # Neither the change nor what is said after it is through while the print is being made.
        assert seen.frames == []
        speaking.set()
        async with asyncio.timeout(5):
            while len(seen.frames) < 2:
                await asyncio.sleep(0.01)
        assert seen.frames == [change, line]
    assert asked == ["charles", "alba"]
    assert voice.score(unit(0, 1)) == pytest.approx(1.0)


def test_the_model_loads_as_a_unit_saying_what_it_loaded() -> None:
    recorded: list[Entry] = []
    embed = voiceprints.load(recorded.append)
    [load] = entries(recorded, "voiceprints.loaded")
    assert (load.outcome, model_facts(load)) == ("ok", {"model": voiceprints.MODEL, "revision": voiceprints.REVISION, "file": voiceprints.FILE, "dimensions": 192})
    # A print is of unit length, whatever is said.
    assert np.linalg.norm(embed(np.random.default_rng(0).normal(0, 0.1, RATE).astype(np.float32))) == pytest.approx(1.0)


def test_a_model_that_does_not_load_fails_loudly_naming_its_file(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    monkeypatch.setattr(voiceprints, "FILE", "no_such_model.onnx")
    with pytest.raises(RuntimeError, match="the voiceprint model no_such_model.onnx from csukuangfj/speaker-embedding-models at 0743f30"):
        voiceprints.load(recorded.append)
    [load] = entries(recorded, "voiceprints.loaded")
    assert load.outcome == "failed" and "no_such_model.onnx" in str(load.error)
