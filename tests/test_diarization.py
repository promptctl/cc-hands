"""Speaker diarization: a hold's speakers from the frames of it the model scored, the run's stream cut into the model's
steps, each hold's diarization told as one event, and the real model telling one person from two in real speech."""

import asyncio
import threading
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile
from loguru import logger

from conftest import CHUNKING
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import diarization
from hands.voice.diarization import Diarized, Diarizer, Model, Speech, Stream, Undiarized
# The real load, bound before conftest stands a model that hears nobody in for it.
from hands.voice.diarization import load as nemotron

FIXTURES = Path(__file__).parent / "fixtures"


def probabilities(*frames: tuple[int, ...]) -> diarization.Probabilities:
    """Frames in which each listed speaker is surely speaking, and every other surely is not."""
    scored = np.zeros((len(frames), 8), np.float32)
    for frame, speaking in enumerate(frames):
        scored[frame, list(speaking)] = 0.9
    return scored


def test_a_holds_speakers_are_the_seconds_each_was_heard_longest_first() -> None:
    held = probabilities((1,), (1, 0), (1,), (0,), (), (3,), (1,))
    assert diarization.speakers(held) == (Speech(1, 0.04), Speech(0, 0.02), Speech(3, 0.01))


def test_a_hold_nobody_was_heard_speaking_in_has_no_speakers() -> None:
    assert diarization.speakers(probabilities((), (), ())) == ()


def test_a_speaker_is_heard_only_above_the_models_threshold() -> None:
    held = np.zeros((4, 8), np.float32)
    held[:, 2] = [0.2, 0.5, 0.51, 0.9]
    assert diarization.speakers(held) == (Speech(2, 0.02),)


# Holds whose samples all say who is speaking: hold k's are (k + 1) / 100, so the scripted step can tell which hold each
# frame it scores came from, and the silence after a hold, all zeros, is nobody's.
def said(hold: int, frames: int) -> diarization.Samples:
    return np.full(frames * diarization.FRAME, (hold + 1) / 100, np.float32)


class Scripted:
    """A model step that hears, in each frame, the speaker the frame's own samples say, as the processor places frames
    in a chunk: from its start in the first, a lead after it in each later one. It counts the frames it scored."""

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.failing = failing
        self.steps = 0

    def __call__(self, chunk: diarization.Samples, first: bool, cache: object) -> tuple[diarization.Probabilities, object]:
        step, self.steps = self.steps, self.steps + 1
        if step in self.failing:
            raise RuntimeError(f"step {step} broke")
        offset = 0 if first else CHUNKING.lead
        heard = chunk[offset : offset + CHUNKING.frames * diarization.FRAME : diarization.FRAME]
        scored = np.zeros((CHUNKING.frames, 8), np.float32)
        for frame, sample in enumerate(heard):
            if sample > 0:
                scored[frame, round(sample * 100) - 1] = 0.9
        return scored, step


def spoken(stream: Stream, hold: int, frames: int, framed: int = 2) -> Diarized | Undiarized:
    """A hold spoken into the stream as Whisper gives it, `framed` 10 ms frames at a time, and ended."""
    samples = said(hold, frames)
    for start in range(0, len(samples), framed * diarization.FRAME):
        stream.heard(samples[start : start + framed * diarization.FRAME])
    return stream.ended()


def test_each_frame_of_a_hold_is_its_own_and_the_silence_after_it_is_no_ones() -> None:
    stream = Stream(Model(Scripted(), CHUNKING))
    # Shorter than a step, a step and a half long, and many steps long: each is its own speaker, for exactly its length.
    lengths = [33, 108, 500, 1, 250]
    for hold, frames in enumerate(lengths):
        diarized = spoken(stream, hold, frames)
        assert isinstance(diarized, Diarized)
        assert diarized.speakers == (Speech(hold, frames / 100),)


def test_a_holds_steps_are_the_ones_that_scored_it() -> None:
    step = Scripted()
    stream = Stream(Model(step, CHUNKING))
    first, second = spoken(stream, 0, 500), spoken(stream, 1, 500)
    assert isinstance(first, Diarized) and isinstance(second, Diarized)
    # Every frame the model scored, the silence after each hold included, was scored by a step one of the holds counts.
    assert first.steps + second.steps == step.steps
    assert first.steps >= 500 // CHUNKING.frames


def test_a_step_that_fails_leaves_its_hold_undiarized_and_the_next_diarized() -> None:
    stream = Stream(Model(Scripted(failing=frozenset({1})), CHUNKING))
    broken = spoken(stream, 0, 200)
    assert isinstance(broken, Undiarized)
    assert "step 1 failed: RuntimeError: step 1 broke" in broken.error
    assert spoken(stream, 1, 200) == Diarized((Speech(1, 2.0),), 4)


def test_a_hold_with_no_audio_has_no_speakers() -> None:
    assert Stream(Model(Scripted(), CHUNKING)).ended() == Diarized((), 0)


class Held:
    """A model step that waits, on the stream's thread, until the test lets it go."""

    def __init__(self) -> None:
        self.go = threading.Event()

    def __call__(self, chunk: diarization.Samples, first: bool, cache: object) -> tuple[diarization.Probabilities, object]:
        self.go.wait()
        return np.zeros((CHUNKING.frames, 8), np.float32), cache


def holds_told(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "diarization.hold"]


async def settled(recorded: list[Entry], told: int) -> list[WideEvent]:
    async with asyncio.timeout(10):
        while len(holds_told(recorded)) < told:
            await asyncio.sleep(0.01)
    return holds_told(recorded)


def pcm(samples: diarization.Samples) -> bytes:
    return (samples * 32768).astype(np.int16).tobytes()


async def test_each_hold_is_one_event_with_its_fate_speakers_and_steps() -> None:
    recorded: list[Entry] = []
    diarizer = Diarizer(lambda: Model(Scripted(), CHUNKING), recorded.append)
    diarizer.heard(pcm(said(0, 100)))
    diarizer.ended(1, "sent")
    diarizer.heard(pcm(said(1, 40)))
    diarizer.ended(2, "dropped")
    sent, dropped = await settled(recorded, 2)
    assert (sent.outcome, sent.facts["hold"], sent.facts["fate"], sent.facts["speakers"]) == ("ok", 1, "sent", (Speech(0, 1.0),))
    assert (dropped.outcome, dropped.facts["hold"], dropped.facts["fate"], dropped.facts["speakers"]) == ("ok", 2, "dropped", (Speech(1, 0.4),))
    assert all(isinstance(event.facts["steps"], int) and event.facts["steps"] > 0 for event in (sent, dropped))


async def test_a_hold_whose_step_failed_is_a_failed_event_with_the_error() -> None:
    recorded: list[Entry] = []
    diarizer = Diarizer(lambda: Model(Scripted(failing=frozenset({0})), CHUNKING), recorded.append)
    diarizer.heard(pcm(said(0, 100)))
    diarizer.ended(1, "sent")
    [told] = await settled(recorded, 1)
    assert told.outcome == "failed"
    assert told.error is not None and "step 0 failed" in told.error
    assert "speakers" not in told.facts


async def test_a_hold_still_unsettled_as_the_next_ends_is_warned_of() -> None:
    recorded: list[Entry] = []
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(message.record["message"]), level="WARNING")
    try:
        step = Held()
        diarizer = Diarizer(lambda: Model(step, CHUNKING), recorded.append)
        diarizer.heard(pcm(said(0, 100)))
        diarizer.ended(1, "sent")
        diarizer.ended(2, "sent")
        assert warned == ["the speakers of hold(s) [1] are not known yet as hold 2 ends: diarization is behind"]
        step.go.set()
        assert [event.facts["hold"] for event in await settled(recorded, 2)] == [1, 2]
    finally:
        logger.remove(sink)


async def test_a_hold_unsettled_as_the_pipeline_stops_ends_cancelled_once() -> None:
    recorded: list[Entry] = []
    step = Held()
    diarizer = Diarizer(lambda: Model(step, CHUNKING), recorded.append)
    diarizer.heard(pcm(said(0, 100)))
    diarizer.ended(1, "sent")
    diarizer.stopped()
    [told] = holds_told(recorded)
    assert (told.outcome, told.facts["hold"], told.facts["fate"]) == ("cancelled", 1, "sent")
    # Its speakers settling after the stop tell it no second time.
    step.go.set()
    await asyncio.sleep(0.2)
    assert len(holds_told(recorded)) == 1


def loaded(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "diarization.loaded"]


async def test_holds_spoken_while_the_model_loads_are_diarized_once_it_has() -> None:
    recorded: list[Entry] = []
    go = threading.Event()

    def slow() -> Model:
        go.wait()
        return Model(Scripted(), CHUNKING)

    diarizer = Diarizer(slow, recorded.append)
    diarizer.heard(pcm(said(0, 100)))
    diarizer.ended(1, "sent")
    await asyncio.sleep(0.1)
    assert (loaded(recorded), holds_told(recorded)) == ([], [])
    go.set()
    [told] = await settled(recorded, 1)
    assert (told.outcome, told.facts["speakers"]) == ("ok", (Speech(0, 1.0),))
    [load] = loaded(recorded)
    assert (load.outcome, load.facts["model"], load.facts["revision"]) == ("ok", diarization.MODEL, diarization.REVISION)


async def test_a_model_that_fails_to_load_leaves_every_hold_a_failed_event_saying_why() -> None:
    recorded: list[Entry] = []

    def unreachable() -> Model:
        raise OSError("the hub is unreachable")

    diarizer = Diarizer(unreachable, recorded.append)
    for hold in (1, 2):
        diarizer.heard(pcm(said(0, 100)))
        diarizer.ended(hold, "sent")
    told = await settled(recorded, 2)
    assert [(event.outcome, event.error, event.facts["steps"]) for event in told] == [("failed", "the model did not load: OSError: the hub is unreachable", 0)] * 2
    [load] = loaded(recorded)
    assert (load.outcome, load.error) == ("failed", "OSError: the hub is unreachable")


# The real model, on real speech: the weights are fetched into the Hugging Face cache the first time, and CI keeps that
# cache. The fixtures are holds separated by half a second of exact zeros, which speech never holds for that long.
GAP = diarization.RATE // 2


def fixture_holds(name: str) -> list[diarization.Samples]:
    audio, rate = cast(tuple[diarization.Samples, int], soundfile.read(FIXTURES / name, dtype="float32"))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
    assert rate == diarization.RATE
    silent = np.convolve(audio == 0, np.ones(GAP), mode="valid") == GAP
    gaps = np.flatnonzero(np.diff(np.concatenate([[False], silent, [False]]).astype(np.int8)))
    edges = [0, *(edge + offset for edge, offset in zip(gaps, [0, GAP - 1] * (len(gaps) // 2))), len(audio)]
    return [audio[start:end] for start, end in zip(edges[0::2], edges[1::2])]


@pytest.fixture(scope="module")
def model() -> Model:
    return nemotron()


def first_heard(diarized: Diarized | Undiarized) -> int | None:
    assert isinstance(diarized, Diarized), diarized
    return diarized.speakers[0].speaker if diarized.speakers else None


def test_short_quiet_holds_of_one_person_are_one_speaker(model: Model) -> None:
    # 24 utterances of 0.8 to 2 s by one LibriVox reader (The Invisible Man, read solo; public domain), each 20 to 30 dB
    # down: the holds the first speaker check (PR #277) filed as a guest's voice.
    holds = fixture_holds("one_reader_quiet.flac")
    assert len(holds) == 24
    stream = Stream(model)
    diarized = [spoken_samples(stream, hold) for hold in holds]
    assert all(isinstance(each, Diarized) for each in diarized), diarized
    assert [{speech.speaker for speech in each.speakers} for each in diarized if isinstance(each, Diarized)] == [{0}] * 24


def test_two_people_are_two_speakers_each_hold_its_speakers(model: Model) -> None:
    # 16 holds of two women, each speaking alone, in the order they spoke: AMI Meeting Corpus ES2004a, mixed headsets,
    # speakers FEE013 and FEE016 (CC BY 4.0, University of Edinburgh).
    holds = fixture_holds("two_speakers_ami.flac")
    stream = Stream(model)
    assert [first_heard(spoken_samples(stream, hold)) for hold in holds] == [int(who) for who in "0001000001011010"]


def spoken_samples(stream: Stream, samples: diarization.Samples) -> Diarized | Undiarized:
    """Real speech spoken into the stream 20 ms at a time, as Whisper gives a hold, and ended."""
    for start in range(0, len(samples), 2 * diarization.FRAME):
        stream.heard(samples[start : start + 2 * diarization.FRAME])
    return stream.ended()
