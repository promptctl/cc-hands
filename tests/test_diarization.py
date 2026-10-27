"""Speaker diarization: a hold's speakers from the frames of it the model scored, the run's stream cut into the model's
steps, each hold's diarization told as one event, and the real model telling one person from two in real speech."""

import asyncio
import threading
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import soundfile

from conftest import CHUNKING
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import diarization
from hands.voice.diarization import Diarized, Diarizer, Model, Speech, Stream, Undiarized
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import Fate, Hold, Sound
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
    """A hold spoken into the stream as the hold owner gives it, `framed` 10 ms frames at a time, and ended."""
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


def holds_told(recorded: list[Entry]) -> list[WideEvent]:
    """Each hold's diarization, as the hold owner tells the diarizer's hearings."""
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hearing" and entry.facts["listener"] == "diarization"]


class Unused:
    """A transcriber no test here gives a hold to."""

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        raise AssertionError("nothing here transcribes")


def owner(diarizer: Diarizer, recorded: list[Entry]) -> HoldOwner:
    """The hold owner, with the diarizer its one listener, whose hearings it runs as a pipeline's would."""
    return HoldOwner(Unused(), (diarizer,), recorded.append)


async def ended(diarizer: Diarizer, owned: HoldOwner, number: int, fate: Fate) -> None:
    """Hold `number` ends, and the hearings the diarizer returns for it are run as the owner runs them."""
    hold = Hold(number, "held key")
    for hearing in diarizer.ended(hold, fate):
        await owned._heard(diarizer, hold, hearing)  # pyright: ignore[reportPrivateUsage]  (what the owner's hearing lane runs)


def heard(diarizer: Diarizer, samples: diarization.Samples) -> None:
    assert diarizer.heard(Sound(pcm(samples), pcm(samples))) == ()


def pcm(samples: diarization.Samples) -> bytes:
    return (samples * 32768).astype(np.int16).tobytes()


async def test_each_hold_is_one_hearing_with_its_fate_speakers_and_steps() -> None:
    recorded: list[Entry] = []
    diarizer = Diarizer(lambda: Model(Scripted(), CHUNKING), recorded.append)
    owned = owner(diarizer, recorded)
    heard(diarizer, said(0, 100))
    await ended(diarizer, owned, 1, "sent")
    heard(diarizer, said(1, 40))
    await ended(diarizer, owned, 2, "dropped")
    sent, dropped = holds_told(recorded)
    assert (sent.outcome, sent.facts["hold"], sent.facts["fate"], sent.facts["speakers"]) == ("ok", 1, "sent", (Speech(0, 1.0),))
    assert (dropped.outcome, dropped.facts["hold"], dropped.facts["fate"], dropped.facts["speakers"]) == ("ok", 2, "dropped", (Speech(1, 0.4),))
    assert all(isinstance(event.facts["steps"], int) and event.facts["steps"] > 0 for event in (sent, dropped))
    # A hold's speakers are no words: nothing they settle cuts hands off.
    assert [(event.facts["heard"], event.facts["cut"]) for event in (sent, dropped)] == [("nothing", False)] * 2


async def test_the_audio_a_hold_kept_from_before_it_opened_is_diarized_as_its_own() -> None:
    recorded: list[Entry] = []
    diarizer = Diarizer(lambda: Model(Scripted(), CHUNKING), recorded.append)
    samples = said(0, 100)
    assert diarizer.opened(Hold(1, "engaged conversation"), Sound(pcm(samples[:3200]), pcm(samples[:3200]))) == ()
    heard(diarizer, samples[3200:])
    await ended(diarizer, owner(diarizer, recorded), 1, "sent")
    [told] = holds_told(recorded)
    assert told.facts["speakers"] == (Speech(0, 1.0),)


async def test_a_hold_whose_step_failed_is_a_failed_hearing_with_the_error() -> None:
    recorded: list[Entry] = []
    diarizer = Diarizer(lambda: Model(Scripted(failing=frozenset({0})), CHUNKING), recorded.append)
    heard(diarizer, said(0, 100))
    await ended(diarizer, owner(diarizer, recorded), 1, "sent")
    [told] = holds_told(recorded)
    assert told.outcome == "failed"
    assert told.error is not None and "step 0 failed" in told.error
    assert "speakers" not in told.facts


def loaded(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "diarization.loaded"]


async def test_holds_spoken_while_the_model_loads_are_diarized_once_it_has() -> None:
    recorded: list[Entry] = []
    go = threading.Event()

    def slow() -> Model:
        go.wait()
        return Model(Scripted(), CHUNKING)

    diarizer = Diarizer(slow, recorded.append)
    heard(diarizer, said(0, 100))
    settling = asyncio.create_task(ended(diarizer, owner(diarizer, recorded), 1, "sent"))
    await asyncio.sleep(0.1)
    assert (loaded(recorded), holds_told(recorded)) == ([], [])
    go.set()
    await settling
    [told] = holds_told(recorded)
    assert (told.outcome, told.facts["speakers"]) == ("ok", (Speech(0, 1.0),))
    [load] = loaded(recorded)
    assert (load.outcome, load.facts["model"], load.facts["revision"]) == ("ok", diarization.MODEL, diarization.REVISION)


async def test_a_model_that_fails_to_load_leaves_every_hold_a_failed_hearing_saying_why() -> None:
    recorded: list[Entry] = []

    def unreachable() -> Model:
        raise OSError("the hub is unreachable")

    diarizer = Diarizer(unreachable, recorded.append)
    owned = owner(diarizer, recorded)
    for number in (1, 2):
        heard(diarizer, said(0, 100))
        await ended(diarizer, owned, number, "sent")
    told = holds_told(recorded)
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
