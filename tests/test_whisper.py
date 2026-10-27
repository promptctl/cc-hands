"""Each hold transcribed in hands' own process, with a stand-in for Parakeet answering what it is asked, and what hands
takes as said from what it heard."""

import wave
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import pytest
from loguru import logger
from parakeet_mlx.alignment import AlignedToken
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame

from conftest import events
from hands.sessions.audit import Entry, HoldHeard, Levels
from hands.voice import transcription
from hands.voice.turnstop import Hold, TurnResolved
from hands.voice.whisper import Whisper, _Recorded  # pyright: ignore[reportPrivateUsage]  (what a release queues)


@dataclass
class Model:
    """Parakeet as these tests answer it: how often it was loaded, the samples each window asked of it, and what it hears
    in each, oldest first; a window it was given no answer for it hears nothing in."""

    answers: list[str | Exception] = field(default_factory=list[str | Exception])
    asked: list[transcription.Samples] = field(default_factory=list[transcription.Samples])
    loads: int = 0

    def load(self) -> transcription.Model:
        self.loads += 1
        return transcription.Model(self.window)

    def window(self, samples: transcription.Samples) -> list[AlignedToken]:
        self.asked.append(samples)
        match self.answers.pop(0) if self.answers else "":
            case Exception() as error:
                raise error
            case text:
                return [AlignedToken(0, text, 0.0, 0.5)] if text else []


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Model:
    model = Model()
    monkeypatch.setattr(transcription, "load", model.load)
    return model


def wav(samples: bytes, rate: int = 16_000) -> bytes:
    """A hold as Pipecat hands it to run_stt: mono 16-bit WAV at the pipeline's input rate, 16 kHz unless said."""
    out = BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples)
    return out.getvalue()


SILENCE = wav(b"\x00\x00" * 16_000)

# How loud each hold queued here was, as the key's release measured it.
LEVELS = Levels(captured_dbfs=-12.5, heard_dbfs=-40.0)


async def transcribe(whisper: Whisper, hold: int, audio: bytes) -> list[Frame]:
    with wave.open(BytesIO(audio)) as held:
        seconds = held.getnframes() / held.getframerate()
    whisper._transcribing.append(_Recorded(Hold(hold, "held key"), seconds, LEVELS))  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
    return [frame async for frame in whisper.run_stt(audio)]


async def test_the_model_is_loaded_once_as_whisper_is_built_and_every_hold_is_heard_by_it(model: Model) -> None:
    recorded: list[Entry] = []
    whisper = Whisper(record=recorded.append)
    for hold in (1, 2):
        await transcribe(whisper, hold, SILENCE)

    assert (model.loads, len(model.asked)) == (1, 2)
    # The load is a unit of work of its own, saying which model the start waited on.
    [event] = events(recorded, "whisper.loaded")
    assert event.outcome == "ok" and event.facts["model"] == transcription.MODEL


async def test_whisper_settings_say_the_model_and_language_its_holds_transcribe_with(model: Model) -> None:
    """Pipecat checks a service's settings at its start and logs an error for any it was never given."""
    whisper = Whisper(record=lambda _: None)
    errors: list[str] = []
    sink = logger.add(errors.append, level="ERROR", filter="pipecat")
    try:
        whisper._settings.validate_complete()  # pyright: ignore[reportPrivateUsage]  (what Pipecat's start checks)
    finally:
        logger.remove(sink)

    assert errors == []
    assert whisper._settings.model == transcription.MODEL  # pyright: ignore[reportPrivateUsage]
    assert whisper._settings.language == "en"  # pyright: ignore[reportPrivateUsage]


async def test_each_hold_is_heard_from_its_own_samples(model: Model) -> None:
    whisper = Whisper(record=lambda _: None)
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    for hold in (1, 2):
        await transcribe(whisper, hold, wav(samples.tobytes()))

    assert all(np.array_equal(asked, samples.astype(np.float32) / 32768.0) for asked in model.asked)


async def test_only_words_heard_are_said_and_every_hold_is_recorded_with_them(model: Model) -> None:
    model.answers += ["", ".", " Okay."]
    recorded: list[Entry] = []
    whisper = Whisper(record=recorded.append)

    said = [frame.text for hold in (1, 2, 3) for frame in await transcribe(whisper, hold, SILENCE) if isinstance(frame, TranscriptionFrame)]

    assert said == ["Okay."]
    # Recorded either way, so "I spoke and nothing happened" can be looked into.
    assert [entry for entry in recorded if isinstance(entry, HoldHeard)] == [HoldHeard(1, None, LEVELS), HoldHeard(2, None, LEVELS), HoldHeard(3, "Okay.", LEVELS)]


async def test_a_hold_the_model_fails_is_said_as_an_error_and_whisper_transcribes_the_next(model: Model) -> None:
    model.answers += [RuntimeError("Metal ran out of memory"), "Okay."]
    recorded: list[Entry] = []
    whisper = Whisper(record=recorded.append)

    failed = await transcribe(whisper, 7, SILENCE)
    [error] = [frame for frame in failed if isinstance(frame, ErrorFrame)]
    await whisper.push_error_frame(error)
    after = await transcribe(whisper, 8, SILENCE)

    assert [type(frame) for frame in failed] == [ErrorFrame, TurnResolved]
    assert "hold 7" in error.error and "Metal ran out of memory" in error.error
    # A permanent error would make Whisper unusable, and Pipecat then gives it no hold to transcribe or fail aloud.
    assert error.category is not None and not error.category.is_permanent and whisper.is_usable
    assert [frame.text for frame in after if isinstance(frame, TranscriptionFrame)] == ["Okay."]
    assert [entry.hold for entry in recorded if isinstance(entry, HoldHeard)] == [8]


async def test_a_hold_at_a_rate_whisper_does_not_hear_is_said_as_an_error_and_never_transcribed(model: Model) -> None:
    whisper = Whisper(record=lambda _: None)

    failed = await transcribe(whisper, 3, wav(b"\x00\x00" * 48_000, rate=48_000))

    [error] = [frame for frame in failed if isinstance(frame, ErrorFrame)]
    assert "48000 Hz" in error.error and model.asked == []
