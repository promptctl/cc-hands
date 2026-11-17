"""Whisper on MLX: each clip transcribed in hands' own process, with mlx_whisper's transcription standing in for the
model, and what hands takes as said from the segments it answers with."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import mlx_whisper
import numpy as np
import pytest

from conftest import events, model_facts
from speech import spoken
from hands.clock import SYSTEM
from hands.sessions.audit import ClipHeard, Entry, Unsaid
from hands.voice import transcription
from hands.voice.clips import Clip
from hands.voice.whisper import Whisper


@dataclass
class Model:
    """mlx_whisper's transcription as these tests answer it: what it was asked, and the segments it gives, oldest first."""

    answers: list[list[tuple[str, float, float]] | Exception] = field(default_factory=list[list[tuple[str, float, float]] | Exception])
    asked: list[tuple[np.ndarray, dict[str, object]]] = field(default_factory=list[tuple[np.ndarray, dict[str, object]]])

    def transcribe(self, audio: np.ndarray, **options: object) -> dict[str, object]:
        self.asked.append((audio, options))
        match self.answers.pop(0) if self.answers else []:
            case Exception() as error:
                raise error
            case segments:
                return {"segments": [{"text": text, "compression_ratio": ratio, "avg_logprob": logprob} for text, ratio, logprob in segments]}


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Model:
    model = Model()
    monkeypatch.setattr(mlx_whisper, "transcribe", model.transcribe)
    return model


def primed(*prompts: str | None) -> Callable[[], Awaitable[str | None]]:
    remaining = iter(prompts)

    async def prompt() -> str | None:
        return next(remaining)

    return prompt


SILENCE = b"\x00\x00" * 16_000

# A clip someone spoke in, as Silero hears it.
SPOKEN = spoken("create a ticket")

# Half a second of silence, which Whisper hears after every clip.
TRAILING = np.zeros(8_000, dtype=np.float32)


def clip(number: int) -> Clip:
    return Clip(number, "shortcut")


async def test_whisper_has_loaded_the_model_its_clips_transcribe_with_once_built(model: Model) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first clip pays for
    no load only if the one done at construction asked for exactly what a clip's transcription asks for."""
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    await whisper.transcribed(clip(1), SILENCE)

    [(_, loaded), (_, transcribed)] = model.asked
    assert loaded == transcribed == {"path_or_hf_repo": transcription.weights(), "temperature": 0.0, "language": "en", "initial_prompt": None, "condition_on_previous_text": False}
    # The load is a unit of work of its own, saying which model the start waited on.
    [event] = events(recorded, "whisper.loaded")
    assert (event.outcome, model_facts(event)) == ("ok", {"model": transcription.MODEL, "revision": transcription.REVISION})


async def test_each_clip_is_transcribed_from_its_samples_primed_with_the_vocabulary_as_it_is_then(model: Model) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed("authMiddleware", None, "sessionStore"), record=lambda _: None)
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    for number in (1, 2, 3):
        await whisper.transcribed(clip(number), samples.tobytes())

    transcribed = model.asked[1:]
    # The clip's own samples, then the silence a segment is heard with after its last word.
    assert all(np.array_equal(audio, np.concatenate([samples.astype(np.float32) / 32768.0, TRAILING])) for audio, _ in transcribed)
    assert [options["initial_prompt"] for _, options in transcribed] == ["authMiddleware", None, "sessionStore"]


async def test_what_a_primed_whisper_makes_of_noise_is_not_said(model: Model) -> None:
    # What Whisper answered, primed: room noise, Gaussian noise read as a guess and as a loop, and "okay" said quietly
    # at -25 dBFS (hands-dictation-2bs.1xq, hands-dictation-d7i).
    model.answers += [
        [],
        [],
        [(".", 0.11, -0.5)],
        [("and slow-talking.", 0.68, -2.84)],
        [("and turnstop, " * 12, 17.1, -0.24)],
        [(" Okay.", 0.38, -1.11)],
    ]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(*["authMiddleware"] * 5), record=recorded.append)

    said = [await whisper.transcribed(clip(number), SPOKEN) for number in range(1, 6)]

    assert said == [None, None, None, None, "Okay."]
    # Each clip is recorded with what was dropped from it and why, so a clip that sent nothing can be looked into.
    heard = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    assert [(entry.clip, entry.said, entry.dropped) for entry in heard] == [
        (1, None, ()),
        (2, None, (Unsaid(".", 0.11, -0.5),)),
        (3, None, (Unsaid("and slow-talking.", 0.68, -2.84),)),
        (4, None, (Unsaid(("and turnstop, " * 12).strip(), 17.1, -0.24),)),
        (5, "Okay.", ()),
    ]
    assert all(entry.voiced > 0.9 for entry in heard)


async def test_what_whisper_hears_in_a_clip_nobody_spoke_in_is_not_said(model: Model) -> None:
    # What Whisper answered unprimed for each of the user's silent holds, sure of it (promptctl-voice-i15t.a5s.x48).
    model.answers += [[], [("Thank you.", 0.56, -0.21)], [("Thank you.", 0.56, -0.21)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None, None), record=recorded.append)

    assert [await whisper.transcribed(clip(number), audio) for number, audio in ((1, SILENCE), (2, SPOKEN))] == [None, "Thank you."]
    [silent, spoke] = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    # Recorded with how sure Silero was that anyone spoke, so a turn that sent nothing says why.
    assert (silent.dropped, silent.voiced < 0.1, spoke.voiced > 0.9) == ((Unsaid("Thank you.", 0.56, -0.21),), True, True)


async def test_a_clip_the_model_fails_raises_and_whisper_transcribes_the_next(model: Model) -> None:
    model.answers += [[], RuntimeError("Metal ran out of memory"), [("Okay.", 0.38, -0.5)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None, None), record=recorded.append)

    with pytest.raises(RuntimeError, match="Metal ran out of memory"):
        await whisper.transcribed(clip(7), SPOKEN)

    assert await whisper.transcribed(clip(8), SPOKEN) == "Okay."
    assert [entry.clip for entry in recorded if isinstance(entry, ClipHeard)] == [8]
