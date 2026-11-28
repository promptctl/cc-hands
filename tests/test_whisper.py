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
from hands.sessions.audit import ClipHeard, Entry
from hands.voice import transcription
from hands.voice.clips import Clip, Played, Sound
from hands.voice.mlxthread import OnMLX
from hands.voice.whisper import Whisper


@dataclass
class Model:
    """mlx_whisper's transcription as these tests answer it: what it was asked, and the segments it gives, oldest first,
    each heard over the whole of the audio."""

    answers: list[list[tuple[str, float, float]] | Exception] = field(default_factory=list[list[tuple[str, float, float]] | Exception])
    asked: list[tuple[np.ndarray, dict[str, object]]] = field(default_factory=list[tuple[np.ndarray, dict[str, object]]])

    def transcribe(self, audio: np.ndarray, **options: object) -> dict[str, object]:
        self.asked.append((audio, options))
        match self.answers.pop(0) if self.answers else []:
            case Exception() as error:
                raise error
            case segments:
                return {"segments": [{"text": text, "start": 0.0, "end": len(audio) / 16_000, "compression_ratio": ratio, "avg_logprob": logprob} for text, ratio, logprob in segments]}


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


def loudness(samples: np.ndarray) -> float:
    """How much sound `samples` hold: what is cut from them of it is lost, and silence holds none."""
    return float(np.abs(samples).sum())


def clip(number: int) -> Clip:
    return Clip(number, "shortcut")


def as_heard(audio: bytes, played: Played = None) -> Sound:
    """A clip's sound as it was heard, and what hands played over it: nothing known of it for None."""
    return Sound(audio, audio, played)


async def test_whisper_has_loaded_the_model_its_clips_transcribe_with_once_built(model: Model) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first clip pays for
    no load only if the one done at construction asked for exactly what a clip's transcription asks for."""
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    await whisper.transcribed(clip(1), as_heard(SILENCE))

    [(heard, loaded), (_, transcribed)] = model.asked
    assert transcribed == {"path_or_hf_repo": transcription.weights(), "temperature": 0.0, "language": "en", "initial_prompt": None, "condition_on_previous_text": False}
    assert (loaded, bool(np.abs(heard).max() > 0.1)) == (transcribed, True)
    # The load is a unit of work of its own, saying which model the start waited on.
    [event] = events(recorded, "whisper.loaded")
    assert (event.outcome, model_facts(event)) == ("ok", {"model": transcription.MODEL, "revision": transcription.REVISION})


async def test_each_clip_is_transcribed_from_its_samples_primed_with_the_vocabulary_as_it_is_then(model: Model) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed("authMiddleware", None, "sessionStore"), record=lambda _: None)
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    for number in (1, 2, 3):
        await whisper.transcribed(clip(number), as_heard(samples.tobytes()))

    transcribed = model.asked[1:]
    # The clip's own samples, then the silence a segment is heard with after its last word.
    assert all(np.array_equal(audio, np.concatenate([samples.astype(np.float32) / 32768.0, TRAILING])) for audio, _ in transcribed)
    assert [options["initial_prompt"] for _, options in transcribed] == ["authMiddleware", None, "sessionStore"]


async def test_a_segment_whisper_repeated_itself_in_or_only_guessed_at_is_not_said(model: Model) -> None:
    # What Whisper answered, primed: room noise, Gaussian noise read as a guess and as a loop, and "okay" said quietly
    # at -25 dBFS (hands-dictation-2bs.1xq, hands-dictation-d7i).
    model.answers += [
        [],
        [],
        [(".", 0.11, -0.5)],
        [("and slow-talking.", 0.68, -2.84)],
        # Heard again unprimed, as a loop is, and nothing in it.
        [("and turnstop, " * 12, 17.1, -0.24)],
        [],
        [(" Okay.", 0.38, -1.11)],
    ]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(*["authMiddleware"] * 5), record=recorded.append)

    said = [await whisper.transcribed(clip(number), as_heard(SPOKEN)) for number in range(1, 6)]

    assert said == [None, None, None, None, "Okay."]
    # Each clip is recorded with what was dropped from it and why, so a clip that sent nothing can be looked into.
    # Spoken in, as Silero hears each: dropped for Whisper's scores alone.
    heard = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    assert [(entry.clip, entry.said, [(unsaid.text, unsaid.compression_ratio, unsaid.avg_logprob) for unsaid in entry.dropped]) for entry in heard] == [
        (1, None, []),
        (2, None, [(".", 0.11, -0.5)]),
        (3, None, [("and slow-talking.", 0.68, -2.84)]),
        (4, None, [(("and turnstop, " * 12).strip(), 17.1, -0.24)]),
        (5, "Okay.", []),
    ]
    assert all(unsaid.voiced >= 0.25 for entry in heard for unsaid in entry.dropped)


async def test_a_clip_whisper_looped_in_primed_is_heard_again_unprimed_and_what_it_hears_then_is_said(model: Model) -> None:
    # "the build is green" through the desk's canceller, primed with the vocabulary: read three times over, at 20.8.
    model.answers += [[], [("the build is green", 20.8, -0.43)], [("The build is green.", 0.7, -0.52)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed("authMiddleware"), record=recorded.append)

    said = await whisper.transcribed(clip(1), as_heard(SPOKEN))

    assert said == "The build is green."
    assert [options["initial_prompt"] for _, options in model.asked[1:]] == ["authMiddleware", None]
    [heard] = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    # The loop is recorded as dropped, so a clip primed to loop can be looked into.
    assert [(unsaid.text, unsaid.compression_ratio) for unsaid in heard.dropped] == [("the build is green", 20.8)]


async def test_a_clip_whisper_loops_in_unprimed_too_says_nothing_and_records_both_loops(model: Model) -> None:
    model.answers += [[], [("the build is green", 20.8, -0.43)], [("and turnstop, " * 12, 17.1, -0.24)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed("authMiddleware"), record=recorded.append)

    said = await whisper.transcribed(clip(1), as_heard(SPOKEN))

    assert said is None
    [heard] = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    assert [unsaid.compression_ratio for unsaid in heard.dropped] == [20.8, 17.1]


async def test_a_clip_whisper_did_not_loop_in_is_decoded_once(model: Model) -> None:
    model.answers += [[], [("The build is green.", 0.7, -0.52)]]

    await Whisper(clock=SYSTEM, prompt=primed("authMiddleware"), record=lambda _: None).transcribed(clip(1), as_heard(SPOKEN))

    assert len(model.asked[1:]) == 1


async def test_a_segment_of_only_punctuation_and_underscores_has_no_word_and_is_not_said(model: Model) -> None:
    # Primed with identifiers, Whisper may write an underscore over noise: no letter, so no word.
    model.answers += [[], [("__ ...", 1.0, -0.2)]]
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)

    assert await whisper.transcribed(clip(1), as_heard(SPOKEN)) is None


async def test_what_whisper_hears_where_nobody_spoke_is_not_said(model: Model) -> None:
    # What Whisper answered unprimed for each of the user's silent holds, sure of it (promptctl-voice-i15t.a5s.x48).
    model.answers += [[], [("Thank you.", 0.56, -0.21)], [("Thank you.", 0.56, -0.21)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None, None), record=recorded.append)

    assert [await whisper.transcribed(clip(number), as_heard(audio)) for number, audio in ((1, SILENCE), (2, SPOKEN))] == [None, "Thank you."]
    # Recorded with how sure Silero was that anyone spoke, so a turn that sent nothing says why.
    [[unsaid], []] = [entry.dropped for entry in recorded if isinstance(entry, ClipHeard)]
    assert (unsaid.text, unsaid.voiced < 0.1) == ("Thank you.", True)


async def test_a_segment_is_heard_as_spoken_in_only_where_it_was_heard(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    # Spoken in, then held on in silence: Whisper's "Thank you." heard after the speech (promptctl-voice-i15t.a5s.x48).
    spoke = len(SPOKEN) / 2 / 16_000
    heard = [transcription.Segment("Create a ticket.", 0.0, spoke, 0.8, -0.3), transcription.Segment("Thank you.", spoke + 0.1, spoke + 0.5, 0.56, -0.28)]

    def segments(_on: OnMLX, _weights: str, _samples: np.ndarray, _prompt: str | None) -> list[transcription.Segment]:
        return heard

    monkeypatch.setattr(transcription, "segments", segments)

    assert await whisper.transcribed(clip(1), as_heard(SPOKEN + SILENCE * 40)) == "Create a ticket."
    [[unsaid]] = [entry.dropped for entry in recorded if isinstance(entry, ClipHeard)]
    assert (unsaid.text, unsaid.voiced < 0.1) == ("Thank you.", True)


async def test_a_quiet_stretch_longer_than_half_a_second_is_cut_to_it_before_whisper_hears_the_clip(model: Model) -> None:
    """Primed with the vocabulary, Whisper answers a long quiet lead with the vocabulary's own words, 0.0 to 30.0 s, and the
    person's words are lost (promptctl-voice-i15t.65f.qem): it is not given the lead, nor any pause that long."""
    whisper = Whisper(clock=SYSTEM, prompt=primed("authMiddleware", "authMiddleware"), record=lambda _: None)
    spoken_in = np.frombuffer(SPOKEN, dtype=np.int16).astype(np.float32) / 32768.0
    # Nine seconds of quiet ahead of the speech, and the same nine in the middle of it.
    await whisper.transcribed(clip(1), as_heard(SILENCE * 9 + SPOKEN))
    await whisper.transcribed(clip(2), as_heard(SPOKEN + SILENCE * 9 + SPOKEN))

    [(lead, _), (pause, _)] = model.asked[1:]
    # Each quiet stretch kept is at most half a second, with the half second heard after the clip: what is not speech is the
    # little added.
    assert (len(lead) <= len(spoken_in) + 3 * 8_000, len(pause) <= 2 * len(spoken_in) + 3 * 8_000) == (True, True)
    # The speech is kept whole: only silence, which has no sample to lose, is cut.
    assert (loudness(lead), loudness(pause)) == (pytest.approx(loudness(spoken_in), rel=1e-6), pytest.approx(2 * loudness(spoken_in), rel=1e-6))


async def test_a_quiet_stretch_of_a_quarter_second_is_heard_as_it_was(model: Model) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    spoken_in = np.frombuffer(SPOKEN, dtype=np.int16).astype(np.float32) / 32768.0
    await whisper.transcribed(clip(1), as_heard(SPOKEN + SILENCE[: len(SILENCE) // 4] + SPOKEN))

    [(heard, _)] = model.asked[1:]
    # Both speeches and the quarter second between them, none of it cut.
    assert len(heard) >= 2 * len(spoken_in) + 4_000


async def test_a_clip_the_model_fails_raises_and_whisper_transcribes_the_next(model: Model) -> None:
    model.answers += [[], RuntimeError("Metal ran out of memory"), [("Okay.", 0.38, -0.5)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None, None), record=recorded.append)

    with pytest.raises(RuntimeError, match="Metal ran out of memory"):
        await whisper.transcribed(clip(7), as_heard(SPOKEN))

    assert await whisper.transcribed(clip(8), as_heard(SPOKEN)) == "Okay."
    assert [entry.clip for entry in recorded if isinstance(entry, ClipHeard)] == [8]
