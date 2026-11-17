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
from hands.sessions.wide import begun
from hands.voice.clips import Clip, Played, Reference, Sound, Synthesis
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


def clip(number: int) -> Clip:
    return Clip(number, "shortcut")


def as_heard(audio: bytes, played: Played = None) -> Sound:
    """A clip's sound as it was heard, and what hands played over it: nothing known of it for None."""
    return Sound(audio, audio, played)


def saying(seconds: float, *sentences: tuple[str, float, float]) -> Sound:
    """`seconds` of a clip, hands saying each of `sentences` over it, its text from its start to its stop in seconds, one
    synthesis each, as the canceller hands what was played on: one reference for each 10 ms."""
    said = [(Synthesis("charles", text, begun(SYSTEM)), start, stop) for text, start, stop in sentences]

    def at(slot: int) -> Reference | None:
        playing = tuple(synthesis for synthesis, start, stop in said if start <= slot / 100 < stop)
        return Reference(bytes(320), 16_000, 1, playing) if playing else None

    return as_heard(bytes(round(seconds * 32_000)), tuple(at(slot) for slot in range(round(seconds * 100))))


def answering(monkeypatch: pytest.MonkeyPatch, *segments: tuple[str, float, float]) -> None:
    """Whisper hearing each of `segments` in the next clip, its text from its start to its end in seconds, each spoken
    in, as Silero hears it: what is heard here is whose words they were."""
    answered = [transcription.Segment(text, start, end, 0.8, -0.3) for text, start, end in segments]

    def heard(_samples: np.ndarray, _prompt: str | None) -> list[transcription.Segment]:
        return answered

    def voiced(_whisper: Whisper, _samples: np.ndarray) -> float:
        return 1.0

    monkeypatch.setattr(transcription, "segments", heard)
    monkeypatch.setattr(Whisper, "_voiced", voiced)


async def told(sound: Sound) -> ClipHeard:
    """What Whisper made of a clip of `sound`, as it records it."""
    recorded: list[Entry] = []
    await Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append).transcribed(clip(1), sound)
    [clip_heard] = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    return clip_heard


async def test_whisper_has_loaded_the_model_its_clips_transcribe_with_once_built(model: Model) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first clip pays for
    no load only if the one done at construction asked for exactly what a clip's transcription asks for."""
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    await whisper.transcribed(clip(1), as_heard(SILENCE))

    [(_, loaded), (_, transcribed)] = model.asked
    assert loaded == transcribed == {"path_or_hf_repo": transcription.weights(), "temperature": 0.0, "language": "en", "initial_prompt": None, "condition_on_previous_text": False}
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
        [("and turnstop, " * 12, 17.1, -0.24)],
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
    # Spoken in, then held on in silence: Whisper's "Thank you." heard from 30 s on (promptctl-voice-i15t.a5s.x48).
    spoke = len(SPOKEN) / 2 / 16_000
    heard = [transcription.Segment("Create a ticket.", 0.0, spoke, 0.8, -0.3), transcription.Segment("Thank you.", 30.0, 40.0, 0.56, -0.28)]

    def segments(_samples: np.ndarray, _prompt: str | None) -> list[transcription.Segment]:
        return heard

    monkeypatch.setattr(transcription, "segments", segments)

    assert await whisper.transcribed(clip(1), as_heard(SPOKEN + SILENCE * 40)) == "Create a ticket."
    [[unsaid]] = [entry.dropped for entry in recorded if isinstance(entry, ClipHeard)]
    assert (unsaid.text, unsaid.voiced < 0.1) == ("Thank you.", True)


async def test_a_clip_the_model_fails_raises_and_whisper_transcribes_the_next(model: Model) -> None:
    model.answers += [[], RuntimeError("Metal ran out of memory"), [("Okay.", 0.38, -0.5)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None, None), record=recorded.append)

    with pytest.raises(RuntimeError, match="Metal ran out of memory"):
        await whisper.transcribed(clip(7), as_heard(SPOKEN))

    assert await whisper.transcribed(clip(8), as_heard(SPOKEN)) == "Okay."
    assert [entry.clip for entry in recorded if isinstance(entry, ClipHeard)] == [8]


async def test_a_segment_of_hands_own_sentence_is_not_said_and_a_persons_beside_it_is(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # hands' sentence come back through the microphone, then a person after it, each a segment of their own.
    answering(monkeypatch, ("The parser is fixed.", 0.1, 1.4), ("Create a ticket.", 2.0, 3.0))
    clip_heard = await told(saying(3.5, ("The parser is fixed.", 0.0, 1.5)))
    # [LAW:nothing-unseen] hands' own words are said apart from noise: dropped as hands' echo, not as nobody's.
    assert (clip_heard.said, clip_heard.echo, clip_heard.dropped) == ("Create a ticket.", ("The parser is fixed.",), ())


async def test_a_segment_with_one_word_that_is_not_hands_is_the_persons_whole(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, ("Yes, run the tests.", 1.6, 2.8))
    clip_heard = await told(saying(3.0, ("Should I run the tests?", 0.0, 1.5)))
    assert (clip_heard.said, clip_heard.echo) == ("Yes, run the tests.", ())


async def test_hands_sentence_heard_with_a_word_the_canceller_took_out_is_still_hands(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, ("Should I the tests?", 0.1, 1.5))
    assert (await told(saying(2.0, ("Should I run the tests?", 0.0, 1.5)))).echo == ("Should I the tests?",)


async def test_a_word_of_hands_said_long_after_hands_said_it_is_the_persons(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # A lone "No." 2 s into hands' sentence: its "no" played 2 s before, and nothing played near it is "no".
    answering(monkeypatch, ("No.", 2.0, 2.4))
    clip_heard = await told(saying(3.0, ("No, I won't delete the branch.", 0.0, 2.5)))
    assert (clip_heard.said, clip_heard.echo) == ("No.", ())


async def test_a_sentence_hands_said_twice_is_hands_each_time_it_is_heard_and_once_said_heard_twice_is_a_persons_after(
    model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, ("Okay.", 0.1, 0.5), ("Okay.", 3.1, 3.5))
    assert (await told(saying(4.0, ("Okay.", 0.0, 0.5), ("Okay.", 3.0, 3.5)))).echo == ("Okay.", "Okay.")
    # Said once, at the start: the second, 3 s on, is someone answering it.
    answering(monkeypatch, ("Okay.", 0.1, 0.5), ("Okay.", 3.1, 3.5))
    clip_heard = await told(saying(4.0, ("Okay.", 0.0, 0.5)))
    assert (clip_heard.said, clip_heard.echo) == ("Okay.", ("Okay.",))


async def test_a_one_word_sentence_of_hands_heard_alone_and_one_spelled_whispers_way_are_hands(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    answering(monkeypatch, ("Done.", 0.05, 0.45), ("The diariser reports one speaker.", 1.1, 3.0))
    clip_heard = await told(saying(3.5, ("Done.", 0.0, 0.4), ("The diarizer reports one speaker.", 1.0, 3.0)))
    assert (clip_heard.said, clip_heard.echo) == (None, ("Done.", "The diariser reports one speaker."))


async def test_nothing_is_hands_where_nothing_is_known_of_what_hands_played(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # At the phone, whose own canceller is told what it plays, and hands is not.
    answering(monkeypatch, ("The parser is fixed.", 0.1, 1.4))
    clip_heard = await told(as_heard(bytes(64_000)))
    assert (clip_heard.said, clip_heard.echo) == ("The parser is fixed.", ())
