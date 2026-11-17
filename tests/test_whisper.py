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
from hands.voice.clips import Clip, Reference, Sentence, Sound
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


def alone(audio: bytes) -> Sound:
    """A clip of `audio` hands played nothing over."""
    return Sound(audio, audio, (None,) * (len(audio) // 320))


def read_over(audio: bytes, *texts: str, seconds: float | None = None) -> Sound:
    """A clip of `audio` hands read `texts` over, for its first `seconds`, all of it for None."""
    slots = len(audio) // 320
    over = slots if seconds is None else round(seconds * 100)
    reading = Reference(bytes(320), 16_000, 1, tuple(Sentence("charles", text) for text in texts))
    return Sound(audio, audio, (reading,) * min(over, slots) + (None,) * max(0, slots - over))


async def test_whisper_has_loaded_the_model_its_clips_transcribe_with_once_built(model: Model) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first clip pays for
    no load only if the one done at construction asked for exactly what a clip's transcription asks for."""
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    await whisper.transcribed(clip(1), alone(SILENCE))

    [(_, loaded), (_, transcribed)] = model.asked
    assert loaded == transcribed == {"path_or_hf_repo": transcription.weights(), "temperature": 0.0, "language": "en", "initial_prompt": None, "condition_on_previous_text": False}
    # The load is a unit of work of its own, saying which model the start waited on.
    [event] = events(recorded, "whisper.loaded")
    assert (event.outcome, model_facts(event)) == ("ok", {"model": transcription.MODEL, "revision": transcription.REVISION})


async def test_each_clip_is_transcribed_from_its_samples_primed_with_the_vocabulary_as_it_is_then(model: Model) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed("authMiddleware", None, "sessionStore"), record=lambda _: None)
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)

    for number in (1, 2, 3):
        await whisper.transcribed(clip(number), alone(samples.tobytes()))

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

    said = [await whisper.transcribed(clip(number), alone(SPOKEN)) for number in range(1, 6)]

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

    assert [await whisper.transcribed(clip(number), alone(audio)) for number, audio in ((1, SILENCE), (2, SPOKEN))] == [None, "Thank you."]
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

    assert await whisper.transcribed(clip(1), alone(SPOKEN + SILENCE * 40)) == "Create a ticket."
    [[unsaid]] = [entry.dropped for entry in recorded if isinstance(entry, ClipHeard)]
    assert (unsaid.text, unsaid.voiced < 0.1) == ("Thank you.", True)


def answering(monkeypatch: pytest.MonkeyPatch, *heard: tuple[str, float, float]) -> None:
    """Whisper hearing `heard` in every clip: each segment's text, and where it starts and ends, in seconds."""

    def segments(_samples: np.ndarray, _prompt: str | None) -> list[transcription.Segment]:
        return [transcription.Segment(text, start, end, 0.8, -0.3) for text, start, end in heard]

    monkeypatch.setattr(transcription, "segments", segments)


# A person, then hands' own voice come back through the microphone, each a segment of their own: 2 s of speech in all.
PERSON_THEN_HANDS = SPOKEN + SPOKEN
SPOKE = len(SPOKEN) / 2 / 16_000


async def test_the_words_of_a_sentence_hands_played_over_them_are_not_said_and_a_persons_in_the_same_clip_are(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    answering(monkeypatch, ("Create a ticket.", 0.0, SPOKE), ("The parser is fixed.", SPOKE, 2 * SPOKE))

    # Whichever speaker the diarizer gave either voice: what hands said over the clip decides.
    assert await whisper.transcribed(clip(1), read_over(PERSON_THEN_HANDS, "The parser is fixed.")) == "Create a ticket."
    # [LAW:nothing-unseen] recorded as hands' own echo, apart from what was dropped as nobody's.
    [heard] = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    assert (heard.dropped, heard.echo) == ((), ("The parser is fixed.",))


async def test_a_person_over_the_tail_of_hands_sentence_keeps_their_words_in_the_segment_they_share(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # hands' sentence, and a person over its tail, heard as one segment.
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("The build finished and two sessions wait. Stop, read me the second one.", 0.0, 2 * SPOKE))

    sound = read_over(PERSON_THEN_HANDS, "The build finished, and two sessions are waiting on you.")
    assert await whisper.transcribed(clip(1), sound) == "wait. Stop, read me the second one."


async def test_hands_words_whisper_spelled_its_own_way_are_still_hands(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # Unprimed, Whisper spells hands' names as it hears them (promptctl-voice-i15t.8vg.06g, on zoj's recordings).
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("The diariser reports one speaker. Starting Houdini MCP now.", 0.0, 2 * SPOKE))

    sound = read_over(PERSON_THEN_HANDS, "The diarizer reports one speaker.", "Starting houdini-mcp now.")
    assert await whisper.transcribed(clip(1), sound) is None


async def test_one_word_a_person_shares_with_hands_sentence_is_theirs(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # "yes" said while hands asks a question with a yes in it: one word alone is as likely a person's.
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("Yes.", 0.0, SPOKE))

    assert await whisper.transcribed(clip(1), read_over(PERSON_THEN_HANDS, "Say yes or no.")) == "Yes."


async def test_a_person_answering_in_words_hands_just_asked_with_keeps_them(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # A press as hands' question ends: the person's answer repeats three of its words, no echo of it.
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("Yes, run the tests.", 0.0, SPOKE), ("No, look at the other one in the config.", SPOKE, 2 * SPOKE))

    sound = read_over(PERSON_THEN_HANDS, "Should I run the tests?", "I'll look at the file in the repo.")
    assert await whisper.transcribed(clip(1), sound) == "Yes, run the tests. No, look at the other one in the config."


async def test_words_hands_said_apart_are_no_run_of_hands_where_a_person_says_them_together(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    # Side by side in what the person said, four of hands' words; apart in hands' sentence, one and then three.
    answering(monkeypatch, ("Please stop the build.", 0.0, SPOKE))

    assert await whisper.transcribed(clip(1), read_over(PERSON_THEN_HANDS, "Please wait while I stop the build.")) == "Please stop the build."


async def test_a_one_word_sentence_of_hands_heard_on_its_own_is_hands(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("Done.", 0.0, SPOKE / 2), ("The tests pass.", SPOKE / 2, SPOKE))

    assert await whisper.transcribed(clip(1), read_over(PERSON_THEN_HANDS, "Done.", "The tests pass.")) is None


def playing(audio: bytes, *plays: tuple[str, float, float]) -> Sound:
    """A clip of `audio` hands said each of `plays` over: its text, and from and to when, in seconds."""
    slots = len(audio) // 320
    over = [tuple(Sentence("charles", text) for text, start, stop in plays if start * 100 <= at < stop * 100) for at in range(slots)]
    return Sound(audio, audio, tuple(Reference(bytes(320), 16_000, 1, sentences) if sentences else None for sentences in over))


async def test_a_sentence_hands_said_twice_is_heard_twice(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("Okay. Okay.", 0.0, SPOKE))

    assert await whisper.transcribed(clip(1), playing(PERSON_THEN_HANDS, ("Okay.", 0.0, 0.3), ("Okay.", 0.5, 0.8))) is None


async def test_a_sentence_hands_began_just_after_whisper_ended_a_segment_is_heard_in_it(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # Whisper ends a segment up to half a second before the last words in it.
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("Create a ticket. The parser is fixed.", 0.0, SPOKE))

    assert await whisper.transcribed(clip(1), playing(PERSON_THEN_HANDS, ("The parser is fixed.", SPOKE + 0.3, 2 * SPOKE))) == "Create a ticket."


async def test_a_sentence_hands_played_long_before_a_segment_is_not_heard_in_it(model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    # hands said it over the clip's first 10 ms, and stopped: a person saying the same a second and more later is heard.
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=lambda _: None)
    answering(monkeypatch, ("The parser is fixed.", SPOKE + 0.2, 2 * SPOKE))

    assert await whisper.transcribed(clip(1), read_over(PERSON_THEN_HANDS, "The parser is fixed.", seconds=0.01)) == "The parser is fixed."


async def test_a_clip_the_model_fails_raises_and_whisper_transcribes_the_next(model: Model) -> None:
    model.answers += [[], RuntimeError("Metal ran out of memory"), [("Okay.", 0.38, -0.5)]]
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None, None), record=recorded.append)

    with pytest.raises(RuntimeError, match="Metal ran out of memory"):
        await whisper.transcribed(clip(7), alone(SPOKEN))

    assert await whisper.transcribed(clip(8), alone(SPOKEN)) == "Okay."
    assert [entry.clip for entry in recorded if isinstance(entry, ClipHeard)] == [8]
