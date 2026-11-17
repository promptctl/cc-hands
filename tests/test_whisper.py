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
from hands.virtualclock import VirtualClock
from hands.voice.clips import Clip, Played, Reference, Sound, Playhead, Synthesis, decoded, track
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
    synthesis each, as the canceller hands what was played on: one reference for each 10 ms, hands' voice as noise at
    -20 dBFS. Its echo is what the microphone captured, and heard through no canceller."""
    said = [(Synthesis("charles", text, begun(VirtualClock())), start, stop) for text, start, stop in sentences]
    voice = np.random.default_rng(0).normal(0, 3_277, round(seconds * 16_000)).astype(np.int16)

    def at(slot: int) -> Reference | None:
        playing = tuple(Playhead(synthesis, slot / 100 - start) for synthesis, start, stop in said if start <= slot / 100 < stop)
        return Reference(voice[160 * slot : 160 * (slot + 1)].tobytes(), 16_000, 1, playing) if playing else None

    played = tuple(at(slot) for slot in range(round(seconds * 100)))
    return as_heard(track(played), played)


def cancelling(sound: Sound, start: float, stop: float) -> Sound:
    """`sound` with the echo canceller taking hands' voice out from `start` to `stop` s: what was heard after the
    canceller there is 40 dB below what the microphone captured."""
    heard = np.frombuffer(sound.heard, np.int16).copy()
    span = slice(round(start * 16_000), round(stop * 16_000))
    heard[span] = (np.frombuffer(sound.captured, np.int16)[span] / 100).astype(np.int16)
    return Sound(heard.tobytes(), sound.captured, sound.played)


def headphones(sound: Sound) -> Sound:
    """`sound` with hands' voice in headphones: none of what was played reached the microphone, which captured a quiet
    room, and the canceller, with no echo to take out, ducked all of it to digital silence."""
    room = np.random.default_rng(1).normal(0, 33, len(sound.captured) // 2).astype(np.int16).tobytes()
    return Sound(bytes(len(room)), room, sound.played)


@dataclass
class Timed:
    """Whisper as these tests answer it with timed words: each segment it hears in the next clip, a word to each
    (text, start, end), each spoken in as Silero hears it; and the samples each clip was heard as, and whether it asked
    for its words timed."""

    segments: list[list[tuple[str, float, float]]] = field(default_factory=list[list[tuple[str, float, float]]])
    samples: list[np.ndarray] = field(default_factory=list[np.ndarray])
    asked: list[bool] = field(default_factory=list[bool])

    def heard(self, samples: np.ndarray, _prompt: str | None, *, words: bool) -> list[transcription.Segment]:
        self.samples.append(samples)
        self.asked.append(words)
        return [
            transcription.Segment("".join(text for text, _, _ in each).strip(), each[0][1], each[-1][2], 0.8, -0.3, tuple(transcription.Word(*word) for word in each) if words else ())
            for each in self.segments
        ]


@pytest.fixture
def timed(model: Model, monkeypatch: pytest.MonkeyPatch) -> Timed:
    timed = Timed()

    def voiced(_whisper: Whisper, _samples: np.ndarray) -> float:
        return 1.0

    monkeypatch.setattr(transcription, "segments", timed.heard)
    monkeypatch.setattr(Whisper, "_voiced", voiced)
    return timed


def spaced(text: str, start: float, step: float) -> list[tuple[str, float, float]]:
    """`text` as Whisper writes its words, from `start`, a word every `step` seconds."""
    return [(f" {word}", start + index * step, start + (index + 1) * step) for index, word in enumerate(text.split())]


async def told(sound: Sound) -> ClipHeard:
    """What Whisper made of a clip of `sound`, as it records it."""
    recorded: list[Entry] = []
    await Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append).transcribed(clip(1), sound)
    [clip_heard] = [entry for entry in recorded if isinstance(entry, ClipHeard)]
    return clip_heard


async def test_whisper_has_loaded_the_model_its_clips_transcribe_with_once_built(model: Model) -> None:
    """MLX Whisper keeps the model it loaded for the process, keyed on what it was asked for, so the first clip pays for
    no load only if the one done at construction asked for exactly what a clip's transcription asks for; and it was
    asked to time the words of speech, so the first clip hands played over pays for no first timing either."""
    recorded: list[Entry] = []
    whisper = Whisper(clock=SYSTEM, prompt=primed(None), record=recorded.append)
    await whisper.transcribed(clip(1), as_heard(SILENCE))

    [(heard, loaded), (_, transcribed)] = model.asked
    assert transcribed == {"path_or_hf_repo": transcription.weights(), "temperature": 0.0, "language": "en", "initial_prompt": None, "condition_on_previous_text": False, "word_timestamps": False}
    assert (loaded, bool(np.abs(heard).max() > 0.1)) == ({**transcribed, "word_timestamps": True}, True)
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
    # Spoken in, then held on in silence: Whisper's "Thank you." heard from 30 s on (promptctl-voice-i15t.a5s.x48).
    spoke = len(SPOKEN) / 2 / 16_000
    heard = [transcription.Segment("Create a ticket.", 0.0, spoke, 0.8, -0.3, ()), transcription.Segment("Thank you.", 30.0, 40.0, 0.56, -0.28, ())]

    def segments(_samples: np.ndarray, _prompt: str | None, *, words: bool) -> list[transcription.Segment]:
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


async def test_a_persons_words_heard_in_one_segment_with_hands_sentence_are_said_and_hands_are_not(timed: Timed) -> None:
    # Primed, Whisper hears both voices in one segment, their words interleaved: the person asking over hands' sentence.
    timed.segments.append([(" Can", 0.1, 0.2), (" starting", 0.3, 0.4), (" you", 0.6, 0.7), (" the", 0.8, 0.9), (" hear", 1.0, 1.1), (" build", 1.2, 1.3), (" me?", 1.4, 1.5), (" now.", 1.7, 1.8)])
    clip_heard = await told(saying(2.5, ("Starting the build now.", 0.0, 2.0)))
    # [LAW:nothing-unseen] hands' own words are said apart from noise: dropped as hands' echo, not as nobody's.
    assert (clip_heard.said, clip_heard.echo, clip_heard.dropped) == ("Can you hear me?", ("starting", "the", "build", "now."), ())


async def test_a_word_of_hands_said_long_after_hands_said_it_is_the_persons(timed: Timed) -> None:
    # A lone "No." 2 s into hands' sentence: its "no" played 2 s before, and nothing played near it is "no".
    timed.segments.append([(" No.", 2.0, 2.4)])
    clip_heard = await told(saying(3.0, ("No, I won't delete the branch.", 0.0, 2.5)))
    assert (clip_heard.said, clip_heard.echo) == ("No.", ())


async def test_hands_sentence_heard_with_a_word_the_canceller_took_out_is_still_hands(timed: Timed) -> None:
    timed.segments.append(spaced("Should I the tests?", 0.1, 0.3))
    clip_heard = await told(saying(2.0, ("Should I run the tests?", 0.0, 1.5)))
    assert (clip_heard.said, clip_heard.echo) == (None, ("Should", "I", "the", "tests?"))


async def test_each_word_of_hands_is_taken_once_so_a_sentence_said_once_and_heard_twice_is_a_persons_the_second_time(timed: Timed) -> None:
    timed.segments += [[(" Okay.", 0.1, 0.5)], [(" Okay.", 1.1, 1.5)]]
    assert (await told(saying(2.0, ("Okay.", 0.0, 0.5), ("Okay.", 1.0, 1.5)))).echo == ("Okay.", "Okay.")
    # Said once, at the start: the second, 1 s on and within reach of it, is someone answering it.
    clip_heard = await told(saying(2.0, ("Okay.", 0.0, 0.5)))
    assert (clip_heard.said, clip_heard.echo) == ("Okay.", ("Okay.",))


async def test_a_one_word_sentence_of_hands_heard_alone_and_one_spelled_whispers_way_are_hands(timed: Timed) -> None:
    timed.segments += [[(" Done.", 0.05, 0.45)], spaced("The diariser reports one speaker.", 1.1, 0.4)]
    clip_heard = await told(saying(3.5, ("Done.", 0.0, 0.4), ("The diarizer reports one speaker.", 1.0, 3.0)))
    assert (clip_heard.said, clip_heard.echo) == (None, ("Done.", "The", "diariser", "reports", "one", "speaker."))


async def test_a_sentence_of_hands_begun_before_the_clip_opened_is_hands_where_the_clip_hears_its_end(timed: Timed) -> None:
    # The clip opened 2.5 s into hands' sentence: it hears "continue with the release", and nothing before.
    timed.segments.append([(" continue", 0.2, 0.5), (" with", 0.6, 0.8), (" the", 0.9, 1.0), (" release.", 1.1, 1.5)])
    clip_heard = await told(saying(3.0, ("Checking the speech models before we continue with the release.", -2.5, 2.0)))
    assert (clip_heard.said, clip_heard.echo) == (None, ("continue", "with", "the", "release."))


async def test_a_person_who_cut_hands_off_is_sent_the_words_hands_never_got_to(timed: Timed) -> None:
    # Cut off 1 s in, hands never said "the release", which it would have 3 s in: the person saying it then is no echo.
    timed.segments.append([(" Starting", 0.1, 0.3), (" the", 0.4, 0.5), (" build", 0.6, 0.9), (" Stop,", 1.2, 1.4), (" ship", 2.9, 3.1), (" the", 3.2, 3.3), (" release.", 3.4, 3.8)])
    clip_heard = await told(saying(4.0, ("Starting the build now, and once it is green, the release.", 0.0, 1.0)))
    assert (clip_heard.said, clip_heard.echo) == ("Stop, ship the release.", ("Starting", "the", "build"))


async def test_a_short_word_of_a_persons_a_letter_off_one_of_hands_is_the_persons(timed: Timed) -> None:
    # "they" is "the" a letter off, and the canceller took hands' "the" out: the person's "They" is theirs.
    timed.segments += [[(" Check", 0.1, 0.2), (" logs.", 0.6, 0.8)], [(" They", 0.4, 0.5), (" said", 0.9, 1.0), (" no.", 1.1, 1.2)]]
    clip_heard = await told(saying(2.0, ("Check the logs.", 0.0, 1.0)))
    assert (clip_heard.said, clip_heard.echo) == ("They said no.", ("Check", "logs."))


async def test_a_word_of_hands_heard_out_of_its_order_leaves_the_rest_of_hands_words_hands(timed: Timed) -> None:
    # "build" heard before "the logs and": it takes hands' "build", and skips none of the words before it.
    timed.segments.append([(" build", 0.2, 0.3), (" the", 0.4, 0.5), (" logs", 0.7, 0.8), (" and", 1.0, 1.1)])
    clip_heard = await told(saying(2.5, ("Check the logs and the build.", 0.0, 2.0)))
    assert (clip_heard.said, clip_heard.echo) == (None, ("build", "the", "logs", "and"))


async def test_a_segment_all_hands_between_two_of_a_persons_leaves_their_words_joined_as_said(timed: Timed) -> None:
    # The segment of hands' sentence also holds a dash, which is no word of anyone's.
    timed.segments += [spaced("Run the tests.", 0.1, 0.2), [*spaced("Build is green.", 1.0, 0.2), (" -", 1.6, 1.7)], spaced("Then deploy.", 2.6, 0.2)]
    clip_heard = await told(saying(3.5, ("Build is green.", 0.9, 1.9)))
    assert clip_heard.said == "Run the tests. Then deploy."


async def test_a_clip_hands_played_nothing_over_is_decoded_without_its_words_timed_and_nothing_of_it_is_hands(timed: Timed) -> None:
    timed.segments.append(spaced("The parser is fixed.", 0.1, 0.3))
    # At the phone, whose own canceller is told what it plays, and hands is not; and at the desk, with nothing played.
    for sound in (as_heard(bytes(64_000)), as_heard(bytes(64_000), (None,) * 200)):
        assert ((await told(sound)).said, timed.asked[-1]) == ("The parser is fixed.", False)
    await told(saying(2.0, ("Done.", 0.0, 0.4)))
    assert timed.asked[-1]


async def test_a_persons_words_that_hands_said_too_are_sent_where_the_canceller_took_hands_voice_out(timed: Timed) -> None:
    # Over hands' "the build is green", a person says "No, the build is broken.", their "the build is" where hands said
    # it: through a canceller taking hands' voice out, nothing Whisper heard is hands'.
    timed.segments.append(spaced("No, the build is broken.", 1.6, 0.3))
    clip_heard = await told(cancelling(saying(3.5, ("Phosphene finished its render, and the build is green.", 0.0, 3.4)), 0.0, 3.5))
    # [LAW:nothing-unseen] every word hands played was taken out, and the clip was decoded with no word timed.
    assert (clip_heard.said, clip_heard.echo, clip_heard.unheard, timed.asked[-1]) == ("No, the build is broken.", (), 9, False)


async def test_hands_words_the_canceller_left_in_are_hands_beside_a_stretch_it_took_hands_voice_out_of(timed: Timed) -> None:
    # The canceller took hands' first sentence out, and left its second, 2 s on, as it came through the room.
    timed.segments += [spaced("Okay, run the tests.", 0.1, 0.3), spaced("The build is green.", 2.6, 0.3)]
    clip_heard = await told(cancelling(saying(4.0, ("Okay, run the tests.", 0.0, 1.5), ("The build is green.", 2.5, 3.8)), 0.0, 1.5))
    assert (clip_heard.said, clip_heard.echo, clip_heard.unheard) == ("Okay, run the tests.", ("The", "build", "is", "green."), 4)


async def test_a_word_hands_played_before_the_clip_opened_is_hands_where_the_canceller_took_out_only_what_came_later(timed: Timed) -> None:
    # The key opened 2 s into hands' sentence, and the echo of its "finished", played 1.4 s before, is heard as the clip
    # opens: the canceller taking hands' voice out from 0.5 s on is no reach of a word that far before.
    timed.segments.append(spaced("finished", 0.05, 0.25))
    clip_heard = await told(cancelling(saying(3.0, ("Phosphene finished its render, and the build is green.", -2.0, 1.4)), 0.5, 3.0))
    assert (clip_heard.said, clip_heard.echo) == (None, ("finished",)), clip_heard


async def test_hands_words_the_canceller_left_in_after_a_minute_of_loud_speech_are_hands(timed: Timed) -> None:
    # 50 s at -10 dBFS, then hands' echo captured at -45 dBFS and heard at -55: the canceller took out 10 dB of it, short
    # of `_CANCELLING`, however long the loud stretch before it.
    timed.segments.append(spaced("The build is green.", 55.1, 0.3))
    sound = saying(60.0, ("The build is green.", 55.0, 56.5))
    loud = np.random.default_rng(0).normal(0, 10_362, 50 * 16_000)
    echo = np.frombuffer(sound.captured, np.int16)[50 * 16_000 :].astype(np.float64)
    captured, heard = (np.concatenate((loud, echo * deviation / 3_277)).astype(np.int16).tobytes() for deviation in (184, 58))
    clip_heard = await told(Sound(heard, captured, sound.played))
    assert (clip_heard.said, clip_heard.unheard) == (None, 0), clip_heard


async def test_a_persons_words_that_hands_said_too_are_sent_where_no_echo_of_hands_reached_the_microphone(timed: Timed) -> None:
    # The same person over the same sentence, with hands' voice in headphones: there was no echo for a canceller to take
    # out, and none of hands' voice is in what Whisper heard.
    timed.segments.append(spaced("No, the build is broken.", 1.6, 0.3))
    sound = headphones(saying(3.5, ("Phosphene finished its render, and the build is green.", 0.0, 3.4)))
    clip_heard = await told(sound)
    # [LAW:nothing-unseen] every word hands played is unheard, and the echo path read no higher than chance.
    assert (clip_heard.said, clip_heard.echo, clip_heard.unheard, timed.asked[-1]) == ("No, the build is broken.", (), 9, False)
    assert clip_heard.echo_path is not None and clip_heard.echo_path < 10, clip_heard
    # Whisper heard the clip as the microphone captured it, not as the canceller ducked it.
    assert np.array_equal(timed.samples[-1][: len(sound.captured) // 2], decoded(sound.captured))


async def test_the_echo_path_of_hands_voice_through_the_room_reads_far_above_chance_and_none_where_hands_played_nothing(timed: Timed) -> None:
    timed.segments.append(spaced("Done.", 0.1, 0.3))
    assert (await told(as_heard(SILENCE, (None,) * 100))).echo_path is None
    echoed = await told(saying(2.0, ("Done.", 0.0, 1.0)))
    assert echoed.echo_path is not None and echoed.echo_path > 50, echoed
