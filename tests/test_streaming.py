"""Streaming speech recognition with diarization: each clip's words heard while it is open, each with its speaker, a
hearing for each step the diarizer commits, one session for the run, and the real pair hearing words in every clip engaged
conversation would close, none in noise, and each word from the speaker who said it."""

import asyncio
import contextlib
import itertools
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import soundfile
from loguru import logger

import recordings
import echoes
from echoes import replayed
from conftest import CALLED, PAIRS, model_facts
from listening import LEAD, SHAPES, Live, Timed, turn
import echoscore
from hands.clock import SYSTEM
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice import engaged, streaming
from hands.voice.clipowner import ClipOwner
from hands.voice.clips import Clip, Fate, Sound, decoded
from hands.voice.streaming import CHUNK, MEL_HOP, STEP, Active, Closed, Decoded, Ended, Said, Samples, Session, Spoken, StreamingRecognizer, steps, through
# The real load, bound before conftest stands a pair that hears nothing in for it.
from hands.voice.streaming import load as nemotron

FIXTURES = Path(__file__).parent / "fixtures"


def said(*texts: str | tuple[int, str]) -> Spoken:
    """What a feed decoded: each text speaker 0's, unless it names its speaker."""
    return tuple(Said(*text) if isinstance(text, tuple) else Said(0, text) for text in texts)


NOTHING_CLOSED = Closed(frozenset(), ())


@dataclass
class Scripted:
    """The run's session as the tests script it: each time it is fed, what it decodes, oldest first, nothing once the
    script runs out; each time it is closed, what that closes, closing nothing once that script runs out; and every feed
    it is to fail, counted from 0."""

    spoken: list[Spoken]
    failing: frozenset[int] = frozenset()
    closing: list[Closed] = field(default_factory=list[Closed])
    fed: list[Samples] = field(default_factory=list[Samples])
    # Where the diarizer heard each speaker, a feed's at a time, as `spoken` is scripted.
    active: list[tuple[Active, ...]] = field(default_factory=list[tuple[Active, ...]])

    def decoded(self, samples: Samples) -> Decoded:
        self.fed.append(samples)
        if len(self.fed) - 1 in self.failing:
            raise RuntimeError("Metal ran out of memory")
        return Decoded(self.spoken.pop(0) if self.spoken else (), self.active.pop(0) if self.active else ())

    def closed(self) -> Closed:
        return self.closing.pop(0) if self.closing else NOTHING_CLOSED


class Unused:
    """A transcriber no test here gives a clip to."""

    async def transcribed(self, clip: Clip, sound: Sound) -> str | None:
        raise AssertionError("nothing here transcribes")


def pcm(samples: Samples) -> bytes:
    return (samples * 32768).astype(np.int16).tobytes()


def decoded_as_captured(samples: Samples) -> Samples:
    """`samples` as a clip carries them: through 16-bit audio and back."""
    return np.frombuffer(pcm(samples), np.int16).astype(np.float32) / np.float32(32768)


# The samples of a diarizer's step after the run's first, which a hearing follows each of.
STEPPED = STEP * MEL_HOP


def hearings(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hearing" and entry.facts["listener"] == "streaming"]


@dataclass
class Rig:
    """The listener, the clip owner running its hearings as a pipeline's would, and what they recorded."""

    listener: StreamingRecognizer
    owned: ClipOwner
    recorded: list[Entry]

    async def run(self, clip: Clip, told: tuple[object, ...]) -> None:
        for hearing in told:
            # As the owner's hearing lane runs each: one that fails is said on its event, and the lane hears on.
            with contextlib.suppress(Exception):
                await self.owned._heard(self.listener, clip, hearing, begun(SYSTEM))  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    async def opened(self, clip: Clip, before: Samples) -> None:
        await self.run(clip, self.listener.opened(clip, Sound(pcm(before), pcm(before), None)))

    async def heard(self, clip: Clip, samples: Samples, frame: int = STEPPED) -> None:
        """`samples` of `clip`, given in frames of `frame` samples, and every hearing they bring run."""
        for start in range(0, len(samples), frame):
            piece = pcm(samples[start : start + frame])
            await self.run(clip, self.listener.heard(Sound(piece, piece, None)))

    async def ended(self, clip: Clip, fate: Fate) -> None:
        await self.run(clip, self.listener.ended(clip, fate, Sound(b"", b"", None)))

    def told(self) -> list[tuple[object, object, object]]:
        """What each hearing came to: its outcome, what it heard, and the words on its event: those each speaker added
        while the clip was open, and each speaker's whole as it ended."""
        return [(event.outcome, event.facts["heard"], event.facts.get("added", event.facts.get("words"))) for event in hearings(self.recorded)]


def rigged(session: Session) -> Rig:
    recorded: list[Entry] = []
    listener = StreamingRecognizer(SYSTEM, lambda _: session, recorded.append)
    return Rig(listener, ClipOwner(SYSTEM, Unused(), (listener,), recorded.append), recorded)


def sound(seconds: float, level: float = 0.1) -> Samples:
    return np.full(round(seconds * streaming.RATE), level, np.float32)


CLIP = Clip(1, "engaged conversation")


def seconds(samples: int) -> float:
    return samples / streaming.RATE


async def test_a_clips_words_are_heard_while_it_is_open_each_hearing_the_words_it_added() -> None:
    rig = rigged(Scripted([(), (), said("wait,"), said(" not"), ()]))
    await rig.opened(CLIP, sound(0))
    # The opening is fed first, with nothing kept from before it. The first step needs 5216 samples, so the second
    # frame of the clip completes it; one hearing for each step after, saying the words it added, until one adds none.
    await rig.heard(CLIP, sound(seconds(4 * STEPPED)))
    assert rig.told() == [("ok", "words", said("wait,")), ("ok", "words", said("not")), ("ok", "nothing", ())]


async def test_a_hearing_follows_the_feed_that_completes_a_step_however_the_clip_is_framed() -> None:
    rig = rigged(Scripted([*[()] * 17, said("stop")]))
    await rig.opened(CLIP, sound(0))
    # 20 ms frames: the diarizer's first step waits for 5216 samples, so the 17th frame completes it, and only it is heard.
    await rig.heard(CLIP, sound(0.32), frame=320)
    assert rig.told() == []
    await rig.heard(CLIP, sound(0.02), frame=320)
    assert rig.told() == [("ok", "words", said("stop"))]
    # The next completes 3840 samples on, in the 29th.
    await rig.heard(CLIP, sound(0.22), frame=320)
    assert len(rig.told()) == 1
    await rig.heard(CLIP, sound(0.02), frame=320)
    assert len(rig.told()) == 2


async def test_each_speakers_words_are_kept_apart_as_they_are_heard_and_as_the_clip_ends() -> None:
    rig = rigged(Scripted([(), (), said("is it", (1, "yes")), said(" done")]))
    await rig.opened(CLIP, sound(0))
    await rig.heard(CLIP, sound(seconds(3 * STEPPED)))
    await rig.ended(CLIP, "sent")
    # What cuts is every speaker's words; which speaker said them is on the event.
    assert [(event.facts["heard"], event.facts.get("added", event.facts.get("words"))) for event in hearings(rig.recorded)] == [
        ("words", said("is it", (1, "yes"))), ("words", said("done")), ("words", said("is it done", (1, "yes"))),
    ]


async def test_the_full_stop_that_ends_the_last_clip_is_no_word_of_this_one() -> None:
    rig = rigged(Scripted([(), (), said("."), said(" he went")]))
    await rig.opened(CLIP, sound(0))
    await rig.heard(CLIP, sound(seconds(3 * STEPPED)))
    assert rig.told() == [("ok", "nothing", ()), ("ok", "words", said("he went"))]


def test_the_silence_after_a_clip_runs_until_the_chunk_holding_its_last_sample_is_committed() -> None:
    # A clip ending at 9600 samples: its last sample is in mel frame 61, chunk 3, so the diarizer must commit through
    # 64 frames, which its third step does, at 12896 samples.
    assert (CHUNK, through(9600)) == (16, 12_896)
    # The silence ends on the sample that completes that step, no later.
    assert steps(12_895) < steps(12_896)


async def test_each_clip_is_heard_from_what_it_kept_before_it_opened_to_the_silence_after_it() -> None:
    session = Scripted([(), said("turn it"), said("."), said(" and the"), (), ()])
    rig = rigged(session)
    first, second = Clip(1, "engaged conversation"), Clip(2, "shortcut")
    await rig.opened(first, sound(0.5, 0.25))
    await rig.heard(first, sound(0.2, 0.25))
    await rig.ended(first, "sent")
    await rig.opened(second, sound(0.1, 0.5))
    await rig.heard(second, sound(0.2, 0.5))
    await rig.ended(second, "dropped")
    # What the stream heard, in order: each clip, and the silence after it until the diarizer has committed the chunk
    # holding its last sample; the second clip starts where that silence ended.
    assert [(len(fed), round(float(fed[0]), 2)) for fed in session.fed] == [
        (8_000, 0.25), (3_200, 0.25), (5_536, 0.0), (1_600, 0.5), (3_200, 0.5), (2_880, 0.0)
    ]
    # The first clip's full stop is its own, decoded in the silence after it, and only its end says it.
    assert rig.told() == [
        ("ok", "nothing", ()), ("ok", "words", said("turn it")), ("ok", "words", said("turn it.")),
        ("ok", "words", said("and the")), ("ok", "words", said("and the")),
    ]


async def test_a_clip_that_ends_is_heard_through_the_silence_after_it_and_told_whole_with_its_fate() -> None:
    rig = rigged(Scripted([(), said("the parser"), said(" is fixed")]))
    await rig.opened(CLIP, sound(0))
    await rig.heard(CLIP, sound(seconds(2 * STEPPED)))
    await rig.ended(CLIP, "sent")
    *_, whole = hearings(rig.recorded)
    # The words after it closed cut nothing: the clip is no longer open.
    assert (whole.facts["fate"], whole.facts["words"], whole.facts["seconds"], whole.facts["barge_in"]) == ("sent", said("the parser is fixed"), 0.48, False)
    # The clip's last sample is in the run's fourth chunk, so the silence runs to the step that commits it: the run fed
    # through 12896 samples.
    assert whole.facts["silence"] == 0.326


async def test_a_clips_close_ends_it_with_its_last_words_and_none_of_them_come_out_in_the_next() -> None:
    # The close decodes each stream's last token, which feeding alone leaves pending: speaker 0's " done", and a word
    # of speaker 1, who said nothing else in the clip.
    rig = rigged(Scripted([(), (), said("is it"), (), (), (), said("stop")], closing=[Closed(frozenset({0, 1}), said(" done", (1, "yes")))]))
    for clip in (Clip(1, "engaged conversation"), Clip(2, "engaged conversation")):
        await rig.opened(clip, sound(0))
        await rig.heard(clip, sound(seconds(2 * STEPPED)))
        await rig.ended(clip, "sent")
    assert rig.told() == [
        ("ok", "words", said("is it")), ("ok", "words", said("is it done", (1, "yes"))),
        ("ok", "nothing", ()), ("ok", "words", said("stop")), ("ok", "words", said("stop")),
    ]
    # [LAW:nothing-unseen] each clip's end says whose streams it closed and what the close, not the feed, decoded.
    assert [(event.facts["closed"], event.facts["closing"]) for event in hearings(rig.recorded) if "closed" in event.facts] == [
        (frozenset({0, 1}), said(" done", (1, "yes"))), (frozenset(), ()),
    ]


async def test_a_clip_nothing_was_said_in_ends_with_no_words() -> None:
    rig = rigged(Scripted([]))
    await rig.opened(CLIP, sound(0))
    await rig.heard(CLIP, sound(seconds(2 * STEPPED)))
    await rig.ended(CLIP, "dropped")
    assert rig.told() == [("ok", "nothing", ()), ("ok", "nothing", ())]


async def test_decoding_that_fails_breaks_the_runs_stream_and_each_clip_from_it_on_fails_once_as_it_ends() -> None:
    # The models' caches are left in a state nothing vouches for, and the grid hearings are timed on with them.
    session = Scripted([(), (), (), said("stop")], failing=frozenset({2}))
    rig = rigged(session)
    logged: list[Any] = []
    sink = logger.add(lambda message: logged.append(message.record), level="ERROR")
    try:
        for number in (1, 2):
            await rig.opened(Clip(number, "engaged conversation"), sound(0))
            await rig.heard(Clip(number, "engaged conversation"), sound(seconds(2 * STEPPED)))
            await rig.ended(Clip(number, "engaged conversation"), "sent")
    finally:
        logger.remove(sink)
    # Where in the pair it broke is said once, with what was raised there.
    [broke] = logged
    assert broke["exception"].value.args == ("Metal ran out of memory",)
    failed = ("failed", "RuntimeError: decoding failed: RuntimeError: Metal ran out of memory")
    # The second clip, further along the run's grid, completes two steps.
    assert [(event.outcome, event.error) for event in hearings(rig.recorded)] == [("ok", None), failed, ("ok", None), ("ok", None), failed]
    # Nothing is fed to the models once they broke.
    assert len(session.fed) == 3


async def test_a_speakers_words_cut_hands_off_from_when_they_were_first_heard() -> None:
    # Half a second kept from before the clip opened; the speaker first heard in the run's eighth frame, 0.06 s after it.
    session = Scripted([(), said("stop")], active=[(), (Active(0, 7), Active(0, 8))])
    stream = streaming.Stream(session)
    stream.opened(sound(0.5), 0)
    stream.heard(sound(seconds(STEPPED)))
    since = stream.said()
    assert since.added == said("stop")
    assert since.barge_in_at == pytest.approx(0.06)


def test_a_sound_the_diarizer_took_for_a_speaker_with_no_words_in_it_cuts_nothing() -> None:
    session = Scripted([(), ()], active=[(), (Active(0, 7), Active(0, 8))])
    stream = streaming.Stream(session)
    stream.opened(sound(0.5), 0)
    stream.heard(sound(seconds(STEPPED)))
    assert stream.said().barge_in_at is None


def test_a_speaker_marked_only_where_the_clip_has_no_samples_was_first_heard_nowhere_and_cuts_nothing() -> None:
    # Speaker 0 only in frame 40, 3.2 s into a run whose clip is 0.48 s: in the silence after it.
    stream = streaming.Stream(Scripted([(), said("stop")], active=[(), (Active(0, 40),)]))
    stream.opened(sound(0), 0)
    stream.heard(sound(seconds(2 * STEPPED)))
    since = stream.said()
    assert since.added == said("stop") and since.barge_in_at is None


def loaded(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "streaming.loaded"]


async def test_clips_spoken_while_the_models_load_are_heard_once_they_have() -> None:
    recorded: list[Entry] = []
    go = threading.Event()

    def slow(_gating: int) -> Session:
        go.wait()
        return Scripted([(), said("hello")])

    listener = StreamingRecognizer(SYSTEM, slow, recorded.append)
    rig = Rig(listener, ClipOwner(SYSTEM, Unused(), (listener,), recorded.append), recorded)
    await rig.opened(CLIP, sound(0))
    hearing = asyncio.create_task(rig.heard(CLIP, sound(seconds(2 * STEPPED))))
    await asyncio.sleep(0.1)
    assert (loaded(recorded), rig.told()) == ([], [])
    go.set()
    await hearing
    assert rig.told() == [("ok", "words", said("hello"))]
    [load] = loaded(recorded)
    assert (load.outcome, model_facts(load)) == (
        "ok",
        {
            "model": streaming.MODEL, "revision": streaming.REVISION, "diarization": streaming.DIARIZATION,
            "diarization_revision": streaming.DIARIZATION_REVISION, "preset": "ultra_low", "gating": streaming.GATING, "transcription_delay_ms": 160,
        },
    )


async def test_models_that_fail_to_load_fail_each_clip_once_saying_why() -> None:
    recorded: list[Entry] = []

    def unreachable(_gating: int) -> Session:
        raise OSError("the hub is unreachable")

    listener = StreamingRecognizer(SYSTEM, unreachable, recorded.append)
    rig = Rig(listener, ClipOwner(SYSTEM, Unused(), (listener,), recorded.append), recorded)
    for number in (1, 2):
        await rig.opened(Clip(number, "engaged conversation"), sound(0))
        await rig.heard(Clip(number, "engaged conversation"), sound(seconds(2 * STEPPED)))
        await rig.ended(Clip(number, "engaged conversation"), "sent")
    failed = ("failed", "RuntimeError: the models did not load: OSError: the hub is unreachable")
    assert [(event.outcome, event.error) for event in hearings(recorded)] == [("ok", None), failed, ("ok", None), ("ok", None), failed]
    [load] = loaded(recorded)
    assert (load.outcome, load.error) == ("failed", "OSError: the hub is unreachable")


def test_a_recognizer_that_frames_the_run_with_another_fft_is_refused_before_a_session_is_made(monkeypatch: pytest.MonkeyPatch) -> None:
    # `through` places a clip's last mel frame by the recognizer's FFT, which the session does not check against the
    # diarizer's.
    import mlx_audio.stt  # pyright: ignore[reportMissingTypeStubs]

    def recognizer(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(preprocessor_config=SimpleNamespace(n_fft=400))

    monkeypatch.setattr(mlx_audio.stt, "load", recognizer)
    with pytest.raises(ValueError, match="FFT of 400; hands places a clip's last mel frame on 512"):
        nemotron(streaming.GATING)


def test_a_speaker_is_the_number_the_session_gave_them() -> None:
    assert streaming.speaker("speaker_3") == 3
    with pytest.raises(ValueError, match="'hands'"):
        streaming.speaker("hands")


# The real pair, on real sound: the weights are fetched into the Hugging Face cache the first time, and CI keeps that
# cache. It runs on MLX's default device, and is the only pair loaded while it is held (conftest's `Pairs`). Each test
# starts the session afresh, as a run starts one. The fixtures are clips separated by half a second of exact zeros, which
# speech never holds for that long.
GAP = streaming.RATE // 2


@pytest.fixture(scope="module")
def pair() -> Any:
    """The real pair, loaded once, as a run loads it, and the only one the suite holds while this module runs."""
    return PAIRS.loaded(lambda: nemotron(streaming.GATING))


@pytest.fixture
def paired(pair: Any) -> Any:
    """The pair's session as a run begins it: nothing heard, nobody numbered."""
    pair.reset()
    return pair


@pytest.fixture
def real(request: pytest.FixtureRequest, pair: Any) -> Iterator[Callable[[str], Session]]:
    """The real pair's session for each run, by name, as a run begins one. With --record-decodes, a test that passes writes
    the recording of each run it made (tests/recordings.py), the only way one is made."""
    recorders = dict[str, recordings.Recorder]()

    def of(name: str) -> Session:
        assert name not in recorders, f"the run {name!r} begun twice in one test: its first recording would be lost"
        pair.reset()
        recorders[name] = recordings.Recorder(pair)
        return recorders[name]

    yield of
    # pytest types `node` as unknown on a fixture request, whatever the fixture's scope.
    called = cast(pytest.Item, request.node).stash.get(CALLED, None)  # pyright: ignore[reportUnknownMemberType]
    if request.config.getoption("record_decodes") and called is not None and called.passed:
        for name, recorder in recorders.items():
            recordings.write(name, recorder.recording())


@pytest.fixture(params=["recorded", pytest.param("real", marks=[pytest.mark.outside, pytest.mark.weights("streaming")])])
def session(request: pytest.FixtureRequest) -> Callable[[str], Session]:
    """The session each run, by name, is heard through, in the unit layer or outside it. `recorded` serves what the real pair
    decoded for the run; `real` is the pair. The test asserts the same of both, which is what holds the recording to the
    pair: a recording the pair no longer decodes as fails a `real` case, and a run fed other than it was recorded from fails
    a `recorded` one."""
    return recordings.read if request.param == "recorded" else request.getfixturevalue("real")


def fixture_clips(name: str) -> list[Samples]:
    audio, rate = cast(tuple[Samples, int], soundfile.read(FIXTURES / name, dtype="float32"))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
    assert rate == streaming.RATE
    # Every window of GAP samples that is all exact zeros, from the running count of zeros: a sliding window's sum in one pass.
    zeros = np.concatenate([[0], np.cumsum(audio == 0)])
    silent = zeros[GAP:] - zeros[:-GAP] == GAP
    gaps = np.flatnonzero(np.diff(np.concatenate([[False], silent, [False]]).astype(np.int8)))
    edges = [0, *(edge + offset for edge, offset in zip(gaps, [0, GAP - 1] * (len(gaps) // 2))), len(audio)]
    return [audio[start:end] for start, end in zip(edges[0::2], edges[1::2])]


def spoken_words(spoken: Spoken) -> list[tuple[int, str]]:
    """Each word each speaker said, in lower case, with no punctuation."""
    return [(said.speaker, word) for said in spoken for word in re.findall(r"[\w']+", said.text.lower())]


class Stepped:
    """mlx-audio's session as `Paired` reads it: what it was fed, feed by feed, and the count of it the session keeps."""

    def __init__(self) -> None:
        self.fed: list[Samples] = []
        self._frontend = SimpleNamespace(total_samples=0)
        self._push_features = None

    def feed(self, samples: Samples) -> list[Any]:
        self.fed.append(samples)
        self._frontend.total_samples += len(samples)
        return []

    def reset(self) -> None:
        self.fed = []
        self._frontend.total_samples = 0


def test_the_session_is_fed_nothing_until_a_feed_completes_a_step_and_then_all_of_it_in_order() -> None:
    session = Stepped()
    paired = streaming.Paired(session)
    # Each diarizer step costs the session a feed, and a feed reads the GPU, so 20 ms frames between steps must not.
    run = np.arange(32000, dtype=np.float32)
    frames = [run[start : start + 320] for start in range(0, len(run), 320)]
    told = [paired.decoded(frame) for frame in frames]
    ends = [len(frame) + index * 320 for index, frame in enumerate(frames)]
    completing = [end for end, before in zip(ends, [0, *ends]) if steps(end) > steps(before)]
    assert list(itertools.accumulate(map(len, session.fed))) == completing
    assert np.array_equal(np.concatenate(session.fed), run[: completing[-1]])
    assert all(decoded == Decoded((), ()) for decoded in told)


def test_a_run_begun_afresh_keeps_nothing_of_one_stopped_part_way_through_a_step() -> None:
    session = Stepped()
    paired = streaming.Paired(session)
    paired.decoded(np.ones(320, np.float32))
    paired.reset()
    first = next(samples for samples in range(320, 32000, 320) if steps(samples) > 0)
    for _ in range(first // 320):
        paired.decoded(np.zeros(320, np.float32))
    assert [len(fed) for fed in session.fed] == [first]
    assert not np.any(session.fed[0])


def test_a_clip_closed_short_of_a_step_fails_rather_than_carrying_its_samples_into_the_next() -> None:
    paired = streaming.Paired(Stepped())
    paired.decoded(np.ones(320, np.float32))
    with pytest.raises(AssertionError, match="ends every clip on a step"):
        paired.closed()


# Clips of one LibriVox reader (The Invisible Man, read solo; public domain), 20 to 30 dB down: utterances of 0.8 to 2 s,
# cut from the reading wherever it paused, so a clip can begin or end inside a phrase. One reader is one speaker, the
# first the run heard.
READER = "one_reader_quiet.flac"
# What engaged conversation's detector waits out before it closes a clip: the only clips whose cut waits on words.
STOP = np.zeros(round(engaged.STOP_SECS * streaming.RATE), np.float32)


def test_every_clip_engaged_conversation_would_close_has_words_heard_while_it_was_open(session: Callable[[str], Session]) -> None:
    live = Live(session("reader-first-8"))
    heard = [live.spoken(np.concatenate([samples, STOP])) for samples in fixture_clips(READER)[:8]]
    assert all(open_ for open_, _ in heard), heard
    assert {said.speaker for open_, whole in heard for said in (*open_, *whole.words)} == {0}, heard
    # What it heard open is the start of what it heard whole: the model never takes back a word it has given.
    assert all(whole.words[0].text.startswith(open_[0].text) for open_, whole in heard), heard
    assert heard[0][1].words[0].text.rstrip(".") == "Door was ajar"


def test_each_clip_opens_on_its_own_first_word_not_the_last_of_the_clip_before(session: Callable[[str], Session]) -> None:
    # Clips 3 to 10 of the reading, each followed by the stop, as engaged conversation closes them. Their source text:
    # "...and found the bottle as he had been directed. But returning with the bottle, he noticed that the bolts of the
    # front door had been shot back, that the door was in fact simply on the latch. And with a flash of inspiration he
    # connected this with the stranger's room upstairs..." Clip 5 is heard and not asserted: its cut falls inside "he
    # noticed that", so the text does not say which word opens it.
    live = Live(session("reader-3-10"))
    first = {number: spoken_words(live.spoken(np.concatenate([samples, STOP]))[1].words)[0] for number, samples in enumerate(fixture_clips(READER)[2:10], 3)}
    del first[5]
    assert first == {3: (0, "and"), 4: (0, "but"), 6: (0, "that"), 7: (0, "simply"), 8: (0, "and"), 9: (0, "he"), 10: (0, "the")}


# A woman coughing three times ("Woman coughing three times.wav", Ashe (OwlStorm on Freesound), CC0, from Wikimedia
# Commons), and 3 s of a computer keyboard typed on ("Computer keyboard.ogg", russiandoll, public domain, from Wikimedia
# Commons, from 1.5 s in), each made mono at 16 kHz.
NOT_SPEECH = [*fixture_clips("cough.flac"), *fixture_clips("keyboard_typing.flac")]


def test_a_clip_with_no_speech_in_it_has_no_words_so_it_cuts_nothing(session: Callable[[str], Session]) -> None:
    # What a clip engaged conversation opens can hold with nobody speaking: the room, a hiss, a tap on the desk, a cough,
    # the keyboard.
    noise = np.random.default_rng(0).standard_normal(2 * streaming.RATE).astype(np.float32)
    clicks = np.where(np.arange(2 * streaming.RATE) % 4_000 == 0, 0.8, 0.0).astype(np.float32)
    clips = [np.zeros(2 * streaming.RATE, np.float32), noise * 0.01, noise * 0.056, clicks, *NOT_SPEECH]
    live = Live(session("no-speech"))
    assert [(open_, whole.words) for open_, whole in map(live.spoken, clips)] == [((), ())] * len(clips)


def test_each_word_is_the_speaker_who_said_it_and_each_clip_closes_its_own_speakers_streams(session: Callable[[str], Session]) -> None:
    # Clips of two women, each speaking alone, in the order they spoke: AMI Meeting Corpus ES2004a, mixed headsets,
    # speakers FEE013 and FEE016 (CC BY 4.0, University of Edinburgh). Each clip's speaker is the one the transformers
    # path diarizing on the CPU gave it, which every word's speaker matched over all 16 clips, the run started at five
    # phases of the grid (promptctl-voice-i15t.qg8.5yl). The first six, so it stays inside the test timeout on a CPU: a
    # run's start is fed as the whole run's is, so it decides as the whole did. A later stretch alone does not: with less
    # of each voice heard, both paths take both speakers to be speaking in clip 15, and both speakers' streams say it.
    live = Live(session("ami-6"))
    ended = [whole for _, whole in map(live.spoken, fixture_clips("two_speakers_ami.flac")[:6])]
    assert [{said.speaker for said in clip.words} for clip in ended] == [{int(who)} for who in "000100"]
    # A clip where one person speaks closes their stream alone: one made from gate history carried across the close
    # before it would show here. In clips 4 and 5 the diarizer also marks, for a moment, a speaker who decodes no word.
    assert [clip.closed.speakers for clip in ended] == [{0}, {0}, {0}, {0, 1}, {0, 2}, {0}]


# Who says each clip of two_speakers_ami.flac that these cases hear.
WOMAN = {1: "FEE013", 2: "FEE013", 3: "FEE013", 4: "FEE016", 12: "FEE016"}


AMI = "two_speakers_ami.flac"


def spoken_alone(live: Live, alone: tuple[int, ...]) -> None:
    """Each woman of `alone` heard alone first, a clip of her own, closed as engaged conversation closes it."""
    clips = fixture_clips(AMI)
    for number in alone:
        live.spoken(np.concatenate([clips[number - 1], STOP]))


def spoken_joined(live: Live, joined: tuple[int, int]) -> tuple[Ended, int]:
    """One clip of a turn of each of `joined`, one after the other: what the clip came to, and how many samples the first
    turn is."""
    clips = fixture_clips(AMI)
    first, second = clips[joined[0] - 1], clips[joined[1] - 1]
    _, whole = live.spoken(np.concatenate([first, second, STOP]))
    return whole, len(first)


def each_persons_words(whole: Ended, alone: tuple[int, ...], joined: tuple[int, int]) -> list[int]:
    """hands gives each person's words as theirs, each the speaker numbered by when the run first heard her, in the order
    each was first heard in the clip: the speakers it gave, asserted."""
    arrived = list(dict.fromkeys(WOMAN[number] for number in alone))
    speakers = [arrived.index(WOMAN[number]) for number in joined]
    assert [said.speaker for said in whole.words] == speakers, whole.words
    return speakers


# Each woman heard alone first, then one clip of a turn of each, joined. One more clip of FEE013 before FEE016's first
# is what gives each her own in 12 then 3 (the next test).
TWO_PEOPLE = pytest.mark.parametrize(
    ("name", "alone", "joined"), [("ami-known-once", (1, 4), (3, 12)), ("ami-known-twice", (1, 2, 4), (12, 3))], ids=["known-once", "known-twice"]
)


@pytest.mark.outside
@pytest.mark.weights("streaming")
@TWO_PEOPLE
def test_two_people_in_one_clip_come_out_as_each_persons_words_on_the_real_pair(real: Callable[[str], Session], pair: Any, monkeypatch: pytest.MonkeyPatch, name: str, alone: tuple[int, ...], joined: tuple[int, int]) -> None:
    # Decided on when each word starts, not on its text, which is about 20% wrong on AMI. The tap goes on the pair's own
    # session, in place: a copy of the pair (`dataclasses.replace`) wraps the session's step again, and only its first
    # decode would carry the diarizer's frames, so a recording made through it would not be the pair's.
    timed = Timed(pair.session)
    monkeypatch.setattr(pair, "session", timed)
    live = Live(real(name))
    spoken_alone(live, alone)
    opened, heard = live.fed, len(timed.tokens)
    whole, first = spoken_joined(live, joined)
    speakers = each_persons_words(whole, alone, joined)
    boundary = seconds(opened + first)
    words = [(speaker, start) for speaker, start, text in timed.tokens[heard:] if any(map(str.isalnum, text))]
    # Each person's words start in their own stretch of the clip, and nobody else's do. mlx-audio dates a word from the
    # chunk the encoder was pushed, and the encoder may still hold up to a chunk of the feed before it, so a word's start
    # is late by as much as a chunk, never early: the first person's last word, said at the end of her stretch, can be
    # dated past it, and the second's first cannot be dated before it.
    late = seconds(CHUNK * MEL_HOP)
    assert {speaker for speaker, _ in words} == set(speakers), words
    assert [(speaker, start) for speaker, start in words if not (start < boundary + late if speaker == speakers[0] else boundary <= start)] == [], (boundary, words)


@TWO_PEOPLE
def test_a_recording_of_two_people_in_one_clip_comes_out_as_each_persons_words(name: str, alone: tuple[int, ...], joined: tuple[int, int]) -> None:
    live = Live(recordings.read(name))
    spoken_alone(live, alone)
    whole, _ = spoken_joined(live, joined)
    each_persons_words(whole, alone, joined)


def test_a_newcomer_first_in_a_clip_after_one_of_the_voice_the_run_knows_is_given_that_voice_s_speaker(session: Callable[[str], Session]) -> None:
    # The miss docs/features.md states ("Who said each word"), pinned so that a change to it fails here and the claim is
    # measured again: after FEE013's clip 1 and FEE016's clip 4, 12 then 3 gives FEE016's words, heard first, FEE013's
    # speaker. What becomes of FEE013's few words after them differs between Metal and MLX's CPU.
    live = Live(session("ami-newcomer"))
    clips = fixture_clips("two_speakers_ami.flac")
    for number in (1, 4):
        live.spoken(np.concatenate([clips[number - 1], STOP]))
    _, whole = live.spoken(np.concatenate([clips[12 - 1], clips[3 - 1], STOP]))
    assert whole.words[0].speaker == 0, whole.words


def test_a_clip_whose_last_chunk_is_committed_by_the_last_step_of_its_silence_keeps_its_last_word(session: Callable[[str], Session]) -> None:
    # Clip 15 of the reading, "There was no answer.", runs to its last sample, and its length is the worst phase of the
    # grid: the step that commits the chunk holding its last sample commits nothing past it, so a silence one step
    # shorter leaves "answer" undecoded at the close, and it comes out in the next clip, "It was as he expected."
    first, second = fixture_clips(READER)[14:16]
    holding_last = ((len(first) - 1 + streaming.N_FFT // 2) // MEL_HOP // CHUNK + 1) * CHUNK
    assert STEP * steps(through(len(first))) == holding_last
    live = Live(session("reader-last-chunk"))
    words = [spoken_words(whole.words) for _, whole in (live.spoken(first), live.spoken(second))]
    assert (words[0][-1], words[1][0]) == ((0, "answer"), (0, "it")), words


# Six turns of TS3003's four people, AMI Meeting Corpus (Mix-Headset; CC BY 4.0, University of Edinburgh), each from 0.1 s
# before its first word to engaged.STOP_SECS past its last, a silence nobody sounds in by AMI's manual word-level
# transcripts, which give each turn's words. scripts/measure-gating.py measures every such turn of the sixteen test
# meetings; these six it found ending right at GATING 1 and not at 4. They stay together: with less of each voice heard
# before them, the diarizer gives one person two numbers (promptctl-voice-i15t.qg8.2d2).
TURNS = "turns_ending_in_silence.flac"
# Each turn's first and last word, in seconds of its meeting, its person, and its words.
KNOWN = [
    (826.98, 828.96, "Yeah uh yeah"),  # TS3003b, B
    (920.09, 924.32, "So that brings us to the discussion about our concepts"),  # TS3003c, A
    (926.95, 927.81, "Mm"),  # TS3003c, A
    (1189.47, 1191.37, "It's a bit higher percentage but"),  # TS3003c, C
    (1157.62, 1159.08, "And we put uh"),  # TS3003d, A
    (1446.94, 1448.52, "I'm glad we could make a bit"),  # TS3003d, A
]


@pytest.mark.parametrize("shape", SHAPES)
def test_a_turn_that_ends_in_silence_ends_on_its_last_words_and_only_its_speaker_says_them(session: Callable[[str], Session], shape: str) -> None:
    # GATING decides how a clip's last words come out. At 4, four of these end otherwise in this run: "yeah" alone,
    # "concept", "but uh", "bit better", and in engaged conversation a second speaker says "But it breaks." after the first.
    live = Live(session(f"turns-{shape}"))
    # Each fixture clip opens LEAD before its turn's first word.
    clips = [turn(samples, LEAD, LEAD + end - start, SHAPES[shape]) for samples, (start, end, _) in zip(fixture_clips(TURNS), KNOWN, strict=True)]
    heard = [spoken_words(live.spoken(clip)[1].words) for clip in clips]
    # Its last two words, or the one of the turn that has one.
    ends = [(len({speaker for speaker, _ in words}), [word for _, word in words][-2:]) for words in heard]
    assert ends == [(1, re.findall(r"[\w']+", known.lower())[-2:]) for _, _, known in KNOWN], heard


@pytest.mark.outside
@pytest.mark.weights("streaming")
def test_the_diarizer_commits_on_exactly_the_feeds_steps_names(paired: Any) -> None:
    # [LAW:one-source-of-truth] the diarizer's own count of the frames it committed, against `steps`, fed 20 ms at a time.
    # The session's own diarizer, as `streaming.diarizer` set it to its preset.
    model = paired.session.diarization_model
    state = model.init_streaming_state()
    audio = np.concatenate(fixture_clips("one_reader_quiet.flac")[:4])
    moved: list[tuple[bool, bool]] = []
    for start in range(0, len(audio), 320):
        committed = state.frames_processed
        _, state = model.feed(audio[start : start + 320], state)
        moved.append((steps(start + len(audio[start : start + 320])) > steps(start), state.frames_processed > committed))
    assert all(timed == stepped for timed, stepped in moved), [at for at, (timed, stepped) in enumerate(moved) if timed != stepped]
    assert sum(stepped for _, stepped in moved) > 5


@dataclass
class Overheard:
    """The real pair's stream, fed a clip 20 ms at a time as the listener feeds it and asked at every step of the diarizer's
    as a hearing asks.

    Each cut is placed in the audio where it was asked for: how far into the clip the stream had heard, and when the
    speaker who cut was first heard. The stream runs on this thread, as the real pair's MLX streams must, so the place is
    the same on every machine."""

    stream: streaming.Stream
    samples: int = 0
    barge_ins: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])

    def held(self, clip: Samples) -> Ended:
        """A clip of `clip`, heard to its end."""
        stream = self.stream
        self.barge_ins.clear()
        stream.opened(np.zeros(0, np.float32), self.samples)
        for start in range(0, len(clip), FRAME_SAMPLES):
            piece = clip[start : start + FRAME_SAMPLES]
            stream.heard(piece)
            self.samples += len(piece)
            if steps(self.samples) > steps(self.samples - len(piece)):
                since = stream.said()
                self.barge_ins += [] if since.barge_in_at is None else [((start + len(piece)) / streaming.RATE, since.barge_in_at)]
        silence = through(self.samples) - self.samples
        stream.silence(silence)
        self.samples += silence
        return stream.ended()


# 20 ms, as the microphone hands the clip owner each frame.
FRAME_SAMPLES = 320

# The longest from a person's first word over hands to the cut, while the clip is open: the recognizer's first words of
# them, a chunk and a step of the diarizer's after they are said. Measured on the recorded takes through the desk's
# canceller: 1.68 s over a reply's opening, 1.44 s over its middle, 0.7 to 1.2 s elsewhere.
CUT_SECONDS = 2.0


@pytest.fixture
def overheard(paired: Session) -> Overheard:
    return Overheard(streaming.Stream(paired))


def desk_heard(fixture: Path) -> tuple[echoscore.Take, Samples]:
    """A take as the desk hears it: through the echo canceller."""
    recorded = echoscore.recorded(fixture)
    sound = replayed(recorded.captured, recorded.reference, recorded.playheads)
    return recorded, decoded(sound.heard)


@pytest.mark.outside
@pytest.mark.weights("streaming")
@pytest.mark.parametrize("fixture", echoscore.TAKES, ids=lambda fixture: fixture.name)
def test_hands_alone_cuts_nothing_on_any_recorded_take_heard_whole_as_one_clip(overheard: Overheard, fixture: Path) -> None:
    recorded, heard = desk_heard(fixture)
    overheard.held(heard)
    # A cut asked for before the person's first word is hands' own voice or the room cutting hands off.
    assert [cut for cut in overheard.barge_ins if cut[0] < recorded.cut_in] == [], (fixture.name, overheard.barge_ins, recorded.cut_in)


@pytest.mark.outside
@pytest.mark.weights("streaming")
@pytest.mark.parametrize("fixture", [fixture for fixture in echoscore.TAKES if echoscore.recorded(fixture).person], ids=lambda fixture: fixture.name)
def test_a_person_over_hands_cuts_it_off_within_two_seconds_of_each_line_they_say_or_has_it_heard_whole_as_their_clip_closes(overheard: Overheard, fixture: Path) -> None:
    recorded, heard = desk_heard(fixture)
    ends = [*[line.start - 0.3 for line in recorded.person[1:]], len(heard) / streaming.RATE]
    for line, end in zip(recorded.person, ends, strict=True):
        # Each line its own clip, from LEAD before its first word to three seconds past it, as a voice-opened turn is.
        first, last = round((line.start - LEAD) * streaming.RATE), round(min(end, line.start + 3.0) * streaming.RATE)
        ended = overheard.held(heard[first:last])
        said = streaming.joined(each.text for each in ended.words)
        match overheard.barge_ins:
            case [(asked, _), *_]:
                assert asked - LEAD <= CUT_SECONDS, (fixture.name, line, overheard.barge_ins)
            case _:
                # A short word the recognizer holds until the clip closes comes out as the clip closes, whole: the
                # transcriber hears it too, and the clip owner cuts on its words.
                assert said, (fixture.name, line)


# The harness that scores a chain (tests/echoscore.py), held to the real pair it hears through.
@pytest.mark.outside
@pytest.mark.weights("streaming")
@pytest.mark.parametrize("fixture", echoscore.TAKES, ids=lambda fixture: fixture.name)
def test_the_real_pair_hears_a_take_with_each_word_dated_by_its_own_token_inside_it(pair: Any, fixture: Path) -> None:
    take = echoscore.recorded(fixture)
    heard = echoscore.heard_words(pair, echoscore.raw(take))  # raises where a word is dated past the clip: a pair not begun afresh
    assert heard, f"{take.name}: the pair decoded nothing"
    # Dated by its own token and not all at the close, and heard on into where the person speaks, or past the middle of a take with no one.
    assert len({word.start for word in heard}) > len(heard) // 2
    assert heard[-1].start > (take.cut_in if take.person else len(take.reference) / echoscore.RATE / 2)


@pytest.mark.outside
@pytest.mark.weights("streaming")
def test_the_microphone_as_captured_is_the_control_that_leaks_hands_at_the_desk(pair: Any) -> None:
    take = echoscore.recorded(echoes.CANCELLED)
    assert echoscore.judged(take, echoscore.heard_words(pair, echoscore.raw(take)))[1]
