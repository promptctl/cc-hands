"""Streaming speech recognition: each hold's words heard while it is open, a hearing for each chunk the model decodes,
one stream for the run, and the real model hearing words in every hold engaged conversation would close, and none in
noise."""

import asyncio
import contextlib
import threading
from dataclasses import dataclass, field

import numpy as np
import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice import engaged, streaming
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import Fate, Hold, Sound
from hands.voice.streaming import CHUNK, MEL_HOP, TRAILING, Samples, Session, StreamingRecognizer
# The real load, bound before conftest stands a model that hears nothing in for it.
from hands.voice.streaming import load as nemotron
from test_diarization import Unused, fixture_holds, pcm


@dataclass
class Scripted:
    """The run's session as the tests script it: each time it is fed, the text it decodes, oldest first, none once the
    script runs out; and every call it is to fail, counted from 0."""

    texts: list[str]
    failing: frozenset[int] = frozenset()
    fed: list[Samples] = field(default_factory=list[Samples])

    def decoded(self, samples: Samples) -> str:
        self.fed.append(samples)
        if len(self.fed) - 1 in self.failing:
            raise RuntimeError("Metal ran out of memory")
        return self.texts.pop(0) if self.texts else ""


# The samples of a chunk after the run's first, which a hearing follows each of.
CHUNKED = CHUNK * MEL_HOP


def hearings(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hearing" and entry.facts["listener"] == "streaming"]


@dataclass
class Rig:
    """The listener, the hold owner running its hearings as a pipeline's would, and what they recorded."""

    listener: StreamingRecognizer
    owned: HoldOwner
    recorded: list[Entry]

    async def run(self, hold: Hold, told: tuple[object, ...]) -> None:
        for hearing in told:
            # As the owner's hearing lane runs each: one that fails is said on its event, and the lane hears on.
            with contextlib.suppress(Exception):
                await self.owned._heard(self.listener, hold, hearing, begun())  # pyright: ignore[reportPrivateUsage, reportArgumentType]

    async def opened(self, hold: Hold, before: Samples) -> None:
        await self.run(hold, self.listener.opened(hold, Sound(pcm(before), pcm(before))))

    async def heard(self, hold: Hold, samples: Samples, frame: int = CHUNKED) -> None:
        """`samples` of `hold`, given in frames of `frame` samples, and every hearing they bring run."""
        for start in range(0, len(samples), frame):
            piece = pcm(samples[start : start + frame])
            await self.run(hold, self.listener.heard(Sound(piece, piece)))

    async def ended(self, hold: Hold, fate: Fate) -> None:
        await self.run(hold, self.listener.ended(hold, fate))

    def told(self) -> list[tuple[object, object, object]]:
        """What each hearing came to: its outcome, what it heard, and the words on its event: those it added while the
        hold was open, and the hold's whole as it ended."""
        return [(event.outcome, event.facts["heard"], event.facts.get("added", event.facts.get("words"))) for event in hearings(self.recorded)]


def rigged(session: Session) -> Rig:
    recorded: list[Entry] = []
    listener = StreamingRecognizer(lambda: session, recorded.append)
    return Rig(listener, HoldOwner(Unused(), (listener,), recorded.append), recorded)


def sound(seconds: float, level: float = 0.1) -> Samples:
    return np.full(round(seconds * streaming.RATE), level, np.float32)


HELD = Hold(1, "engaged conversation")


def seconds(samples: int) -> float:
    return samples / streaming.RATE


async def test_a_holds_words_are_heard_while_it_is_open_each_hearing_the_words_it_added() -> None:
    rig = rigged(Scripted(["", "wait,", " not", ""]))
    await rig.opened(HELD, sound(0))
    # One hearing for each chunk the run completes, saying the words it added, until one adds none.
    await rig.heard(HELD, sound(seconds(3 * CHUNKED)))
    assert rig.told() == [("ok", "words", "wait,"), ("ok", "words", "not"), ("ok", "nothing", None)]


async def test_a_hearing_follows_the_feed_that_completes_a_chunk_however_the_hold_is_framed() -> None:
    rig = rigged(Scripted(["", "stop"]))
    await rig.opened(HELD, sound(0))
    # 20 ms frames: the run's first chunk needs 1536 samples, so the fifth completes it, and only it is heard.
    await rig.heard(HELD, sound(0.08), frame=320)
    assert rig.told() == []
    await rig.heard(HELD, sound(0.02), frame=320)
    assert rig.told() == [("ok", "words", "stop")]
    # The next completes 2560 samples on, in the thirteenth.
    await rig.heard(HELD, sound(0.14), frame=320)
    assert len(rig.told()) == 1
    await rig.heard(HELD, sound(0.02), frame=320)
    assert len(rig.told()) == 2


async def test_the_full_stop_that_ends_the_last_hold_is_no_word_of_this_one() -> None:
    rig = rigged(Scripted(["", ".", " he went"]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(2 * CHUNKED)))
    assert rig.told() == [("ok", "nothing", None), ("ok", "words", "he went")]


async def test_each_hold_is_heard_from_what_it_kept_before_it_opened_to_the_silence_after_it() -> None:
    session = Scripted(["turn it", ".", "", " and the", "", ""])
    rig = rigged(session)
    first, second = Hold(1, "engaged conversation"), Hold(2, "held key")
    await rig.opened(first, sound(0.5, 0.25))
    await rig.heard(first, sound(0.1, 0.25))
    await rig.ended(first, "sent")
    await rig.opened(second, sound(0.1, 0.5))
    await rig.heard(second, sound(0.2, 0.5))
    await rig.ended(second, "dropped")
    # What the stream heard, in order: each hold, and the silence after it.
    assert [(len(fed), round(float(fed[0]), 2)) for fed in session.fed] == [
        (8_000, 0.25), (1_600, 0.25), (TRAILING, 0.0), (1_600, 0.5), (CHUNKED, 0.5), (640, 0.5), (TRAILING, 0.0)
    ]
    # The first hold's full stop is its own, decoded in it, and only its end says it.
    assert rig.told() == [
        ("ok", "words", "turn it"), ("ok", "nothing", None), ("ok", "words", "turn it."),
        ("ok", "words", "and the"), ("ok", "nothing", None), ("ok", "words", "and the"),
    ]


async def test_a_hold_that_ends_is_heard_through_the_silence_after_it_and_told_whole_with_its_fate() -> None:
    rig = rigged(Scripted(["", "the parser", " is fixed"]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(CHUNKED)))
    await rig.ended(HELD, "sent")
    *_, whole = hearings(rig.recorded)
    # The words after it closed cut nothing: the hold is no longer open.
    assert (whole.facts["fate"], whole.facts["words"], whole.facts["seconds"], whole.facts["cut"]) == ("sent", "the parser is fixed", 0.16, False)


async def test_a_hold_nothing_was_said_in_ends_with_no_words() -> None:
    rig = rigged(Scripted([]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(CHUNKED)))
    await rig.ended(HELD, "dropped")
    assert rig.told() == [("ok", "nothing", None), ("ok", "nothing", None)]


async def test_decoding_that_fails_breaks_the_runs_stream_and_each_hold_from_it_on_fails_once_as_it_ends() -> None:
    # The model's caches are left in a state nothing vouches for, and the grid hearings are timed on with them.
    session = Scripted(["", "", "", "stop"], failing=frozenset({1}))
    rig = rigged(session)
    for number in (1, 2):
        await rig.opened(Hold(number, "engaged conversation"), sound(0))
        await rig.heard(Hold(number, "engaged conversation"), sound(seconds(CHUNKED)))
        await rig.ended(Hold(number, "engaged conversation"), "sent")
    failed = ("failed", "RuntimeError: decoding failed: RuntimeError: Metal ran out of memory")
    assert [(event.outcome, event.error) for event in hearings(rig.recorded)] == [("ok", None), failed] * 2
    # Nothing is fed to the model once it broke.
    assert len(session.fed) == 2


def loaded(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "streaming.loaded"]


async def test_holds_spoken_while_the_model_loads_are_heard_once_it_has() -> None:
    recorded: list[Entry] = []
    go = threading.Event()

    def slow() -> Session:
        go.wait()
        return Scripted(["", "hello"])

    listener = StreamingRecognizer(slow, recorded.append)
    rig = Rig(listener, HoldOwner(Unused(), (listener,), recorded.append), recorded)
    await rig.opened(HELD, sound(0))
    hearing = asyncio.create_task(rig.heard(HELD, sound(seconds(CHUNKED))))
    await asyncio.sleep(0.1)
    assert (loaded(recorded), rig.told()) == ([], [])
    go.set()
    await hearing
    assert rig.told() == [("ok", "words", "hello")]
    [load] = loaded(recorded)
    assert (load.outcome, load.facts["model"], load.facts["revision"], load.facts["lookahead_ms"]) == ("ok", streaming.MODEL, streaming.REVISION, 160)


async def test_a_model_that_fails_to_load_fails_each_hold_once_saying_why() -> None:
    recorded: list[Entry] = []

    def unreachable() -> Session:
        raise OSError("the hub is unreachable")

    listener = StreamingRecognizer(unreachable, recorded.append)
    rig = Rig(listener, HoldOwner(Unused(), (listener,), recorded.append), recorded)
    for number in (1, 2):
        await rig.opened(Hold(number, "engaged conversation"), sound(0))
        await rig.heard(Hold(number, "engaged conversation"), sound(seconds(CHUNKED)))
        await rig.ended(Hold(number, "engaged conversation"), "sent")
    failed = ("failed", "RuntimeError: the model did not load: OSError: the hub is unreachable")
    assert [(event.outcome, event.error) for event in hearings(recorded)] == [("ok", None), failed] * 2
    [load] = loaded(recorded)
    assert (load.outcome, load.error) == ("failed", "OSError: the hub is unreachable")


# The real model, on real sound: the weights are fetched into the Hugging Face cache the first time, and CI keeps that
# cache. It runs on MLX's default device, the CPU where there is no Metal, as on CI's VMs, so it hears the first eight
# holds of the fixture, which there take about as long to hear as to say. Each test is one stream, as a run is.
@pytest.fixture
def stream() -> streaming.Stream:
    return streaming.Stream(nemotron())


def spoken(stream: streaming.Stream, samples: Samples) -> tuple[str | None, str | None]:
    """A hold of `samples` fed 20 ms at a time, as the microphone gives one: its words heard while it was open, and once
    it ended."""
    stream.opened(np.zeros(0, np.float32))
    for start in range(0, len(samples), 320):
        stream.heard(samples[start : start + 320])
    return stream.said(), stream.ended().words


def test_every_hold_engaged_conversation_would_close_has_words_heard_while_it_was_open(stream: streaming.Stream) -> None:
    # Utterances of 0.8 to 2 s by one LibriVox reader (The Invisible Man, read solo; public domain), 20 to 30 dB down, each
    # followed by the silence engaged conversation's detector waits out before it closes a hold: the only holds whose cut
    # waits on words.
    silence = np.zeros(round(engaged.STOP_SECS * streaming.RATE), np.float32)
    heard = [spoken(stream, np.concatenate([samples, silence])) for samples in fixture_holds("one_reader_quiet.flac")[:8]]
    assert all(live is not None for live, _ in heard), heard
    # What it heard open is the start of what it heard whole: the model never takes back a word it has given.
    assert all(whole is not None and live is not None and whole.startswith(live) for live, whole in heard), heard
    assert heard[0][1] is not None and heard[0][1].rstrip(".") == "Door was ajar"


def test_a_hold_with_no_speech_in_it_has_no_words_so_it_cuts_nothing(stream: streaming.Stream) -> None:
    # What a hold engaged conversation opens can hold with nobody speaking: the room, a hiss, a tap on the desk.
    noise = np.random.default_rng(0).standard_normal(2 * streaming.RATE).astype(np.float32)
    clicks = np.where(np.arange(2 * streaming.RATE) % 4_000 == 0, 0.8, 0.0).astype(np.float32)
    holds = [np.zeros(2 * streaming.RATE, np.float32), noise * 0.01, noise * 0.056, clicks]
    assert [spoken(stream, samples) for samples in holds] == [(None, None)] * 4


def test_the_model_decodes_on_the_grid_its_hearings_are_timed_on() -> None:
    # [LAW:one-source-of-truth] `chunks` is what times a hearing; the model's own encoder says when a chunk is decoded.
    session = nemotron()
    assert isinstance(session, streaming._Decoder)  # pyright: ignore[reportPrivateUsage]  (the real load's own session)
    audio = np.concatenate([*fixture_holds("one_reader_quiet.flac")[:2]])
    moved: list[tuple[bool, bool]] = []
    fed = 0
    for start in range(0, len(audio), 320):
        frame = session.frame
        session.decoded(audio[start : start + 320])
        moved.append((streaming.chunks(fed + len(audio[start : start + 320])) > streaming.chunks(fed), session.frame > frame))
        fed += len(audio[start : start + 320])
    assert all(timed == decoded for timed, decoded in moved), [at for at, (timed, decoded) in enumerate(moved) if timed != decoded]
    assert sum(decoded for _, decoded in moved) > 10
