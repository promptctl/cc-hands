"""Streaming speech recognition with diarization: each hold's words heard while it is open, each with its speaker, a
hearing for each step the diarizer commits, one session for the run, and the real pair hearing words in every hold engaged
conversation would close, none in noise, and each word from the speaker who said it."""

import asyncio
import contextlib
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import soundfile

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice import engaged, streaming
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import Fate, Hold, Sound
from hands.voice.streaming import CHUNK, GATING, MEL_HOP, STEP, Said, Samples, Session, Spoken, StreamingRecognizer, steps, through
# The real load, bound before conftest stands a pair that hears nothing in for it.
from hands.voice.streaming import load as nemotron

FIXTURES = Path(__file__).parent / "fixtures"


def said(*texts: str | tuple[int, str]) -> Spoken:
    """What a feed decoded: each text speaker 0's, unless it names its speaker."""
    return tuple(Said(*text) if isinstance(text, tuple) else Said(0, text) for text in texts)


@dataclass
class Scripted:
    """The run's session as the tests script it: each time it is fed, what it decodes, oldest first, nothing once the
    script runs out; and every call it is to fail, counted from 0."""

    spoken: list[Spoken]
    failing: frozenset[int] = frozenset()
    fed: list[Samples] = field(default_factory=list[Samples])

    def decoded(self, samples: Samples) -> Spoken:
        self.fed.append(samples)
        if len(self.fed) - 1 in self.failing:
            raise RuntimeError("Metal ran out of memory")
        return self.spoken.pop(0) if self.spoken else ()


class Unused:
    """A transcriber no test here gives a hold to."""

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        raise AssertionError("nothing here transcribes")


def pcm(samples: Samples) -> bytes:
    return (samples * 32768).astype(np.int16).tobytes()


# The samples of a diarizer's step after the run's first, which a hearing follows each of.
STEPPED = STEP * MEL_HOP


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

    async def heard(self, hold: Hold, samples: Samples, frame: int = STEPPED) -> None:
        """`samples` of `hold`, given in frames of `frame` samples, and every hearing they bring run."""
        for start in range(0, len(samples), frame):
            piece = pcm(samples[start : start + frame])
            await self.run(hold, self.listener.heard(Sound(piece, piece)))

    async def ended(self, hold: Hold, fate: Fate) -> None:
        await self.run(hold, self.listener.ended(hold, fate))

    def told(self) -> list[tuple[object, object, object]]:
        """What each hearing came to: its outcome, what it heard, and the words on its event: those each speaker added
        while the hold was open, and each speaker's whole as it ended."""
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
    rig = rigged(Scripted([(), (), said("wait,"), said(" not"), ()]))
    await rig.opened(HELD, sound(0))
    # The opening is fed first, with nothing kept from before it. The first step needs 5216 samples, so the second
    # frame of the hold completes it; one hearing for each step after, saying the words it added, until one adds none.
    await rig.heard(HELD, sound(seconds(4 * STEPPED)))
    assert rig.told() == [("ok", "words", said("wait,")), ("ok", "words", said("not")), ("ok", "nothing", ())]


async def test_a_hearing_follows_the_feed_that_completes_a_step_however_the_hold_is_framed() -> None:
    rig = rigged(Scripted([*[()] * 17, said("stop")]))
    await rig.opened(HELD, sound(0))
    # 20 ms frames: the diarizer's first step waits for 5216 samples, so the 17th frame completes it, and only it is heard.
    await rig.heard(HELD, sound(0.32), frame=320)
    assert rig.told() == []
    await rig.heard(HELD, sound(0.02), frame=320)
    assert rig.told() == [("ok", "words", said("stop"))]
    # The next completes 3840 samples on, in the 29th.
    await rig.heard(HELD, sound(0.22), frame=320)
    assert len(rig.told()) == 1
    await rig.heard(HELD, sound(0.02), frame=320)
    assert len(rig.told()) == 2


async def test_each_speakers_words_are_kept_apart_as_they_are_heard_and_as_the_hold_ends() -> None:
    rig = rigged(Scripted([(), (), said("is it", (1, "yes")), said(" done")]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(3 * STEPPED)))
    await rig.ended(HELD, "sent")
    # What cuts is every speaker's words; which speaker said them is on the event.
    assert [(event.facts["heard"], event.facts.get("added", event.facts.get("words"))) for event in hearings(rig.recorded)] == [
        ("words", said("is it", (1, "yes"))), ("words", said("done")), ("words", said("is it done", (1, "yes"))),
    ]


async def test_the_full_stop_that_ends_the_last_hold_is_no_word_of_this_one() -> None:
    rig = rigged(Scripted([(), (), said("."), said(" he went")]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(3 * STEPPED)))
    assert rig.told() == [("ok", "nothing", ()), ("ok", "words", said("he went"))]


def test_the_silence_after_a_hold_runs_until_its_speakers_are_gated_off() -> None:
    # A hold ending at 9600 samples: its last sample is in mel frame 61, chunk 3. Its speaker's stream is stepped for
    # GATING chunks after it, so the diarizer must commit through chunk 3 + GATING, 128 frames, which its sixth step
    # does, at 24416 samples.
    assert (GATING, CHUNK, through(9600)) == (4, 16, 24_416)
    # The silence ends on the sample that completes that step, no later.
    assert steps(24_415) < steps(24_416)


async def test_each_hold_is_heard_from_what_it_kept_before_it_opened_to_the_silence_after_it() -> None:
    session = Scripted([(), said("turn it"), said("."), said(" and the"), (), ()])
    rig = rigged(session)
    first, second = Hold(1, "engaged conversation"), Hold(2, "held key")
    await rig.opened(first, sound(0.5, 0.25))
    await rig.heard(first, sound(0.2, 0.25))
    await rig.ended(first, "sent")
    await rig.opened(second, sound(0.1, 0.5))
    await rig.heard(second, sound(0.2, 0.5))
    await rig.ended(second, "dropped")
    # What the stream heard, in order: each hold, and the silence after it until the diarizer has committed GATING
    # chunks past it; the second hold starts where that silence ended.
    assert [(len(fed), round(float(fed[0]), 2)) for fed in session.fed] == [
        (8_000, 0.25), (3_200, 0.25), (13_216, 0.0), (1_600, 0.5), (3_200, 0.5), (14_400, 0.0)
    ]
    # The first hold's full stop is its own, decoded in the silence after it, and only its end says it.
    assert rig.told() == [
        ("ok", "nothing", ()), ("ok", "words", said("turn it")), ("ok", "words", said("turn it.")),
        ("ok", "words", said("and the")), ("ok", "words", said("and the")),
    ]


async def test_a_hold_that_ends_is_heard_through_the_silence_after_it_and_told_whole_with_its_fate() -> None:
    rig = rigged(Scripted([(), said("the parser"), said(" is fixed")]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(2 * STEPPED)))
    await rig.ended(HELD, "sent")
    *_, whole = hearings(rig.recorded)
    # The words after it closed cut nothing: the hold is no longer open.
    assert (whole.facts["fate"], whole.facts["words"], whole.facts["seconds"], whole.facts["cut"]) == ("sent", said("the parser is fixed"), 0.48, False)


async def test_a_hold_nothing_was_said_in_ends_with_no_words() -> None:
    rig = rigged(Scripted([]))
    await rig.opened(HELD, sound(0))
    await rig.heard(HELD, sound(seconds(2 * STEPPED)))
    await rig.ended(HELD, "dropped")
    assert rig.told() == [("ok", "nothing", ()), ("ok", "nothing", ())]


async def test_decoding_that_fails_breaks_the_runs_stream_and_each_hold_from_it_on_fails_once_as_it_ends() -> None:
    # The models' caches are left in a state nothing vouches for, and the grid hearings are timed on with them.
    session = Scripted([(), (), (), said("stop")], failing=frozenset({2}))
    rig = rigged(session)
    for number in (1, 2):
        await rig.opened(Hold(number, "engaged conversation"), sound(0))
        await rig.heard(Hold(number, "engaged conversation"), sound(seconds(2 * STEPPED)))
        await rig.ended(Hold(number, "engaged conversation"), "sent")
    failed = ("failed", "RuntimeError: decoding failed: RuntimeError: Metal ran out of memory")
    # The second hold, further along the run's grid, completes two steps.
    assert [(event.outcome, event.error) for event in hearings(rig.recorded)] == [("ok", None), failed, ("ok", None), ("ok", None), failed]
    # Nothing is fed to the models once they broke.
    assert len(session.fed) == 3


def loaded(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "streaming.loaded"]


async def test_holds_spoken_while_the_models_load_are_heard_once_they_have() -> None:
    recorded: list[Entry] = []
    go = threading.Event()

    def slow() -> Session:
        go.wait()
        return Scripted([(), said("hello")])

    listener = StreamingRecognizer(slow, recorded.append)
    rig = Rig(listener, HoldOwner(Unused(), (listener,), recorded.append), recorded)
    await rig.opened(HELD, sound(0))
    hearing = asyncio.create_task(rig.heard(HELD, sound(seconds(2 * STEPPED))))
    await asyncio.sleep(0.1)
    assert (loaded(recorded), rig.told()) == ([], [])
    go.set()
    await hearing
    assert rig.told() == [("ok", "words", said("hello"))]
    [load] = loaded(recorded)
    assert (load.outcome, {name: load.facts[name] for name in ("model", "revision", "diarization", "diarization_revision", "preset", "gating", "transcription_delay_ms")}) == (
        "ok",
        {
            "model": streaming.MODEL, "revision": streaming.REVISION, "diarization": streaming.DIARIZATION,
            "diarization_revision": streaming.DIARIZATION_REVISION, "preset": "ultra_low", "gating": 4, "transcription_delay_ms": 160,
        },
    )


async def test_models_that_fail_to_load_fail_each_hold_once_saying_why() -> None:
    recorded: list[Entry] = []

    def unreachable() -> Session:
        raise OSError("the hub is unreachable")

    listener = StreamingRecognizer(unreachable, recorded.append)
    rig = Rig(listener, HoldOwner(Unused(), (listener,), recorded.append), recorded)
    for number in (1, 2):
        await rig.opened(Hold(number, "engaged conversation"), sound(0))
        await rig.heard(Hold(number, "engaged conversation"), sound(seconds(2 * STEPPED)))
        await rig.ended(Hold(number, "engaged conversation"), "sent")
    failed = ("failed", "RuntimeError: the models did not load: OSError: the hub is unreachable")
    assert [(event.outcome, event.error) for event in hearings(recorded)] == [("ok", None), failed, ("ok", None), ("ok", None), failed]
    [load] = loaded(recorded)
    assert (load.outcome, load.error) == ("failed", "OSError: the hub is unreachable")


def test_a_speaker_is_the_number_the_session_gave_them() -> None:
    assert streaming.speaker("speaker_3") == 3
    with pytest.raises(ValueError, match="'hands'"):
        streaming.speaker("hands")


# The real pair, on real sound: the weights are fetched into the Hugging Face cache the first time, and CI keeps that
# cache. It runs on MLX's default device, the CPU where there is no Metal, as on CI's VMs. Each test is one session, as a
# run is. The fixtures are holds separated by half a second of exact zeros, which speech never holds for that long.
GAP = streaming.RATE // 2


def fixture_holds(name: str) -> list[Samples]:
    audio, rate = cast(tuple[Samples, int], soundfile.read(FIXTURES / name, dtype="float32"))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
    assert rate == streaming.RATE
    silent = np.convolve(audio == 0, np.ones(GAP), mode="valid") == GAP
    gaps = np.flatnonzero(np.diff(np.concatenate([[False], silent, [False]]).astype(np.int8)))
    edges = [0, *(edge + offset for edge, offset in zip(gaps, [0, GAP - 1] * (len(gaps) // 2))), len(audio)]
    return [audio[start:end] for start, end in zip(edges[0::2], edges[1::2])]


@dataclass
class Live:
    """One stream of the real pair, fed holds as the listener feeds it: 20 ms at a time, each followed by the silence
    `through` says."""

    stream: streaming.Stream = field(default_factory=lambda: streaming.Stream(nemotron()))
    fed: int = 0

    def spoken(self, samples: Samples) -> tuple[Spoken, Spoken]:
        """A hold of `samples`: each speaker's words heard while it was open, and once it ended."""
        self.stream.opened(np.zeros(0, np.float32))
        for start in range(0, len(samples), 320):
            self.stream.heard(samples[start : start + 320])
        self.fed += len(samples)
        live = self.stream.said()
        silence = through(self.fed) - self.fed
        self.stream.silence(silence)
        self.fed += silence
        return live, self.stream.ended().words


def test_every_hold_engaged_conversation_would_close_has_words_heard_while_it_was_open() -> None:
    # Utterances of 0.8 to 2 s by one LibriVox reader (The Invisible Man, read solo; public domain), 20 to 30 dB down, each
    # followed by the silence engaged conversation's detector waits out before it closes a hold: the only holds whose cut
    # waits on words. One reader is one speaker, the first the run heard.
    silence = np.zeros(round(engaged.STOP_SECS * streaming.RATE), np.float32)
    live = Live()
    heard = [live.spoken(np.concatenate([samples, silence])) for samples in fixture_holds("one_reader_quiet.flac")[:8]]
    assert all(open_ for open_, _ in heard), heard
    assert {said.speaker for open_, whole in heard for said in (*open_, *whole)} == {0}, heard
    # What it heard open is the start of what it heard whole: the model never takes back a word it has given.
    assert all(whole[0].text.startswith(open_[0].text) for open_, whole in heard), heard
    assert heard[0][1][0].text.rstrip(".") == "Door was ajar"


def test_a_hold_with_no_speech_in_it_has_no_words_so_it_cuts_nothing() -> None:
    # What a hold engaged conversation opens can hold with nobody speaking: the room, a hiss, a tap on the desk.
    noise = np.random.default_rng(0).standard_normal(2 * streaming.RATE).astype(np.float32)
    clicks = np.where(np.arange(2 * streaming.RATE) % 4_000 == 0, 0.8, 0.0).astype(np.float32)
    holds = [np.zeros(2 * streaming.RATE, np.float32), noise * 0.01, noise * 0.056, clicks]
    live = Live()
    assert [live.spoken(samples) for samples in holds] == [((), ())] * 4


@dataclass
class Run:
    """One session of the real pair as mlx-audio gives it, fed holds as the listener feeds it: 20 ms at a time, each
    followed by the silence `through` says. Each token is kept with the hold it came out while feeding, and each feed it
    came out on that completed no step of the diarizer's is counted."""

    session: Any = field(default_factory=lambda: cast(Any, nemotron()).session)
    fed: int = 0
    holds: list[int] = field(default_factory=list[int])
    tokens: list[tuple[int, int, Any]] = field(default_factory=list[tuple[int, int, Any]])
    off_grid: int = 0

    def feed(self, samples: Samples) -> None:
        decoded = self.session.feed(samples)
        self.off_grid += bool(decoded) and steps(self.fed + len(samples)) == steps(self.fed)
        self.tokens += [(len(self.holds) - 1, streaming.speaker(delta.speaker), token) for delta in decoded for token in delta.tokens]
        self.fed += len(samples)

    def spoken(self, samples: Samples) -> None:
        self.holds.append(self.fed)
        for start in range(0, len(samples), 320):
            self.feed(samples[start : start + 320])
        self.feed(np.zeros(through(self.fed) - self.fed, np.float32))

    def speakers(self) -> list[set[int]]:
        """The speakers of each hold's words: whose tokens came out while it and the silence after it were fed."""
        return [{speaker for fed_in, speaker, token in self.tokens if fed_in == hold and any(map(str.isalnum, token.text))} for hold in range(len(self.holds))]

    def crossed(self) -> list[tuple[int, str]]:
        """Each token that came out while one hold was fed, of audio from before that hold began: a word of the one before."""
        return [(hold, token.text) for hold, _, token in self.tokens if token.start * streaming.RATE < self.holds[hold]]


def test_each_word_is_the_speaker_who_said_it_and_stays_in_the_hold_it_was_said_in() -> None:
    # Holds of two women, each speaking alone, in the order they spoke: AMI Meeting Corpus ES2004a, mixed headsets,
    # speakers FEE013 and FEE016 (CC BY 4.0, University of Edinburgh). Each hold's speaker is the one the transformers
    # path diarizing on the CPU gave it, which every word's speaker matched over all 16 holds, the run started at five
    # phases of the grid (promptctl-voice-i15t.qg8.5yl). The first six, so it stays inside the test timeout on a CPU: a
    # run's start is fed as the whole run's is, so it decides as the whole did. A later stretch alone does not: with less
    # of each voice heard, both paths take both speakers to be speaking in hold 15, and both speakers' streams say it.
    run = Run()
    for samples in fixture_holds("two_speakers_ami.flac")[:6]:
        run.spoken(samples)
    assert run.speakers() == [{int(who)} for who in "000100"]
    assert run.crossed() == []
    # [LAW:one-source-of-truth] `steps` is what times a hearing: no word comes out on a feed it says no step completed.
    assert run.off_grid == 0


def test_a_hold_whose_silence_ends_exactly_where_its_speaker_is_gated_off_keeps_its_last_word() -> None:
    # The worst phase: the hold ends where the step that commits GATING chunks past its last sample has none to spare, so
    # its speaker's stream is stepped for the last time in the last chunk of silence. Then another speaker's hold.
    first, second = fixture_holds("two_speakers_ami.flac")[3:5]

    def committed(samples: int) -> int:
        return ((samples - 1 + streaming.N_FFT // 2) // MEL_HOP // CHUNK + 1 + GATING) * CHUNK

    # The phases repeat every frame count both a chunk and a step divide.
    cut = max(length for length in range(len(first) - math.lcm(CHUNK, STEP) * MEL_HOP, len(first)) if committed(length) % STEP == 0)
    assert STEP * steps(through(cut)) == committed(cut)
    run = Run()
    run.spoken(first[:cut])
    run.spoken(second)
    # The run's first speaker is 0, so the first hold's is 0 here and the second's 1.
    assert (run.speakers(), run.crossed()) == ([{0}, {1}], [])


def test_the_diarizer_commits_on_exactly_the_feeds_steps_names() -> None:
    # [LAW:one-source-of-truth] the diarizer's own count of the frames it committed, against `steps`, fed 20 ms at a time.
    model = streaming.diarizer()
    state = model.init_streaming_state()
    audio = np.concatenate(fixture_holds("one_reader_quiet.flac")[:4])
    moved: list[tuple[bool, bool]] = []
    for start in range(0, len(audio), 320):
        committed = state.frames_processed
        _, state = model.feed(audio[start : start + 320], state)
        moved.append((steps(start + len(audio[start : start + 320])) > steps(start), state.frames_processed > committed))
    assert all(timed == stepped for timed, stepped in moved), [at for at, (timed, stepped) in enumerate(moved) if timed != stepped]
    assert sum(stepped for _, stepped in moved) > 5
