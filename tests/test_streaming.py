"""Streaming speech recognition with diarization: each hold's words heard while it is open, each with its speaker, a
hearing for each step the diarizer commits, one session for the run, and the real pair hearing words in every hold engaged
conversation would close, none in noise, and each word from the speaker who said it."""

import asyncio
import contextlib
import dataclasses
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import soundfile
from loguru import logger

from conftest import PAIRS, model_facts
from listening import LEAD, SHAPES, Live, turn
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice import engaged, streaming
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import Fate, Hold, Sound
from hands.voice.streaming import CHUNK, MEL_HOP, STEP, Closed, Said, Samples, Session, Spoken, StreamingRecognizer, steps, through
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

    def decoded(self, samples: Samples) -> Spoken:
        self.fed.append(samples)
        if len(self.fed) - 1 in self.failing:
            raise RuntimeError("Metal ran out of memory")
        return self.spoken.pop(0) if self.spoken else ()

    def closed(self) -> Closed:
        return self.closing.pop(0) if self.closing else NOTHING_CLOSED


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
    listener = StreamingRecognizer(lambda _: session, recorded.append)
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


def test_the_silence_after_a_hold_runs_until_the_chunk_holding_its_last_sample_is_committed() -> None:
    # A hold ending at 9600 samples: its last sample is in mel frame 61, chunk 3, so the diarizer must commit through
    # 64 frames, which its third step does, at 12896 samples.
    assert (CHUNK, through(9600)) == (16, 12_896)
    # The silence ends on the sample that completes that step, no later.
    assert steps(12_895) < steps(12_896)


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
    # What the stream heard, in order: each hold, and the silence after it until the diarizer has committed the chunk
    # holding its last sample; the second hold starts where that silence ended.
    assert [(len(fed), round(float(fed[0]), 2)) for fed in session.fed] == [
        (8_000, 0.25), (3_200, 0.25), (5_536, 0.0), (1_600, 0.5), (3_200, 0.5), (2_880, 0.0)
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
    # The hold's last sample is in the run's fourth chunk, so the silence runs to the step that commits it: the run fed
    # through 12896 samples.
    assert whole.facts["silence"] == 0.326


async def test_a_holds_close_ends_it_with_its_last_words_and_none_of_them_come_out_in_the_next() -> None:
    # The close decodes each stream's last token, which feeding alone leaves pending: speaker 0's " done", and a word
    # of speaker 1, who said nothing else in the hold.
    rig = rigged(Scripted([(), (), said("is it"), (), (), (), said("stop")], closing=[Closed(frozenset({0, 1}), said(" done", (1, "yes")))]))
    for hold in (Hold(1, "engaged conversation"), Hold(2, "engaged conversation")):
        await rig.opened(hold, sound(0))
        await rig.heard(hold, sound(seconds(2 * STEPPED)))
        await rig.ended(hold, "sent")
    assert rig.told() == [
        ("ok", "words", said("is it")), ("ok", "words", said("is it done", (1, "yes"))),
        ("ok", "nothing", ()), ("ok", "words", said("stop")), ("ok", "words", said("stop")),
    ]
    # [LAW:nothing-unseen] each hold's end says whose streams it closed and what the close, not the feed, decoded.
    assert [(event.facts["closed"], event.facts["closing"]) for event in hearings(rig.recorded) if "closed" in event.facts] == [
        (frozenset({0, 1}), said(" done", (1, "yes"))), (frozenset(), ()),
    ]


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
    logged: list[Any] = []
    sink = logger.add(lambda message: logged.append(message.record), level="ERROR")
    try:
        for number in (1, 2):
            await rig.opened(Hold(number, "engaged conversation"), sound(0))
            await rig.heard(Hold(number, "engaged conversation"), sound(seconds(2 * STEPPED)))
            await rig.ended(Hold(number, "engaged conversation"), "sent")
    finally:
        logger.remove(sink)
    # Where in the pair it broke is said once, with what was raised there.
    [broke] = logged
    assert broke["exception"].value.args == ("Metal ran out of memory",)
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

    def slow(_gating: int) -> Session:
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
    assert (load.outcome, model_facts(load)) == (
        "ok",
        {
            "model": streaming.MODEL, "revision": streaming.REVISION, "diarization": streaming.DIARIZATION,
            "diarization_revision": streaming.DIARIZATION_REVISION, "preset": "ultra_low", "gating": streaming.GATING, "transcription_delay_ms": 160,
        },
    )


async def test_models_that_fail_to_load_fail_each_hold_once_saying_why() -> None:
    recorded: list[Entry] = []

    def unreachable(_gating: int) -> Session:
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


def test_a_recognizer_that_frames_the_run_with_another_fft_is_refused_before_a_session_is_made(monkeypatch: pytest.MonkeyPatch) -> None:
    # `through` places a hold's last mel frame by the recognizer's FFT, which the session does not check against the
    # diarizer's.
    import mlx_audio.stt  # pyright: ignore[reportMissingTypeStubs]

    def recognizer(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(preprocessor_config=SimpleNamespace(n_fft=400))

    monkeypatch.setattr(mlx_audio.stt, "load", recognizer)
    with pytest.raises(ValueError, match="FFT of 400; hands places a hold's last mel frame on 512"):
        nemotron(streaming.GATING)


def test_a_speaker_is_the_number_the_session_gave_them() -> None:
    assert streaming.speaker("speaker_3") == 3
    with pytest.raises(ValueError, match="'hands'"):
        streaming.speaker("hands")


# The real pair, on real sound: the weights are fetched into the Hugging Face cache the first time, and CI keeps that
# cache. It runs on MLX's default device, and is the only pair loaded while it is held (conftest's `Pairs`). Each test
# starts the session afresh, as a run starts one. The fixtures are holds separated by half a second of exact zeros, which
# speech never holds for that long.
GAP = streaming.RATE // 2


@pytest.fixture(scope="module")
def pair() -> Any:
    """The real pair, loaded once, as a run loads it, and the only one the suite holds while this module runs."""
    return PAIRS.loaded(lambda: nemotron(streaming.GATING))


@pytest.fixture
def paired(pair: Any) -> Any:
    """The pair's session as a run begins it: nothing heard, nobody numbered."""
    pair.session.reset()
    return pair


def fixture_holds(name: str) -> list[Samples]:
    audio, rate = cast(tuple[Samples, int], soundfile.read(FIXTURES / name, dtype="float32"))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
    assert rate == streaming.RATE
    silent = np.convolve(audio == 0, np.ones(GAP), mode="valid") == GAP
    gaps = np.flatnonzero(np.diff(np.concatenate([[False], silent, [False]]).astype(np.int8)))
    edges = [0, *(edge + offset for edge, offset in zip(gaps, [0, GAP - 1] * (len(gaps) // 2))), len(audio)]
    return [audio[start:end] for start, end in zip(edges[0::2], edges[1::2])]


def spoken_words(spoken: Spoken) -> list[tuple[int, str]]:
    """Each word each speaker said, in lower case, with no punctuation."""
    return [(said.speaker, word) for said in spoken for word in re.findall(r"[\w']+", said.text.lower())]


# Holds of one LibriVox reader (The Invisible Man, read solo; public domain), 20 to 30 dB down: utterances of 0.8 to 2 s,
# cut from the reading wherever it paused, so a hold can begin or end inside a phrase. One reader is one speaker, the
# first the run heard.
READER = "one_reader_quiet.flac"
# What engaged conversation's detector waits out before it closes a hold: the only holds whose cut waits on words.
STOP = np.zeros(round(engaged.STOP_SECS * streaming.RATE), np.float32)


def test_every_hold_engaged_conversation_would_close_has_words_heard_while_it_was_open(paired: Session) -> None:
    live = Live(paired)
    heard = [live.spoken(np.concatenate([samples, STOP])) for samples in fixture_holds(READER)[:8]]
    assert all(open_ for open_, _ in heard), heard
    assert {said.speaker for open_, whole in heard for said in (*open_, *whole.words)} == {0}, heard
    # What it heard open is the start of what it heard whole: the model never takes back a word it has given.
    assert all(whole.words[0].text.startswith(open_[0].text) for open_, whole in heard), heard
    assert heard[0][1].words[0].text.rstrip(".") == "Door was ajar"


def test_each_hold_opens_on_its_own_first_word_not_the_last_of_the_hold_before(paired: Session) -> None:
    # Holds 3 to 10 of the reading, each followed by the stop, as engaged conversation closes them. Their source text:
    # "...and found the bottle as he had been directed. But returning with the bottle, he noticed that the bolts of the
    # front door had been shot back, that the door was in fact simply on the latch. And with a flash of inspiration he
    # connected this with the stranger's room upstairs..." Hold 5 is heard and not asserted: its cut falls inside "he
    # noticed that", so the text does not say which word opens it.
    live = Live(paired)
    first = {number: spoken_words(live.spoken(np.concatenate([samples, STOP]))[1].words)[0] for number, samples in enumerate(fixture_holds(READER)[2:10], 3)}
    del first[5]
    assert first == {3: (0, "and"), 4: (0, "but"), 6: (0, "that"), 7: (0, "simply"), 8: (0, "and"), 9: (0, "he"), 10: (0, "the")}


# A woman coughing three times ("Woman coughing three times.wav", Ashe (OwlStorm on Freesound), CC0, from Wikimedia
# Commons), and 3 s of a computer keyboard typed on ("Computer keyboard.ogg", russiandoll, public domain, from Wikimedia
# Commons, from 1.5 s in), each made mono at 16 kHz.
NOT_SPEECH = [*fixture_holds("cough.flac"), *fixture_holds("keyboard_typing.flac")]


def test_a_hold_with_no_speech_in_it_has_no_words_so_it_cuts_nothing(paired: Session) -> None:
    # What a hold engaged conversation opens can hold with nobody speaking: the room, a hiss, a tap on the desk, a cough,
    # the keyboard.
    noise = np.random.default_rng(0).standard_normal(2 * streaming.RATE).astype(np.float32)
    clicks = np.where(np.arange(2 * streaming.RATE) % 4_000 == 0, 0.8, 0.0).astype(np.float32)
    holds = [np.zeros(2 * streaming.RATE, np.float32), noise * 0.01, noise * 0.056, clicks, *NOT_SPEECH]
    live = Live(paired)
    assert [(open_, whole.words) for open_, whole in map(live.spoken, holds)] == [((), ())] * len(holds)


def test_each_word_is_the_speaker_who_said_it_and_each_hold_closes_its_own_speakers_streams(paired: Session) -> None:
    # Holds of two women, each speaking alone, in the order they spoke: AMI Meeting Corpus ES2004a, mixed headsets,
    # speakers FEE013 and FEE016 (CC BY 4.0, University of Edinburgh). Each hold's speaker is the one the transformers
    # path diarizing on the CPU gave it, which every word's speaker matched over all 16 holds, the run started at five
    # phases of the grid (promptctl-voice-i15t.qg8.5yl). The first six, so it stays inside the test timeout on a CPU: a
    # run's start is fed as the whole run's is, so it decides as the whole did. A later stretch alone does not: with less
    # of each voice heard, both paths take both speakers to be speaking in hold 15, and both speakers' streams say it.
    live = Live(paired)
    ended = [whole for _, whole in map(live.spoken, fixture_holds("two_speakers_ami.flac")[:6])]
    assert [{said.speaker for said in hold.words} for hold in ended] == [{int(who)} for who in "000100"]
    # A hold where one person speaks closes their stream alone: one made from gate history carried across the close
    # before it would show here. In holds 4 and 5 the diarizer also marks, for a moment, a speaker who decodes no word.
    assert [hold.closed.speakers for hold in ended] == [{0}, {0}, {0}, {0, 1}, {0, 2}, {0}]
    # [LAW:one-source-of-truth] `steps` is what times a hearing: no word comes out on a feed it says no step completed.
    assert live.session.off_grid == 0


@dataclass
class Timed:
    """The pair's mlx-audio session, keeping each token it decodes as it is fed, with its speaker and when it starts, in
    seconds of the run: mlx-audio's own alignment, which `Said` leaves out."""

    session: Any
    tokens: list[tuple[int, float, str]] = field(default_factory=list[tuple[int, float, str]])

    def feed(self, samples: Samples) -> Any:
        decoded = self.session.feed(samples)
        self.tokens += [(streaming.speaker(delta.speaker), token.start, token.text) for delta in decoded for token in delta.tokens]
        return decoded

    def __getattr__(self, name: str) -> Any:
        # Everything else the pair's close reaches is the session's own.
        return getattr(self.session, name)


# Who says each hold of two_speakers_ami.flac that these cases hear.
WOMAN = {1: "FEE013", 2: "FEE013", 3: "FEE013", 4: "FEE016", 12: "FEE016"}


# Each woman heard alone first, then one hold of a turn of each, joined (docs/features.md, "Who said each word"): after
# holds 1 and 4, 12 then 3 gives most of FEE016's words to FEE013's speaker; one more hold of FEE013 before FEE016's
# first gives each her own.
@pytest.mark.parametrize(
    ("alone", "joined"),
    [
        ((1, 4), (3, 12)),
        pytest.param((1, 4), (12, 3), marks=pytest.mark.xfail(raises=AssertionError, strict=True, reason="the voice the run knows is heard in one hold")),
        ((1, 2, 4), (12, 3)),
    ],
    ids=["known-once", "known-once-newcomer-first", "known-twice"],
)
def test_two_people_in_one_hold_come_out_as_each_persons_words(paired: Any, alone: tuple[int, ...], joined: tuple[int, int]) -> None:
    # Decided on when each word starts, not on its text, which is about 20% wrong on AMI.
    timed = Timed(paired.session)
    live = Live(dataclasses.replace(paired, session=timed))
    holds = fixture_holds("two_speakers_ami.flac")
    for number in alone:
        live.spoken(np.concatenate([holds[number - 1], STOP]))
    first, second = holds[joined[0] - 1], holds[joined[1] - 1]
    opened, heard = live.session.fed, len(timed.tokens)
    _, whole = live.spoken(np.concatenate([first, second, STOP]))
    # hands gives each person's words as theirs, each the speaker numbered by when the run first heard her, in the order
    # each was first heard in the hold.
    arrived = list(dict.fromkeys(WOMAN[number] for number in alone))
    speakers = [arrived.index(WOMAN[number]) for number in joined]
    assert [said.speaker for said in whole.words] == speakers, whole.words
    boundary = seconds(opened + len(first))
    words = [(speaker, start) for speaker, start, text in timed.tokens[heard:] if any(map(str.isalnum, text))]
    # Each person's words start in their own stretch of the hold, and nobody else's do. mlx-audio dates a word from the
    # chunk the encoder was pushed, and the encoder may still hold up to a chunk of the feed before it, so a word's start
    # is late by as much as a chunk, never early: the first person's last word, said at the end of her stretch, can be
    # dated past it, and the second's first cannot be dated before it.
    late = seconds(CHUNK * MEL_HOP)
    assert {speaker for speaker, _ in words} == set(speakers), words
    assert [(speaker, start) for speaker, start in words if not (start < boundary + late if speaker == speakers[0] else boundary <= start)] == [], (boundary, words)


def test_a_hold_whose_last_chunk_is_committed_by_the_last_step_of_its_silence_keeps_its_last_word(paired: Session) -> None:
    # Hold 15 of the reading, "There was no answer.", runs to its last sample, and its length is the worst phase of the
    # grid: the step that commits the chunk holding its last sample commits nothing past it, so a silence one step
    # shorter leaves "answer" undecoded at the close, and it comes out in the next hold, "It was as he expected."
    first, second = fixture_holds(READER)[14:16]
    holding_last = ((len(first) - 1 + streaming.N_FFT // 2) // MEL_HOP // CHUNK + 1) * CHUNK
    assert STEP * steps(through(len(first))) == holding_last
    live = Live(paired)
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


@pytest.mark.parametrize("after", SHAPES.values(), ids=SHAPES.keys())
def test_a_turn_that_ends_in_silence_ends_on_its_last_words_and_only_its_speaker_says_them(paired: Session, after: float) -> None:
    # GATING decides how a hold's last words come out. At 4, four of these end otherwise in this run: "yeah" alone,
    # "concept", "but uh", "bit better", and in engaged conversation a second speaker says "But it breaks." after the first.
    live = Live(paired)
    # Each fixture hold opens LEAD before its turn's first word.
    holds = [turn(samples, LEAD, LEAD + end - start, after) for samples, (start, end, _) in zip(fixture_holds(TURNS), KNOWN, strict=True)]
    heard = [spoken_words(live.spoken(hold)[1].words) for hold in holds]
    # Its last two words, or the one of the turn that has one.
    ends = [(len({speaker for speaker, _ in words}), [word for _, word in words][-2:]) for words in heard]
    assert ends == [(1, re.findall(r"[\w']+", known.lower())[-2:]) for _, _, known in KNOWN], heard


def test_the_diarizer_commits_on_exactly_the_feeds_steps_names(paired: Any) -> None:
    # [LAW:one-source-of-truth] the diarizer's own count of the frames it committed, against `steps`, fed 20 ms at a time.
    # The session's own diarizer, as `streaming.diarizer` set it to its preset.
    model = paired.session.diarization_model
    state = model.init_streaming_state()
    audio = np.concatenate(fixture_holds("one_reader_quiet.flac")[:4])
    moved: list[tuple[bool, bool]] = []
    for start in range(0, len(audio), 320):
        committed = state.frames_processed
        _, state = model.feed(audio[start : start + 320], state)
        moved.append((steps(start + len(audio[start : start + 320])) > steps(start), state.frames_processed > committed))
    assert all(timed == stepped for timed, stepped in moved), [at for at, (timed, stepped) in enumerate(moved) if timed != stepped]
    assert sum(stepped for _, stepped in moved) > 5
