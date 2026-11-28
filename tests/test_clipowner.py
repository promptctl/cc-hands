"""The clip owner: clips cut where the gate cut them, every listener given the same samples of each in capture order, the
transcriber's words in clip order, a listener's words cutting hands off only while their clip is open, and one event for
every clip and every hearing however it ends."""

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import pytest
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import Running, running, unprimed
from hands.clock import SYSTEM, Clock
from hands.sessions.audit import Entry, Levels
from hands.sessions.wide import WideEvent, begun
from hands.virtualclock import VirtualClock
from hands.voice import clipowner, transcription
from hands.voice.clipowner import ClipOwner
from hands.voice.clips import Clip, Fate, Heard, Hearing, Hearings, Recording, Reference, Sound, Playhead, Synthesis, Transcriber, voices
from hands.voice.ptt import Gate
from hands.voice.trigger import Trigger, by_hand
from hands.voice.turnstop import ClipDiscarded, ClipSent, SpeechHeard, TurnOpened, TurnResolved, TurnUnheard, Typed, Words
from hands.voice.mlxthread import OnMLX
from hands.voice.whisper import Whisper


@dataclass
class Heeding:
    """A listener that writes down everything it is given, and returns, for each frame of a clip open, the hearings
    `hearings` makes for it."""

    name: str
    given: list[tuple[str, object]] = field(default_factory=list[tuple[str, object]])
    hearings: Callable[[Clip, Sound], Hearings] = lambda _clip, _sound: ()
    after: Callable[[Clip, Fate], Hearings] = lambda _clip, _fate: ()
    _open: Clip | None = None

    def opened(self, clip: Clip, before: Sound) -> Hearings:
        self._open = clip
        self.given.append(("opened", (clip, before)))
        return ()

    def heard(self, sound: Sound) -> Hearings:
        self.given.append(("heard", sound))
        assert self._open is not None
        return self.hearings(self._open, sound)

    def ended(self, clip: Clip, fate: Fate, sound: Sound) -> Hearings:
        self.given.append(("ended", (clip, fate, sound)))
        return self.after(clip, fate)


@dataclass
class Scripted:
    """A transcriber that says, for each clip in turn, what it was given: words, nothing, a failure, or to wait."""

    said: list[str | None | Exception | asyncio.Event] = field(default_factory=list[str | None | Exception | asyncio.Event])
    # What each clip's transcription was given: the clip as heard, and what hands played over it.
    given: list[Sound] = field(default_factory=list[Sound])

    async def transcribed(self, clip: Clip, sound: Sound) -> str | None:
        self.given.append(sound)
        match self.said.pop(0):
            case asyncio.Event() as go:
                await go.wait()
                return None
            case Exception() as error:
                raise error
            case text:
                return text


class Seen(FrameProcessor):
    """What leaves the owner, in order, as (kind, clip number or words)."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.seen: list[tuple[object, ...]] = []
        # How each turn the gate ended was said to end, as (kind, clip or turn number), in the order it was said.
        self.ends: list[tuple[str, int]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TurnOpened(clip=clip):
                self.seen.append(("opened", clip.number))
            case ClipSent(clip=clip):
                self.ends.append(("sent", clip.number))
            case ClipDiscarded(clip=clip):
                self.ends.append(("discarded", clip.number))
            case TurnUnheard(turn=turn):
                self.ends.append(("unheard", turn))
            case TurnResolved(clip=clip):
                self.seen.append(("resolved", clip.number))
            case Words(text=text):
                self.seen.append(("words", text))
            case SpeechHeard(clip=clip, first=first):
                self.seen.append(("barge_in", clip.number, first))
            case _:
                pass
        await self.push_frame(frame, direction)


@dataclass
class Rig:
    run: Running
    owner: ClipOwner
    seen: Seen
    recorded: list[Entry]
    gate: Gate = field(default_factory=Gate)

    async def frame(self, gate: Gate, sound: bytes, voices: tuple[str, ...] = (), reached: bool = True) -> None:
        """A frame captured under `gate`, hands playing in `voices` over it, its voice `reached` the microphone or not."""
        self.gate = gate
        await self.run.worker.queue_frame(gate.framed(sound, sound, played_with(sound, voices, reached), 16000, 1, "desk"))

    async def clip(self, sounds: Sequence[bytes], fate: Fate = "sent", by: Trigger = "engaged conversation", voices: tuple[str, ...] = (), reached: bool = True) -> None:
        """A clip opened by `by`, a frame of each sound captured while it is open, hands playing in `voices` over it, its
        voice `reached` the microphone or not, then sent or thrown away."""
        down = started(self.gate, by)
        for sound in sounds:
            await self.frame(down, sound, voices, reached)
        await self.frame(down.after("stop" if fate == "sent" else "drop", by), b"\x00\x00" * 320, voices, reached)

    def events(self, name: str) -> list[WideEvent]:
        return [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == name]

    async def until(self, what: Callable[[], bool]) -> None:
        async with asyncio.timeout(10):
            while not what():
                await asyncio.sleep(0.01)


@asynccontextmanager
async def rigged(transcriber: Transcriber, listeners: Sequence[Heeding] = (), clock: Clock = SYSTEM) -> AsyncGenerator[Rig]:
    recorded: list[Entry] = []
    owner, seen = ClipOwner(clock, transcriber, listeners, recorded.append), Seen()
    async with running([owner, seen]) as run:
        yield Rig(run, owner, seen, recorded)


def started(gate: Gate, by: Trigger) -> Gate:
    """The gate once `by` has opened a clip: a hand's press opens it, and a voice's is listened for, armed and started."""
    return gate.after("start", by) if by_hand(by) else gate.after("listen", by).after("arm", by).after("start", by)


def tap(gate: Gate, by: Trigger = "shortcut") -> Gate:
    """The gate once a press and its release both came, with no frame captured between them."""
    return started(gate, by).after("stop", by)


def frame_of(n: int) -> bytes:
    """20 ms of a sound every sample of which is `n`."""
    return n.to_bytes(2, "little") * 320


def played_with(sound: bytes, voices: tuple[str, ...] = (), reached: bool = True) -> tuple[Reference, ...]:
    """What the speaker played over `sound`, as the canceller hands it on, one reference for each 10 ms: the same sound,
    a sentence of hands' TTS in each of `voices`, that `reached` the microphone or not."""
    syntheses = tuple(Playhead(Synthesis(voice, "The parser is fixed.", begun(SYSTEM)), 0.0) for voice in voices)
    return tuple(Reference(sound[start : start + 320], 16000, 1, syntheses, reached) for start in range(0, len(sound), 320))


async def test_every_listener_is_given_the_same_samples_in_capture_order_what_the_clip_kept_from_before_it_included() -> None:
    first, second = Heeding("first"), Heeding("second")
    async with rigged(Scripted([None]), (first, second)) as rig:
        # The desk listening, then speech opening the clip: what it heard while arming is the clip's start.
        listening = rig.gate.after("listen", "engaged conversation")
        await rig.frame(listening, frame_of(1))
        arming = listening.after("arm", "engaged conversation")
        await rig.frame(arming, frame_of(2))
        down = arming.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(4))
        await rig.frame(down.after("stop", "engaged conversation"), frame_of(5))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
    clip = Clip(1, "engaged conversation")
    def sound(*levels: int) -> Sound:
        audio = b"".join(map(frame_of, levels))
        return Sound(audio, audio, played_with(audio))

    # And as it ends, the whole of it, with what the speaker played over every 10 ms of it.
    assert first.given == second.given == [
        ("opened", (clip, sound(1, 2))),
        ("heard", sound(3)),
        ("heard", sound(4)),
        ("ended", (clip, "sent", sound(1, 2, 3, 4))),
    ]


def test_the_second_a_clip_keeps_from_before_it_keeps_what_was_played_over_that_second_and_no_more() -> None:
    recording = Recording()
    for level in range(60):
        audio = frame_of(level)
        recording.add(Sound(audio, audio, played_with(audio)), gathering=False)
    # 60 frames of 20 ms, the last 50 kept: one second, and a reference for each 10 ms of it.
    kept = b"".join(map(frame_of, range(10, 60)))
    assert recording.sound == Sound(kept, kept, played_with(kept))
    # A frame the phone heard has nothing of the desk's speaker under it: the recording says nothing was referenced.
    recording.add(Sound(frame_of(1), frame_of(1), None), gathering=True)
    assert recording.sound.played is None


async def test_hands_voice_is_held_in_what_a_clip_opened_now_would_keep_until_a_clip_takes_that_away() -> None:
    async with rigged(Scripted([None])) as rig:
        listening = rig.gate.after("listen", "wake word")
        # Hands played over the second the desk heard: a clip opened now would keep hands' voice.
        await rig.frame(listening, frame_of(1), ("alba",))
        await rig.until(rig.owner.holds_hands)
        # The clip opened, kept it, and was sent: what a clip opened after it would keep starts again, without hands.
        await rig.clip([frame_of(2)], by="wake word")
        await rig.until(lambda: rig.seen.ends == [("sent", 1)])
        assert not rig.owner.holds_hands()
        # Played where it could not reach the microphone, as in headphones, hands' voice is in nothing a clip keeps.
        await rig.frame(rig.gate, frame_of(3), ("alba",), reached=False)
        await rig.clip([frame_of(4)], by="wake word", voices=("alba",), reached=False)
        await rig.until(lambda: rig.seen.ends == [("sent", 1), ("sent", 2)])
        assert not rig.owner.holds_hands()


async def test_the_transcribers_words_go_on_in_the_order_the_clips_ended_and_a_typed_clip_waits_its_turn() -> None:
    go = asyncio.Event()
    async with rigged(Scripted([go, "and the tests"])) as rig:
        await rig.clip([frame_of(3)], by="shortcut")
        await rig.until(lambda: ("opened", 1) in rig.seen.seen)
        # Typed words are a data frame, which the microphone's system frames would overtake: they go in once taken.
        await rig.run.worker.queue_frame(Typed(text="typed after"))
        await rig.until(lambda: ("opened", 2) in rig.seen.seen)
        await rig.clip([frame_of(3)], by="shortcut")
        await rig.until(lambda: ("opened", 3) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == []
        go.set()
        await rig.until(lambda: ("resolved", 3) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("resolved", 1), ("words", "typed after"), ("resolved", 2), ("words", "and the tests"), ("resolved", 3)]


def worded(words: str) -> Callable[[Clip, Sound], Hearings]:
    """Hearings that hear `words` in every frame."""

    async def hearing() -> Heard:
        return Heard(words, 0.0)

    return lambda _clip, _sound: (hearing,)


async def test_a_listener_slow_to_hear_clips_up_no_other_listeners_cut() -> None:
    stuck = asyncio.Event()

    async def waiting() -> Heard:
        await stuck.wait()
        return Heard(None, None)

    slow, quick = Heeding("slow", hearings=lambda _clip, _sound: (waiting,)), Heeding("quick", hearings=worded("wait"))
    async with rigged(Scripted([None]), (slow, quick)) as rig:
        await rig.frame(started(rig.gate, "engaged conversation"), frame_of(3))
        await rig.until(lambda: ("barge_in", 1, 0.0) in rig.seen.seen)
        stuck.set()


async def test_words_heard_once_their_clip_closed_cut_nothing_and_never_land_behind_its_resolution() -> None:
    go = asyncio.Event()

    async def late() -> Heard:
        await go.wait()
        return Heard("too late", 0.0)

    listener = Heeding("late", hearings=lambda _clip, _sound: (late,))
    async with rigged(Scripted([None]), (listener,)) as rig:
        await rig.clip([frame_of(3)], fate="dropped")
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        go.set()
        await rig.until(lambda: len(rig.events("hearing")) == 1)
        [hearing] = rig.events("hearing")
        assert (hearing.outcome, hearing.facts["listener"], hearing.facts["clip"], hearing.facts["heard"], hearing.facts["barge_in"]) == ("ok", "late", 1, "words", False)
        assert ("barge_in", 1, 0.0) not in rig.seen.seen


async def test_words_heard_while_their_clip_is_open_cut_hands_off_ahead_of_its_resolution() -> None:
    async with rigged(Scripted([None]), (Heeding("quick", hearings=worded("wait")),)) as rig:
        down = started(rig.gate, "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: ("barge_in", 1, 0.0) in rig.seen.seen)
        await rig.frame(down.after("drop", "engaged conversation"), frame_of(0))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert rig.seen.seen.index(("barge_in", 1, 0.0)) < rig.seen.seen.index(("resolved", 1))
        [hearing] = rig.events("hearing")
        assert (hearing.facts["heard"], hearing.facts["barge_in"]) == ("words", True)


async def test_a_listeners_work_after_its_clip_ends_is_a_hearing_of_its_own() -> None:
    async def settled() -> Heard:
        return Heard(None, None)

    listener = Heeding("settling", after=lambda _clip, _fate: (settled,))
    async with rigged(Scripted([None]), (listener,)) as rig:
        await rig.clip([frame_of(3)])
        await rig.until(lambda: len(rig.events("hearing")) == 1)
        [hearing] = rig.events("hearing")
        assert (hearing.outcome, hearing.facts["clip"], hearing.facts["heard"], hearing.facts["barge_in"]) == ("ok", 1, "nothing", False)


async def test_a_hearing_that_never_returns_fails_on_its_event_and_the_listener_hears_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipowner, "HEARING_SECONDS", 0.1)
    hung = asyncio.Event()
    answers: list[Hearing] = []

    async def never() -> Heard:
        await hung.wait()
        return Heard(None, None)

    async def words() -> Heard:
        return Heard("still here", 0.0)

    answers += [never, words]
    listener = Heeding("hung", hearings=lambda _clip, _sound: (answers.pop(0),))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = started(rig.gate, "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: ("barge_in", 1, 0.0) in rig.seen.seen)
        failed, heard = rig.events("hearing")
        assert (failed.outcome, failed.error) == ("failed", "TimeoutError: ")
        assert heard.outcome == "ok"


async def test_a_hearing_is_bounded_from_when_its_listener_returned_it_so_one_queued_behind_a_hung_one_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clipowner, "HEARING_SECONDS", 0.1)
    hung = asyncio.Event()
    answers: list[Hearing] = []

    async def never() -> Heard:
        await hung.wait()
        return Heard(None, None)

    async def words() -> Heard:
        # A hearing that has to wait at all: one whose words are already in hand is no hearing still to come back.
        await asyncio.sleep(0)
        return Heard("too long in line", 0.0)

    answers += [never, words]
    listener = Heeding("hung", hearings=lambda _clip, _sound: (answers.pop(0),))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = started(rig.gate, "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: len(rig.events("hearing")) == 2)
        # Its minute ran out while it waited behind the hung one: its event says how long that was.
        hung_one, queued = rig.events("hearing")
        assert [event.error for event in (hung_one, queued)] == ["TimeoutError: "] * 2
        assert queued.duration_ms >= 100
        assert ("barge_in", 1, 0.0) not in rig.seen.seen


def at_48k(gate: Gate) -> Frame:
    return gate.framed(frame_of(3), frame_of(3), None, 48_000, 1, "desk")


def unkeyed(_gate: Gate) -> Frame:
    return InputAudioRawFrame(audio=frame_of(3), sample_rate=16_000, num_channels=1)


@pytest.mark.parametrize(("framed", "why"), [(at_48k, "at 48000 Hz"), (unkeyed, "carries no key")])
async def test_a_frame_no_clip_can_be_cut_from_fails_as_it_arrives_and_no_listener_hears_it(framed: Callable[[Gate], Frame], why: str) -> None:
    listener = Heeding("given")
    async with rigged(Scripted([]), (listener,)) as rig:
        await rig.run.worker.queue_frame(framed(started(rig.gate, "engaged conversation")))
        await rig.until(lambda: len(rig.run.errors) == 1)
        [error] = rig.run.errors
        assert why in error.error
        assert (listener.given, rig.seen.seen) == ([], [])


async def test_a_listener_that_raises_as_it_is_told_fails_its_own_hearing_and_the_clip_still_resolves() -> None:
    def broken(_clip: Clip, _fate: Fate) -> Hearings:
        raise RuntimeError("listener broke")

    async with rigged(Scripted(["still sent"]), (Heeding("broken", after=broken),)) as rig:
        await rig.clip([frame_of(3)])
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert ("words", "still sent") in rig.seen.seen
        [hearing] = rig.events("hearing")
        assert (hearing.outcome, hearing.error, hearing.facts["listener"], hearing.facts["clip"]) == ("failed", "RuntimeError: listener broke", "broken", 1)


async def test_a_clip_sent_by_a_move_to_the_phone_is_measured_over_its_own_audio() -> None:
    async with rigged(Scripted([None])) as rig:
        down = started(rig.gate, "shortcut")
        await rig.frame(down, frame_of(16384))
        await rig.frame(down, frame_of(16384))
        await rig.run.worker.queue_frame(down.moved("phone").framed(frame_of(0), frame_of(0), None, 16000, 1, "phone"))
        await rig.until(lambda: len(rig.events("clip")) == 1)
        [clip] = rig.events("clip")
        assert (clip.facts["fate"], clip.facts["seconds"], clip.facts["levels"]) == ("sent", 0.04, Levels(-6.0, -6.0))


async def test_each_clip_is_one_event_sent_dropped_typed_and_failed() -> None:
    # The transcriber is given the spoken clips sent alone: the first, the fourth and the fifth.
    async with rigged(Scripted([None, RuntimeError("model failed"), "what time is it"])) as rig:
        await rig.clip([frame_of(16384)], by="shortcut")
        await rig.clip([frame_of(16384)], fate="dropped", by="shortcut")
        await rig.until(lambda: ("opened", 2) in rig.seen.seen)
        await rig.run.worker.queue_frame(Typed(text="typed"))
        await rig.until(lambda: ("opened", 3) in rig.seen.seen)
        await rig.clip([frame_of(16384)], by="shortcut")
        await rig.clip([frame_of(16384)], by="shortcut")
        await rig.until(lambda: len(rig.events("clip")) == 5)
        told = [(event.outcome, event.facts["clip"], event.facts["opener"], event.facts.get("fate"), event.facts.get("transcript")) for event in rig.events("clip")]
        assert told == [
            ("ok", 1, "shortcut", "sent", "none"),
            ("ok", 2, "shortcut", "dropped", None),
            ("ok", 3, "typed", None, "words"),
            ("ok", 4, "shortcut", "sent", "failed"),
            ("ok", 5, "shortcut", "sent", "words"),
        ]
        assert rig.events("clip")[0].facts["levels"] == Levels(-6.0, -6.0) and rig.events("clip")[0].facts["seconds"] == 0.02


async def test_clips_still_to_be_resolved_as_the_pipeline_stops_end_cancelled() -> None:
    go = asyncio.Event()
    async with rigged(Scripted([go, None])) as rig:
        await rig.clip([frame_of(3)], by="shortcut")
        await rig.clip([frame_of(3)], by="shortcut")
        await rig.until(lambda: ("opened", 2) in rig.seen.seen)
        await asyncio.sleep(0.05)
    assert [(event.outcome, event.facts["clip"], event.facts["fate"]) for event in rig.events("clip")] == [("cancelled", 1, "sent"), ("cancelled", 2, "sent")]


async def test_hearings_still_to_run_as_the_pipeline_stops_end_cancelled() -> None:
    stuck = asyncio.Event()

    async def waiting() -> Heard:
        await stuck.wait()
        return Heard(None, None)

    listener = Heeding("stuck", hearings=lambda _clip, _sound: (waiting,))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = started(rig.gate, "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: len(listener.given) == 3)
        await asyncio.sleep(0.05)
    assert [(event.outcome, event.facts["listener"], event.facts["clip"]) for event in rig.events("hearing")] == [("cancelled", "stuck", 1)] * 2


def hands_words(_clip: Clip, _sound: Sound) -> Hearings:
    """Hearings of a clip in which no speech is left after the echo canceller: no speaker, so no words and no cut."""

    async def hearing() -> Heard:
        return Heard(None, None)

    return (hearing,)


def barge_ins(rig: Rig) -> list[tuple[object, ...]]:
    return [seen for seen in rig.seen.seen if seen[0] == "barge_in"]


async def test_a_clip_with_no_speech_left_in_it_after_the_canceller_cuts_nothing_whatever_hands_played_over_it() -> None:
    # The whole clip hands' own reading, taken out of the sound: the transcriber hears nobody in it.
    transcriber = Scripted([None])
    async with rigged(transcriber, (Heeding("streaming", hearings=hands_words),)) as rig:
        await rig.clip([frame_of(3), frame_of(3)], voices=("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert barge_ins(rig) == []
        assert [seen for seen in rig.seen.seen if seen[0] == "words"] == []
        assert {event.facts["barge_in"] for event in rig.events("hearing")} == {False}
        [given] = transcriber.given
        assert (given.heard, voices(given)) == (frame_of(3) * 2, ("charles",))
        # [LAW:nothing-unseen] a turn with no one in it says so: no words.
        [clip] = rig.events("clip")
        assert clip.facts["transcript"] == "none"


@pytest.mark.parametrize(
    ("played", "by"),
    [((), "wake word"), (("charles",), "wake word"), (("charles",), "engaged conversation")],
    ids=["nothing played", "hands played, wake word", "hands played, engaged"],
)
async def test_a_voice_opened_clip_nothing_cut_while_open_cuts_on_the_transcribers_words_as_it_closes_ahead_of_them(played: tuple[str, ...], by: Trigger) -> None:
    # A wake phrase said alone, too brief for the recognizer to say its words before the clip closed.
    async with rigged(Scripted(["hey hands"]), (Heeding("streaming", hearings=hands_words),)) as rig:
        await rig.clip([frame_of(3)], by=by, voices=played)
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("barge_in", 1, 0.0), ("words", "hey hands"), ("resolved", 1)]


async def test_a_clip_that_cut_while_it_was_open_is_not_cut_again_as_it_closes() -> None:
    async with rigged(Scripted(["stop there"]), (Heeding("streaming", hearings=worded("stop")),)) as rig:
        await rig.clip([frame_of(3)], by="engaged conversation", voices=("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("barge_in", 1, 0.0), ("words", "stop there"), ("resolved", 1)]


async def test_a_clip_the_voice_opened_that_the_transcriber_heard_nobody_in_cuts_nothing_as_it_closes() -> None:
    async with rigged(Scripted([None, None]), (Heeding("streaming", hearings=hands_words),)) as rig:
        await rig.clip([frame_of(3)], by="wake word")
        await rig.clip([frame_of(3)], by="wake word", voices=("charles",))
        await rig.until(lambda: ("resolved", 2) in rig.seen.seen)
        assert barge_ins(rig) == []


async def test_a_clip_the_voice_opened_whose_transcription_does_not_return_is_given_up_on_at_the_bound_and_cuts_nothing() -> None:
    # Nothing was heard while the clip was open, so hands talks on until its transcription returns: `_closing` cuts on the
    # words of one that does. One that does not is given up on at the bound, the most hands talks on.
    clock, bound, transcriber = VirtualClock(), clipowner.TRANSCRIBING_SECONDS, Scripted([asyncio.Event()])
    async with rigged(transcriber, (Heeding("streaming", hearings=hands_words),), clock) as rig:
        await rig.clip([frame_of(3)], by="engaged conversation")
        # The transcriber records a clip once it is inside the bound, whose deadline is then set.
        await rig.until(lambda: len(transcriber.given) == 1)
        clock.advance(bound)
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen and len(rig.run.errors) == 1)
        assert barge_ins(rig) == []
        [clip] = rig.events("clip")
        assert (clip.facts["transcript"], "cut" in clip.facts, clip.duration_ms) == ("failed", False, bound * 1000)
        # [LAW:no-silent-failure] the owner says it, as the error the system speaks.
        [failed] = rig.run.errors
        assert failed.processor is rig.owner and failed.error == f"Clip 1 could not be transcribed: nothing after {bound:g} s"


async def test_a_transcription_given_up_on_is_not_run_beside_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    returns, ran_on = threading.Event(), list[threading.Thread]()

    def stuck(_on: OnMLX, _weights: str, samples: np.ndarray, prompt: str | None) -> list[transcription.Segment]:
        ran_on.append(threading.current_thread())
        returns.wait()
        return []

    # Whisper as it is, down to the model, whose one thread the first clip's run holds.
    monkeypatch.setattr(transcription, "segments", stuck)
    clock, bound = VirtualClock(), clipowner.TRANSCRIBING_SECONDS
    try:
        async with rigged(Whisper(clock=clock, prompt=unprimed, record=lambda _: None), (), clock) as rig:
            await rig.clip([frame_of(3)])
            # The model is inside the first clip before its bound is spent.
            await rig.until(lambda: len(ran_on) == 1)
            clock.advance(bound)
            await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
            await rig.clip([frame_of(4)])
            await clock.scheduled(clock.at + bound)
            clock.advance(bound)
            await rig.until(lambda: ("resolved", 2) in rig.seen.seen)
            # The first clip's run ends, and the third clip's is the model's next.
            returns.set()
            await rig.clip([frame_of(5)])
            await rig.until(lambda: ("resolved", 3) in rig.seen.seen)
            assert [event.facts["transcript"] for event in rig.events("clip")] == ["failed", "failed", "none"]
            # The model ran on the first clip and the third on the MLX thread, and never on the second, which failed in
            # its own time behind the first.
            assert ran_on == [ran_on[0]] * 2
    finally:
        returns.set()


async def test_a_clip_the_hand_opened_was_cut_as_it_opened_so_its_close_cuts_nothing_whoever_it_heard() -> None:
    transcriber = Scripted(["stop there"])
    async with rigged(transcriber, (Heeding("streaming"),)) as rig:
        await rig.clip([frame_of(3)], by="shortcut", voices=("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert barge_ins(rig) == []
        # Hands played over it, so its transcriber is given what hands played, to hear only the people in it by.
        [given] = transcriber.given
        assert voices(given) == ("charles",)


async def test_a_turn_that_opens_and_ends_between_two_frames_is_said_unheard_once_and_the_next_clip_is_numbered_as_ever() -> None:
    async with rigged(Scripted([None])) as rig:
        await rig.frame(rig.gate, frame_of(1))
        await rig.frame(tap(rig.gate), frame_of(2))
        await rig.frame(rig.gate, frame_of(3))
        await rig.clip([frame_of(4)])
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
    assert rig.seen.ends == [("unheard", 0), ("sent", 1)]
    # Each is an event of its own, so a turn nothing was heard in is never mistaken for one that never ran.
    assert [(event.facts["turn"], event.facts["fate"]) for event in rig.events("turn.unheard")] == [(0, "sent")]


async def test_every_turn_the_gate_ended_between_two_frames_is_said_unheard_in_the_order_it_ended() -> None:
    async with rigged(Scripted([])) as rig:
        await rig.frame(rig.gate, frame_of(1))
        await rig.frame(tap(tap(rig.gate)), frame_of(2))
        await rig.until(lambda: len(rig.seen.ends) == 2)
    assert rig.seen.ends == [("unheard", 0), ("unheard", 1)]


async def test_a_clip_thrown_away_is_said_discarded_and_never_sent_though_a_turn_ended_beside_it_before_the_next_frame() -> None:
    """The clip open is dropped, and a tap opens and ends, with no frame between: the record says in order how each ended, so
    the clip is not taken for the one the totals alone would call sent."""
    async with rigged(Scripted([])) as rig:
        down = started(rig.gate, "shortcut")
        await rig.frame(down, frame_of(1))
        await rig.frame(tap(down.after("drop", "shortcut")), frame_of(2))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
    assert rig.seen.ends == [("discarded", 1), ("unheard", 1)]
