"""The clip owner: clips cut where the gate cut them, every listener given the same samples of each in capture order, the
transcriber's words in clip order, a listener's words cutting hands off only while their clip is open, and one event for
every clip and every hearing however it ends."""

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import Running, running
from hands.clock import SYSTEM, Clock
from hands.sessions.audit import Entry, Levels
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import clipowner
from hands.voice.clipowner import IDENTIFIED_SECONDS, ClipOwner
from hands.voice.clips import Clip, Cleared, Fate, Heard, Hearing, Hearings, Recording, Reference, Settle, Sound
from hands.voice.ptt import Gate
from hands.voice.trigger import Trigger
from hands.voice.turnstop import SpeakerCleared, TurnOpened, TurnResolved, Typed, Words


def unjudged(_settle: Settle) -> None:
    """A listener that identifies nobody, so settles no clip's end."""


@dataclass
class Heeding:
    """A listener that writes down everything it is given, and returns, for each frame of a clip open, the hearings
    `hearings` makes for it; it keeps what to tell of a speaker cleared in the clip open, and tells `settling` what to
    tell of each clip's end."""

    name: str
    given: list[tuple[str, object]] = field(default_factory=list[tuple[str, object]])
    hearings: Callable[[Clip, Sound], Hearings] = lambda _clip, _sound: ()
    after: Callable[[Clip, Fate], Hearings] = lambda _clip, _fate: ()
    settling: Callable[[Settle], None] = unjudged
    cleared: Cleared | None = None
    _open: Clip | None = None

    def opened(self, clip: Clip, before: Sound, cleared: Cleared) -> Hearings:
        self._open, self.cleared = clip, cleared
        self.given.append(("opened", (clip, before)))
        return ()

    def heard(self, sound: Sound) -> Hearings:
        self.given.append(("heard", sound))
        assert self._open is not None
        return self.hearings(self._open, sound)

    def ended(self, clip: Clip, fate: Fate, sound: Sound, settle: Settle) -> Hearings:
        self.given.append(("ended", (clip, fate, sound)))
        self.settling(settle)
        return self.after(clip, fate)


def elsewhere(told: Callable[[], None]) -> None:
    """`told` from a thread of its own, as the voiceprints' lane tells the owner."""
    thread = threading.Thread(target=told)
    thread.start()
    thread.join()


@dataclass
class Scripted:
    """A transcriber that says, for each clip in turn, what it was given: words, nothing, a failure, or to wait."""

    said: list[str | None | Exception | asyncio.Event] = field(default_factory=list[str | None | Exception | asyncio.Event])

    async def transcribed(self, clip: Clip, audio: bytes) -> str | None:
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

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TurnOpened(clip=clip):
                self.seen.append(("opened", clip.number))
            case TurnResolved(clip=clip):
                self.seen.append(("resolved", clip.number))
            case Words(text=text):
                self.seen.append(("words", text))
            case SpeakerCleared(clip=clip, first=first):
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

    async def frame(self, gate: Gate, sound: bytes, voices: tuple[str, ...] = ()) -> None:
        """A frame captured under `gate`, hands playing in `voices` over it."""
        self.gate = gate
        await self.run.worker.queue_frame(gate.framed(sound, sound, played_with(sound, voices), 16000, 1, "desk"))

    async def clip(self, sounds: Sequence[bytes], fate: Fate = "sent", by: Trigger = "engaged conversation", voices: tuple[str, ...] = ()) -> None:
        """A clip opened by `by`, a frame of each sound captured while it is open, hands playing in `voices` over it, then
        sent or thrown away."""
        down = self.gate.after("start", by)
        for sound in sounds:
            await self.frame(down, sound, voices)
        await self.frame(down.after("stop" if fate == "sent" else "drop", by), b"\x00\x00" * 320, voices)

    def events(self, name: str) -> list[WideEvent]:
        return [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == name]

    async def until(self, what: Callable[[], bool]) -> None:
        async with asyncio.timeout(10):
            while not what():
                await asyncio.sleep(0.01)


@asynccontextmanager
async def rigged(transcriber: Scripted, listeners: Sequence[Heeding] = (), clock: Clock = SYSTEM) -> AsyncGenerator[Rig]:
    recorded: list[Entry] = []
    owner, seen = ClipOwner(clock, transcriber, listeners, recorded.append), Seen()
    async with running([owner, seen]) as run:
        yield Rig(run, owner, seen, recorded)


def frame_of(n: int) -> bytes:
    """20 ms of a sound every sample of which is `n`."""
    return n.to_bytes(2, "little") * 320


def played_with(sound: bytes, voices: tuple[str, ...] = ()) -> tuple[Reference, ...]:
    """What the speaker played over `sound`, as the canceller hands it on, one reference for each 10 ms: the same sound,
    hands' TTS in `voices`."""
    return tuple(Reference(sound[start : start + 320], 16000, 1, voices) for start in range(0, len(sound), 320))


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
        await rig.frame(rig.gate.after("start", "engaged conversation"), frame_of(3))
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
        down = rig.gate.after("start", "engaged conversation")
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
        down = rig.gate.after("start", "engaged conversation")
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
        down = rig.gate.after("start", "engaged conversation")
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
        await rig.run.worker.queue_frame(framed(rig.gate.after("start", "engaged conversation")))
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


async def test_a_clip_dropped_by_a_move_to_the_phone_is_measured_over_its_own_audio() -> None:
    async with rigged(Scripted([])) as rig:
        down = rig.gate.after("start", "shortcut")
        await rig.frame(down, frame_of(16384))
        await rig.frame(down, frame_of(16384))
        await rig.run.worker.queue_frame(down.moved("phone").framed(frame_of(0), frame_of(0), None, 16000, 1, "phone"))
        await rig.until(lambda: len(rig.events("clip")) == 1)
        [clip] = rig.events("clip")
        assert (clip.facts["fate"], clip.facts["seconds"], clip.facts["levels"]) == ("dropped", 0.04, Levels(-6.0, -6.0))


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
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: len(listener.given) == 3)
        await asyncio.sleep(0.05)
    assert [(event.outcome, event.facts["listener"], event.facts["clip"]) for event in rig.events("hearing")] == [("cancelled", "stuck", 1)] * 2


def settling(first: float | None) -> Callable[[Settle], None]:
    """A clip's last identification coming to `first`, told from the voiceprints' lane, a thread of its own."""
    return lambda settle: elsewhere(lambda: settle(first))


def hands_words(_clip: Clip, _sound: Sound) -> Hearings:
    """Hearings of hands' own reading, come back through the microphone: words, which clear nobody."""

    async def hearing() -> Heard:
        return Heard("the parser is fixed", None)

    return (hearing,)


def barge_ins(rig: Rig) -> list[tuple[object, ...]]:
    return [seen for seen in rig.seen.seen if seen[0] == "barge_in"]


async def test_hands_own_echo_over_a_reading_cuts_nothing_while_its_clip_is_open_or_as_it_closes() -> None:
    echo = Heeding("streaming", hearings=hands_words, settling=settling(None))
    async with rigged(Scripted(["the parser is fixed"]), (echo,)) as rig:
        await rig.clip([frame_of(3), frame_of(3)], voices=("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert barge_ins(rig) == []
        assert {event.facts["barge_in"] for event in rig.events("hearing")} == {False}
        [clip] = rig.events("clip")
        assert (clip.facts["transcript"], clip.facts["identified"]) == ("words", "nobody")


async def test_a_person_over_hands_cuts_it_off_while_their_clip_is_open_and_never_once_it_has_closed() -> None:
    listener = Heeding("streaming", settling=settling(None))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3), ("charles",))
        await rig.until(lambda: listener.cleared is not None)
        cleared = listener.cleared
        assert cleared is not None
        # Identified as a person on the voiceprints' lane, first heard 0.4 s into the clip.
        elsewhere(lambda: cleared(0.4))
        await rig.until(lambda: barge_ins(rig) == [("barge_in", 1, 0.4)])
        await rig.frame(down.after("stop", "engaged conversation"), frame_of(0), ("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert rig.seen.seen.index(("barge_in", 1, 0.4)) < rig.seen.seen.index(("resolved", 1))
        # An identification that lands once the clip has closed cuts nothing: not this turn, nor the next one's.
        elsewhere(lambda: cleared(0.9))
        await rig.run.worker.queue_frame(Typed(text="and the next"))
        await rig.until(lambda: ("resolved", 2) in rig.seen.seen)
        assert barge_ins(rig) == [("barge_in", 1, 0.4)]


@pytest.mark.parametrize("by", ["wake word", "engaged conversation"])
async def test_a_wake_phrase_said_alone_over_hands_cuts_it_off_as_its_clip_closes_ahead_of_its_words(by: Trigger) -> None:
    # Too short to be told from hands' echo while it was open: its last identification, at the close, finds a person.
    listener = Heeding("streaming", hearings=hands_words, settling=settling(-0.3))
    async with rigged(Scripted(["hey hands"]), (listener,)) as rig:
        await rig.clip([frame_of(3)], by=by, voices=("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("barge_in", 1, -0.3), ("words", "hey hands"), ("resolved", 1)]
        [clip] = rig.events("clip")
        assert clip.facts["identified"] == "a person"


async def test_a_clip_hands_played_over_is_cut_on_its_identification_while_its_transcription_is_still_running() -> None:
    transcribing = asyncio.Event()
    async with rigged(Scripted([transcribing]), (Heeding("streaming", settling=settling(0.3)),)) as rig:
        await rig.clip([frame_of(3)], by="wake word", voices=("charles",))
        await rig.until(lambda: barge_ins(rig) == [("barge_in", 1, 0.3)])
        assert ("resolved", 1) not in rig.seen.seen
        transcribing.set()
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)


async def test_a_clip_that_cut_while_it_was_open_resolves_without_waiting_on_its_last_identification() -> None:
    clock = VirtualClock()
    listener = Heeding("streaming")
    # Nothing settles the clip's end, and the clock is never advanced: a wait on it would never end.
    async with rigged(Scripted(["stop there"]), (listener,), clock) as rig:
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3), ("charles",))
        await rig.until(lambda: listener.cleared is not None)
        cleared = listener.cleared
        assert cleared is not None
        elsewhere(lambda: cleared(0.4))
        await rig.until(lambda: barge_ins(rig) == [("barge_in", 1, 0.4)])
        await rig.frame(down.after("stop", "engaged conversation"), frame_of(0), ("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("barge_in", 1, 0.4), ("words", "stop there"), ("resolved", 1)]
        assert "identified" not in rig.events("clip")[0].facts


async def test_a_clip_hands_played_nothing_over_cuts_on_its_words_as_it_closes_and_waits_on_no_voiceprint() -> None:
    # Nothing settles its identification: a clip with no echo of hands in it never waits on one.
    async with rigged(Scripted(["stop there", None]), (Heeding("streaming"),)) as rig:
        await rig.clip([frame_of(3)])
        await rig.clip([frame_of(3)])
        await rig.until(lambda: ("resolved", 2) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("barge_in", 1, 0.0), ("words", "stop there"), ("resolved", 1), ("resolved", 2)]
        assert ["identified" in event.facts for event in rig.events("clip")] == [False, False]


async def test_a_clip_the_hand_opened_was_cut_as_it_opened_so_its_close_waits_on_no_voiceprint() -> None:
    async with rigged(Scripted(["stop there"]), (Heeding("streaming"),)) as rig:
        await rig.clip([frame_of(3)], by="shortcut", voices=("charles",))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert barge_ins(rig) == [] and "identified" not in rig.events("clip")[0].facts


async def test_a_clip_whose_last_identification_misses_its_bound_cuts_nothing_and_says_so() -> None:
    clock = VirtualClock()
    # The voiceprints' lane is behind the model's load: nothing settles the clip's end.
    async with rigged(Scripted(["hey hands"]), (Heeding("streaming"),), clock) as rig:
        await rig.clip([frame_of(3)], by="wake word", voices=("charles",))
        while ("resolved", 1) not in rig.seen.seen:
            clock.advance(IDENTIFIED_SECONDS)
            await asyncio.sleep(0.01)
        assert barge_ins(rig) == [] and ("words", "hey hands") in rig.seen.seen
        [clip] = rig.events("clip")
        assert (clip.outcome, clip.facts["identified"]) == ("ok", f"nothing after {IDENTIFIED_SECONDS:g} s")
