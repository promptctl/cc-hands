"""The hold owner: holds cut where the gate cut them, every listener given the same samples of each in capture order, the
transcriber's words in hold order, a listener's words cutting hands off only while their hold is open, and one event for
every hold and every hearing however it ends."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
from pipecat.frames.frames import Frame, InputAudioRawFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import Running, running
from hands.sessions.audit import Entry, Levels
from hands.sessions.wide import WideEvent
from hands.voice import holdowner
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import Fate, Heard, Hearing, Hearings, Hold, Sound
from hands.voice.ptt import Gate
from hands.voice.trigger import Edge
from hands.voice.turnstop import TurnOpened, TurnResolved, Typed, Words


@dataclass
class Heeding:
    """A listener that writes down everything it is given, and returns, for each frame of a hold open, the hearings
    `hearings` makes for it."""

    name: str
    given: list[tuple[str, object]] = field(default_factory=list[tuple[str, object]])
    hearings: Callable[[Hold, Sound], Hearings] = lambda _hold, _sound: ()
    after: Callable[[Hold, Fate], Hearings] = lambda _hold, _fate: ()
    _open: Hold | None = None

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        self._open = hold
        self.given.append(("opened", (hold, before)))
        return ()

    def heard(self, sound: Sound) -> Hearings:
        self.given.append(("heard", sound))
        assert self._open is not None
        return self.hearings(self._open, sound)

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        self.given.append(("ended", (hold, fate)))
        return self.after(hold, fate)


@dataclass
class Scripted:
    """A transcriber that says, for each hold in turn, what it was given: words, nothing, a failure, or to wait."""

    said: list[str | None | Exception | asyncio.Event] = field(default_factory=list[str | None | Exception | asyncio.Event])

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        match self.said.pop(0):
            case asyncio.Event() as go:
                await go.wait()
                return None
            case Exception() as error:
                raise error
            case text:
                return text


class Seen(FrameProcessor):
    """What leaves the owner, in order, as (kind, hold number or words)."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.seen: list[tuple[str, object]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case TurnOpened(hold=hold):
                self.seen.append(("opened", hold.number))
            case TurnResolved(hold=hold):
                self.seen.append(("resolved", hold.number))
            case Words(text=text):
                self.seen.append(("words", text))
            case InterimTranscriptionFrame(text=text):
                self.seen.append(("interim", text))
            case _:
                pass
        await self.push_frame(frame, direction)


@dataclass
class Rig:
    run: Running
    owner: HoldOwner
    seen: Seen
    recorded: list[Entry]
    gate: Gate = field(default_factory=Gate)

    async def frame(self, gate: Gate, sound: bytes, by: Edge = "held key") -> None:
        self.gate = gate
        await self.run.worker.queue_frame(gate.framed(sound, sound, 16000, 1, "desk"))

    async def held(self, sounds: Sequence[bytes], fate: Fate = "sent", by: Edge = "engaged conversation") -> None:
        """A hold opened by `by`, a frame of each sound captured while it is open, then sent or thrown away."""
        down = self.gate.after("start", by)
        for sound in sounds:
            await self.frame(down, sound)
        await self.frame(down.after("stop" if fate == "sent" else "drop", by), b"\x00\x00" * 320)

    def events(self, name: str) -> list[WideEvent]:
        return [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == name]

    async def until(self, what: Callable[[], bool]) -> None:
        async with asyncio.timeout(10):
            while not what():
                await asyncio.sleep(0.01)


@asynccontextmanager
async def rigged(transcriber: Scripted, listeners: Sequence[Heeding] = ()) -> AsyncGenerator[Rig]:
    recorded: list[Entry] = []
    owner, seen = HoldOwner(transcriber, listeners, recorded.append), Seen()
    async with running([owner, seen]) as run:
        yield Rig(run, owner, seen, recorded)


def frame_of(n: int) -> bytes:
    """20 ms of a sound every sample of which is `n`."""
    return n.to_bytes(2, "little") * 320


async def test_every_listener_is_given_the_same_samples_in_capture_order_what_the_hold_kept_from_before_it_included() -> None:
    first, second = Heeding("first"), Heeding("second")
    async with rigged(Scripted([None]), (first, second)) as rig:
        # The desk listening, then speech opening the hold: what it heard while arming is the hold's start.
        listening = rig.gate.after("listen", "engaged conversation")
        await rig.frame(listening, frame_of(1))
        arming = listening.after("arm", "engaged conversation")
        await rig.frame(arming, frame_of(2))
        down = arming.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(4))
        await rig.frame(down.after("stop", "engaged conversation"), frame_of(5))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
    hold = Hold(1, "engaged conversation")
    assert first.given == second.given == [
        ("opened", (hold, Sound(frame_of(1) + frame_of(2), frame_of(1) + frame_of(2)))),
        ("heard", Sound(frame_of(3), frame_of(3))),
        ("heard", Sound(frame_of(4), frame_of(4))),
        ("ended", (hold, "sent")),
    ]


async def test_the_transcribers_words_go_on_in_the_order_the_holds_ended_and_a_typed_hold_waits_its_turn() -> None:
    go = asyncio.Event()
    async with rigged(Scripted([go, "and the tests"])) as rig:
        await rig.held([frame_of(3)], by="held key")
        await rig.until(lambda: ("opened", 1) in rig.seen.seen)
        # Typed words are a data frame, which the microphone's system frames would overtake: they go in once taken.
        await rig.run.worker.queue_frame(Typed(text="typed after"))
        await rig.until(lambda: ("opened", 2) in rig.seen.seen)
        await rig.held([frame_of(3)], by="held key")
        await rig.until(lambda: ("opened", 3) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == []
        go.set()
        await rig.until(lambda: ("resolved", 3) in rig.seen.seen)
        assert [seen for seen in rig.seen.seen if seen[0] != "opened"] == [("resolved", 1), ("words", "typed after"), ("resolved", 2), ("words", "and the tests"), ("resolved", 3)]


def worded(words: str) -> Callable[[Hold, Sound], Hearings]:
    """Hearings that hear `words` in every frame."""

    async def hearing() -> Heard:
        return Heard(words)

    return lambda _hold, _sound: (hearing,)


async def test_a_listener_slow_to_hear_holds_up_no_other_listeners_cut() -> None:
    stuck = asyncio.Event()

    async def waiting() -> Heard:
        await stuck.wait()
        return Heard(None)

    slow, quick = Heeding("slow", hearings=lambda _hold, _sound: (waiting,)), Heeding("quick", hearings=worded("wait"))
    async with rigged(Scripted([None]), (slow, quick)) as rig:
        await rig.frame(rig.gate.after("start", "engaged conversation"), frame_of(3))
        await rig.until(lambda: ("interim", "wait") in rig.seen.seen)
        stuck.set()


async def test_words_heard_once_their_hold_closed_cut_nothing_and_never_land_behind_its_resolution() -> None:
    go = asyncio.Event()

    async def late() -> Heard:
        await go.wait()
        return Heard("too late")

    listener = Heeding("late", hearings=lambda _hold, _sound: (late,))
    async with rigged(Scripted([None]), (listener,)) as rig:
        await rig.held([frame_of(3)], fate="dropped")
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        go.set()
        await rig.until(lambda: len(rig.events("hearing")) == 1)
        [hearing] = rig.events("hearing")
        assert (hearing.outcome, hearing.facts["listener"], hearing.facts["hold"], hearing.facts["heard"], hearing.facts["cut"]) == ("ok", "late", 1, "words", False)
        assert ("interim", "too late") not in rig.seen.seen


async def test_words_heard_while_their_hold_is_open_cut_hands_off_ahead_of_its_resolution() -> None:
    async with rigged(Scripted([None]), (Heeding("quick", hearings=worded("wait")),)) as rig:
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: ("interim", "wait") in rig.seen.seen)
        await rig.frame(down.after("drop", "engaged conversation"), frame_of(0))
        await rig.until(lambda: ("resolved", 1) in rig.seen.seen)
        assert rig.seen.seen.index(("interim", "wait")) < rig.seen.seen.index(("resolved", 1))
        [hearing] = rig.events("hearing")
        assert (hearing.facts["heard"], hearing.facts["cut"]) == ("words", True)


async def test_a_listeners_work_after_its_hold_ends_is_a_hearing_of_its_own() -> None:
    async def settled() -> Heard:
        return Heard(None)

    listener = Heeding("settling", after=lambda _hold, _fate: (settled,))
    async with rigged(Scripted([None]), (listener,)) as rig:
        await rig.held([frame_of(3)])
        await rig.until(lambda: len(rig.events("hearing")) == 1)
        [hearing] = rig.events("hearing")
        assert (hearing.outcome, hearing.facts["hold"], hearing.facts["heard"], hearing.facts["cut"]) == ("ok", 1, "nothing", False)


async def test_a_hearing_that_never_returns_fails_on_its_event_and_the_listener_hears_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(holdowner, "HEARING_SECONDS", 0.1)
    hung = asyncio.Event()
    answers: list[Hearing] = []

    async def never() -> Heard:
        await hung.wait()
        return Heard(None)

    async def words() -> Heard:
        return Heard("still here")

    answers += [never, words]
    listener = Heeding("hung", hearings=lambda _hold, _sound: (answers.pop(0),))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: ("interim", "still here") in rig.seen.seen)
        failed, heard = rig.events("hearing")
        assert (failed.outcome, failed.error) == ("failed", "TimeoutError: ")
        assert heard.outcome == "ok"


async def test_a_hearing_is_bounded_from_when_its_listener_returned_it_so_one_queued_behind_a_hung_one_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(holdowner, "HEARING_SECONDS", 0.1)
    hung = asyncio.Event()
    answers: list[Hearing] = []

    async def never() -> Heard:
        await hung.wait()
        return Heard(None)

    async def words() -> Heard:
        # A hearing that has to wait at all: one whose words are already in hand is no hearing still to come back.
        await asyncio.sleep(0)
        return Heard("too long in line")

    answers += [never, words]
    listener = Heeding("hung", hearings=lambda _hold, _sound: (answers.pop(0),))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: len(rig.events("hearing")) == 2)
        # Its minute ran out while it waited behind the hung one: its event says how long that was.
        hung_one, queued = rig.events("hearing")
        assert [event.error for event in (hung_one, queued)] == ["TimeoutError: "] * 2
        assert queued.duration_ms >= 100
        assert ("interim", "too long in line") not in rig.seen.seen


def at_48k(gate: Gate) -> Frame:
    return gate.framed(frame_of(3), frame_of(3), 48_000, 1, "desk")


def unkeyed(_gate: Gate) -> Frame:
    return InputAudioRawFrame(audio=frame_of(3), sample_rate=16_000, num_channels=1)


@pytest.mark.parametrize(("framed", "why"), [(at_48k, "at 48000 Hz"), (unkeyed, "carries no key")])
async def test_a_frame_no_hold_can_be_cut_from_fails_as_it_arrives_and_no_listener_hears_it(framed: Callable[[Gate], Frame], why: str) -> None:
    listener = Heeding("given")
    async with rigged(Scripted([]), (listener,)) as rig:
        await rig.run.worker.queue_frame(framed(rig.gate.after("start", "engaged conversation")))
        await rig.until(lambda: len(rig.run.errors) == 1)
        [error] = rig.run.errors
        assert why in error.error
        assert (listener.given, rig.seen.seen) == ([], [])


async def test_each_hold_is_one_event_sent_dropped_typed_and_failed() -> None:
    # The transcriber is given the spoken holds sent alone: the first, the fourth and the fifth.
    async with rigged(Scripted([None, RuntimeError("model failed"), "what time is it"])) as rig:
        await rig.held([frame_of(16384)], by="held key")
        await rig.held([frame_of(16384)], fate="dropped", by="held key")
        await rig.until(lambda: ("opened", 2) in rig.seen.seen)
        await rig.run.worker.queue_frame(Typed(text="typed"))
        await rig.until(lambda: ("opened", 3) in rig.seen.seen)
        await rig.held([frame_of(16384)], by="held key")
        await rig.held([frame_of(16384)], by="held key")
        await rig.until(lambda: len(rig.events("hold")) == 5)
        told = [(event.outcome, event.facts["hold"], event.facts["opener"], event.facts.get("fate"), event.facts.get("transcript")) for event in rig.events("hold")]
        assert told == [
            ("ok", 1, "held key", "sent", "none"),
            ("ok", 2, "held key", "dropped", None),
            ("ok", 3, "typed", None, "words"),
            ("ok", 4, "held key", "sent", "failed"),
            ("ok", 5, "held key", "sent", "words"),
        ]
        assert rig.events("hold")[0].facts["levels"] == Levels(-6.0, -6.0) and rig.events("hold")[0].facts["seconds"] == 0.02


async def test_holds_still_to_be_resolved_as_the_pipeline_stops_end_cancelled() -> None:
    go = asyncio.Event()
    async with rigged(Scripted([go, None])) as rig:
        await rig.held([frame_of(3)], by="held key")
        await rig.held([frame_of(3)], by="held key")
        await rig.until(lambda: ("opened", 2) in rig.seen.seen)
        await asyncio.sleep(0.05)
    assert [(event.outcome, event.facts["hold"], event.facts["fate"]) for event in rig.events("hold")] == [("cancelled", 1, "sent"), ("cancelled", 2, "sent")]


async def test_hearings_still_to_run_as_the_pipeline_stops_end_cancelled() -> None:
    stuck = asyncio.Event()

    async def waiting() -> Heard:
        await stuck.wait()
        return Heard(None)

    listener = Heeding("stuck", hearings=lambda _hold, _sound: (waiting,))
    async with rigged(Scripted([None]), (listener,)) as rig:
        down = rig.gate.after("start", "engaged conversation")
        await rig.frame(down, frame_of(3))
        await rig.frame(down, frame_of(3))
        await rig.until(lambda: len(listener.given) == 3)
        await asyncio.sleep(0.05)
    assert [(event.outcome, event.facts["listener"], event.facts["hold"]) for event in rig.events("hearing")] == [("cancelled", "stuck", 1)] * 2
