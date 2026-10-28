"""The user's turns through Pipecat's real turn machinery: key-tagged microphone frames in, what the model is sent out.

No keyboard and no microphone, so nothing anyone does at this computer can change a result. Each hold is a run of frames
tagged with the key they were captured under, as the keyed microphone makes them, and each transcription waits until the
test hands Whisper its text, so a hold that lands while another is still being transcribed is set up on purpose rather
than hoped for. Whisper, the user aggregator, and the stop strategy are the ones `build_voice` wires.
"""

import asyncio
import heapq
import itertools
import queue
import struct
import threading
from pathlib import Path
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Literal

import pytest
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import events, model_facts, patiently, running, unprimed
from hands.voice.backends import Account, ClaudeCodeBackend
from hands.sessions.audit import CutOff, Entry, HoldHeard, Levels, Replied, Transcribed, TurnStart, Unsaid, UserTurn
from hands.voice import transcription
from hands.sessions.wide import Fact, WideEvent
from hands.voice import pipeline as built
from hands.voice.conversation import AssistantTurns, UserTurns, UserWords, cue_receipt, record_turns
from hands.voice.floor import Floor
from hands.voice.latency import LatencyObserver
from hands.voice.mark import Mark
from hands.voice import player as played
from hands.voice.player import Player
from hands.voice.ptt import Gate, Key, KeyedAudio, PushToTalk
from hands.voice.trigger import Edge, place_of
from hands.core.effects import Asking, Narrate, SessionGone
from hands.core.pending import Finished, News
from hands.core.session import Held, Membership, Permission, RequestId, Running, Session, SessionId
from hands.core.status import Busy, Stamp
from hands.voice.speech import Aloud, Narrated, Unprompted
from hands.voice.holds import Hold
from hands.voice.turnstop import TurnOpened, TurnResolved, Typed
from hands.voice import desk, hold, holdowner, streaming
from hands.voice.desk import Act
from hands.voice.hold import Move as HoldMove
from hands.voice.hold import Idle, KeyEvent, Move, Pressed, Released
from hands.voice.holdowner import HoldOwner
from hands.voice.whisper import Whisper
from test_narrator import heard
from test_playback import ended, spoken
from hands.voice import voices

CLOSING = "that is all"

# Whisper's own transcribing of a hold, which the rig stands in for.
WHISPER_HEARD: object = vars(Whisper)["_heard"]

class Recorded(FrameProcessor):
    """What leaves the user aggregator for the model, the turns it ended, and the holds Whisper opened."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.sent: list[str] = []
        self.started = 0
        self.stopped = 0
        self.holds: list[int] = []
        self.resolved: list[int] = []
        # How many interruptions the user's turns broadcast to what stands behind them.
        self.interrupted = 0
        # How many holds have ended, sent or thrown away.
        self.released = 0
        # The user's turn and what hands said around it, in the order the model's stage would take them.
        self.order: list[str] = []
        # What hands handed the brain to say in its own words, in order.
        self.told: list[str] = []
        # How many frames the test's microphone captured have come through to here: it moves while the pipeline still
        # has that audio to work through, and stops once it has.
        self.audio = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case KeyedAudio():
                self.audio += 1
            case UserWords(words=words):
                self.sent.append(words)
                self.order.append(f"sent: {words}")
            case UserStartedSpeakingFrame():
                self.started += 1
                self.order.append("started")
            case UserStoppedSpeakingFrame():
                self.stopped += 1
                self.order.append("stopped")
            case TTSSpeakFrame(text=text) | Aloud(spoken=TTSSpeakFrame(text=text)):
                self.order.append(f"said: {text}")
            case Narrated(text=text):
                self.told.append(text)
                self.order.append(f"told: {text}")
            case TextFrame(text=text):
                self.order.append(text)
            case TurnResolved(hold=hold):
                self.resolved.append(hold.number)
            case TurnOpened(hold=hold):
                self.holds.append(hold.number)
            case VADUserStoppedSpeakingFrame():
                self.released += 1
            case InterruptionFrame():
                self.interrupted += 1
            case _:
                pass
        await self.push_frame(frame, direction)


async def until(out: Recorded, what: Callable[[], bool]) -> None:
    """Until `what` holds, moved by the captured audio coming through to `out`: three minutes of it take longer to come
    through on a loaded machine than a turn is given to end once it has. Nothing but the key's holds ends a turn, so a
    turn left open fails here instead of ending late."""
    await patiently(what, lambda: out.audio)


class NoSpeech(FrameProcessor):
    """Stands in for pocket-tts, which loads its weights when it is made."""

    Settings = built.PocketTTSService.Settings

    def __init__(self, **_: object) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)


@dataclass
class Streamed:
    """The streaming recognizer's session as the rig scripts it: each time it is fed, the next text the test said it
    decodes, speaker 0's unless it names its speaker, or the failure it raises, oldest first, and none once there is none;
    it decodes nothing while `flowing` is clear, so a test can hold the words back until the hold they were said in has
    ended."""

    said: "queue.SimpleQueue[str | streaming.Said | Exception]" = field(default_factory=queue.SimpleQueue[str | streaming.Said | Exception])
    flowing: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.flowing.set()

    def decoded(self, samples: streaming.Samples) -> streaming.Spoken:
        assert self.flowing.wait(timeout=60), "the test never let the words through"
        try:
            text = self.said.get_nowait()
        except queue.Empty:
            return ()
        match text:
            case Exception() as failure:
                raise failure
            case str():
                return (streaming.Said(0, text),)
            case streaming.Said():
                return (text,)

    def closed(self) -> streaming.Closed:
        """No stream to close: every word is decoded as it is said."""
        return streaming.Closed(frozenset(), ())


@dataclass
class Clock:
    """The floor's clock, read where the test sets it."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now


# A key position, or "dropped": the turn open thrown away, and the key at rest.
Captured = Key | Literal["dropped"]


@dataclass
class Rig:
    worker: PipelineWorker
    stt: HoldOwner
    out: Recorded
    recorded: list[Entry]
    clock: Clock
    # Each live session, as the floor reads it when it lets go.
    live: dict[SessionId, Session]
    # What each transcription, oldest first, will find; it waits for the test to say.
    texts: asyncio.Queue[str] = field(default_factory=asyncio.Queue[str])
    # The audio each transcription was given.
    heard: list[bytes] = field(default_factory=list[bytes])
    # Each turn hands said it received, as its words went to the brain.
    received: list[None] = field(default_factory=list[None])
    # Each mark the latency observer told of the turns, as the phone's page would be told.
    told: list[Mark] = field(default_factory=list[Mark])
    # Both sides of the conversation, as build_voice wires them.
    conversation: tuple[UserTurns, AssistantTurns] = field(kw_only=True)
    # Each time Whisper read the vocabulary, which is none.
    primed: list[None] = field(kw_only=True)
    # What the streaming recognizer hears in each hold while it is open.
    streamed: Streamed = field(kw_only=True)

    # The gate the last frame was captured under.
    gate: Gate = field(default_factory=Gate)

    async def hold(self, keys: Sequence[Captured], sound: bytes = b"\x00\x00" * 320, by: Edge = "held key", captured: bytes | None = None) -> None:
        """A frame captured under each key, by `by`'s microphone, counted as the gate counts them: the key leaving down
        for a rest, or for a press (one that finds the key down, its release gone by unseen), sends the turn, and for
        "dropped" throws it away; a hold it opens is opened by `by`. `captured` is the sound before the echo canceller,
        where the canceller changed it."""
        frames: list[KeyedAudio] = []
        for held in keys:
            match self.gate.key, held:
                case "down", "up" | "listening" | "arming":
                    self.gate = replace(self.gate, sent=self.gate.sent + 1)
                case "down", "dropped":
                    self.gate = replace(self.gate, dropped=self.gate.dropped + 1)
                case _:
                    pass
            self.gate = replace(self.gate, key="up" if held == "dropped" else held, opened=by)
            frames.append(self.gate.framed(sound, sound if captured is None else captured, 16000, 1, place_of(by)))
        await self.worker.queue_frames(frames)

    async def capture(self, gate: Gate, sound: bytes) -> None:
        """A frame captured under `gate`, as it stands after whatever moves made it."""
        self.gate = gate
        await self.worker.queue_frame(gate.framed(sound, sound, 16000, 1, gate.place))

    async def until(self, what: Callable[[], bool]) -> None:
        await until(self.out, what)

    async def everything_sent(self, holds: int) -> list[str]:
        """What the model was sent once the test's `holds` holds are all resolved, closed by one more spoken hold.

        Whether a hold joins the turn the one before it is still open in depends on whether its opening, a system
        frame, overtakes the last one's resolution, a data frame, so what is waited for is every hold resolved and
        every turn that took over ended, not a number of turns; a test with a turn that took nothing over, which says
        only that it stopped, waits for its line. The aggregator says a turn stopped, a system frame,
        ahead of what it sends the model, a data frame, which keeps its order among the data; so once the closing
        hold's words are in, so is everything sent before them.
        """
        out = self.out
        await self.until(lambda: len(out.holds) == len(out.resolved) == holds and out.started <= out.stopped)
        await self.hold(["down", "up"])
        await self.texts.put(CLOSING)
        await self.until(lambda: CLOSING in self.out.sent)
        return self.out.sent[:-1]


@pytest.fixture
async def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncGenerator[Rig, None]:
    async with rigged(monkeypatch, tmp_path, FrameProcessor(), []) as made:
        yield made


@asynccontextmanager
async def rigged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: FrameProcessor, behind: list[FrameProcessor], watching: Sequence[BaseObserver] = ()
) -> AsyncGenerator[Rig]:
    """The rig, with `behind` run behind what records the user's turns, and `watching` beside the latency observer."""
    monkeypatch.setattr(built, "PocketTTSService", NoSpeech)
    recorded: list[Entry] = []
    primed: list[None] = []
    streamed = Streamed()
    monkeypatch.setattr(streaming, "load", lambda: streamed)

    async def vocabulary() -> None:
        primed.append(None)

    voice = built.build_voice(
        built.VoiceConfig(llm=ClaudeCodeBackend(model="unused", config_dir=tmp_path, account=Account("claude.ai", None)), voice=voices.DEFAULT),
        llm=llm,
        key=PushToTalk(recorded.append),
        player=Player(recorded.append),
        floor=Floor(lambda id: id, dict),
        prompt=vocabulary,
        record=recorded.append,
    )
    out, clock, live = Recorded(), Clock(), dict[SessionId, Session]()
    texts: asyncio.Queue[str] = asyncio.Queue()
    heard: list[bytes] = []

    async def transcribe(_self: Whisper, hold: int, audio: bytes, _prompt: str | None) -> HoldHeard:
        # The samples the model would be given: what the microphone heard.
        heard.append(audio)
        return HoldHeard(hold, await texts.get() or None, ())

    monkeypatch.setattr(Whisper, "_heard", transcribe)
    # The floor sits where build_voice puts it, between the hold owner and the user aggregator.
    received: list[None] = []
    cue_receipt(voice.user_turns, lambda: received.append(None))
    told: list[Mark] = []
    # The assistant aggregator ends the pipeline, as build_voice puts it: a cut goes on once it has written what it cut off.
    stages = [voice.stt, Floor(lambda id: id, lambda: live, clock), voice.user_turns, out, *behind, voice.assistant_turns]
    async with running(stages, [LatencyObserver(told.append), *watching]) as run:
        yield Rig(run.worker, voice.stt, out, recorded, clock, live, texts, heard, received, told, conversation=(voice.user_turns, voice.assistant_turns), primed=primed, streamed=streamed)


async def test_pocket_tts_loading_as_the_voice_is_built_is_a_load_of_its_own(rig: Rig) -> None:
    [load] = events(rig.recorded, "tts.loaded")
    assert (load.outcome, model_facts(load)) == ("ok", {"voice": voices.DEFAULT})


async def test_a_spoken_hold_is_sent(rig: Rig) -> None:
    await rig.hold(["down", "down", "up"])
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]


def holds_heard(rig: Rig) -> list[HoldHeard]:
    return [entry for entry in rig.recorded if isinstance(entry, HoldHeard)]


def holds_told(rig: Rig) -> list[WideEvent]:
    """Each hold's `hold` event, as the owner resolved it."""
    return [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "hold"]


async def test_a_holds_event_says_how_loud_it_was_before_and_after_the_echo_canceller(rig: Rig) -> None:
    # Half of full scale captured, a tenth of that left by the canceller: -6.0 and -26.0 dBFS, 20 dB of echo taken out.
    loud, residual = struct.pack("<h", 16384) * 320, struct.pack("<h", 1638) * 320
    # What the key was up over is no part of the hold, however loud: the levels are of the audio transcribed alone.
    await rig.hold(["up", "arming", "down", "down", "up"], sound=residual, captured=loud)
    await rig.texts.put("run")
    await rig.until(lambda: len(holds_told(rig)) == 1)
    [told] = holds_told(rig)
    assert (told.facts["hold"], told.facts["fate"], told.facts["levels"], told.facts["transcript"]) == (1, "sent", Levels(captured_dbfs=-6.0, heard_dbfs=-26.0), "words")
    assert holds_heard(rig) == [HoldHeard(1, "run", ())]
    # The event is of the audio Whisper was given: the three frames from the press on.
    assert rig.heard[0] == residual * 3


async def test_a_hold_of_digital_silence_has_no_level_and_a_hold_at_the_phone_has_one_level(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.texts.put("")
    await rig.until(lambda: len(holds_told(rig)) == 1)
    quiet = struct.pack("<h", 3277) * 320  # a tenth of full scale
    await rig.hold(["down", "up"], sound=quiet, by="phone button")
    await rig.texts.put("hello")
    await rig.until(lambda: len(holds_told(rig)) == 2)
    assert [told.facts["levels"] for told in holds_told(rig)] == [Levels(None, None), Levels(-20.0, -20.0)]


async def test_a_turn_is_received_once_its_words_are_written_and_a_turn_with_none_is_not(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: rig.out.sent == ["what time is it"])
    await rig.until(lambda: len(rig.received) == 1)
    # Heard nothing, then dropped: the key ended both, and neither reached the model.
    await rig.hold(["down", "up"])
    await rig.texts.put("")
    await rig.hold(["down", "down", "dropped"])
    assert await rig.everything_sent(holds=3) == ["what time is it"]
    await rig.until(lambda: len(rig.received) == 2)  # the closing hold's
    await asyncio.sleep(0.05)
    assert len(rig.received) == 2


async def test_a_hold_hears_from_its_press_and_nothing_before_it(rig: Rig) -> None:
    shift, early, held = (bytes([n, n]) * 320 for n in (1, 2, 3))
    # A press that was Shift after all, then one said into before it meant talk.
    await rig.hold(["arming"], sound=shift)
    await rig.hold(["up"])
    await rig.hold(["arming", "arming"], sound=early)
    await rig.hold(["down", "up"], sound=held)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(early + early + held)
    assert shift not in rig.heard[0]


async def test_a_hold_keeps_everything_from_its_press_however_long_it_takes_to_mean_talk(rig: Rig) -> None:
    # Pipecat keeps the last second of audio nobody is speaking in; 60 frames of 20 ms arm for 1.2 s.
    early, held = (bytes([n, n]) * 320 for n in (2, 3))
    arming: list[Key] = ["arming"] * 60
    await rig.hold(arming, sound=early)
    await rig.hold(["down", "up"], sound=held)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(early * 60 + held)


async def test_a_press_that_was_shift_leaves_nothing_behind_for_the_next_hold(rig: Rig) -> None:
    shift, held = (bytes([n, n]) * 320 for n in (1, 3))
    arming: list[Key] = ["arming"] * 60
    up: list[Key] = ["up"] * 60
    await rig.hold(arming, sound=shift)
    await rig.hold(up)
    await rig.hold(["arming", "down", "up"], sound=held)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert shift not in rig.heard[0]
    assert rig.heard[0].startswith(held * 2)


async def test_an_engaged_turn_keeps_the_second_the_desk_heard_before_it_and_listens_on_after(rig: Rig) -> None:
    # Pipecat keeps the last second of audio nobody is speaking in: 50 frames of 20 ms. The desk listens for 1.6 s
    # while the room is quiet, then the user's first word comes as the detector makes sure of it.
    room, onset, said, after = (bytes([n, n]) * 320 for n in (1, 2, 3, 4))
    listening: list[Key] = ["listening"]
    await rig.hold(listening * 70, sound=room)
    await rig.hold(listening * 5, sound=onset)
    await rig.hold(["arming", "down", "down"], sound=said)
    await rig.hold(["listening"], sound=after)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(room * 45 + onset * 5 + said * 3)
    assert after not in rig.heard[0]


async def test_an_engaged_start_that_was_only_noise_opens_no_turn_and_the_desk_listens_on(rig: Rig) -> None:
    noise, said = (bytes([n, n]) * 320 for n in (1, 3))
    listening: list[Key] = ["listening"]
    await rig.hold(["listening", "arming", "arming", "listening"], sound=noise)
    await rig.hold(listening * 60)
    await rig.hold(["arming", "down", "listening"], sound=said)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert noise not in rig.heard[0]


async def test_a_phone_turn_opened_while_the_desk_listens_hears_nothing_the_desk_heard(rig: Rig) -> None:
    room, said = (bytes([n, n]) * 320 for n in (1, 3))
    listening: list[Key] = ["listening"]
    await rig.hold(listening * 20, sound=room)
    await rig.hold(["down", "down", "up"], sound=said, by="phone button")
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert room not in rig.heard[0]


async def test_a_turn_ended_and_the_next_armed_between_two_frames_is_sent(rig: Rig) -> None:
    # The verdict ends an engaged turn and the voice arms the next before the microphone captures a frame at rest.
    said, more = (bytes([n, n]) * 320 for n in (3, 5))
    turn = Gate().after("listen", "engaged conversation").after("arm", "engaged conversation").after("start", "engaged conversation")
    for _ in range(3):
        await rig.capture(turn, said)
    await rig.capture(turn.after("stop", "engaged conversation").after("arm", "engaged conversation"), more)
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.heard[0].startswith(said * 3) and more not in rig.heard[0]


class Speaker(FrameProcessor):
    """Stands in for the output transport: a mark waits here, as behind audio still playing, until a barge-in drops it."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.playing: list[played.Mark] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case played.Mark():
                self.playing.append(frame)
            case InterruptionFrame():
                self.playing.clear()
                await self.push_frame(frame, direction)
            case _:
                await self.push_frame(frame, direction)


@asynccontextmanager
async def reading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncGenerator[tuple[Rig, list[Entry], "asyncio.Task[bool]"]]:
    """The rig with the player behind it, as the pipeline stands it ahead of the speaker, partway through reading a reply,
    and a line said through `heard` waiting on the speaker: what the player records, and that line's waiter."""
    cut: list[Entry] = []
    player, speaker, output = Player(cut.append), IdentityFilter(), Speaker()
    async with rigged(monkeypatch, tmp_path, FrameProcessor(), [player.lines, speaker, output], [player.watching(speaker, output)]) as rig:
        await rig.worker.queue_frames([spoken("The parser is fixed."), spoken("Its tests pass.")])
        line = asyncio.create_task(player.heard("Shall I push it?"))
        try:
            await rig.until(lambda: len(output.playing) == 1)
            yield rig, cut, line
        finally:
            line.cancel()


# Hands' own reply heard back through the echo canceller, or someone speaking: either way the desk's detector opens it.
OPENED_BY_THE_VOICE: list[Captured] = ["listening", "arming", "down", "down", "listening"]


def user_turns(rig: Rig) -> list[tuple[TurnStart, bool]]:
    """Each user turn's line, as it ended: when its edge had it cut, and whether it did."""
    return [(entry.start, entry.cut is not None) for entry in rig.recorded if isinstance(entry, UserTurn)]


async def test_a_hold_the_voice_opened_that_heard_no_words_cuts_nothing_off_and_is_a_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        await rig.hold(OPENED_BY_THE_VOICE, by="engaged conversation")
        await rig.texts.put("")
        await rig.until(lambda: user_turns(rig) == [("on words", False)])
        # The turn took the floor and nothing over: nothing behind it heard the user start, so the model's reply streams
        # on and runs on a tool's result, the reading plays on, and the line waits.
        assert (rig.out.started, rig.out.stopped, rig.out.interrupted, cut, line.done()) == (0, 1, 0, [], False)
        assert await rig.everything_sent(holds=1) == []
        # The closing hold is the held key's, which cut the reading at once.
        assert user_turns(rig) == [("on words", False), ("on the hold", True)]


async def test_a_hold_the_voice_opened_cuts_hands_off_once_whisper_hears_words_in_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        await rig.hold(OPENED_BY_THE_VOICE, by="engaged conversation")
        await rig.until(lambda: rig.out.released == 1)
        # The turn is the user's from the hold's opening, but it takes nothing over until its words are heard.
        assert (rig.out.started, rig.out.interrupted, cut, line.done()) == (0, 0, [], False)
        await rig.texts.put("wait, not yet")
        assert await line is False
        assert cut == [CutOff("The parser is fixed.", 1)] and (rig.out.started, rig.out.interrupted) == (1, 1)
        assert await rig.everything_sent(holds=1) == ["wait, not yet"]
        assert user_turns(rig)[0] == ("on words", True)


# A second of the user speaking, in 20 ms frames: the diarizer's first step of the run commits 0.33 s in and one every
# 0.24 s after, so three of the streaming recognizer's hearings.
SPEAKING: list[Captured] = ["down"] * 50


def hearings(rig: Rig) -> list[WideEvent]:
    """The event of each of the streaming recognizer's hearings, in order."""
    return [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "hearing" and entry.facts["listener"] == "streaming"]


def overheard(rig: Rig) -> list[Fact]:
    """What came of each hearing, in order."""
    return [event.facts["heard"] for event in hearings(rig)]


async def test_words_said_over_hands_in_a_hold_the_voice_opened_cut_it_off_while_the_user_is_still_speaking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # As the audit log had it (hands-voice-o6o): engaged conversation, hands reading a reply, and the user speaking over
    # it for seconds; hands was cut off only once end-of-turn detection closed the hold and Whisper heard it whole.
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        rig.streamed.said.put("wait, not")
        await rig.hold(["listening", "arming", *SPEAKING], by="engaged conversation")
        await rig.until(line.done)
        # Cut while the hold is still open: the user has not stopped speaking.
        assert (line.result(), rig.out.released) == (False, 0)
        assert cut == [CutOff("The parser is fixed.", 1)] and (rig.out.started, rig.out.interrupted) == (1, 1)
        await rig.hold([*SPEAKING, "listening"], by="engaged conversation")
        await rig.texts.put("wait, not yet, read me the diff first")
        # What was heard while the hold was open only cut; the hold heard whole by the transcriber is what is sent.
        assert await rig.everything_sent(holds=1) == ["wait, not yet, read me the diff first"]
        assert user_turns(rig)[0] == ("on words", True)
        # What cut hands off is on the hearing's event, to be looked into where the hold heard whole says nothing.
        [first, *_] = hearings(rig)
        assert (first.facts["hold"], first.facts["heard"], first.facts["cut"], first.facts["added"]) == (1, "words", True, (streaming.Said(0, "wait, not"),))


async def test_a_hold_the_voice_opened_cuts_nothing_off_where_no_words_are_heard_in_it_while_it_is_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Hands' own reply heard back through the canceller as no word, or a cough, for as long as a user's sentence.
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        await rig.hold(["listening", "arming", *SPEAKING, "listening"], by="engaged conversation")
        await rig.texts.put("")
        await rig.until(lambda: user_turns(rig) == [("on words", False)])
        assert set(overheard(rig)) == {"nothing"}
        assert (rig.out.started, rig.out.interrupted, cut, line.done()) == (0, 0, [], False)


async def test_a_hold_the_voice_opened_dropped_before_its_words_are_heard_cuts_nothing_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        # The model is behind: what was said in the hold is decoded only once it has been thrown away.
        rig.streamed.flowing.clear()
        await rig.hold(["listening", "arming", *SPEAKING, "dropped"], by="engaged conversation")
        await rig.until(lambda: rig.out.released == 1)
        rig.streamed.said.put("stop")
        rig.streamed.flowing.set()
        await rig.until(lambda: len(hearings(rig)) == 4)
        # What it heard in a hold thrown away is no one's, and cuts nothing.
        assert (overheard(rig)[0], {event.facts["cut"] for event in hearings(rig)}) == ("words", {False})
        await rig.until(lambda: user_turns(rig) == [("on words", False)])
        assert (rig.out.started, rig.out.interrupted, cut, line.done()) == (0, 0, [], False)
        assert await rig.everything_sent(holds=1) == []


async def test_a_hold_the_voice_opened_is_heard_open_and_only_its_transcription_reads_the_vocabulary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Hands' echo, or a cough, keeping a hold open: it is heard at every chunk the model decodes, which no vocabulary primes.
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        await rig.hold(["listening", "arming", *SPEAKING, "listening"], by="engaged conversation")
        await rig.texts.put("")
        await rig.until(lambda: user_turns(rig) == [("on words", False)])
        await rig.until(lambda: len(hearings(rig)) == 4)
        # Three while it is open, for the 1.04 s it gathered, a step of the diarizer's committing 0.33 s in and every
        # 0.24 s after; one as it ends.
        assert [event.facts.get("fate") for event in hearings(rig)] == [None] * 3 + ["sent"]
        assert len(rig.primed) == 1
        assert (rig.out.started, rig.out.interrupted, cut, line.done()) == (0, 0, [], False)


async def test_a_hold_the_voice_opened_not_heard_while_open_cuts_on_the_words_it_is_heard_whole_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        rig.streamed.said.put(RuntimeError("Metal ran out of memory"))
        await rig.hold(["listening", "arming", *SPEAKING], by="engaged conversation")
        await rig.until(lambda: len(hearings(rig)) == 3)
        # Its words are not known, so nothing heard while it is open cuts.
        assert (set(overheard(rig)), line.done()) == ({"nothing"}, False)
        # Heard whole as it closes, and cut on its words then; its end says why it was not heard open.
        await rig.hold([*SPEAKING, "listening"], by="engaged conversation")
        await rig.texts.put("wait, not yet")
        await rig.until(line.done)
        assert line.result() is False
        assert cut == [CutOff("The parser is fixed.", 1)]
        await rig.until(lambda: user_turns(rig) == [("on words", True)])
        *_, whole = hearings(rig)
        assert (whole.outcome, whole.error, whole.facts["hold"]) == ("failed", "RuntimeError: decoding failed: RuntimeError: Metal ran out of memory", 1)


async def test_a_turn_the_voice_opened_cuts_once_for_all_its_holds_words_and_ends_once_every_hold_resolves(rig: Rig) -> None:
    await rig.hold(OPENED_BY_THE_VOICE, by="engaged conversation")
    await rig.hold(OPENED_BY_THE_VOICE, by="engaged conversation")
    await rig.until(lambda: rig.out.released == 2)
    await rig.texts.put("turn it off")
    await rig.texts.put("and the lights")
    # The cut lands behind the first hold's words, ahead of the second's: neither is dropped by it.
    assert await rig.everything_sent(holds=2) == ["turn it off and the lights"]
    assert rig.out.interrupted == 2  # the closing hold's is the second
    assert user_turns(rig)[0] == ("on words", True)


async def test_a_held_key_pressed_in_a_turn_the_voice_opened_cuts_at_once(rig: Rig) -> None:
    await rig.hold(OPENED_BY_THE_VOICE, by="engaged conversation")
    await rig.until(lambda: rig.out.released == 1)
    await rig.hold(["down", "down", "up"])
    await rig.until(lambda: rig.out.interrupted == 1)
    await rig.texts.put("")
    await rig.texts.put("stop")
    assert await rig.everything_sent(holds=2) == ["stop"]
    assert user_turns(rig)[0] == ("on words", True)


@pytest.mark.parametrize(("by", "held"), [("held key", ["down", "down", "up"]), ("engaged conversation", OPENED_BY_THE_VOICE)])
async def test_the_reply_a_turn_cut_off_is_written_ahead_of_the_words_that_cut_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, by: Edge, held: list[Captured]
) -> None:
    # The pipeline's stages between the user's turn and the assistant aggregator, which writes the reply as cut off.
    stages: list[FrameProcessor] = [IdentityFilter() for _ in range(8)]
    async with rigged(monkeypatch, tmp_path, FrameProcessor(), stages) as rig:
        written: list[Entry] = []
        record_turns(*rig.conversation, written.append)
        await rig.worker.queue_frames([LLMFullResponseStartFrame(), ended("The parser is fixed.")])
        await rig.until(lambda: "The parser is fixed." in rig.out.order)
        await rig.hold(held, by=by)
        await rig.until(lambda: rig.out.released == 1)
        # A turn the voice opened cuts as its words arrive, a few stages ahead of them; the audit log still reads in order.
        await rig.texts.put("wait, not yet")
        await rig.until(lambda: len(written) == 2)
        assert written == [Replied("The parser is fixed.", interrupted=True), Transcribed("wait, not yet")]


@pytest.mark.parametrize("by", ["held key", "phone button", "wake word"])
async def test_a_hold_the_user_opened_on_purpose_cuts_hands_off_at_once_with_or_without_words(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, by: Edge) -> None:
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        await rig.hold(["down", "down", "up"], by=by)
        # Cut before Whisper has heard anything: its transcription waits on the test.
        assert await line is False
        assert cut == [CutOff("The parser is fixed.", 1)]
        await rig.texts.put("")
        assert await rig.everything_sent(holds=1) == []
        assert user_turns(rig)[0] == ("on the hold", True)


async def test_words_typed_are_sent_as_a_turn_of_their_own(rig: Rig) -> None:
    await rig.worker.queue_frame(Typed(text="what time is it"))
    assert await rig.everything_sent(holds=1) == ["what time is it"]
    assert rig.received == [None, None]


async def test_words_typed_cut_hands_off_at_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with reading(monkeypatch, tmp_path) as (rig, cut, line):
        await rig.worker.queue_frame(Typed(text="stop there"))
        assert await line is False
        assert cut == [CutOff("The parser is fixed.", 1)]
        assert await rig.everything_sent(holds=1) == ["stop there"]
        assert user_turns(rig)[0] == ("on the hold", True)


async def test_words_typed_while_a_hold_is_open_join_its_turn_and_leave_its_audio_to_it(rig: Rig) -> None:
    await rig.hold(["down", "down"], sound=b"\x01\x00" * 320)
    await rig.until(lambda: rig.out.holds == [1])
    await rig.worker.queue_frame(Typed(text="and the tests"))
    await rig.until(lambda: rig.out.resolved == [2])
    await rig.hold(["up"], sound=b"\x01\x00" * 320)
    await rig.texts.put("run the parser")
    assert await rig.everything_sent(holds=2) == ["and the tests run the parser"]
    # The key's hold is heard from its own audio: both frames captured while it was down.
    assert rig.heard[0].startswith(b"\x01\x00" * 320 * 2)


async def test_words_typed_while_a_hold_is_being_transcribed_come_after_its_words(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.until(lambda: rig.out.holds == [1])
    await rig.worker.queue_frame(Typed(text="and the tests"))
    await rig.until(lambda: rig.out.holds == [1, 2])
    await rig.texts.put("run the parser")
    assert await rig.everything_sent(holds=2) == ["run the parser and the tests"]


async def test_a_dropped_hold_ends_its_turn_and_sends_nothing(rig: Rig) -> None:
    await rig.hold(["down", "down", "dropped"])
    assert await rig.everything_sent(holds=1) == []


async def test_a_hold_that_heard_nothing_ends_its_turn_and_sends_nothing(rig: Rig) -> None:
    await rig.hold(["down", "up"])
    await rig.texts.put("")
    assert await rig.everything_sent(holds=1) == []


async def test_a_hold_a_press_finds_open_is_sent_and_the_press_is_a_hold_of_its_own(rig: Rig) -> None:
    # A press that finds the key down sends the hold, its release gone by unseen, and arms again between two frames, so
    # no frame sees the key at rest.
    await rig.hold(["down", "arming", "down", "up"])
    await rig.texts.put("what time is it")
    await rig.texts.put("and the date")
    assert " ".join(await rig.everything_sent(holds=2)) == "what time is it and the date"


# 20 ms of speech at 16 kHz, as the keyed microphone captures it.
FRAME_SECONDS = 0.02
SPEECH = bytes([3, 3]) * 320


@dataclass
class Timeline:
    """What a timeline captured: each frame, as the gate tagged it, and every move the edge made."""

    frames: list[KeyedAudio]
    moves: list[Move]

    def held(self) -> int:
        """How many frames were captured with the turn open or its press arming: the audio the turn is made of."""
        return sum(frame.key in ("arming", "down") for frame in self.frames)


def play_out[S, E](
    step: Callable[[S, E], tuple[S, tuple[Act | HoldMove, ...]]],
    state: S,
    asked: Callable[[S, E, S], tuple[tuple[float, E], ...]],
    told: Sequence[tuple[float, E]],
    seconds: float,
    by: Edge,
) -> Timeline:
    """Speech captured every 20 ms for `seconds`, under the gate as the edge's own `step` moves it: each event of `told`,
    and each that a step `asked` to be told later, stepped at its instant, as the loop delivers them. No keyboard, no
    microphone and no clock: hours play out in the time it takes to step them."""
    # In order of instant, then of arrival: the loop's own order for two calls due at once.
    arrival = itertools.count()
    due: list[tuple[float, int, E]] = [(at, next(arrival), event) for at, event in told]
    heapq.heapify(due)
    gate, frames, moves = Gate(), list[KeyedAudio](), list[Move]()
    for n in range(round(seconds / FRAME_SECONDS)):
        while due and due[0][0] <= n * FRAME_SECONDS:
            _, _, event = heapq.heappop(due)
            was = state
            state, acts = step(state, event)
            for at, later in asked(was, event, state):
                heapq.heappush(due, (at, next(arrival), later))
            for act in acts:
                match act:
                    case "judge" | desk.Leveled():
                        pass
                    case desk.Made(move=move, by=edge):
                        moves.append(move)
                        gate = gate.after(move, edge)
                    case move:
                        moves.append(move)
                        gate = gate.after(move, by)
        frames.append(gate.framed(SPEECH, SPEECH, 16000, 1, place_of(by)))
    return Timeline(frames, moves)


def held_key(told: Sequence[tuple[float, KeyEvent]], seconds: float) -> Timeline:
    """The talk key's own hold, told `told`, with its press's Ripe and Overlong told as `keys.tapped` tells them."""
    return play_out(hold.step, Idle(), lambda _was, event, _now: hold.asked(event), told, seconds, "held key")


def engaged_conversation(told: Sequence[tuple[float, desk.Event]], seconds: float) -> Timeline:
    """The desk, told `told`, with its key's presses told as `keys.tapped` tells them and each turn's limit as
    `engaged.drive` tells it."""

    def asked(was: desk.Desk, event: desk.Event, now: desk.Desk) -> tuple[tuple[float, desk.Event], ...]:
        return (*(hold.asked(event) if isinstance(event, KeyEvent) else ()), *desk.deadline(was.phase, now.phase))

    return play_out(desk.step, desk.Desk(), asked, told, seconds, "engaged conversation")


async def test_a_held_key_turn_of_three_minutes_of_speech_reaches_the_brain_whole_as_one_turn(rig: Rig) -> None:
    timeline = held_key([(0.0, Pressed(0.0)), (180.0, Released())], 181.0)
    assert timeline.moves == ["arm", "start", "stop"]
    await rig.worker.queue_frames(timeline.frames)
    await rig.texts.put("three minutes of dictation")
    assert await rig.everything_sent(holds=1) == ["three minutes of dictation"]
    # Every frame from the press to the release, then Pipecat's silence: 180 s of it, none cut.
    assert timeline.held() == 9000
    assert rig.heard[0].rstrip(b"\x00") == SPEECH * timeline.held()


async def test_an_engaged_turn_of_three_minutes_of_speech_reaches_the_brain_whole_as_one_turn(rig: Rig) -> None:
    engage = [(1.0, desk.Chosen("engaged"))]
    timeline = engaged_conversation([*engage, (2.0, desk.SpeechStarted(2.0)), (182.0, desk.TurnEnded("verdict"))], 183.0)
    assert timeline.moves == ["listen", "arm", "start", "stop"]
    await rig.worker.queue_frames(timeline.frames)
    await rig.texts.put("three minutes of conversation")
    assert await rig.everything_sent(holds=1) == ["three minutes of conversation"]
    # The turn is all 180 s it was open, after the second the desk heard before it.
    assert timeline.held() == 9000
    assert rig.heard[0].rstrip(b"\x00") == SPEECH * (timeline.held() + 50)


async def test_a_turn_whose_release_never_comes_is_sent_at_the_limit_with_all_it_recorded(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.hold.TURN_LIMIT_SECONDS", 3.0)
    # Pressed, and never released: a key stuck down, or a release macOS never reported.
    timeline = held_key([(0.0, Pressed(0.0))], 5.0)
    assert timeline.moves == ["arm", "start", "timeout"]
    await rig.worker.queue_frames(timeline.frames)
    await rig.texts.put("what I said before the limit")
    assert await rig.everything_sent(holds=1) == ["what I said before the limit"]
    # The three seconds up to the limit, every one; after it the microphone is closed, and the room is no part of it.
    assert timeline.held() == 150
    assert rig.heard[0].rstrip(b"\x00") == SPEECH * timeline.held()
    assert timeline.frames[-1].audio == bytes(len(SPEECH))


async def test_an_engaged_turn_that_never_ends_is_sent_at_the_limit_with_all_it_recorded(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    # The limit as the desk schedules it: desk.deadline reads its own name for it.
    monkeypatch.setattr("hands.voice.desk.TURN_LIMIT_SECONDS", 3.0)
    engage = [(1.0, desk.Chosen("engaged"))]
    # Speech that never stops and is never judged ended.
    timeline = engaged_conversation([*engage, (2.0, desk.SpeechStarted(2.0))], 6.0)
    assert timeline.moves == ["listen", "arm", "start", "timeout"]
    await rig.worker.queue_frames(timeline.frames)
    await rig.texts.put("what I said before the limit")
    assert await rig.everything_sent(holds=1) == ["what I said before the limit"]
    # The three seconds up to the limit, every one, after the second the desk heard before it.
    assert timeline.held() == 150
    assert rig.heard[0].rstrip(b"\x00") == SPEECH * (timeline.held() + 50)


async def test_a_hold_dropped_while_the_last_is_transcribed_leaves_the_last_to_be_sent(rig: Rig) -> None:
    # The second press joins the turn the first is still open in; its drop must not end that turn before the first
    # hold's words are in it.
    await rig.hold(["down", "down", "up", "down", "dropped"])
    await rig.until(lambda: rig.out.holds == [1, 2])
    await rig.texts.put("what time is it")
    assert await rig.everything_sent(holds=2) == ["what time is it"]


async def test_a_hold_that_heard_nothing_does_not_end_the_hold_pressed_after_it(rig: Rig) -> None:
    await rig.hold(["down", "up", "down", "down"])
    await rig.until(lambda: rig.out.holds == [1, 2])
    # The first hold's empty transcription arrives while the second is still held.
    await rig.texts.put("")
    await rig.hold(["up"])
    await rig.texts.put("how many sessions are running")
    assert await rig.everything_sent(holds=2) == ["how many sessions are running"]


async def held(rig: Rig, hold: int) -> None:
    """A hold as a hand makes it: the key down until its press has crossed the pipeline, then up until its release has.

    The latency observer takes a press and a release by the user going from one to the other, so a release queued in
    the same burst as its press, as the other tests here queue them, reaches it between two pushes of that press and
    reads as a second hold."""
    await rig.hold(["down"])
    await rig.until(lambda: rig.out.holds[-1:] == [hold])
    await rig.hold(["up"])
    await rig.until(lambda: rig.out.released == hold)


async def test_a_turn_with_no_words_is_told_as_that_once_it_ends(rig: Rig) -> None:
    await held(rig, 1)
    await rig.texts.put("")
    await rig.until(lambda: rig.out.stopped == 1)
    assert rig.told == ["released", "no words"]


async def test_a_silent_hold_in_a_turn_that_has_words_is_not_told_as_no_words(rig: Rig) -> None:
    await held(rig, 1)
    await rig.hold(["down"])
    await rig.until(lambda: rig.out.holds == [1, 2])
    # The first hold's words arrive while the second is still held, so both are one turn.
    await rig.texts.put("what time is it")
    await rig.until(lambda: rig.out.resolved == [1])
    await rig.hold(["up"])
    await rig.texts.put("")
    await rig.until(lambda: rig.out.sent == ["what time is it"])
    assert rig.told == ["released", "transcript", "released"]


async def test_a_hold_whisper_is_slow_to_transcribe_is_sent_with_its_own_turn(rig: Rig) -> None:
    await held(rig, 1)
    # Longer than the 5 s after which Pipecat's user aggregator, left to its default, ends a turn on its own.
    await asyncio.sleep(5.6)
    assert rig.out.stopped == 0
    await rig.texts.put("what time is it")
    await rig.until(lambda: rig.out.sent == ["what time is it"])
    assert rig.out.stopped == 1
    assert rig.told == ["released", "transcript"]


async def test_a_hold_whisper_could_not_transcribe_is_told_as_failed_and_not_as_no_words(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    async def refused(_self: Whisper, hold: int, audio: bytes, prompt: str | None) -> HoldHeard:
        raise ConnectionError("LowTalker is not serving")

    monkeypatch.setattr(Whisper, "_heard", refused)
    await held(rig, 1)
    await rig.until(lambda: rig.out.stopped == 1)
    assert rig.told == ["released", "failed"]


async def test_a_hold_whisper_never_finishes_transcribing_fails_and_ends_its_turn(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    async def hung(_self: Whisper, hold: int, audio: bytes, prompt: str | None) -> HoldHeard:
        await asyncio.Event().wait()
        raise AssertionError("a transcription that never returns returned")

    monkeypatch.setattr(Whisper, "_heard", hung)
    monkeypatch.setattr(holdowner, "TRANSCRIBING_SECONDS", 0.2)
    await held(rig, 1)
    await rig.until(lambda: rig.out.stopped == 1)
    assert rig.told == ["released", "failed"]


async def test_a_long_turn_has_as_long_as_it_ran_to_be_transcribed(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    # The least a transcription is given, made far shorter than this turn's transcription takes.
    monkeypatch.setattr(holdowner, "TRANSCRIBING_SECONDS", 0.2)
    timeline = held_key([(0.0, Pressed(0.0)), (180.0, Released())], 181.0)
    await rig.worker.queue_frames(timeline.frames)
    await rig.until(lambda: len(rig.heard) == 1)
    await asyncio.sleep(0.5)
    await rig.texts.put("three minutes of dictation")
    assert await rig.everything_sent(holds=1) == ["three minutes of dictation"]


async def test_a_transcription_given_up_on_is_not_run_beside_the_next(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    returns, begun = threading.Event(), list[None]()

    def stuck(samples: bytes, prompt: str | None) -> list[Unsaid]:
        begun.append(None)
        returns.wait()
        return []

    # Whisper as it is, down to the model: the rig's stand-in is put aside.
    monkeypatch.setattr(Whisper, "_heard", WHISPER_HEARD)
    monkeypatch.setattr(transcription, "segments", stuck)
    monkeypatch.setattr(holdowner, "TRANSCRIBING_SECONDS", 0.2)
    try:
        await held(rig, 1)
        await rig.until(lambda: rig.out.stopped == 1)
        await held(rig, 2)
        await rig.until(lambda: rig.out.stopped == 2)
        assert rig.told == ["released", "failed", "released", "failed"]
        # The second hold's turn failed in its own time without the model being run on it beside the first.
        assert len(begun) == 1
    finally:
        returns.set()


API, WEB = SessionId("api"), SessionId("web")
def waiting() -> Unprompted:
    return Unprompted(SessionGone(API), (heard(),))


def asking(session: SessionId, request: str) -> Unprompted:
    return Unprompted(Narrate(Asking(session, RequestId(request), Permission("Bash", {"command": "ls"}))), (heard(),))


def finished(session: SessionId, reply: str) -> Unprompted:
    return Unprompted(Finished(session, (News(None, reply, "", "", (), frozenset()),), "full"), (heard(),))


def let_go(unprompted: Unprompted) -> dict[str, Fact]:
    """What the floor found of the one thing hands heard that `unprompted` tells, and its fate where the floor settled it."""
    [utterance] = unprompted.utterances
    settled: dict[str, Fact] = {"fate": utterance.settled.result()} if utterance.settled.done() else {}
    return {key: value for key, value in utterance.facts.items() if key in ("held_ms", "told", "folded")} | settled


# A hold the held key opens and lets go of, and one the voice opens and its end of turn closes.
HOLDS: list[tuple[Edge, list[Captured], list[Captured]]] = [
    ("held key", ["down", "down"], ["up"]),
    ("engaged conversation", ["listening", "arming", "down", "down"], ["listening"]),
]


@pytest.mark.parametrize(("by", "held", "let_go_of", "order"), [
    # The held key's turn takes over as it opens; the voice's only once its words are heard, so after the floor held.
    (*HOLDS[0], ["started", "marker", "stopped"]),
    (*HOLDS[1], ["marker", "started", "stopped"]),
])
async def test_a_session_waiting_while_the_key_is_held_is_said_after_the_users_turn_is_sent_and_not_before(
    rig: Rig, by: Edge, held: list[Captured], let_go_of: list[Captured], order: list[str]
) -> None:
    rig.clock.now = 10.0
    # A turn the voice opens holds the floor from its opening, though it takes nothing over until its words are heard.
    await rig.hold(held, by=by)
    await rig.until(lambda: rig.out.holds == [1])
    gone = waiting()
    await rig.worker.queue_frames([gone, TextFrame("marker")])
    # Frames keep their order, so the marker past the floor with the announcement not is the announcement held.
    await rig.until(lambda: "marker" in rig.out.order)
    rig.clock.now = 13.5
    await rig.hold(let_go_of, by=by)
    await rig.texts.put("what time is it")
    await rig.until(lambda: "said: The session api is gone." in rig.out.order)
    assert rig.out.order == [*order, "sent: what time is it", "said: The session api is gone."]
    assert let_go(gone) == {"held_ms": 3500.0, "told": "SessionGone", "folded": 1}


async def test_a_press_that_passes_the_floor_ahead_of_the_last_turns_stop_keeps_the_floor_until_its_own_turn_stops() -> None:
    """The stop comes back up from the user aggregator while a press goes down past the floor, so the press can arrive
    first: the floor stays the user's through the turn that press opened."""
    out = Recorded()
    one, two = Hold(1, "held key"), Hold(2, "engaged conversation")
    async with running([Floor(lambda id: id, dict), out], []) as run:
        await run.worker.queue_frames([TurnOpened(hold=one), TurnResolved(hold=one)])
        await until(out, lambda: out.resolved == [1])
        # The aggregator stops the first turn behind its hold's resolution, as the second hold opens.
        await run.worker.queue_frames([TurnOpened(hold=two), UserStoppedSpeakingFrame(), waiting(), TextFrame("marker")])
        await until(out, lambda: "marker" in out.order)
        assert not any(each.startswith("said:") for each in out.order)
        await run.worker.queue_frame(TurnResolved(hold=two))
        await until(out, lambda: out.resolved == [1, 2])
        await run.worker.queue_frame(UserStoppedSpeakingFrame())
        await until(out, lambda: "said: The session api is gone." in out.order)
        assert out.order == ["stopped", "marker", "stopped", "said: The session api is gone."]


def waiting_on(session: SessionId, request: str) -> Session:
    """The session live, its dialog waiting on the user to answer `request`."""
    member = Membership(session, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl"))
    dialog = Held(Permission("Bash", {"command": "ls"}), RequestId(request))
    return Session(member, Running(Busy(), Stamp(1000), None), mode=None, dialogs=(dialog,))


async def test_a_request_heard_while_the_key_is_held_is_handed_the_brain_after_the_users_turn(rig: Rig) -> None:
    rig.live[API] = waiting_on(API, "r1")
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    await rig.worker.queue_frames([asking(API, "r1"), TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: len(rig.out.order) == 5)
    assert rig.out.order[:4] == ["started", "marker", "stopped", "sent: what time is it"]
    assert rig.out.order[4].startswith("told: [hands] The Claude Code session api is waiting for permission to use Bash")


async def test_two_sessions_finishing_during_one_held_key_are_told_asks_first_and_one_telling_a_session(rig: Rig) -> None:
    """What a session waits on the user for is told before what a session did; a session's turns that finished while
    the user talked are one telling; and a request answered at the keyboard meanwhile is not told at all."""
    rig.live[WEB] = waiting_on(WEB, "w2")
    await rig.hold(["down", "down"])
    await rig.until(lambda: rig.out.started == 1)
    # web's first request was answered at the keyboard while the key was down; its second still waits.
    held = [finished(API, "Fixed the parser."), asking(WEB, "w1"), finished(API, "Pushed it."), asking(WEB, "w2")]
    rig.clock.now = 2.0
    await rig.worker.queue_frames([*held, TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    # Each is held from when it came, not from the press.
    rig.clock.now = 5.0
    await rig.hold(["up"])
    await rig.texts.put("what time is it")
    await rig.until(lambda: len(rig.out.told) == 2)
    asked, told = rig.out.told
    assert rig.out.sent == ["what time is it"]
    assert asked.startswith("[hands] The Claude Code session web is waiting for permission") and "Request id: w2." in asked
    assert told.startswith("[hands] The Claude Code session api (id api) finished 2 turns. The last thing it said was:\n\nFixed the parser.\n\nThen, in the turn after that: The last thing it said was:\n\nPushed it.")
    # Each thing heard is followed to its fate: both turns told in one telling, and the request answered meanwhile dropped.
    assert [let_go(each) for each in held] == [
        {"held_ms": 3000.0, "told": "Finished", "folded": 2},
        {"held_ms": 3000.0, "fate": "dropped"},
        {"held_ms": 3000.0, "told": "Finished", "folded": 2},
        {"held_ms": 3000.0, "told": "Narrate", "folded": 1},
    ]


async def test_a_hold_that_was_shift_after_all_gives_back_what_waited_for_it(rig: Rig) -> None:
    await rig.hold(["down"])
    await rig.until(lambda: rig.out.started == 1)
    await rig.worker.queue_frame(waiting())
    await rig.hold(["dropped"])
    await rig.until(lambda: "said: The session api is gone." in rig.out.order)
    assert rig.out.order == ["started", "stopped", "said: The session api is gone."]


async def test_what_hands_says_with_no_turn_open_passes_at_once_and_records_no_wait(rig: Rig) -> None:
    gone = waiting()
    await rig.worker.queue_frame(gone)
    await rig.until(lambda: rig.out.order == ["said: The session api is gone."])
    assert let_go(gone) == {"held_ms": 0.0, "told": "SessionGone", "folded": 1}


async def test_a_request_answered_before_it_reaches_the_floor_is_not_told(rig: Rig) -> None:
    answered = asking(API, "gone")
    await rig.worker.queue_frames([answered, TextFrame("marker")])
    await rig.until(lambda: "marker" in rig.out.order)
    assert rig.out.order == ["marker"]
    assert let_go(answered) == {"held_ms": 0.0, "fate": "dropped"}


def owned() -> HoldOwner:
    return HoldOwner(Whisper(prompt=unprimed, record=lambda _: None), (), lambda _: None)


async def test_the_hold_owner_hears_only_the_keyed_microphone() -> None:
    with pytest.raises(TypeError, match="carries no key"):
        await owned()._captured(InputAudioRawFrame(b"\x00\x00", 16000, 1))  # pyright: ignore[reportPrivateUsage]  (what every frame of audio meets)


async def test_the_hold_owner_hears_only_mono_audio_at_the_rate_every_model_hears() -> None:
    with pytest.raises(ValueError, match="48000 Hz"):
        await owned()._captured(Gate().framed(b"\x00\x00", b"\x00\x00", 48000, 1, "desk"))  # pyright: ignore[reportPrivateUsage]


async def test_over_many_turns_each_is_sent_as_its_own_words_and_neither_side_keeps_any_of_them(rig: Rig) -> None:
    """hands-brain-80y: each turn was appended to a message list the brain's stage was handed whole, which grew for as
    long as the daemon ran; the stage counted what it had read to pick the new words out."""
    said = [f"turn {n}" for n in range(10)]
    for words in said:
        await rig.hold(["down", "up"])
        await rig.texts.put(words)
        await rig.until(lambda: rig.out.sent[-1:] == [words])
    assert rig.out.sent == said
    user, assistant = rig.conversation
    assert user.messages == assistant.messages == []


async def test_each_hold_the_key_sent_or_dropped_is_told_with_each_speakers_words_beside_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with rigged(monkeypatch, tmp_path, FrameProcessor(), []) as rig:
        rig.streamed.said.put("what time")
        rig.streamed.said.put(streaming.Said(1, "is it"))
        await rig.hold(["down", "down", "down", "up"], sound=struct.pack("<h", 1000) * 320)
        await rig.texts.put("what time is it")
        # Once the first hold's silence has been decoded, so the next words are the second's.
        await rig.until(lambda: len(ends(rig)) == 1)
        rig.streamed.said.put(streaming.Said(1, "never mind"))
        await rig.hold(["down", "dropped"], sound=struct.pack("<h", 1000) * 320)
        await rig.until(lambda: len(ends(rig)) == 2)
        sent, dropped = ends(rig)
        assert (sent.facts["hold"], sent.facts["fate"], sent.facts["words"]) == (1, "sent", (streaming.Said(0, "what time"), streaming.Said(1, "is it")))
        assert (dropped.facts["hold"], dropped.facts["fate"], dropped.facts["words"]) == (2, "dropped", (streaming.Said(1, "never mind"),))


def ends(rig: Rig) -> list[WideEvent]:
    """The streaming recognizer's hearing of each hold as it ended."""
    return [event for event in hearings(rig) if "fate" in event.facts]


async def test_a_hold_whose_streaming_breaks_still_sends_its_words(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async with rigged(monkeypatch, tmp_path, FrameProcessor(), []) as rig:
        rig.streamed.said.put(RuntimeError("the model broke"))
        await rig.hold(["down", "down", "up"], sound=struct.pack("<h", 1000) * 320)
        await rig.texts.put("what time is it")
        assert await rig.everything_sent(holds=1) == ["what time is it"]
        # The broken stream fails every hold from it on, the one `everything_sent` closes the run with too.
        await rig.until(lambda: len(ends(rig)) == 2)
        assert [(event.facts["hold"], event.outcome) for event in ends(rig)] == [(1, "failed"), (2, "failed")]
