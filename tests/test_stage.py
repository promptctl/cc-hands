"""The brain's LLM stage in a running pipeline: a turn goes to the brain, its words come off the wire as LLM text.

The brain is a stand-in whose turns end when the test says, and the wire is what the proxy would tell: each request as
it leaves, each text delta as it arrives, each exchange once it ends. So every order the real pieces can arrive in is
set up on purpose, not waited for.
"""

import asyncio
import json
from pathlib import Path
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import pytest
from datetime import UTC, datetime
from pipecat.utils.errors import ErrorCategory
from pipecat.frames.frames import (
    DataFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import patiently, running
from hands.brain.process import BROKEN, NOBODY, SPOKEN_OVER, STOPPED, UNANSWERED, Asked, BrainAnswered, BrainEnded, BrainStopped, Untaken, model
from hands.core.brainwork import AlreadyEnded, Idle, NotWorking, Running, Stopping, Working
from hands.core.effects import Allow, Deny
from hands.core import place
from hands.core.place import Modality
from hands.core.playback import Bookmark
from hands.core.events import Ending, Joined, PermissionRequested
from hands.core.session import DenyAndInterrupt, Membership, Permission, RequestId
from hands.sessions.registry import Sessions
from hands.brain.mcp import CallSpans
from hands.brain.stage import ACTED_UNTOLD, ANSWERS, FAILED_AFTER, OWED, SILENT, BrainStage, HandsAsked, UserAsked, Broken
from hands.core.front import FrontUnread, InFront, NoSessionInFront, SessionInFront, told
from hands.core.session import SessionId
from hands.core.trace import Span
from hands.core.wire import (
    Answering,
    BlockStarted,
    BlockStopped,
    Exchanged,
    Fork,
    Garbled,
    Heard,
    Hold,
    JsonDelta,
    Kind,
    MainTurn,
    Reached,
    Route,
    Send,
    Sent,
    StreamError,
    Tail,
    TextDelta,
    Unknown,
    Unreached,
    Unreadable,
    UsageLimitReached,
    unstreamed,
)
from hands.core.stopfailure import StopFailed
from hands.sessions.model_facts import ModelActedUntold, ModelFact, ModelFailed, ModelFault, ModelReplyEmpty, ModelUnreachable
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, root
from hands.voice.conversation import UserWords
from hands.voice.player import Mark
from hands.voice.trigger import Edge
from hands.voice.turnstop import HoldDiscarded
from hands.voice.speech import Aloud, Narrated
from hands.voice.utterance import Uttered, Uttering

from test_narrator import heard as unasked
from hands.voice.tool import Result, Tool, tool
from hands.voice.tools import stay_silent_tool

BRAIN = SessionId("brain-session")
ANSWERED = BrainAnswered("p1", None)
TAIL = "[hands] The Claude Code sessions running now: none of note."
# The span a routed request is compared with, its own kept apart in the rig's spans.
APART = Span("", "", None)
UNREAD = FrontUnread("not read in this test")
# A user's turn as the rig records it: the screen left unread, read in no time on the rig's clock.
ASKED = UserAsked("screen", "held key", UNREAD, 0.0)


def heard(words: str) -> str:
    """A user's turn as the brain is asked it at the desk, with the screen left unread."""
    return f"{words}\n\n{place.told('screen')}"


async def stage_draft(session: str, text: str) -> Result:
    """Stage a draft."""
    return {"readback": f"staged for {session}: {text}"}


async def amend_draft(session: str, text: str) -> Result:
    """Amend a draft, which hands reads back itself."""
    return {"says": f"amended for {session}: {text}"}


async def read_session(session: str) -> Result:
    """Read a session."""
    return {"steps": []}


TOOLS: Sequence[Tool] = (
    tool(stage_draft),
    tool(amend_draft, then="silence"),
    # hands' own: whether a turn may close on it is read off its `then`, so the rig's must be the real one's.
    stay_silent_tool(),
    tool(read_session),
)


@dataclass
class FakeBrain:
    """A brain whose every turn ends when the test ends it."""

    session: SessionId = BRAIN
    asked: list[str] = field(default_factory=list[str])
    turns: list[asyncio.Future[BrainEnded]] = field(default_factory=list[asyncio.Future[BrainEnded]])
    # How many times the stage asked the brain to stop the turn in flight.
    interrupts: int = 0
    # Each turn's teller of the permissions it holds.
    tellers: list[Callable[[Asked], None]] = field(default_factory=list[Callable[[Asked], None]])

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainEnded:
        self.asked.append(text)
        self.tellers.append(asks)
        turn = asyncio.get_running_loop().create_future()
        self.turns.append(turn)
        return await asyncio.shield(turn)

    # The brain's session, as the brain's own model holds it: what a permission is held in, and settled through.
    sessions: Sessions = field(default_factory=lambda: model(lambda _: None, lambda _: None, lambda _: None))
    held: int = 0

    def permit(self, tool: str, input: dict[str, object]) -> Asked:
        """The turn in flight holding a permission its setup asks about, as the real brain does at its hook: held in the
        brain's model, and put to the user as a view of it."""
        held = self.holding(tool, input)
        self.tellers[-1](held)
        return held

    def holding(self, tool: str, input: dict[str, object]) -> Asked:
        """A permission held in the brain's model, as the brain holds one at its hook, not yet put to any turn."""
        if self.sessions.live_session(self.session) is None:
            self.sessions.hear(Joined(Membership(self.session, pid=1, cwd=Path("/brain"), transcript=Path("/brain/t.jsonl")), "startup"))
        self.held += 1
        permission = Permission(tool, input)
        asked = PermissionRequested(self.session, at=0.0, request=RequestId(f"r{self.held}"), on=permission, mode=None)
        return Asked(permission, asked.request, self.sessions.hold(asked), self.sessions, self.session)

    # Each turn Claude Code opened that nobody waits on, as the stage heard its opening request.
    unawaiting: list[asyncio.Future[BrainEnded]] = field(default_factory=list[asyncio.Future[BrainEnded]])

    def unawaited(self) -> asyncio.Future[BrainEnded]:
        told: asyncio.Future[BrainEnded] = asyncio.get_running_loop().create_future()
        self.unawaiting.append(told)
        return told

    def interrupt(self) -> bool:
        """Asked to stop the turn in flight, as the real brain is: whether there was one. The test ends it, as Escape would."""
        self.interrupts += 1
        return bool(self.turns) and not self.turns[-1].done()

    def end(self, answered: BrainEnded = ANSWERED) -> None:
        self.turns[-1].set_result(answered)

    def fail(self, error: Exception) -> None:
        self.turns[-1].set_exception(error)


@dataclass
class Received(DataFrame):
    """Queued behind what the stage has handed on: `reached` is called once the rig's speaker has it, and so has every
    frame queued ahead of it, whether or not it is holding. A data frame reaches each processor in the order it was
    queued, and the stage passes on what it does not know."""

    reached: Callable[[], None]


class Spoken(FrameProcessor):
    """What leaves the stage for TTS, played at once, as the speaker and the marks behind it would play it, unless held."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        # How many frames have reached it: what a test's wait sees move while the stage has work it is getting through.
        self.received = 0
        self.frames: list[Frame] = []
        # Marks not yet played, while `holding`: what is ahead of them is still being said.
        self.holding = False
        self.marks: list[Mark] = []
        # How many key releases have passed the stage, the ones it was told to throw away included.
        self.releases = 0
        # How many of the user's turns have stopped past the stage.
        self.stops = 0
        # What is said of utterances: the frames that lead and close each, and the words between them, in order.
        self.uttering: list[str] = []
        # Each reading of the speaker asked for, settled as the next barge-in reaches it: with `playing`, where the test
        # sets one, and otherwise with all that was said, played to its end.
        self.cutting: list[asyncio.Future[Bookmark]] = []
        self.playing: Bookmark | None = None

    def cut_off(self) -> asyncio.Future[Bookmark]:
        cut = asyncio.get_running_loop().create_future()
        self.cutting.append(cut)
        return cut

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.received += 1
        match frame:
            case Received(reached=reached):
                reached()
            case Mark(played=played) if not self.holding:
                played()
            case Mark():
                self.marks.append(frame)
            case VADUserStoppedSpeakingFrame():
                self.releases += 1
                await self.push_frame(frame, direction)
            case UserStoppedSpeakingFrame():
                self.stops += 1
                await self.push_frame(frame, direction)
            case LLMFullResponseStartFrame() | LLMFullResponseEndFrame() | LLMTextFrame() | TTSSpeakFrame() | InterruptionFrame():
                if isinstance(frame, InterruptionFrame):
                    said = tuple(self.said())
                    cutting, self.cutting = self.cutting, []
                    for cut in cutting:
                        cut.set_result(Bookmark(said, len(said)) if self.playing is None else self.playing)
                self.frames.append(frame)
                if isinstance(frame, LLMTextFrame | TTSSpeakFrame):
                    self.uttering.append(frame.text)
                await self.push_frame(frame, direction)
            case Uttering() | Uttered():
                # A Resumed is an Uttering, named as itself.
                self.uttering.append(type(frame).__name__)
                await self.push_frame(frame, direction)
            case _:
                await self.push_frame(frame, direction)

    def said(self) -> list[str]:
        return [frame.text for frame in self.frames if isinstance(frame, LLMTextFrame | TTSSpeakFrame)]

    def shape(self) -> list[str]:
        return [type(frame).__name__ for frame in self.frames]


@dataclass
class Rig:
    worker: PipelineWorker
    stage: BrainStage
    brain: FakeBrain
    out: Spoken
    recorded: list[Entry]
    errors: list[ErrorFrame]
    # How the sessions stand, as the registry would compose it when a request leaves.
    standing: list[str]
    # The stage's clock, which the test moves.
    now: list[float]
    # Each session the stage moved the focus to, with how many turns the brain had been asked as it moved.
    refocused: list[tuple[SessionId, int]]
    # What is in front on the Mac's screen as a user's turn is submitted: the last of these, read when it is asked.
    fronts: list[InFront]
    # The stage asking the brain each turn, as the daemon runs it beside the pipeline.
    asking: asyncio.Task[None]
    # Whether the user can see a screen as their turn is submitted: the last of these.
    modalities: list[Modality]
    # The edge that opened the gate's last turn: the last of these.
    edges: list[Edge]
    # Each running call's span, as hands' MCP server finds it.
    call_spans: CallSpans
    # Each pause the stage is waiting out, with the task waiting it out: its time runs only when the test says.
    pauses: list[tuple[asyncio.Future[None], asyncio.Task[object]]]
    exchanges: int = 0
    # The span each request sent on carries, in the order they left.
    spans: list[Span] = field(default_factory=list[Span])
    # The exchanges whose reply has a text block streaming, as the wire opens one ahead of its first delta, with its index.
    voicing: dict[str, int] = field(default_factory=dict[str, int])
    # How many blocks each exchange's reply has opened: the next one's index, as the wire numbers them.
    blocks: dict[str, int] = field(default_factory=dict[str, int])

    async def until(self, what: Callable[[], bool]) -> None:
        """Until `what` holds and the speaker has everything the stage handed on by then, moved by anything the stage does
        that the rig sees: a frame reaching the speaker, an entry recorded, a turn asked of the brain. What the stage
        records, it records as it hands its words on, which is before the speaker has them."""
        await self._moving(what)
        await self.heard_out()

    async def heard_out(self) -> None:
        reached = asyncio.Event()
        await self.worker.queue_frame(Received(reached.set))
        await self._moving(reached.is_set)

    async def _moving(self, what: Callable[[], bool]) -> None:
        await patiently(what, lambda: (self.out.received, len(self.recorded), len(self.brain.asked)))

    async def turned(self, count: int = 1) -> None:
        """Until `count` of the brain's turns have ended."""
        await self.until(lambda: len(turns(self.recorded)) >= count)

    async def say(self, words: str) -> None:
        """The user aggregator handing on the words of a turn as it ends."""
        asked = len(self.brain.asked)
        await self.worker.queue_frame(UserWords(words))
        await self.until(lambda: len(self.brain.asked) > asked)

    def request(self, body: dict[str, object] | None = None, kind: Kind = MainTurn(None), session: SessionId | None = BRAIN) -> tuple[str, Route]:
        """A request leaving on the wire, asking for a stream as the brain's do unless its body says not: the stage hears it
        and decides where it goes."""
        self.exchanges += 1
        exchange = f"x{self.exchanges}"
        # By default the prompt the brain was last typed, as Claude Code opens the turn with what it took.
        sent = Sent(exchange, session, kind, None, {"stream": True, **(body or {"messages": [{"role": "user", "content": self.brain.asked[-1] if self.brain.asked else "hi"}]})})
        self.stage.hear(sent)
        route = self.stage.route(sent)
        # The span apart: where the request goes is what each test asserts, and what trace it is in is a few tests' own.
        self.spans.append(route.span)
        return exchange, replace(route, span=APART)

    def stream(self, exchange: str, *texts: str) -> None:
        """The reply's text, in the text block streaming, which the wire opens ahead of the first of it."""
        if exchange not in self.voicing:
            self.voicing[exchange] = self.block(exchange)
            self.stage.hear(Heard(exchange, BlockStarted(self.voicing[exchange], {"type": "text", "text": ""})))
        for text in texts:
            self.stage.hear(Heard(exchange, TextDelta(self.voicing[exchange], text)))

    def close(self, exchange: str) -> None:
        """The text block streaming whole: every word of it is on the wire."""
        self.stage.hear(Heard(exchange, BlockStopped(self.voicing.pop(exchange))))

    def calls(self, exchange: str, *calls: tuple[str, str]) -> None:
        """The reply making a whole block for each call, by id and name, as the stream carries it before the reply ends.
        The text block streaming ends as the first of them starts."""
        self.voicing.pop(exchange, None)
        for call, name in calls:
            index = self.block(exchange)
            self.stage.hear(Heard(exchange, BlockStarted(index, {"type": "tool_use", "id": call, "name": name, "input": {}})))
            self.stage.hear(Heard(exchange, BlockStopped(index)))

    def call(self, exchange: str, call: str, name: str, *pieces: str) -> None:
        """The reply making one whole call block, its input streamed in `pieces`, as the wire carries a call's input."""
        self.voicing.pop(exchange, None)
        index = self.block(exchange)
        self.stage.hear(Heard(exchange, BlockStarted(index, {"type": "tool_use", "id": call, "name": name, "input": {}})))
        for piece in pieces:
            self.stage.hear(Heard(exchange, JsonDelta(index, piece)))
        self.stage.hear(Heard(exchange, BlockStopped(index)))

    def block(self, exchange: str) -> int:
        """The index of the next block the exchange's reply opens."""
        index = self.blocks[exchange] = self.blocks.get(exchange, -1) + 1
        return index

    async def release(self, frame: VADUserStoppedSpeakingFrame | None = None) -> None:
        """The key let go, as Whisper tells it on its way to the stage: on words to transcribe, unless told otherwise."""
        released = self.out.releases
        await self.worker.queue_frame(VADUserStoppedSpeakingFrame() if frame is None else frame)
        await self.until(lambda: self.out.releases > released)

    async def interrupt(self) -> None:
        """The cut, as the user aggregator broadcasts it: that the user started speaking, then the interruption."""
        cuts = self.out.shape().count("InterruptionFrame")
        await self.worker.queue_frame(UserStartedSpeakingFrame())
        await self.worker.queue_frame(InterruptionFrame())
        await self.until(lambda: self.out.shape().count("InterruptionFrame") > cuts)

    async def stop(self) -> None:
        """The user's turn stopping, as the user aggregator says of every turn it started, ahead of its words."""
        stops = self.out.stops
        await self.worker.queue_frame(UserStoppedSpeakingFrame())
        await self.until(lambda: self.out.stops > stops)

    async def cut_in(self, words: str) -> None:
        """The user's words, behind the cut their turn opens with, as every turn that carries words is cut, and the turn's
        stop."""
        await self.interrupt()
        await self.stop()
        await self.worker.queue_frame(UserWords(words))

    async def elapse(self) -> None:
        """The pause the stage waits out runs its time, once it is waited out, and what waited it out runs to its end."""
        await self.until(lambda: any(not pause.done() for pause, _ in self.pauses))
        [(pause, waiting)] = [(pause, waiting) for pause, waiting in self.pauses if not pause.done()]
        pause.set_result(None)
        await asyncio.wait({waiting})


@pytest.fixture
async def rig() -> AsyncGenerator[Rig, None]:
    brain = FakeBrain()
    recorded: list[Entry] = []
    standing = [TAIL]
    now = [0.0]
    refocused: list[tuple[SessionId, int]] = []
    # A screen left unread by default: its note is nothing, so the user's words reach the brain as they were said.
    fronts: list[InFront] = [UNREAD]

    async def refocus(session: SessionId) -> None:
        refocused.append((session, len(brain.asked)))

    async def front() -> InFront:
        return fronts[-1]

    modalities: list[Modality] = ["screen"]
    edges: list[Edge] = ["held key"]
    call_spans = CallSpans()
    pauses: list[tuple[asyncio.Future[None], asyncio.Task[object]]] = []

    async def pause(_seconds: float) -> None:
        paused = asyncio.get_running_loop().create_future()
        waiting = asyncio.current_task()
        assert waiting is not None
        pauses.append((paused, waiting))
        await paused

    out = Spoken()
    stage = BrainStage(brain, TOOLS, lambda: standing[-1], refocus, front, lambda: modalities[-1], lambda: edges[-1], recorded.append, call_spans, out.cut_off, clock=lambda: now[0], pause=pause)
    async with running([stage, out]) as run:
        # As the daemon runs it: a watch beside the pipeline.
        asking = asyncio.create_task(stage.ask_each())
        try:
            yield Rig(run.worker, stage, brain, out, recorded, run.errors, standing, now, refocused, fronts, asking, modalities, edges, call_spans, pauses)
        finally:
            asking.cancel()


def turns(recorded: Sequence[Entry]) -> list[WideEvent]:
    """The event each brain turn left, in the order they ended."""
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "voice.turn"]


def spoke(recorded: Sequence[Entry]) -> list[tuple[object, ...]]:
    """What each turn handed the speaker: its exchanges, words, readbacks, whether it was spoken over, who asked it, how
    long it waited behind the turn before it, and what it failed of."""
    facts = ("exchanges", "text", "readbacks", "interrupted", "asker", "queued_ms", "failed")
    return [tuple(turn.facts[fact] for fact in facts) for turn in turns(recorded)]


def answering(name: str, result: object, call_id: str = "t1") -> dict[str, object]:
    """A request whose last message hands the result of the brain's call to `name` back to the model."""
    return {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": [{"type": "text", "text": json.dumps(result)}]}]},
            # As Claude Code sends it: a message of its own after the results.
            {"role": "system", "content": [{"type": "text", "text": "<system-reminder>tokens left</system-reminder>"}]},
        ]
    }


async def test_a_finished_turn_hands_narrates_reaches_the_brain_as_a_typed_turn_of_its_own(rig: Rig) -> None:
    """What hands says aloud of a session's turn is the brain's own turn, so it can answer about what the user heard."""
    await rig.worker.queue_frame(Narrated("[hands] The Claude Code session api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: rig.brain.asked == ["[hands] The Claude Code session api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.turned()
    assert rig.out.said() == ["api opened pull request 68."]
    assert ((exchange,), "api opened pull request 68.", (), False, HandsAsked(), 0.0, None) in spoke(rig.recorded)


async def test_the_brain_s_turn_telling_a_finished_turn_is_part_of_its_utterance_and_sent_between_its_marks(rig: Rig) -> None:
    """The utterance is open from when hands heard the turn to when it was heard: the brain's turn telling it is a part
    of it, in its trace, and what it says is led and closed by the frames the output transport reads its fate off."""
    utterance = unasked()
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (utterance,)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    [turn] = turns(rig.recorded)
    assert (turn.trace_id, turn.parent_id) == (utterance.begun.span.trace_id, utterance.begun.span.span_id)
    assert rig.out.uttering == ["Uttering", "api opened pull request 68.", "Uttered"]


async def test_a_barge_in_the_telling_goes_on_through_leads_the_rest_of_it_on_again(rig: Rig) -> None:
    """None of the turn's requests had left, so the barge-in stops nothing: what the turn says is still the telling's."""
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (unasked(),)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    await rig.interrupt()
    await rig.stop()
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    assert rig.out.uttering == ["Uttering", "Resumed", "api opened pull request 68.", "Uttered"]


async def test_a_barge_in_that_stops_the_telling_leads_nothing_more_on(rig: Rig) -> None:
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (unasked(),)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened ")
    await rig.until(lambda: "api opened " in rig.out.uttering)
    await rig.interrupt()
    rig.stream(exchange, "pull request 68.")
    rig.brain.end()
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    assert rig.out.uttering == ["Uttering", "api opened ", "Uttered"]


async def test_a_telling_the_brain_fails_fails_its_utterance_with_why(rig: Rig) -> None:
    utterance = unasked()
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (utterance,)))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    rig.brain.end(BrainAnswered("p1", StopFailed("unknown", "API Error: 500 overloaded")))
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    [turn] = turns(rig.recorded)
    assert turn.error is not None and utterance.failure == turn.error


async def test_a_line_said_as_written_in_hands_lane_is_sent_between_its_marks(rig: Rig) -> None:
    await rig.worker.queue_frame(Aloud(TTSSpeakFrame("The session api is gone."), (unasked(),)))
    await rig.until(lambda: rig.out.uttering[-1:] == ["Uttered"])
    assert rig.out.uttering == ["Uttering", "The session api is gone.", "Uttered"]
    assert rig.out.shape() == ["TTSSpeakFrame"]


async def test_a_narration_moves_the_focus_as_the_brain_takes_it_ahead_of_what_the_user_says_meanwhile(rig: Rig) -> None:
    """The user's next words are taken as said to the session told of: so the focus moves before the brain is asked to
    tell it, and words the user speaks while it is said are asked after it."""
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    await rig.worker.queue_frame(UserWords("push it"))
    await rig.until(lambda: "api opened pull request 68." in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: rig.brain.asked[1:] == [heard("push it")])
    assert rig.refocused == [(SessionId("api"), 0)]


async def test_the_users_turn_goes_ahead_of_a_narration_waiting_for_the_brain(rig: Rig) -> None:
    await rig.say("what is running?")
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.worker.queue_frame(UserWords("and the backlog?"))
    # Frames pass the stage in order, so once this is out, both before it are waiting in the stage.
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    # Words the user spoke before the telling was taken are not taken as said to the session it tells of.
    assert rig.refocused == []
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 3)
    assert rig.brain.asked == [heard("what is running?"), heard("and the backlog?"), "[hands] api finished a turn."]
    assert rig.refocused == [(SessionId("api"), 2)]


async def test_a_narration_records_how_long_it_waited_behind_the_users_turn(rig: Rig) -> None:
    await rig.say("what is running?")
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.now[0] = 2.5
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.stream(exchange, "api finished.")
    rig.brain.end()
    await rig.until(lambda: any(turn.facts["asker"] == HandsAsked() for turn in turns(rig.recorded)))
    assert ((exchange,), "api finished.", (), False, HandsAsked(), 2500.0, None) in spoke(rig.recorded)


async def test_what_hands_says_as_written_is_heard_after_the_narration_ahead_of_it(rig: Rig) -> None:
    """A session's end is said after its last turn, though that turn waits for the brain and the end needs no model."""
    await rig.say("what is running?")
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.worker.queue_frame(Aloud(TTSSpeakFrame("The session api is gone."), ()))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.stream(exchange, "api opened pull request 68.")
    await rig.until(lambda: "api opened pull request 68." in rig.out.said())
    assert "The session api is gone." not in rig.out.said()
    rig.brain.end()
    await rig.until(lambda: "The session api is gone." in rig.out.said())
    assert rig.out.said() == ["marker", "api opened pull request 68.", "The session api is gone."]


async def test_a_narration_the_brain_fails_is_said_as_written_with_its_question(rig: Rig) -> None:
    unsaid = "api finished a turn, and I could not tell it. Want me to push it?"
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", unsaid, SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    rig.brain.end(BrainAnswered("p1", StopFailed("unknown", "API Error: 500 overloaded")))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.said() == [unsaid]


async def test_a_turns_event_says_what_was_typed_into_the_brain(rig: Rig) -> None:
    await rig.say("what is api doing?")
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert turn.facts["asked"] == rig.brain.asked[0]


async def test_a_users_turn_the_brain_fails_has_nothing_said_for_it(rig: Rig) -> None:
    await rig.say("hello")
    rig.brain.end(BrainAnswered("p1", StopFailed("unknown", "API Error: 500 overloaded")))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.said() == []


async def test_a_turn_goes_to_the_brain_and_its_words_come_off_the_wire(rig: Rig) -> None:
    await rig.say("what is running?")
    assert rig.brain.asked == [heard("what is running?")]
    exchange, route = rig.request()
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.stream(exchange, "Two sessions ", "are running.")
    await rig.until(lambda: len(rig.out.said()) == 2)
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert rig.out.said() == ["Two sessions ", "are running."]
    # The audit log ties what was spoken to the exchange on the wire it came from.
    assert ((exchange,), "Two sessions are running.", (), False, ASKED, 0.0, None) in spoke(rig.recorded)


async def test_a_users_turn_carries_what_was_in_front_as_it_was_submitted_and_its_event_says_what_was_read(rig: Rig) -> None:
    front = SessionInFront("iTerm2", SessionId("s1"), "hands, docs")
    rig.fronts.append(front)
    await rig.say("what am I looking at?")
    assert rig.brain.asked == [heard(f"what am I looking at?\n\n{told(front)}")]
    exchange, _ = rig.request()
    rig.stream(exchange, "The docs session.")
    rig.brain.end()
    await rig.turned()
    assert ((exchange,), "The docs session.", (), False, UserAsked("screen", "held key", front, 0.0), 0.0, None) in spoke(rig.recorded)
    # Read again for the next turn, not kept: the screen has changed since.
    rig.fronts.append(NoSessionInFront("Safari"))
    await rig.say("and now?")
    assert rig.brain.asked[-1] == heard(f"and now?\n\n{told(NoSessionInFront('Safari'))}")


async def test_words_that_wait_behind_a_turn_carry_what_was_in_front_as_they_arrived_and_the_wait_leaves_out_the_read(rig: Rig) -> None:
    await rig.say("what is running?")
    spoken_at = SessionInFront("iTerm2", SessionId("s1"), "hands, docs")
    rig.fronts.append(spoken_at)
    await rig.worker.queue_frame(UserWords("what's this one doing?"))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    # The user looked elsewhere while the first turn was still being answered.
    rig.fronts.append(NoSessionInFront("Safari"))
    rig.now[0] = 2.5
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[-1] == heard(f"what's this one doing?\n\n{told(spoken_at)}")
    exchange, _ = rig.request()
    rig.stream(exchange, "Writing docs.")
    rig.brain.end()
    await rig.turned(2)
    assert ((exchange,), "Writing docs.", (), False, UserAsked("screen", "held key", spoken_at, 0.0), 2500.0, None) in spoke(rig.recorded)


async def test_a_users_turn_carries_whether_they_could_see_a_screen_as_their_words_arrived(rig: Rig) -> None:
    await rig.say("what is running?")
    await rig.worker.queue_frame(UserWords("go audio only"))
    await rig.worker.queue_frame(TTSSpeakFrame("marker"))
    await rig.until(lambda: "marker" in rig.out.said())
    # Switched while the words waited behind the first turn: they carry what held as they arrived.
    rig.modalities.append("audio-only")
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[-1] == heard("go audio only")
    rig.brain.end()
    await rig.say("what now?")
    assert rig.brain.asked[-1] == f"what now?\n\n{place.told('audio-only')}"
    exchange, _ = rig.request()
    rig.stream(exchange, "Nothing.")
    rig.brain.end()
    await rig.turned(3)
    assert ((exchange,), "Nothing.", (), False, UserAsked("audio-only", "held key", UNREAD, 0.0), 0.0, None) in spoke(rig.recorded)


async def test_a_users_turn_carries_the_edge_that_opened_it_though_another_opens_before_its_words_arrive(rig: Rig) -> None:
    await rig.release()
    # The phone's button pressed while the desk's words are still transcribed.
    rig.edges.append("phone button")
    await rig.say("what is running?")
    exchange, _ = rig.request()
    rig.stream(exchange, "Nothing.")
    rig.brain.end()
    await rig.turned()
    assert spoke(rig.recorded)[0][4] == UserAsked("screen", "held key", UNREAD, 0.0)


async def test_a_turn_hands_narrates_is_not_read_against_the_screen(rig: Rig) -> None:
    rig.fronts.append(SessionInFront("iTerm2", SessionId("s1"), "hands, docs"))
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    assert rig.brain.asked == ["[hands] api finished a turn."]


async def test_a_turn_is_one_event_saying_how_long_the_user_waited_and_where_the_time_went(rig: Rig) -> None:
    # Let go of the key, transcribed 0.4 s on, and written to the brain at once.
    rig.now[0] = 1000.0
    await rig.release()
    rig.now[0] = 1000.4
    await rig.say("what is api doing?")
    # The model asked, its answer's head heard, and a call made whole and run.
    rig.now[0] = 1000.5
    first, _ = rig.request()
    rig.now[0] = 1001.0
    rig.stage.hear(Answering(first, 200, None))
    rig.now[0] = 1001.2
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    # The span hands' MCP server runs the call inside, while it runs.
    running = rig.call_spans.span("t1")
    # The call's result going back is the end of its run, and the second round trip's start.
    rig.now[0] = 1001.5
    second, _ = rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.now[0] = 1001.9
    rig.stage.hear(Answering(second, 200, None))
    rig.now[0] = 1002.0
    rig.stream(second, "api is running its tests.")
    rig.now[0] = 1002.3
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    events = [entry for entry in rig.recorded if isinstance(entry, WideEvent)]
    assert (turn.outcome, turn.counts) == ("ok", {"round_trips": 2, "tools": 1, "refused": 0, "cut_in": 0, "silence_refused": 0})
    # Two seconds from letting go to the first word: 0.4 transcribing, none waiting behind another turn, and the rest
    # the model's and its tool's, as the parts show.
    assert {fact: turn.facts[fact] for fact in ("asker", "waited_ms", "transcribed_ms", "queued_ms")} == {"asker": ASKED, "waited_ms": 2000.0, "transcribed_ms": 400.0, "queued_ms": 0.0}
    # Not spoken over: nothing went unheard.
    assert (turn.facts["interrupted"], turn.facts["spoken_over"]) == (False, "")
    [call] = [event for event in events if event is not turn]
    assert (call.event, call.duration_ms, call.outcome, call.facts) == ("tool.call", 300.0, "ok", {"call": "t1", "tool": "mcp__hands__read_session"})
    # The call's run is the child of the very span the call is emitted as, and the turn's end forgets it.
    assert running is not None and (call.trace_id, call.span_id, call.parent_id) == (running.trace_id, running.span_id, running.parent_id)
    assert rig.call_spans.span("t1") is None
    # Each round trip is the proxy's record of its request, which carries a span of its own inside the turn's.
    assert [(span.trace_id, span.parent_id) for span in rig.spans] == [(turn.trace_id, turn.span_id)] * 2
    assert len({span.span_id for span in rig.spans}) == 2
    # One trace: each part is the turn's child.
    assert {(event.trace_id, event.parent_id) for event in events if event is not turn} == {(turn.trace_id, turn.span_id)}


async def test_a_turn_spoken_over_before_its_answer_and_its_call_came_back_counts_them_cancelled(rig: Rig) -> None:
    # Spoken over while its words still play.
    rig.out.holding = True
    await rig.say("what is api doing?")
    exchange, _ = rig.request()
    rig.stream(exchange, "Let me look.")
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    await rig.interrupt()
    # The model asked again, and the user spoke over the turn before its answer came.
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    [call] = [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "tool.call"]
    assert call.outcome == "cancelled" and turn.facts["interrupted"] is True
    # A turn nobody asked aloud has no transcription to time.
    assert turn.facts["transcribed_ms"] is None


async def test_a_turn_stopped_mid_way_still_says_what_it_did(rig: Rig) -> None:
    await rig.say("what is api doing?")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    # hands stopping while the brain is still answering.
    rig.asking.cancel()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.outcome, turn.counts, turn.facts["exchanges"]) == ("cancelled", {"round_trips": 1, "tools": 1, "refused": 0, "cut_in": 0, "silence_refused": 0}, (exchange,))
    [call] = [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "tool.call"]
    assert (call.outcome, call.parent_id) == ("cancelled", turn.span_id)


async def test_words_that_waited_behind_a_turn_together_are_timed_from_the_last_of_them(rig: Rig) -> None:
    await rig.say("what is api doing?")
    # Two holds said while the brain answers, each let go of and transcribed in its turn.
    for released, arrived, words in ((1000.0, 1000.4, "and auth?"), (1002.0, 1002.4, "and web?")):
        rig.now[0] = released
        await rig.release()
        rig.now[0] = arrived
        await rig.worker.queue_frame(UserWords(words))
        # Frames pass the stage in order, so once this is out, the words before it are waiting in the stage.
        await rig.worker.queue_frame(TTSSpeakFrame(words))
        await rig.until(lambda: words in rig.out.said())
    rig.now[0] = 1003.0
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    exchange, _ = rig.request()
    rig.now[0] = 1004.0
    rig.stream(exchange, "Both are idle.")
    rig.brain.end()
    await rig.turned(2)
    assert rig.brain.asked[1] == heard("and auth?\n\nand web?")
    # Two seconds from the last let-go: 0.4 transcribing it, 0.6 behind the turn before, and 1.0 to the first word.
    timed = {fact: turns(rig.recorded)[1].facts[fact] for fact in ("waited_ms", "transcribed_ms", "queued_ms")}
    assert timed == {"waited_ms": 2000.0, "transcribed_ms": 400.0, "queued_ms": 600.0}


async def test_a_hold_thrown_away_is_no_release_to_time_a_wait_from(rig: Rig) -> None:
    rig.now[0] = 1000.0
    await rig.release()
    rig.now[0] = 1003.0
    await rig.release(HoldDiscarded())
    rig.now[0] = 1003.5
    await rig.say("hello")
    exchange, _ = rig.request()
    rig.now[0] = 1004.0
    rig.stream(exchange, "Hi.")
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["waited_ms"], turn.facts["transcribed_ms"]) == (4000.0, 3500.0)


async def test_a_turn_the_brain_failed_is_a_failed_event_saying_why(rig: Rig) -> None:
    await rig.say("hello")
    rig.brain.end(BrainAnswered("p1", StopFailed("server_error", "API Error: 529 Overloaded")))
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.outcome, turn.error) == ("failed", "the brain's turn ended in error: server_error: API Error: 529 Overloaded")
    # Nothing was said, so there is no wait to a first word: absent, not zero.
    assert (turn.facts["waited_ms"], turn.counts) == (None, {"round_trips": 0, "tools": 0, "refused": 0, "cut_in": 0, "silence_refused": 0})


async def test_each_request_of_a_turn_carries_how_the_sessions_stand_as_it_leaves(rig: Rig) -> None:
    await rig.say("start the tests in auth")
    exchange, first = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.standing.append("[hands] The Claude Code sessions running now: auth, working.")
    _, second = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged"}))
    # Composed as each request leaves, never kept from the one before.
    assert (first, second) == (Send((Tail(TAIL),), refusal="final", span=APART), Send((Tail("[hands] The Claude Code sessions running now: auth, working."),), refusal="final", span=APART))
    rig.brain.end()


async def test_only_the_brains_own_main_turns_are_spoken(rig: Rig) -> None:
    await rig.say("hello")
    others = [rig.request(kind=Fork()), rig.request(kind=Unknown("a messages request with no tools")), rig.request(session=SessionId("a summary"))]
    # Every request of the brain's is final, said or not; another session's goes as it came (hands-wire-zi2).
    assert [route for _, route in others] == [Send(refusal="final", span=APART), Send(refusal="final", span=APART), Send(span=APART)]
    # The brain's are its turn's, in one trace under one parent; another session's is the root of a trace of its own.
    fork, unknown, summary = rig.spans
    assert (fork.trace_id, fork.parent_id) == (unknown.trace_id, unknown.parent_id) and fork.parent_id is not None
    assert summary.parent_id is None and summary.trace_id != fork.trace_id
    for exchange, _ in others:
        rig.stream(exchange, "not for the user")
    exchange, _ = rig.request()
    rig.stream(exchange, "Hello.")
    await rig.until(lambda: rig.out.said() == ["Hello."])
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.said() == ["Hello."]


async def test_a_main_turn_no_turn_asked_of_is_final_so_a_spent_limit_is_never_continued(rig: Rig) -> None:
    # Asked again at the reset, it would run the brain's tools with nobody there (hands-wire-zi2).
    _, route = rig.request()
    assert isinstance(route, Send) and route.refusal == "final"
    # It opens a turn nobody waits on, the root of a trace of its own, and its request is a part of it.
    assert len(rig.brain.unawaiting) == 1
    rig.brain.unawaiting[0].set_result(ANSWERED)
    await rig.turned()
    (event,) = turns(rig.recorded)
    assert event.parent_id is None and rig.spans[0].parent_id == event.span_id


async def test_each_turn_the_brain_is_asked_that_turn_s_words_and_nothing_said_before_them(rig: Rig) -> None:
    await rig.say("anything new?")
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    await rig.say("is api done?")
    assert rig.brain.asked == [heard("anything new?"), heard("is api done?")]


async def test_a_turn_is_written_only_once_the_one_before_it_has_ended(rig: Rig) -> None:
    await rig.say("first")
    await rig.cut_in("second")
    # The stage has taken the second turn and is waiting on the first, which the brain has not ended.
    await asyncio.sleep(0.1)
    assert rig.brain.asked == [heard("first")]
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[1] == heard("second")


async def test_stay_silent_holds_the_next_request_so_nothing_follows_it(rig: Rig) -> None:
    await rig.say("(to a colleague) back in five")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}))
    assert route == Hold(SILENT, APART)
    # Held, it is still the turn's round trip, in the turn's trace beside the one before it.
    sent, held = rig.spans
    assert (held.trace_id, held.parent_id) == (sent.trace_id, sent.parent_id) and held.span_id != sent.span_id
    rig.brain.end()
    await rig.until(lambda: "LLMFullResponseEndFrame" in rig.out.shape())
    assert rig.out.said() == []
    # The model chose to say nothing: not a failure, so nothing is reported either.
    await asyncio.sleep(0.1)
    assert rig.errors == []
    # The held reply closed the call in the brain's history, so the silence does not carry into the next turn.
    await rig.say("are you there?")
    closed = answering("mcp__hands__stay_silent", {"silent": True})
    closed["messages"] = [*closed["messages"], {"role": "assistant", "content": [{"type": "text", "text": SILENT}]}, {"role": "user", "content": "are you there?"}]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    exchange, route = rig.request(closed)
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.stream(exchange, "I am.")
    await rig.until(lambda: rig.out.said() == ["I am."])
    rig.brain.end()


async def acted_then_silent(rig: Rig, act: tuple[str, object], *closing: str) -> tuple[str, Route]:
    """The user's turn: one call that acts, answered, then a reply of `closing` words calling stay_silent; the request
    answering that stay_silent, and where it goes."""
    await rig.say("file a ticket for the flaky mic test")
    exchange, _ = rig.request()
    name, result = act
    rig.calls(exchange, ("t1", name))
    exchange, _ = rig.request(answering(name, result))
    rig.stream(exchange, *closing)
    rig.calls(exchange, ("t2", "mcp__hands__stay_silent"))
    return rig.request(answering("mcp__hands__stay_silent", {"silent": True}, "t2"))


async def test_a_stay_silent_closing_a_turn_that_acted_is_refused_and_the_brain_says_what_it_did(rig: Rig) -> None:
    exchange, route = await acted_then_silent(rig, ("Bash", "filed"))
    # Sent on, told it acted and was heard saying nothing since: not held.
    assert route == Send((Tail(OWED), Tail(TAIL)), refusal="final", span=APART)
    rig.stream(exchange, "Filed it.")
    rig.brain.end()
    await rig.until(lambda: rig.out.said() == ["Filed it."])
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (None, 1)
    assert rig.errors == []


async def test_a_stay_silent_after_a_readback_handed_to_the_brain_is_refused(rig: Rig) -> None:
    """A result the brain is to say, as send_draft's readback is, is not said by hands: closing on stay_silent drops it."""
    _, route = await acted_then_silent(rig, ("mcp__hands__stage_draft", {"readback": "Sent the draft to api."}))
    assert route == Send((Tail(OWED), Tail(TAIL)), refusal="final", span=APART)


async def test_the_request_asked_again_after_the_refusal_broke_is_told_to_say_what_it_did_too(rig: Rig) -> None:
    """Claude Code asks again whole for a reply that broke before any block of it finished: the same request, so the
    same note, or the one refusal is spent on a reply that never came."""
    await acted_then_silent(rig, ("Bash", "filed"))
    _, route = rig.request({"stream": False, **answering("mcp__hands__stay_silent", {"silent": True}, "t2")})
    assert route == Send((Tail(OWED), Tail(TAIL)), refusal="final", span=APART)


async def test_a_stay_silent_after_an_act_beside_a_readback_hands_says_is_held(rig: Rig) -> None:
    """A reply that read a session and amended a draft goes on, since the read is the model's to answer; the readback
    hands says at the turn's end is heard after the read, so the stay_silent that follows is held."""
    await rig.say("check api and fix the draft")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"), ("t2", "mcp__hands__amend_draft"))
    both = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": json.dumps({"steps": []})}]},
        {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": json.dumps({"says": "Amended it."})}]},
    ]}
    exchange, route = rig.request({"messages": [{"role": "user", "content": "hi"}, both]})
    assert isinstance(route, Send)
    rig.calls(exchange, ("t3", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}, "t3"))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.until(lambda: rig.out.said() == ["Amended it."])
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (None, 0)


async def test_a_stay_silent_beside_whitespace_closes_on_nothing_and_is_refused(rig: Rig) -> None:
    _, route = await acted_then_silent(rig, ("Bash", "filed"), "\n\n")
    assert route == Send((Tail(OWED), Tail(TAIL)), refusal="final", span=APART)


async def test_a_second_stay_silent_after_the_refusal_is_held_and_the_turn_is_said_as_acted_untold(rig: Rig) -> None:
    exchange, _ = await acted_then_silent(rig, ("Bash", "filed"))
    rig.calls(exchange, ("t3", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}, "t3"))
    # Once refused is enough: the model's obedience is not what ends the turn.
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == ModelActedUntold()
    [turn] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (ModelActedUntold(), 1)
    await rig.say("what happened?")
    assert rig.brain.asked[-1] == f"{heard('what happened?')}\n\n{ACTED_UNTOLD}"


async def test_a_stay_silent_closing_a_turn_that_acted_is_held_where_something_is_heard_after_the_act(rig: Rig) -> None:
    """Words in the closing reply are heard after every act before it."""
    _, route = await acted_then_silent(rig, ("Bash", "filed"), "Filed it.")
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (None, 0)


async def test_a_turn_that_acted_and_closes_on_a_reply_hands_says_is_held(rig: Rig) -> None:
    """A Bash that found what a draft needs, then the draft, read back by hands: the readback is heard after the act."""
    await rig.say("tell api to fix the parser")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "Bash"))
    exchange, _ = rig.request(answering("Bash", "parser.py"))
    rig.calls(exchange, ("t2", "mcp__hands__amend_draft"))
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: fix the parser"}, "t2"))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.until(lambda: "amended for api: fix the parser" in rig.out.said())
    [turn] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (None, 0)


async def test_a_turn_hands_asked_may_close_on_stay_silent_after_it_read(rig: Rig) -> None:
    """What hands hands the brain to tell says whether to speak of it: the user did not ask, so nothing is owed them."""
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    exchange, _ = rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.calls(exchange, ("t2", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}, "t2"))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.turned()
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_turn_the_user_spoke_over_may_close_on_stay_silent_after_it_acted(rig: Rig) -> None:
    """Nothing of a turn spoken over is said from then on, and the user's words are already the next turn: a word owed
    them would be heard by nobody, and keep their words waiting."""
    await rig.say("file a ticket for the flaky mic test")
    exchange, _ = rig.request()
    rig.stream(exchange, "Filing it now. ")
    await rig.until(lambda: rig.out.said() == ["Filing it now. "])
    rig.out.playing = Bookmark(("Filing it now.",), 0)
    await rig.interrupt()
    await rig.worker.queue_frame(UserWords("actually wait"))
    rig.calls(exchange, ("t1", "Bash"))
    exchange, _ = rig.request(answering("Bash", "filed"))
    rig.calls(exchange, ("t2", "mcp__hands__stay_silent"))
    _, route = rig.request(answering("mcp__hands__stay_silent", {"silent": True}, "t2"))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    [turn, *_] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (None, 0)


async def test_a_barge_in_mid_reply_stops_it_being_said_and_the_brains_turn_runs_on_to_its_end(rig: Rig) -> None:
    await rig.say("tell me everything")
    exchange, _ = rig.request()
    rig.stream(exchange, "First, it rained. Then it snowed. ")
    await rig.until(lambda: rig.out.said() == ["First, it rained. Then it snowed. "])
    # The speaker was part way through the second sentence.
    rig.out.playing = Bookmark(("First, it rained.", "Then it snowed."), 1)
    await rig.interrupt()
    # What the user said over it waits for the turn's end.
    await rig.worker.queue_frame(UserWords("and the logs?"))
    rig.stream(exchange, "Then it cleared.")
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    # The call runs, and its result goes on to the model: nothing is held or answered by hands, and no key is pressed.
    after, route = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged for api: add tests"}))
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.stream(after, " Staged.")
    rig.brain.end()
    # Then it is the next turn: the user's words, told what they did not hear of the turn they spoke over.
    await rig.until(lambda: len(rig.brain.asked) == 2)
    over = '[hands] The user spoke over you, and nothing more was said aloud. They last heard "First, it rained.". They did not hear: "Then it snowed. Then it cleared. Staged.".'
    assert rig.brain.asked == [heard("tell me everything"), f"{heard('and the logs?')}\n\n{over}"]
    # Nothing of the turn was said after the barge-in.
    assert rig.out.said() == ["First, it rained. Then it snowed. "]
    [turn] = turns(rig.recorded)
    assert (turn.facts["exchanges"], turn.facts["interrupted"], turn.facts["spoken_over"]) == ((exchange, after), True, over)
    rig.brain.end()


async def test_a_sentence_on_its_way_to_the_speaker_when_the_user_spoke_over_it_is_told_unheard_whole(rig: Rig) -> None:
    await rig.say("what did you do?")
    exchange, _ = rig.request()
    rig.stream(exchange, "First, it rained. I closed the ")
    await rig.until(lambda: rig.out.said() == ["First, it rained. I closed the "])
    # The speaker played the first sentence; the start of the second never reached it.
    rig.out.playing = Bookmark(("First, it rained.",), 1)
    await rig.interrupt()
    await rig.worker.queue_frame(UserWords("and then?"))
    rig.stream(exchange, "session.")
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    over = '[hands] The user spoke over you, and nothing more was said aloud. They last heard "First, it rained.". They did not hear: "I closed the session.".'
    assert rig.brain.asked[1] == f"{heard('and then?')}\n\n{over}"
    rig.brain.end()


async def test_a_turn_spoken_over_that_then_fails_tells_the_next_what_the_user_heard_and_that_it_failed(rig: Rig) -> None:
    await rig.say("tell me everything")
    exchange, _ = rig.request()
    rig.stream(exchange, "First, it rained. Then it snowed. ")
    await rig.until(lambda: rig.out.said() == ["First, it rained. Then it snowed. "])
    rig.out.playing = Bookmark(("First, it rained.", "Then it snowed."), 1)
    await rig.interrupt()
    await rig.worker.queue_frame(UserWords("and the logs?"))
    rig.brain.end(BrainAnswered("p1", StopFailed("server_error", "API Error: Connection lost mid-response.")))
    await rig.until(lambda: len(rig.brain.asked) == 2)
    # Not also that the user heard all the speaker was handed: the speaker says what they heard.
    over = '[hands] The user spoke over you, and nothing more was said aloud. They last heard "First, it rained.". They did not hear: "Then it snowed.".'
    assert rig.brain.asked[1] == f"{heard('and the logs?')}\n\n{over}\n\n{FAILED_AFTER}"
    rig.brain.end()


async def test_a_reply_that_breaks_after_the_user_spoke_over_the_turn_is_told_the_brain_as_heard_by_nobody(rig: Rig) -> None:
    await rig.say("tell me everything")
    exchange, _ = rig.request()
    rig.stream(exchange, "First, it rained. ")
    await rig.until(lambda: rig.out.said() == ["First, it rained. "])
    rig.out.playing = Bookmark(("First, it rained.",), 1)
    await rig.interrupt()
    await rig.worker.queue_frame(UserWords("and the logs?"))
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    _, retry = broken_then_asked_again(rig, ("Then it ",), "Then it snowed.")
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert turns(rig.recorded)[0].facts["broken"] == (Broken(retry, ""),)
    # Only the spoken-over note: no account of the broken reply as heard, which nobody heard.
    [words, over] = rig.brain.asked[1].rsplit("\n\n", 1)
    assert words == heard("and the logs?") and over.startswith('[hands] The user spoke over you, and nothing more was said aloud. They last heard "First, it rained.".')
    rig.brain.end()


async def test_a_word_the_brain_streams_after_the_user_spoke_over_its_turn_is_no_wait_of_theirs(rig: Rig) -> None:
    await rig.say("what is api doing?")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    # Spoken over as a text block opens after the call, before a word of it.
    rig.stream(exchange)
    await rig.interrupt()
    after, _ = rig.request(answering("mcp__hands__read_session", {"readback": "api is idle"}))
    rig.stream(after, "It is idle.")
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["interrupted"], turn.facts["waited_ms"]) == (True, None)


async def test_a_readback_a_call_hands_hands_ends_the_turn_and_is_said_by_hands_as_written(rig: Rig) -> None:
    await rig.say("make it add tests too")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: add tests too"}))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    await rig.until(lambda: rig.out.said() == ["amended for api: add tests too"])
    # hands-readback-ddk: said after the turn's reply has ended, where it is a turn of its own, over once it is said.
    await rig.until(lambda: rig.out.shape() == ["LLMFullResponseStartFrame", "LLMFullResponseEndFrame", "TTSSpeakFrame"])


async def test_a_readback_of_a_turn_the_user_spoke_over_is_told_the_brain_never_said(rig: Rig) -> None:
    await rig.say("make it add tests too")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    # Spoken over as a text block opens after the call, before a word of it.
    rig.stream(exchange)
    await rig.interrupt()
    await rig.worker.queue_frame(UserWords("wait, no"))
    _, route = rig.request(answering("mcp__hands__amend_draft", {"says": "amended for api: add tests too"}))
    assert route == Hold(SILENT, APART)
    rig.brain.end()
    # The draft changed, and the brain, told the user never heard how, judges whether to say it.
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[1] == f'{heard("wait, no")}\n\n[hands] The user spoke over you, and nothing more was said aloud. They did not hear: "amended for api: add tests too".'
    assert "amended for api: add tests too" not in rig.out.said()
    rig.brain.end()


async def test_a_refused_call_to_a_silence_tool_is_the_models_to_answer(rig: Rig) -> None:
    await rig.say("make it add tests too")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"))
    refused, route = rig.request(answering("mcp__hands__amend_draft", {"error": "the draft text is empty"}))
    # Nothing was said and nothing changed: the model is asked to go on, to retry or to say what went wrong.
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.stream(refused, "I couldn't change it.")
    await rig.until(lambda: rig.out.said() == ["I couldn't change it."])
    rig.brain.end()


async def test_a_reply_with_one_draft_said_and_one_refused_is_the_models_to_answer(rig: Rig) -> None:
    await rig.say("make both add tests too")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__amend_draft"), ("t2", "mcp__hands__amend_draft"))
    results: tuple[tuple[str, dict[str, str]], ...] = (("t1", {"says": "amended for api: add tests too"}), ("t2", {"error": "There is no session web."}))
    body: dict[str, object] = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": call, "name": "mcp__hands__amend_draft", "input": {}} for call, _ in results]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call, "content": [{"type": "text", "text": json.dumps(result)}]} for call, result in results]},
        ]
    }
    _, route = rig.request(body)
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.brain.end()


async def test_frames_behind_a_turn_pass_while_the_brain_is_still_on_it(rig: Rig) -> None:
    await rig.say("check every session")
    # A session asking permission is announced while the brain works through its tools, not once it is done.
    await rig.worker.queue_frame(TTSSpeakFrame("api asks to run tests."))
    await rig.until(lambda: rig.out.said() == ["api asks to run tests."])
    rig.brain.end()


async def test_what_the_user_said_while_a_turn_ran_is_asked_once_it_ends_through_any_number_of_presses(rig: Rig) -> None:
    await rig.say("first")
    await rig.cut_in("second")
    await rig.interrupt()
    await rig.stop()
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked == [heard("first"), heard("second")]


async def test_a_message_list_reaching_the_stage_is_an_error_not_a_turn(rig: Rig) -> None:
    """A turn Pipecat ended other than through UserTurns carries its words in a list hands keeps none of: they were lost."""
    await rig.worker.queue_frame(LLMContextFrame(LLMContext([{"role": "user", "content": "lost"}])))
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(rig.errors[0].exception, TypeError) and rig.errors[0].processor is rig.stage
    assert rig.brain.asked == []


async def test_a_turn_whose_replies_carried_nothing_is_said_as_the_models_empty_reply(rig: Rig) -> None:
    unsaid = "api finished a turn, and I could not tell it."
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", unsaid, SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    # Whitespace is not words.
    exchange, _ = rig.request()
    rig.stream(exchange, " ")
    rig.brain.end()
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == ModelReplyEmpty()
    assert rig.errors[0].processor is rig.stage
    # What hands had for the brain to tell is said as written, as for any turn the brain failed.
    await rig.until(lambda: unsaid in rig.out.said())
    # [LAW:nothing-unseen] the empty reply is on the turn's line in the audit log.
    [turn] = turns(rig.recorded)
    assert turn.facts["failed"] == ModelReplyEmpty()
    # An empty reply is whole and in the brain's history: its next turn is not told the API broke one off.
    await rig.say("hello?")
    assert rig.brain.asked[-1] == heard("hello?")


async def test_a_turn_that_acted_and_then_answered_nothing_is_said_as_acted_untold(rig: Rig) -> None:
    """Claude often ends a turn with an empty reply to a tool's result, and the user, who heard only what came before the
    call, hears nothing of what it found or did: the turn fails aloud, never as an empty reply, since something was done."""
    await rig.say("what is api doing?")
    exchange, _ = rig.request()
    rig.stream(exchange, "Let me look.")
    rig.calls(exchange, ("t1", "mcp__hands__read_session"))
    rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.brain.end()
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == ModelActedUntold()
    [turn] = turns(rig.recorded)
    assert (turn.facts["failed"], turn.counts["silence_refused"]) == (ModelActedUntold(), 0)
    # Its history ends in silence: the next turn is told the user heard the failure in its place.
    await rig.say("hello?")
    assert rig.brain.asked[-1] == f"{heard('hello?')}\n\n{ACTED_UNTOLD}"


async def test_a_turn_ending_on_a_call_that_lands_is_no_empty_reply(rig: Rig) -> None:
    await rig.say("tell the api session to add tests")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    rig.brain.end()
    await rig.turned()
    await asyncio.sleep(0.1)
    assert rig.errors == []


async def test_a_turn_the_brain_ended_in_error_is_reported_as_the_model_stages_error(rig: Rig) -> None:
    await rig.say("hello")
    rig.brain.end(BrainAnswered("p1", StopFailed("unknown", "API Error: 500 overloaded")))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].processor is rig.stage
    assert "API Error: 500 overloaded" in rig.errors[0].error


RESETS = datetime(2026, 9, 30, 18, 0, tzinfo=UTC).timestamp()


def unreached(exchange: str) -> Exchanged:
    """The proxy's record of a request it could not get to the API, told before it answers the brain 502."""
    return Exchanged(exchange, BRAIN, MainTurn(None), None, "POST", "/v1/messages", 2, (), 0.0, 0.0, Unreached("ClientConnectorError: no route", 0.0), True, root())


@pytest.mark.parametrize(
    ("answers", "failure", "fact"),
    [
        # As Claude Code meets each (measured, hands-wire-6ic.gfq): a spent limit is asked once, an API the proxy
        # cannot reach eleven times, and what failed the turn is its latest answer.
        ([429], StopFailed("rate_limit", "You've hit your session limit · resets 12:00pm"), UsageLimitReached(RESETS)),
        ([None, None, None], StopFailed("server_error", "API Error: 502 hands' proxy could not reach https://api.anthropic.com"), ModelUnreachable()),
        ([529], StopFailed("server_error", "API Error: 529 Overloaded"), ModelFailed(ErrorCategory.SERVER)),
        # A 200 names no cause: the reply it began broke, and Claude Code says so.
        ([None, 529, 200], LOST := StopFailed("server_error", "API Error: Connection lost mid-response."), LOST),
        # A request whose answer never came, its handler cancelled before the head: an earlier answer is not its reason.
        ([None, "unanswered"], TIMED_OUT := StopFailed("server_error", "API Error: Request timed out."), TIMED_OUT),
        # Failed in Claude Code before any request, as the audit log saw an expired login on 2026-10-24: only the StopFailure says why.
        ([], EXPIRED := StopFailed("invalid_request", "Login expired · Run /login to sign in again, or re-authenticate your Anthropic profile"), EXPIRED),
    ],
)
async def test_a_failed_turn_is_said_as_the_wire_says_it_failed(rig: Rig, answers: list[int | None | Literal["unanswered"]], failure: StopFailed, fact: ModelFact) -> None:
    await rig.say("hello")
    for status in answers:
        exchange, _ = rig.request()
        match status:
            case None:
                rig.stage.hear(unreached(exchange))
            case 429:
                rig.stage.hear(Answering(exchange, 429, UsageLimitReached(RESETS)))
            case int():
                rig.stage.hear(Answering(exchange, status, None))
            case "unanswered":
                pass
    rig.brain.end(BrainAnswered("p1", failure))
    await rig.until(lambda: len(rig.errors) == 1)
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == fact
    # [LAW:nothing-unseen] which failure the turn was said as is on its line in the audit log.
    [turn] = turns(rig.recorded)
    assert turn.facts["failed"] == fact


async def test_a_stop_failure_that_shows_no_text_is_told_by_its_kind_of_error(rig: Rig) -> None:
    # Claude Code's StopFailure always names its kind of error, and its text only when it showed one.
    await rig.say("hello")
    rig.brain.end(BrainAnswered("p1", unsaid := StopFailed("authentication_failed", None)))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].error == "the brain's turn ended in error: authentication_failed"
    assert isinstance(error := rig.errors[0].exception, ModelFault) and error.fact == unsaid


async def test_a_reply_the_api_breaks_mid_stream_is_said_once_as_far_as_it_came_then_the_failure(rig: Rig) -> None:
    # As Claude Code ends it (hands-wire-6ic.6dz, measured): no retry and no fallback request, only StopFailure.
    await rig.say("tell me about lighthouses")
    exchange, _ = rig.request()
    rig.stream(exchange, "1. Lighthouses stand ", "on rocky coasts and h")
    await rig.until(lambda: len(rig.out.said()) == 2)
    rig.brain.end(BrainAnswered("p1", lost := StopFailed("server_error", "API Error: Connection lost mid-response. The response above may be incomplete.")))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.out.shape() == ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMTextFrame", "LLMFullResponseEndFrame"]
    assert rig.out.said() == ["1. Lighthouses stand ", "on rocky coasts and h"]
    assert rig.errors[0].processor is rig.stage
    assert "Connection lost mid-response" in rig.errors[0].error
    assert ((exchange,), "1. Lighthouses stand on rocky coasts and h", (), False, ASKED, 0.0, lost) in spoke(rig.recorded)
    # The broken reply is not in the brain's history, so its next turn tells it what the user heard, and only that one.
    await rig.say("what were you saying?")
    rig.brain.end()
    await rig.say("thanks")
    assert rig.brain.asked[1:] == [
        f"{heard('what were you saying?')}\n\n"
        '[hands] Your last turn was broken off. The user heard you say "1. Lighthouses stand on rocky coasts and h", then that it failed. '
        "Mention this only if the user asks.",
        heard("thanks"),
    ]


def broken_then_asked_again(rig: Rig, heard: Sequence[str], retried: str, ahead: Sequence[dict[str, object]] = ()) -> tuple[str, str]:
    """A reply the API breaks once `heard` has streamed, before its text block finishes, and the request Claude Code asks it
    again with, without a stream, answered `retried` whole after the blocks `ahead`: each told as the proxy tells it."""
    broken, _ = rig.request()
    rig.stream(broken, *heard)
    overloaded = {"type": "overloaded_error", "message": "Overloaded"}
    rig.stage.hear(Heard(broken, StreamError(overloaded)))
    reply = Reached(200, 0.0, 0.0, 9, Garbled(f"the API sent an error mid-stream: {json.dumps(overloaded)}"))
    rig.stage.hear(Exchanged(broken, BRAIN, MainTurn(None), None, "POST", "/v1/messages", 2, (), 0.0, 0.0, reply, False, root()))
    retry, _ = rig.request({"stream": False, "messages": [{"role": "user", "content": "hi"}]})
    message: dict[str, object] = {"type": "message", "id": "msg_1", "role": "assistant", "model": "claude-opus-5-5", "content": [*ahead, {"type": "text", "text": retried}], "stop_reason": "end_turn", "usage": {}}
    events = unstreamed(json.dumps(message).encode())
    assert isinstance(events, tuple)
    for event in events:
        rig.stage.hear(Heard(retry, event))
    return broken, retry


async def test_a_reply_asked_again_that_opens_as_the_broken_one_did_goes_on_from_where_the_user_stopped_hearing_it(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    broken, retry = broken_then_asked_again(rig, ("Lighthouses stand ", "on rocky"), "Lighthouses stand on rocky coasts.")
    await rig.until(lambda: len(rig.out.said()) == 3)
    rig.brain.end()
    await rig.turned()
    assert rig.out.said() == ["Lighthouses stand ", "on rocky", " coasts."]
    # One reply, so one sentence: the block the break left streaming goes on in the reply asked again, never ended at the break.
    texts = [at for at, frame in enumerate(rig.out.frames) if isinstance(frame, LLMTextFrame)]
    assert "LLMFullResponseEndFrame" not in rig.out.shape()[texts[1] : texts[2]]
    [turn] = turns(rig.recorded)
    assert turn.facts["text"] == "Lighthouses stand on rocky coasts."
    assert turn.facts["exchanges"] == (broken, retry)
    # [LAW:nothing-unseen] the break, what of it was heard, and that the reply asked again said it once.
    assert turn.facts["broken"] == (Broken(retry, "Lighthouses stand on rocky", repeated=True),)
    # What the user heard is all in the brain's history, as the reply asked again: there is nothing to tell it.
    await rig.say("go on")
    assert rig.brain.asked[1:] == [heard("go on")]
    rig.brain.end()


async def test_a_reply_asked_again_that_opens_with_thinking_goes_on_with_the_block_the_break_left_streaming(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    thinking: dict[str, object] = {"type": "thinking", "thinking": "Carry on with the sentence.", "signature": "sig"}
    broken_then_asked_again(rig, ("Lighthouses stand ", "on rocky"), "Lighthouses stand on rocky coasts.", ahead=(thinking,))
    await rig.until(lambda: len(rig.out.said()) == 3)
    rig.brain.end()
    await rig.turned()
    assert rig.out.said() == ["Lighthouses stand ", "on rocky", " coasts."]
    # Neither the thinking's start nor its stop ends the block: still one sentence.
    texts = [at for at, frame in enumerate(rig.out.frames) if isinstance(frame, LLMTextFrame)]
    assert "LLMFullResponseEndFrame" not in rig.out.shape()[texts[1] : texts[2]]


async def test_a_reply_asked_again_that_opens_otherwise_is_said_whole_and_the_brain_told_what_the_user_heard_of_the_broken_one(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    _, retry = broken_then_asked_again(rig, ("Lighthouses stand ", "on rocky"), "Most lighthouses stand on rocky coasts.")
    await rig.until(lambda: len(rig.out.said()) == 3)
    rig.brain.end()
    await rig.turned()
    # A text block of its own after the broken one, as the brain's next words after a finished block are.
    assert rig.out.said() == ["Lighthouses stand ", "on rocky", "\n\nMost lighthouses stand on rocky coasts."]
    assert turns(rig.recorded)[0].facts["broken"] == (Broken(retry, "Lighthouses stand on rocky"),)
    await rig.say("what were you saying?")
    rig.brain.end()
    await rig.say("thanks")
    note = '[hands] A reply of yours last turn broke off and was asked for again; your history holds only the reply asked again. Before it broke, the user heard you say "Lighthouses stand on rocky". Mention this only if the user asks.'
    # Told once: the turn after it broke nothing.
    assert rig.brain.asked[1:] == [f"{heard('what were you saying?')}\n\n{note}", heard("thanks")]


async def test_a_reply_broken_before_it_said_a_word_is_asked_again_with_nothing_to_tell(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    _, retry = broken_then_asked_again(rig, (), "Lighthouses stand on rocky coasts.")
    await rig.until(lambda: len(rig.out.said()) == 1)
    rig.brain.end()
    await rig.turned()
    assert rig.out.said() == ["Lighthouses stand on rocky coasts."]
    assert turns(rig.recorded)[0].facts["broken"] == (Broken(retry, ""),)
    await rig.say("go on")
    assert rig.brain.asked[1:] == [heard("go on")]
    rig.brain.end()


async def test_a_turn_that_fails_after_a_reply_was_asked_again_tells_the_brain_all_the_user_heard_of_it(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    _, retry = broken_then_asked_again(rig, ("Lighthouses stand ", "on rocky"), "Most lighthouses stand on rocky coasts.")
    await rig.until(lambda: len(rig.out.said()) == 3)
    rig.brain.end(BrainAnswered("p1", StopFailed("server_error", "API Error: Connection lost mid-response.")))
    await rig.until(lambda: len(rig.errors) == 1)
    assert turns(rig.recorded)[0].facts["broken"] == (Broken(retry, "Lighthouses stand on rocky"),)
    await rig.say("what were you saying?")
    # The turn's failure says all of it, the broken reply's words with the rest: one note, not one more for the break.
    spoken = "Lighthouses stand on rocky\n\nMost lighthouses stand on rocky coasts."
    note = f'[hands] Your last turn was broken off. The user heard you say "{spoken}", then that it failed. Mention this only if the user asks.'
    assert rig.brain.asked[1:] == [f"{heard('what were you saying?')}\n\n{note}"]
    rig.brain.end()


async def test_what_the_user_heard_of_a_broken_turn_is_told_with_the_first_turn_the_brain_takes(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    rig.stream(rig.request()[0], "Lighthouses stand")
    await rig.until(lambda: len(rig.out.said()) == 1)
    rig.brain.end(BrainAnswered("p1", StopFailed("server_error", "API Error: Connection lost mid-response.")))
    await rig.until(lambda: len(rig.errors) == 1)
    await rig.say("what were you saying?")
    rig.brain.fail(Untaken("the brain did not take the turn typed into it in 30s"))
    await rig.until(lambda: len(rig.errors) == 2)
    await rig.say("again")
    note = '[hands] Your last turn was broken off. The user heard you say "Lighthouses stand", then that it failed. Mention this only if the user asks.'
    assert rig.brain.asked[1:] == [f"{heard('what were you saying?')}\n\n{note}", f"{heard('again')}\n\n{note}"]
    rig.brain.end()


async def test_a_turn_that_breaks_after_the_brain_spoke_tells_the_next_where_the_user_stopped_hearing_it(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    rig.stream(rig.request()[0], "Lighthouses stand")
    await rig.until(lambda: len(rig.out.said()) == 1)
    # Broken in hands' own work, after the brain took the turn: its asker hears the failure, not an answer.
    rig.brain.fail(RuntimeError("no voice"))
    await rig.until(lambda: len(rig.errors) == 1)
    await rig.say("what were you saying?")
    note = '[hands] Your last turn was broken off. The user heard you say "Lighthouses stand", then that it failed. Mention this only if the user asks.'
    assert rig.brain.asked[1:] == [f"{heard('what were you saying?')}\n\n{note}"]
    rig.brain.end()


async def test_a_turn_the_brain_never_took_is_reported_as_the_model_stages_error_and_the_next_is_asked(rig: Rig) -> None:
    await rig.say("hello")
    # The brain is still running, so no watch reports it: the stage does.
    rig.brain.fail(Untaken("the brain did not take the turn typed into it in 30s"))
    await rig.until(lambda: len(rig.errors) == 1)
    assert rig.errors[0].processor is rig.stage
    assert "did not take the turn" in rig.errors[0].error
    await rig.say("again")
    await rig.until(lambda: rig.brain.asked == [heard("hello"), heard("again")])
    rig.brain.end()


async def test_a_stay_silent_answered_in_an_earlier_turn_does_not_hold_the_next(rig: Rig) -> None:
    await rig.say("(to a colleague) back in five")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "mcp__hands__stay_silent"))
    rig.brain.end()
    # The next turn's first request carries the old call's result beside the new question: it is not this turn's call.
    await rig.say("are you there?")
    body = answering("mcp__hands__stay_silent", {"silent": True})
    body["messages"] = [*body["messages"], {"role": "user", "content": "are you there?"}]  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    _, route = rig.request(body)
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.brain.end()


async def test_a_barge_in_before_the_brain_has_sent_the_turn_stops_nothing_and_the_turn_is_answered(rig: Rig) -> None:
    await rig.say("are you listening?")
    # Still waiting for the input: nothing of it has left, so there is nothing to stop, and the user's words follow it.
    await rig.interrupt()
    # The reply goes on through the barge-in, which ended it for whatever writes it: started again, so it is written whole.
    await rig.until(lambda: rig.out.shape() == ["LLMFullResponseStartFrame", "InterruptionFrame", "LLMFullResponseStartFrame"])
    _, route = rig.request()
    assert route == Send((Tail(TAIL),), refusal="final", span=APART)
    rig.brain.end()


async def test_each_text_block_of_a_turn_is_its_own_paragraph_so_a_closing_fence_stays_on_its_own_line(rig: Rig) -> None:
    """hands-narration-2mc.1zu: joined bare, the next block's words ran onto the closer and held the rest of the turn as code."""
    await rig.say("run the tests")
    exchange, _ = rig.request()
    rig.stage.hear(Heard(exchange, BlockStarted(0, {"type": "text", "text": ""})))
    rig.stream(exchange, "Here is what I ran:\n```bash\nnpm test\n```")
    rig.calls(exchange, ("t1", "mcp__hands__stage_draft"))
    second, _ = rig.request(answering("mcp__hands__stage_draft", {"readback": "staged"}))
    rig.stage.hear(Heard(second, BlockStarted(0, {"type": "text", "text": ""})))
    rig.stream(second, "All 42 tests passed.")
    rig.brain.end()
    await rig.turned()
    assert "".join(rig.out.said()) == "Here is what I ran:\n```bash\nnpm test\n```\n\nAll 42 tests passed."


async def test_a_permission_the_brain_holds_is_asked_after_its_words_and_the_users_yes_lets_its_turn_go_on(rig: Rig) -> None:
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt", "content": "hello"})
    await rig.until(lambda: "May I use Write on notes.txt? Say yes to allow it." in rig.out.said())
    assert rig.out.said() == ["I'll write it.", "May I use Write on notes.txt? Say yes to allow it."]
    # The user starting to answer stops what is playing, and stops nothing of the turn: it waits on their answer.
    await rig.cut_in("Yes, go ahead.")
    await rig.until(lambda: not asked.open)
    assert (await asked.decided()) == Allow()
    # The turn goes on, and what it says next is heard.
    after, route = rig.request()
    assert isinstance(route, Send) and route.refusal == "final"
    rig.stream(after, "Done, it says hello.")
    # A text block of its own, a paragraph after the one before the question.
    await rig.until(lambda: "\n\nDone, it says hello." in rig.out.said())
    rig.brain.end()
    await rig.turned()
    # The answer was the turn's, never a turn of its own.
    assert rig.brain.asked == [heard("make the notes say hello")]


async def test_anything_but_a_plain_yes_refuses_the_permission_and_hands_the_brain_the_users_words(rig: Rig) -> None:
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Edit", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: "May I use Edit on notes.txt? Say yes to allow it." in rig.out.said())
    await rig.cut_in("No, put it in the to-do list instead.")
    await rig.until(lambda: not asked.open)
    match (await asked.decided()):
        case Deny(message=message):
            assert '"No, put it in the to-do list instead."' in message
        case other:
            pytest.fail(f"a no allowed the edit: {other}")
    # Their next words, once nothing is asked, are a turn of their own.
    rig.brain.end()
    await rig.say("thanks")
    assert rig.brain.asked == [heard("make the notes say hello"), heard("thanks")]


async def refuse(rig: Rig, tool: str, input: dict[str, object], question: str) -> str:
    """The brain asks `question` for a permission, which the user hears to its end and refuses; the request that carries
    the refusal."""
    asked = rig.brain.permit(tool, input)
    before = rig.out.said().count(question)
    await rig.until(lambda: rig.out.said().count(question) > before)
    await rig.cut_in("no")
    await rig.until(lambda: not asked.open)
    after, _ = rig.request()
    return after


async def refuse_the_edit(rig: Rig) -> str:
    """A turn whose permission to edit the notes the user hears asked and refuses; the request that carries the refusal."""
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    return await refuse(rig, "Edit", {"file_path": "/Users/bmf/notes.txt"}, "May I use Edit on notes.txt? Say yes to allow it.")


async def test_a_refusal_the_brain_says_nothing_after_is_said_by_hands_so_silence_never_passes_for_it_done(rig: Rig) -> None:
    after = await refuse_the_edit(rig)
    rig.stream(after, "  ")
    rig.brain.end()
    await rig.turned()
    assert rig.out.said()[-1] == "I did not use Edit on notes.txt, so that is not done."
    assert acknowledgements(rig) == ["I did not use Edit on notes.txt, so that is not done."]
    [turn] = turns(rig.recorded)
    assert (turn.facts["refusals"], turn.counts["refused"]) == (("I did not use Edit on notes.txt, so that is not done.",), 1)


@pytest.mark.parametrize("why", [STOPPED, BROKEN], ids=["hands stopped its turn", "hands broke settling it"])
async def test_a_permission_heard_asked_then_refused_as_hands_ends_its_turn_is_counted_and_said_as_a_refusal(rig: Rig, why: str) -> None:
    """The refusal that ends a turn is a refusal the user heard asked all the same: silence never passes for it done."""
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    question = "May I use Edit on notes.txt? Say yes to allow it."
    asked = rig.brain.permit("Edit", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: question in rig.out.said())
    rig.brain.sessions.hear(Ending(rig.brain.session, DenyAndInterrupt(why)))
    assert (await asked.decided()) == DenyAndInterrupt(why)
    after, _ = rig.request()
    rig.stream(after, "  ")
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == ["I did not use Edit on notes.txt, so that is not done."]
    [turn] = turns(rig.recorded)
    assert turn.counts["refused"] == 1


async def test_a_refusal_the_brain_speaks_to_is_not_said_again_by_hands(rig: Rig) -> None:
    after = await refuse_the_edit(rig)
    rig.stream(after, "Okay, I left the notes alone.")
    rig.brain.end()
    await rig.turned()
    assert rig.out.said()[-1] == "\n\nOkay, I left the notes alone."
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert (turn.facts["refusals"], turn.counts["refused"]) == ((), 1)


async def test_a_refusal_the_user_speaks_over_is_left_to_the_turn_they_begin(rig: Rig) -> None:
    after = await refuse_the_edit(rig)
    rig.stream(after, "  ")
    await rig.until(lambda: "\n\n  " in rig.out.said())
    await rig.interrupt()
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert (turn.facts["refusals"], turn.counts["refused"], turn.facts["interrupted"]) == ((), 1, True)


async def test_a_permission_asked_and_refused_again_in_silence_is_said_once(rig: Rig) -> None:
    after = await refuse_the_edit(rig)
    rig.stream(after, "  ")
    after = await refuse(rig, "Edit", {"file_path": "/Users/bmf/notes.txt"}, "May I use Edit on notes.txt? Say yes to allow it.")
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == ["I did not use Edit on notes.txt, so that is not done."]
    [turn] = turns(rig.recorded)
    assert turn.counts["refused"] == 2


async def test_a_refused_command_is_named_by_its_opening_words(rig: Rig) -> None:
    await rig.say("build it")
    exchange, _ = rig.request()
    rig.stream(exchange, "On it.")
    await refuse(rig, "Bash", {"command": "make build && " * 40 + "rm -rf ~/x"}, f"May I use Bash to run {'make build && ' * 40}rm -rf ~/x? Say yes to allow it.")
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == ["I did not use Bash to run make build && make build && and so on, so that is not done."]


async def test_a_refusal_of_a_question_never_heard_to_its_end_is_not_said(rig: Rig) -> None:
    rig.out.holding = True
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Edit", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: "May I use Edit on notes.txt? Say yes to allow it." in rig.out.said())
    asked.settle(Deny(UNANSWERED))
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert (turn.facts["refusals"], turn.counts["refused"]) == ((), 0)


async def test_a_permission_held_after_the_user_spoke_over_its_turn_is_refused_and_never_asked(rig: Rig) -> None:
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    await rig.until(lambda: "I'll write it." in rig.out.said())
    await rig.interrupt()
    asked = rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt"})
    assert (await asked.decided()) == Deny(SPOKEN_OVER)
    assert rig.out.said() == ["I'll write it."]
    rig.brain.end()


async def test_words_said_before_a_permission_is_asked_are_a_turn_of_their_own_and_never_part_of_its_answer(rig: Rig) -> None:
    await rig.say("make the notes say hello")
    # Said before the turn's first request left: it waits as the next turn.
    await rig.cut_in("also check the logs")
    await asyncio.sleep(0.1)
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    asked = rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt"})
    await rig.until(lambda: "May I use Write on notes.txt? Say yes to allow it." in rig.out.said())
    await rig.cut_in("yes")
    await rig.until(lambda: not asked.open)
    assert (await asked.decided()) == Allow()
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked == [heard("make the notes say hello"), heard("also check the logs")]


async def test_a_permission_to_run_a_command_is_asked_with_the_command_it_would_run(rig: Rig) -> None:
    await rig.say("clean up the old branches")
    exchange, _ = rig.request()
    rig.stream(exchange, "On it.")
    rig.brain.permit("Bash", {"command": "git branch -D old", "description": "Delete the old branch"})
    await rig.until(lambda: "May I use Bash to run git branch -D old? Say yes to allow it." in rig.out.said())
    rig.brain.end()


async def test_a_command_is_asked_whole_however_long_and_a_fetch_with_its_address(rig: Rig) -> None:
    command = "make build && " * 40 + "rm -rf ~/x"
    await rig.say("build it")
    exchange, _ = rig.request()
    rig.stream(exchange, "On it.")
    run = rig.brain.permit("Bash", {"command": command})
    await rig.until(lambda: f"May I use Bash to run {command}? Say yes to allow it." in rig.out.said())
    run.settle(Deny("no"))
    rig.brain.permit("WebFetch", {"url": "https://example.com/x", "prompt": "read it"})
    await rig.until(lambda: "May I use WebFetch on https://example.com/x? Say yes to allow it." in rig.out.said())
    rig.brain.end()


async def test_a_yes_said_before_the_question_has_played_allows_nothing_and_the_brain_is_told_it_went_unheard(rig: Rig) -> None:
    question = "May I use Bash to run rm -rf build? Say yes to allow it."
    rig.out.holding = True
    await rig.say("clear out the build directory")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll clear out the build directory.")
    rig.calls(exchange, ("c1", "Bash"))
    asked = rig.brain.permit("Bash", {"command": "rm -rf build"})
    await rig.until(lambda: question in rig.out.said())
    # Said over what is still playing: the question was never heard to its end, so nothing waits on the user's words.
    rig.out.playing = Bookmark(("I'll clear out the build directory.", question), 0)
    await rig.interrupt()
    assert (await asked.decided()) == Deny(SPOKEN_OVER)
    await rig.worker.queue_frame(UserWords("yeah"))
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    over = f"[hands] The user spoke over you, and nothing more was said aloud. They did not hear: \"I'll clear out the build directory. {question}\"."
    assert rig.brain.asked == [heard("clear out the build directory"), f"{heard('yeah')}\n\n{over}"]
    rig.brain.end()


async def test_permissions_held_at_once_are_asked_one_at_a_time_and_each_answer_goes_to_the_one_heard(rig: Rig) -> None:
    await rig.say("build both")
    exchange, _ = rig.request()
    rig.stream(exchange, "Both, then.")
    first = rig.brain.permit("Bash", {"command": "make a"})
    second = rig.brain.permit("Bash", {"command": "make b"})
    await rig.until(lambda: "May I use Bash to run make a? Say yes to allow it." in rig.out.said())
    await asyncio.sleep(0.1)
    assert "May I use Bash to run make b? Say yes to allow it." not in rig.out.said()
    await rig.cut_in("yes")
    await rig.until(lambda: "May I use Bash to run make b? Say yes to allow it." in rig.out.said())
    assert (await first.decided()) == Allow() and second.open
    await rig.cut_in("not that one")
    await rig.until(lambda: not second.open)
    assert isinstance((await second.decided()), Deny)
    rig.brain.end()


async def test_a_permission_settled_before_its_turn_came_to_be_said_is_never_asked(rig: Rig) -> None:
    await rig.say("make the notes say hello")
    exchange, _ = rig.request()
    rig.stream(exchange, "I'll write it.")
    rig.brain.permit("Write", {"file_path": "/Users/bmf/notes.txt"}).settle(Deny(UNANSWERED))
    rig.stream(exchange, " Or not.")
    await rig.until(lambda: " Or not." in rig.out.said())
    assert rig.out.said() == ["I'll write it.", " Or not."]
    rig.brain.end()


async def test_a_permission_held_for_a_turn_the_stage_no_longer_asks_is_refused(rig: Rig) -> None:
    await rig.say("make the notes say hello")
    teller = rig.brain.tellers[-1]
    rig.brain.end()
    await rig.turned()
    asked = rig.brain.holding("Write", {"file_path": "/Users/bmf/notes.txt"})
    teller(asked)
    assert (await asked.decided()) == Deny(NOBODY)


def acknowledgements(rig: Rig) -> list[str]:
    """What hands said of the turns as its own lines: never the brain's words, and kept out of the context."""
    return [frame.text for frame in rig.out.frames if isinstance(frame, TTSSpeakFrame) and not frame.append_to_context]


async def test_a_users_turn_whose_call_runs_on_with_nothing_said_is_acknowledged_before_its_answer(rig: Rig) -> None:
    rig.now[0] = 1000.0
    await rig.release()
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.now[0] = 1001.0
    rig.calls(first, ("t1", "Bash"))
    # The call runs past the pause, and nothing of the turn has been said.
    rig.now[0] = 1003.0
    await rig.elapse()
    await rig.until(lambda: acknowledgements(rig) == ["One moment."])
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.stream(second, "The build broke on a missing import.")
    rig.brain.end()
    await rig.turned()
    # Said as a sentence of its own ahead of the answer, which is the brain's alone.
    assert rig.out.said() == ["One moment.", "The build broke on a missing import."]
    assert rig.out.shape()[:4] == ["LLMFullResponseStartFrame", "LLMFullResponseEndFrame", "TTSSpeakFrame", "LLMFullResponseStartFrame"]
    [turn] = turns(rig.recorded)
    assert (turn.facts["asides"], turn.facts["text"]) == ((("One moment.", 3000.0),), "The build broke on a missing import.")


async def test_a_turn_whose_answer_was_said_before_its_time_is_not_acknowledged(rig: Rig) -> None:
    await rig.say("what is api doing?")
    first, _ = rig.request()
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    second, _ = rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.stream(second, "api is running its tests.")
    await rig.until(lambda: rig.out.said() == ["api is running its tests."])
    await rig.elapse()
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert turn.facts["asides"] == ()


async def test_a_turn_that_said_what_it_would_do_before_its_call_is_not_acknowledged(rig: Rig) -> None:
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.stream(first, "Let me look at the build log.")
    rig.calls(first, ("t1", "Bash"))
    await rig.elapse()
    rig.brain.end()
    await rig.turned()
    assert rig.out.said() == ["Let me look at the build log."]


async def test_a_turn_hands_narrates_is_never_acknowledged_since_nobody_waits_on_it(rig: Rig) -> None:
    await rig.worker.queue_frame(Narrated("[hands] The Claude Code session api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: len(rig.brain.asked) == 1)
    first, _ = rig.request()
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    await rig.elapse()
    rig.brain.end()
    await rig.turned()
    assert acknowledgements(rig) == []


async def test_a_turn_is_acknowledged_once_however_many_calls_run_on_and_the_next_differently(rig: Rig) -> None:
    for asked in ("find out why the build broke", "and the tests?"):
        await rig.say(asked)
        first, _ = rig.request()
        rig.calls(first, ("t1", "Bash"), ("t2", "Grep"))
        await rig.elapse()
        rig.brain.end()
        await rig.until(lambda: len(turns(rig.recorded)) == len(rig.brain.asked))
    assert acknowledgements(rig) == ["One moment.", "On it."]


async def test_words_said_while_a_call_runs_with_nothing_to_hear_stop_nothing_and_are_answered_at_once(rig: Rig) -> None:
    rig.now[0] = 1000.0
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.stream(first, "Let me look at the build log.")
    rig.calls(first, ("t1", "Bash"))
    # Everything said has played, and the call runs: the user adds to what they asked.
    await rig.interrupt()
    rig.now[0] = 1004.0
    await rig.release()
    rig.now[0] = 1005.0
    await rig.stop()
    await rig.worker.queue_frame(UserWords("and check the tests too"))
    await rig.until(lambda: "Got it. I'll come to that when this is done." in rig.out.said())
    # The turn goes on being said, and is acknowledged no more: the user has heard hands answer them.
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.stream(second, "The build broke on a missing import.")
    await rig.until(lambda: "\n\nThe build broke on a missing import." in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert acknowledgements(rig) == ["Got it. I'll come to that when this is done."]
    [turn] = turns(rig.recorded)
    # Timed from the end of the words, the release, as the turn's own wait is.
    assert (turn.facts["interrupted"], turn.counts["cut_in"], turn.facts["asides"], turn.facts["spoken_over"]) == (False, 1, (("Got it. I'll come to that when this is done.", 1000.0),), "")
    # The words are the next turn, with no note: nothing was spoken over.
    assert rig.brain.asked[1] == heard("and check the tests too")
    rig.brain.end()


async def test_each_utterance_waiting_behind_a_turn_is_answered_and_differently(rig: Rig) -> None:
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    await rig.cut_in("and the tests")
    await rig.until(lambda: len(acknowledgements(rig)) == 1)
    # Said over hands' own answer, which is no word of the brain's: no barge-in, and answered again.
    rig.out.holding = True
    await rig.cut_in("and the lint")
    await rig.until(lambda: len(acknowledgements(rig)) == 2)
    assert acknowledgements(rig) == ["Got it. I'll come to that when this is done.", "Heard you. That's next, once this finishes."]
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    [turn] = turns(rig.recorded)
    assert turn.facts["interrupted"] is False
    # Asked as one turn, as all the words waiting are.
    assert rig.brain.asked[1] == heard("and the tests\n\nand the lint")
    rig.brain.end()


async def test_words_said_while_the_brains_words_still_play_are_a_barge_in_and_unanswered(rig: Rig) -> None:
    rig.out.holding = True
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.stream(first, "Let me look at the build log.")
    rig.calls(first, ("t1", "Bash"))
    await rig.cut_in("and the tests")
    await asyncio.sleep(0.1)
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert turn.facts["interrupted"] is True
    rig.brain.end()


async def test_words_said_while_a_text_block_streams_are_a_barge_in(rig: Rig) -> None:
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.stream(first, "Let me look")
    await rig.interrupt()
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert turn.facts["interrupted"] is True


async def test_a_text_blocks_last_sentence_is_said_as_it_closes_and_a_cut_after_loses_none_of_it(rig: Rig) -> None:
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.stream(first, "Let me look.")
    rig.close(first)
    # The reply ends at the block's end, so what TTS holds of it is said now, ahead of the call it makes.
    await rig.until(lambda: rig.out.shape()[-2:] == ["LLMFullResponseEndFrame", "LLMFullResponseStartFrame"])
    assert rig.out.said() == ["Let me look."]
    rig.calls(first, ("t1", "Bash"))
    await rig.interrupt()
    await rig.stop()
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["interrupted"], turn.counts["cut_in"], turn.facts["text"]) == (False, 1, "Let me look.")


async def test_words_that_arrive_while_a_text_block_streams_are_answered_after_it(rig: Rig) -> None:
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    # Cut in while nothing was to be heard; the call came back while the user spoke, and the brain speaks once they stop.
    await rig.interrupt()
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.stream(second, "Let me ch")
    await rig.stop()
    await rig.worker.queue_frame(UserWords("and the tests"))
    await rig.until(lambda: "Let me ch" in rig.out.said())
    await asyncio.sleep(0.1)
    assert "Got it. I'll come to that when this is done." not in rig.out.said()
    rig.stream(second, "eck the file.")
    rig.close(second)
    await rig.until(lambda: "Got it. I'll come to that when this is done." in rig.out.said())
    assert rig.out.said()[-3:] == ["Let me ch", "eck the file.", "Got it. I'll come to that when this is done."]
    rig.brain.end()


async def test_what_a_turn_says_while_the_user_speaks_over_its_silence_waits_for_them_and_it_is_not_acknowledged(rig: Rig) -> None:
    rig.now[0] = 1000.0
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    # The user cuts in on the call's silence: no barge-in, and the floor is theirs until their turn stops. The pause before
    # an acknowledgement runs out while they speak, and the call comes back with the answer.
    await rig.interrupt()
    await rig.elapse()
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.now[0] = 1001.0
    rig.stream(second, "The build broke on a missing import.")
    rig.close(second)
    await asyncio.sleep(0.1)
    assert rig.out.said() == []
    rig.now[0] = 1004.0
    await rig.stop()
    await rig.until(lambda: rig.out.said() == ["The build broke on a missing import."])
    rig.brain.end()
    await rig.turned()
    # A turn the user spoke in is answered, never acknowledged: they said nothing here, so nothing of hands' own is said.
    assert acknowledgements(rig) == []
    [turn] = turns(rig.recorded)
    assert (turn.facts["interrupted"], turn.counts["cut_in"], turn.facts["asides"]) == (False, 1, ())
    # The user waited until the first word reached hands: the time they spent speaking after it is their own.
    assert turn.facts["waited_ms"] == 1000.0


async def test_words_said_over_a_tellings_silence_are_the_next_turn_and_not_answered_in_it(rig: Rig) -> None:
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), (unasked(),)))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    first, _ = rig.request()
    rig.calls(first, ("t1", "mcp__hands__read_session"))
    await rig.cut_in("what about web?")
    await asyncio.sleep(0.1)
    second, _ = rig.request(answering("mcp__hands__read_session", {"steps": []}))
    rig.stream(second, "api opened pull request 68.")
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    # Nothing but the telling is said in it: an answer there would be heard as the telling's own.
    assert acknowledgements(rig) == []
    assert rig.out.uttering == ["Uttering", "Resumed", "api opened pull request 68.", "Uttered"]
    assert rig.brain.asked[1] == heard("what about web?")
    rig.brain.end()


async def test_an_answer_waiting_behind_a_block_the_turn_ends_in_is_said(rig: Rig) -> None:
    await rig.say("tell me about lighthouses")
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    await rig.interrupt()
    await rig.stop()
    second, _ = rig.request(answering("Bash", {"output": "ok"}))
    rig.stream(second, "Lighthouses stand ")
    await rig.worker.queue_frame(UserWords("and the tests"))
    await asyncio.sleep(0.1)
    assert "Got it. I'll come to that when this is done." not in rig.out.said()
    # The API breaks the reply past its retry, and the turn ends with the block open.
    rig.brain.end(BrainAnswered("p1", StopFailed("server_error", "API Error: Connection lost mid-response.")))
    await rig.until(lambda: "Got it. I'll come to that when this is done." in rig.out.said())
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert turn.facts["asides"] == (("Got it. I'll come to that when this is done.", 0.0),)
    rig.brain.end()


async def test_a_question_asked_after_the_cut_is_not_answered_by_its_words(rig: Rig) -> None:
    await rig.say("clean up the old branches")
    first, _ = rig.request()
    rig.calls(first, ("c1", "Bash"))
    # The user cuts in while the call runs, and nothing was to be heard: no barge-in.
    await rig.interrupt()
    # The brain asks while they speak: the question waits for them to stop, then plays to its end before their words arrive.
    asked = rig.brain.permit("Bash", {"command": "git branch -D old"})
    question = "May I use Bash to run git branch -D old? Say yes to allow it."
    await asyncio.sleep(0.1)
    assert question not in rig.out.said()
    await rig.stop()
    await rig.until(lambda: question in rig.out.said())
    await rig.worker.queue_frame(UserWords("and the stale tags"))
    await asyncio.sleep(0.1)
    # Not its answer: it is still open, and the words wait as the next turn. Hands' answer to them waits on the question
    # the user heard last (docs/barge-in.md, the accepted windows).
    assert asked.open
    assert "Got it. I'll come to that when this is done." not in rig.out.said()
    await rig.cut_in("yes")
    await rig.until(lambda: not asked.open)
    assert (await asked.decided()) == Allow()
    await rig.until(lambda: "Got it. I'll come to that when this is done." in rig.out.said())
    rig.brain.end()
    await rig.until(lambda: len(rig.brain.asked) == 2)
    assert rig.brain.asked[1] == heard("and the stale tags")
    rig.brain.end()


async def test_a_question_queued_behind_one_heard_does_not_make_its_answer_a_barge_in(rig: Rig) -> None:
    await rig.say("build both")
    exchange, _ = rig.request()
    rig.calls(exchange, ("c1", "Bash"), ("c2", "Bash"))
    first = rig.brain.permit("Bash", {"command": "make a"})
    second = rig.brain.permit("Bash", {"command": "make b"})
    await rig.until(lambda: "May I use Bash to run make a? Say yes to allow it." in rig.out.said())
    await rig.cut_in("yes")
    await rig.until(lambda: not first.open)
    assert (await first.decided()) == Allow() and second.open
    rig.brain.end()


async def test_a_reply_broken_in_a_call_block_runs_no_call(rig: Rig) -> None:
    await rig.say("what is api doing?")
    first, _ = rig.request()
    rig.stage.hear(Heard(first, BlockStarted(0, {"type": "tool_use", "id": "t1", "name": "mcp__hands__read_session", "input": {}})))
    # Broken before the block was whole, and asked again without a stream.
    retry, _ = rig.request({"stream": False, "messages": [{"role": "user", "content": "hi"}]})
    rig.stream(retry, "api is idle.")
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.counts["tools"], [entry for entry in rig.recorded if isinstance(entry, WideEvent) and entry.event == "tool.call"]) == (0, [])
    assert not rig.pauses


async def test_a_retry_that_opens_with_a_call_ends_the_text_the_broken_reply_left_streaming(rig: Rig) -> None:
    await rig.say("what is api doing?")
    first, _ = rig.request()
    rig.stream(first, "Let me look.")
    retry, _ = rig.request({"stream": False, "messages": [{"role": "user", "content": "hi"}]})
    rig.calls(retry, ("t1", "mcp__hands__read_session"))
    # Ended ahead of the call: the reply's end says what TTS held of it.
    await rig.until(lambda: rig.out.shape()[-2:] == ["LLMFullResponseEndFrame", "LLMFullResponseStartFrame"])
    await rig.interrupt()
    await rig.stop()
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert turn.facts["interrupted"] is False


async def test_tool_work_of_short_calls_that_runs_on_with_nothing_said_is_acknowledged(rig: Rig) -> None:
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.calls(first, ("t1", "Grep"))
    # Each call comes straight back, and the model goes on to the next: the work runs on, unheard.
    second, _ = rig.request(answering("Grep", {"matches": []}))
    rig.calls(second, ("t2", "Read"))
    await rig.elapse()
    await rig.until(lambda: acknowledgements(rig) == ["One moment."])
    rig.brain.end()


async def test_a_turn_that_ends_while_its_acknowledgement_still_plays_tells_it(rig: Rig) -> None:
    rig.now[0] = 1000.0
    rig.out.holding = True
    await rig.say("find out why the build broke")
    first, _ = rig.request()
    rig.calls(first, ("t1", "Bash"))
    rig.now[0] = 1002.0
    await rig.elapse()
    second, _ = rig.request(answering("Bash", {"output": "error"}))
    rig.stream(second, "A missing import.")
    rig.brain.end()
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert turn.facts["asides"] == (("One moment.", 2000.0),)


# What the next turn carries of a turn stopped before Claude Code held any of its reply.
UNKEPT = "your history does not hold"


def turn_of(rig: Rig) -> str:
    """The turn the stage has in flight, as the phone's page is shown it."""
    match rig.stage.work():
        case Working(turn=turn):
            return turn
        case idle:
            raise AssertionError(f"no turn in flight: {idle}")


async def test_the_work_shown_is_the_call_the_turn_runs_with_its_input_until_its_result_leaves(rig: Rig) -> None:
    assert rig.stage.work() == Idle()
    await rig.say("wait thirty seconds")
    match rig.stage.work():
        case Working(calls=()):
            pass
        case other:
            raise AssertionError(f"a turn the model is still replying to runs no call: {other}")
    exchange, _ = rig.request()
    rig.call(exchange, "t1", "Bash", '{"command": "sl', 'eep 30"}')
    rig.call(exchange, "t2", "mcp__hands__read_session", '{"session": "api"}')
    match rig.stage.work():
        case Working(calls=(Running(tool="Bash", input={"command": "sleep 30"}), Running(tool="read_session", input={"session": "api"}))):
            pass
        case other:
            raise AssertionError(f"each call is shown by the name the user knows, with its input: {other}")
    # The request carrying a result is the end of that call's run: only the other runs on.
    rig.request(answering("Bash", {"output": ""}, "t1"))
    match rig.stage.work():
        case Working(calls=(Running(tool="read_session"),)):
            pass
        case other:
            raise AssertionError(f"a call whose result left is not running: {other}")
    rig.brain.end()
    await rig.until(lambda: rig.stage.work() == Idle())


async def test_a_call_whose_input_is_not_json_is_shown_as_unreadable_never_as_an_input(rig: Rig) -> None:
    await rig.say("run it")
    exchange, _ = rig.request()
    rig.call(exchange, "t1", "Bash", '{"command": ')
    match rig.stage.work():
        case Working(calls=(Running(tool="Bash", input=Unreadable(reason=reason)),)):
            assert "Bash" in reason
        case other:
            raise AssertionError(f"an input the wire cannot read is said to be unreadable: {other}")
    rig.brain.end()


async def test_a_stop_of_the_turn_shown_ends_it_as_the_user_asked_and_never_as_a_failure(rig: Rig) -> None:
    await rig.say("wait thirty seconds")
    exchange, _ = rig.request()
    rig.call(exchange, "t1", "Bash", '{"command": "sleep 30"}')
    assert rig.stage.stop(turn_of(rig)) == Stopping() and rig.brain.interrupts == 1
    # Stopped mid-call, as Escape stops it: a turn that acted and said nothing of it, which the user asked for.
    rig.brain.end(BrainStopped())
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["stopped"], turn.facts["held_reply"], turn.facts["failed"], turn.outcome, rig.errors) == (True, True, None, "ok", [])
    # Claude Code held the call's block, so its history keeps the turn: the next carries nothing of it.
    await rig.say("never mind")
    assert rig.brain.asked[-1].startswith("never mind") and UNKEPT not in rig.brain.asked[-1]
    rig.brain.end()


async def test_a_turn_stopped_before_any_of_its_reply_was_held_is_told_the_brain_with_the_next(rig: Rig) -> None:
    await rig.say("write me a long story")
    exchange, _ = rig.request()
    # Text streaming: no block of the reply has ended, so Claude Code holds none of it, and Escape gives the prompt back.
    rig.stream(exchange, "Once upon")
    assert rig.stage.stop(turn_of(rig)) == Stopping()
    rig.brain.end(BrainStopped())
    await rig.turned()
    [turn] = turns(rig.recorded)
    assert (turn.facts["stopped"], turn.facts["held_reply"]) == (True, False)
    await rig.say("something shorter")
    assert rig.brain.asked[-1].startswith("something shorter")
    assert f'{UNKEPT} what the user asked in it: "write me a long story". The user heard you say "Once upon" before they stopped it.' in rig.brain.asked[-1]
    rig.brain.end()


async def test_turns_stopped_one_after_another_before_any_reply_tell_the_last_by_its_words_alone(rig: Rig) -> None:
    for words in ("write me a long story", "something shorter"):
        await rig.say(words)
        rig.request()
        assert rig.stage.stop(turn_of(rig)) == Stopping()
        rig.brain.end(BrainStopped())
        await rig.until(lambda: rig.stage.work() == Idle())
    # What the user could see as they asked, typed beside each turn's words: the same screen for every turn here.
    seen = rig.brain.asked[0].removeprefix("write me a long story\n\n")
    assert seen and seen != rig.brain.asked[0]
    await rig.say("just a poem")
    told = rig.brain.asked[-1]
    # The stopped turn is told by its words, never by the screen typed beside them, which this turn says once, nor by the
    # note it was typed with, of the turn before it, which the user's stop left behind.
    assert (told.count(seen), told.count(UNKEPT)) == (1, 1)
    assert f'{UNKEPT} what the user asked in it: "something shorter". Do not act' in told
    rig.brain.end()


async def test_a_telling_of_hands_stopped_before_any_reply_is_told_the_brain_as_hands_words_never_the_user_s(rig: Rig) -> None:
    await rig.worker.queue_frame(Narrated("[hands] api finished a turn.", "api finished a turn, and I could not tell it.", SessionId("api"), ()))
    await rig.until(lambda: rig.brain.asked == ["[hands] api finished a turn."])
    rig.stream(rig.request()[0], "api opened")
    assert rig.stage.stop(turn_of(rig)) == Stopping()
    rig.brain.end(BrainStopped())
    await rig.until(lambda: rig.stage.work() == Idle())
    await rig.say("what happened")
    assert f'{UNKEPT} what hands told you in it: "[hands] api finished a turn."' in rig.brain.asked[-1]
    rig.brain.end()


async def test_a_stop_of_a_turn_that_has_ended_or_of_no_turn_stops_nothing(rig: Rig) -> None:
    assert rig.stage.stop("any") == NotWorking()
    await rig.say("hello")
    shown = turn_of(rig)
    rig.brain.end()
    await rig.turned()
    await rig.say("and again")
    # The turn shown has ended, and the brain is on another: that one goes on.
    assert rig.stage.stop(shown) == AlreadyEnded() and rig.brain.interrupts == 0
    rig.brain.end()


# A turn Claude Code opens for a background task's result, as its opening request carries it (backgrounded_then_notified.jsonl, p2).
NOTICE = "<task-notification>\n<task-id>b74jw7b5h</task-id>\n<status>completed</status>\n</task-notification>"


def notified() -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "Filing it."}, {"role": "user", "content": NOTICE}]}


async def test_a_turns_requests_after_its_opening_and_a_notification_queued_into_it_are_all_its_own(rig: Rig) -> None:
    await rig.say("run it")
    exchange, _ = rig.request()
    rig.calls(exchange, ("t1", "Bash"))
    # The notification queued beside the call's result is the running turn's, under its id (notified_mid_turn.jsonl, p1).
    queued = answering("Bash", {"output": ""})
    queued["messages"][-1]["content"].append({"type": "text", "text": NOTICE})  # pyright: ignore[reportIndexIssue, reportUnknownMemberType, reportAttributeAccessIssue]
    after, _ = rig.request(queued)
    rig.stream(after, "Both done.")
    rig.brain.end()
    await rig.turned()
    assert rig.brain.unawaiting == []
    (event,) = turns(rig.recorded)
    assert (event.facts["origin"], event.facts["opened_by"], event.facts["text"]) == ("awaited", "asked", "Both done.")


async def test_a_turn_claude_code_opens_for_a_notification_is_spoken_and_ends_on_its_own_end(rig: Rig) -> None:
    exchange, _ = rig.request(notified())
    rig.stream(exchange, "The ticket is filed.")
    await rig.until(lambda: rig.out.said() == ["The ticket is filed."])
    (told,) = rig.brain.unawaiting
    told.set_result(ANSWERED)
    await rig.turned()
    (event,) = turns(rig.recorded)
    assert (event.facts["origin"], event.facts["opened_by"], event.facts["text"], event.outcome) == ("unawaited", "notified", "The ticket is filed.", "ok")
    # Nobody waited on it, so nothing acknowledged it.
    assert event.facts["asides"] == ()


async def test_the_users_turn_spoken_while_a_notified_turn_runs_waits_behind_it_and_each_says_only_its_own(rig: Rig) -> None:
    exchange, _ = rig.request(notified())
    rig.stream(exchange, "The ticket is filed.")
    await rig.until(lambda: rig.out.said() == ["The ticket is filed."])
    await rig.worker.queue_frame(UserWords("and the other one?"))
    # Heard, and told it is next, but not typed while the notified turn runs.
    await asyncio.sleep(0.05)
    assert rig.brain.asked == []
    rig.now[0] += 4.0
    rig.brain.unawaiting[0].set_result(ANSWERED)
    await rig.until(lambda: rig.brain.asked == [heard("and the other one?")])
    mine, _ = rig.request()
    rig.stream(mine, "Filing that too.")
    rig.brain.end()
    await rig.turned(2)
    notice, asked = turns(rig.recorded)
    assert (notice.facts["origin"], notice.facts["text"]) == ("unawaited", "The ticket is filed.")
    # Told, in the notified turn's voice, that what they said is next.
    assert [aside[0] for aside in cast(tuple[tuple[str, float], ...], notice.facts["asides"])] == [ANSWERS[0]]
    assert (asked.facts["origin"], asked.facts["text"]) == ("awaited", "Filing that too.")
    # It waited the notified turn out, and nothing was stopped.
    assert asked.facts["queued_ms"] == 4000.0 and rig.brain.interrupts == 0


async def test_a_notified_turn_opened_as_the_users_turn_is_typed_goes_first_and_each_ends_on_its_own(rig: Rig) -> None:
    await rig.say("what's next?")
    # Claude Code opened its own turn as hands typed: the user's waits in its queue behind it.
    first, _ = rig.request(notified())
    rig.stream(first, "The ticket is filed.")
    rig.brain.unawaiting[0].set_result(ANSWERED)
    await rig.turned()
    mine, _ = rig.request()
    rig.stream(mine, "The backlog is next.")
    rig.brain.end()
    await rig.turned(2)
    notice, asked = turns(rig.recorded)
    assert (notice.facts["origin"], notice.facts["text"]) == ("unawaited", "The ticket is filed.")
    assert (asked.facts["origin"], asked.facts["text"], asked.facts["failed"]) == ("awaited", "The backlog is next.", None)



async def test_a_users_turn_failed_untaken_behind_a_long_notified_turn_is_said_to_fail_and_its_late_answer_is_still_spoken(rig: Rig) -> None:
    await rig.say("what's next?")
    first, _ = rig.request(notified())
    rig.stream(first, "The ticket is filed.")
    # The notified turn outlasts the user's turn's time to be taken: that turn fails, aloud.
    rig.brain.turns[0].set_exception(Untaken("the brain did not take the turn typed into it in 30s"))
    rig.brain.unawaiting[0].set_result(ANSWERED)
    await rig.turned(2)
    # What was typed for it then runs in Claude Code's queue, a turn nobody waits on, and its answer is spoken.
    late, _ = rig.request()
    rig.stream(late, "The backlog is next.")
    await rig.until(lambda: "The backlog is next." in rig.out.said())
    rig.brain.unawaiting[1].set_result(ANSWERED)
    await rig.turned(3)
    events = turns(rig.recorded)
    assert sorted((event.facts["origin"], event.facts["opened_by"], event.outcome) for event in events) == [
        ("awaited", "asked", "failed"),
        ("unawaited", "asked", "ok"),
        ("unawaited", "notified", "ok"),
    ]
    assert rig.errors and "did not take" in rig.errors[0].error
