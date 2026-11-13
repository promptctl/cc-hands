"""The start of a user turn, and the barge-in: the moment it cuts off what hands is saying, which the edge that opened it decides.

Every turn starts as the hold that opens it opens, so its words gather from the first hold and the floor is the user's
from the moment they speak (`hands.voice.floor`). Starting is not cutting. The barge-in is the user taking over: Pipecat's
start of a turn, which the user aggregator broadcasts as `UserStartedSpeakingFrame` and then the interruption that stops
the speaker, cancels the reply still streaming from the model and its calls in flight, and settles a line waiting to be
heard as cut off. A turn the user's hand opened cuts as it opens; one the voice opened cuts once the hold owner pushes words for
it, Pipecat's transcription start: the first a listener hears in a hold still open (`InterimWords`), or the hold's own words as it
closes, where it heard none before. A turn that ends uncut has taken nothing over: nothing is cut off, so nothing has to
be undone, and the reply it found on its way goes on as if no turn had opened, a tool's result included. The barge-in can
land while the turn's own words are on their way, so they are uninterruptible on both sides of the user aggregator
(`hands.voice.turnstop.Words`, `hands.voice.conversation.UserWords`): an interruption stops hands, never the user. And it lands just
ahead of them, so the user aggregator takes none of them in until the reply the barge-in stopped is written as cut off
(`BargeInWritten`): the audit log reads the reply, then the words that cut it off.
"""

import asyncio
from dataclasses import dataclass

from pipecat.frames.frames import Frame, SystemFrame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import BaseUserTurnStartStrategy

from hands.clock import Clock
from hands.sessions.audit import Record, TurnStart, UserTurn
from hands.voice.trigger import turn_start
from hands.voice.turnstop import InterimWords, TurnOpened, Words


class BargeInWritten(SystemFrame):
    """A barge-in reached the end of the pipeline, and the reply it stopped is written as cut off: sent back up
    by the assistant aggregator (`hands.voice.conversation.AssistantTurns`), the last thing every interruption reaches."""


@dataclass
class _Open:
    """The turn open: when the edge that opened it has it barge in, when it opened, and how long after that it did, None until
    it has."""

    start: TurnStart
    opened: float
    barge_in_after: float | None = None


class EdgeTurnStart(BaseUserTurnStartStrategy):
    """Starts the user turn as its first hold opens, and barges in on hands when the edge that opened each hold says
    (`hands.voice.trigger.turn_start`).

    [LAW:single-enforcer] the one place a turn's start and its barge-in are decided: the pipeline runs no other start strategy,
    and Pipecat says nothing of its own as the turn starts, neither that the user started speaking nor an interruption.
    """

    def __init__(self, clock: Clock, record: Record) -> None:
        super().__init__(enable_interruptions=False)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._record = record
        self._now = clock.now
        # Between turns, None: Pipecat ends a turn, and this is told so (`handle_user_turn_stopped`).
        self._turn: _Open | None = None
        # Each barge-in, for the user aggregator to broadcast, and each written, for it to go on behind (`interrupting`).
        self._register_event_handler("on_barge_in", sync=True)
        self._register_event_handler("on_barge_in_written", sync=True)

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        match frame, self._turn:
            case TurnOpened(hold=hold), None:
                start = turn_start(hold.opener)
                self._turn = _Open(start, self._now())
                await self.trigger_user_turn_started(enable_user_speaking_frames=False)
                await self._barge_in_where(start, self._turn)
            case TurnOpened(hold=hold), _Open() as turn:
                # A hold pressed while the hold owner is still on the last joins its turn, and cuts as its own edge says.
                await self._barge_in_where(turn_start(hold.opener), turn)
            case Words() | InterimWords(), _Open() as turn:
                # Words, in whichever hold, heard while it was open or as it closed, are someone speaking.
                await self._barge_in(turn)
            case Words() | InterimWords(), None:
                # [LAW:no-silent-failure] a turn ends only once the hold owner is done with every hold, behind each one's words.
                raise RuntimeError("the hold owner pushed words outside any user turn")
            case BargeInWritten(), _:
                await self._call_event_handler("on_barge_in_written")  # pyright: ignore[reportUnknownMemberType]  (Pipecat's *args is untyped)
            case _:
                pass
        return ProcessFrameResult.CONTINUE

    async def handle_user_turn_stopped(self) -> None:
        match self._turn:
            case _Open(start=start, barge_in_after=barge_in_after):
                # [LAW:nothing-unseen] every turn is a line: how its edge had it barge in, and when, or that it did not.
                self._record(UserTurn(start, barge_in_after))
                self._turn = None
            case None:
                # [LAW:no-silent-failure] Pipecat stops only a turn it started, and nothing but this starts one.
                raise RuntimeError("Pipecat stopped a user turn this never started")

    async def _barge_in_where(self, start: TurnStart, turn: _Open) -> None:
        match start:
            case "on the hold":
                await self._barge_in(turn)
            case "on words":
                pass

    async def _barge_in(self, turn: _Open) -> None:
        """Barge in on hands, once a turn."""
        match turn.barge_in_after:
            case None:
                turn.barge_in_after = self._now() - turn.opened
                await self._call_event_handler("on_barge_in")  # pyright: ignore[reportUnknownMemberType]  (Pipecat's *args is untyped)
            case _:
                pass


def interrupting(start: EdgeTurnStart, turns: LLMUserAggregator) -> None:
    """Have `turns`, the user aggregator `start` is the start strategy of, broadcast each barge-in `start` makes, and go on
    with the turn only once the barge-in is written.

    The aggregator broadcasts it as it broadcasts the start of a turn Pipecat starts, in the same order: that the user
    started speaking, which the assistant side reads as the user having taken over, so it runs the model on no tool's
    result until the turn is sent; then the interruption. Both go inline, from within the frame being processed, so they
    are pushed ahead of anything behind that frame. Queued as frames of their own, the turn's words could leave for the
    model ahead of them, and the interruption would then cancel the reply to the very words that made it.
    """
    # [LAW:no-ambient-temporal-coupling] the reply a barge-in stops is written as cut off where the interruption reaches
    # the end of the pipeline, and the words that cut it where they reach this aggregator, a few hops apart. What orders
    # the two is state: no frame behind the barge-in is taken in until the end of the pipeline says the reply is written.
    written = asyncio.Event()

    @start.event_handler("on_barge_in")
    async def barge_in(_start: EdgeTurnStart) -> None:  # pyright: ignore[reportUnusedFunction]
        written.clear()
        await turns.broadcast_frame(UserStartedSpeakingFrame)  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # The interruption resets what the aggregator holds back, so the hold is taken behind it.
        await turns.broadcast_interruption()
        await turns.pause_processing_frames()
        if written.is_set():
            await turns.resume_processing_frames()

    @start.event_handler("on_barge_in_written")
    async def barge_in_written(_start: EdgeTurnStart) -> None:  # pyright: ignore[reportUnusedFunction]
        written.set()
        await turns.resume_processing_frames()
