"""The conversation with the brain: the user's words handed to its stage turn by turn, and each turn written to the audit log.

Pipecat's aggregators are built for a model service that re-reads a whole message list every turn. The brain is no such
service: Claude Code keeps its own transcript, so hands keeps no list for it. The user side hands on the words of the turn
it ended, and nothing else; the list both sides write to keeps nothing. The turns themselves stay Pipecat's: when each
starts and ends, what was said of a reply before a barge-in cut it off, and the events the audit log is written from.
"""

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    UninterruptibleFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.processors.frame_processor import FrameDirection

from hands.sessions.audit import Record, Replied, Transcribed
from hands.voice.turnstart import CutWritten


@dataclass
class UserWords(DataFrame, UninterruptibleFrame):
    """The words of a user's turn as it ends, for the brain's stage to ask the brain.

    Uninterruptible: they are the only copy, so a barge-in landing as they travel must not drop them.
    """

    words: str


class Unkept(LLMContext):
    """A message list that keeps nothing written to it: what both aggregators write to, so nothing grows with the turns.

    [LAW:one-source-of-truth] the brain's transcript in Claude Code is the conversation; the audit log is hands' record of it.
    Every write Pipecat makes ends in one of these three (1.10.0): `transform_messages` sets, and the image and audio
    messages add.
    """

    def add_message(self, message: LLMContextMessage) -> None:
        pass

    def add_messages(self, messages: list[LLMContextMessage]) -> None:
        pass

    def set_messages(self, messages: list[LLMContextMessage]) -> None:
        pass


class UserTurns(LLMUserAggregator):
    """Pipecat's user aggregator, which hands the turn it ends on as its words alone rather than as the message list.

    Every path that ends a turn writes it in `_push_aggregation` (1.10.0, `llm_response_universal.py:888`): this is that
    write, with the words pushed where Pipecat appends them and pushes the list, and the same event after it. hands runs
    no speculative inference, so every turn's words are asked.
    """

    async def _push_aggregation(self, *, run_llm: bool = True) -> str:
        if not self._aggregation:
            return ""
        words = self.aggregation_string()
        await self.reset()
        await self.push_frame(UserWords(words))
        await self._call_event_handler("on_user_turn_message_added", UserTurnMessageAddedMessage(content=words, timestamp=self._user_turn_start_timestamp))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        return words


class AssistantTurns(LLMAssistantAggregator):
    """Pipecat's assistant aggregator, which also ends the turn of a line hands says as written and counts as a
    reply: written once it is said, as heard whole, or with the brain's reply where one is open.

    [LAW:single-enforcer] the one place such a line's turn is ended, so whoever says a line sends the line alone.
    Pipecat's TTS service ends it only while it takes no reply as under way, and one that pushes its own text frames, as
    pocket-tts does, takes a reply as under way from its start frame to the next barge-in (1.10.0, `tts_service.py:858`):
    the line stayed an open turn, written at the next key press as cut off. After a barge-in it takes none as under way
    whatever the brain's reply, so the end it sends is not taken inside one.

    [LAW:no-ambient-temporal-coupling] the extent of the brain's reply is held here, where its turn is written: open
    from its start frame to its end frame. A line said inside it is written with it, as the reply ends.
    """

    _replying = False
    # How much of what is said and unwritten was said before the reply under way started: the lines it answers behind.
    _ahead = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        match frame:
            case LLMAssistantPushAggregationFrame() if self._replying:
                # The TTS service's end of a line's turn, sent after a barge-in whatever the brain's reply (1.10.0).
                return
            case LLMFullResponseStartFrame():
                self._replying, self._ahead = True, len(self._aggregation)
            case LLMFullResponseEndFrame():
                self._replying = False
            case InterruptionFrame():
                # A barge-in drops the reply under way. A line waiting on it was heard whole, its text arriving behind its
                # audio: written as that, ahead of Pipecat writing a reply as cut off once a sentence of its own has been heard.
                if self._aggregation and not (self._replying and len(self._aggregation) > self._ahead):
                    await self._end_turn()
                self._replying = False
            case _:
                pass
        await super().process_frame(frame, direction)
        match frame:
            case InterruptionFrame():
                # Written as cut off: the user's turn that made the cut is taken in behind it (`hands.voice.turnstart`).
                await self.push_frame(CutWritten(), FrameDirection.UPSTREAM)
            case _:
                pass
        # Whatever the frame, so whichever one closes the brain's reply writes what was said inside it: a line itself,
        # which the output transport passes on once its audio has played, where no reply is open.
        if self._aggregation and not self._replying:
            await self._end_turn()

    async def _end_turn(self) -> None:
        # Ended as Pipecat's own service ends a line's turn, by the frame it sends for one.
        await super().process_frame(LLMAssistantPushAggregationFrame(), FrameDirection.DOWNSTREAM)


def turns(user_params: LLMUserAggregatorParams) -> tuple[UserTurns, AssistantTurns]:
    """The two sides of the conversation over one list that keeps nothing, wired as Pipecat's `LLMContextAggregatorPair`
    wires its own (1.10.0), which builds neither of hands' sides."""
    context = Unkept()
    user = UserTurns(context, params=user_params)
    return user, AssistantTurns(context, _paired_user_aggregator=user)


def record_turns(user_turns: LLMUserAggregator, assistant_turns: LLMAssistantAggregator, record: Record) -> None:
    """Write each user transcript as its words go to the brain, and each reply as its turn ends."""

    # Handed on, not merely stopped: the user text is final here, and it is what the brain is asked.
    @user_turns.event_handler("on_user_turn_message_added")
    async def heard(_aggregator: LLMUserAggregator, message: UserTurnMessageAddedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        record(Transcribed(message.content))
        # The words themselves, in the terminal as well as the audit log: the latency line says only that a
        # transcript came, and whoever is talking wants to see what was heard: the plain words, on one line.
        logger.info(f"heard: {' '.join(message.content.split())}")

    @assistant_turns.event_handler("on_assistant_turn_stopped")
    async def replied(_aggregator: LLMAssistantAggregator, message: AssistantTurnStoppedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        match message:
            case AssistantTurnStoppedMessage(content="", interrupted=False):
                # A turn that only called a tool said nothing; its tool's event is the record of it.
                pass
            case _:
                record(Replied(message.content, message.interrupted))


def cue_receipt(user_turns: LLMUserAggregator, received: Callable[[], None]) -> None:
    """Tell `received` of each user turn as its words go to the brain: a turn that heard nothing sends none."""

    @user_turns.event_handler("on_user_turn_message_added")
    async def written(_aggregator: LLMUserAggregator, _message: UserTurnMessageAddedMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        received()
