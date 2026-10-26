"""The conversation as the audit log tells it: each reply written as the speaker finishes it, whoever's words it was."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.pocket_tts.tts import PocketTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from conftest import Running, patiently, running
from hands.sessions.audit import Entry, Replied
from hands.voice.conversation import AssistantTurns, record_turns, turns
from hands.voice.player import Mark, Marks
from hands.voice.spoken import FenceAggregator


class Speaker(TTSService):
    """The speaker with the synthesis taken out: started, stopped, and its text pushed by Pipecat's service as pocket-tts's are."""

    def __init__(self) -> None:
        # Pipecat's own wait for a context's first audio, as PocketTTSService is made with it.
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            push_start_frame=True, push_stop_frames=True, settings=PocketTTSSettings(model=None, voice="alba", language=Language.EN)
        )
        # How many frames it has handed on: what a test's wait sees move while the pipeline works through what it was sent.
        self.pushed = 0

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        self.pushed += 1
        await super().push_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        yield TTSAudioRawFrame(b"\x00\x00", self.sample_rate, 1, context_id=context_id)


@dataclass
class Conversation:
    """The audit log's replies, behind Pipecat's own speaker and hands' assistant side."""

    pipeline: Running
    assistant: AssistantTurns
    speaker: Speaker
    recorded: list[Entry] = field(default_factory=list[Entry])

    def moved(self) -> tuple[int, int]:
        return self.speaker.pushed, len(self.recorded)

    async def said(self, *frames: Frame) -> None:
        """Send the frames as the model's stage sends them, and wait until the last of them has been taken."""
        taken = asyncio.Event()
        await self.pipeline.worker.queue_frames([*frames, Mark(taken.set)])
        await patiently(taken.is_set, self.moved)

    async def barge_in(self) -> None:
        """A barge-in, which overtakes whatever is still on its way and drops a mark sent behind it."""
        await self.pipeline.worker.queue_frame(InterruptionFrame())

    async def replies(self, count: int) -> list[Entry]:
        """What the audit log is told, once `count` replies are written."""
        await patiently(lambda: len(self.recorded) >= count, self.moved)
        return self.recorded


@asynccontextmanager
async def conversing() -> AsyncGenerator[Conversation]:
    user, assistant = turns(LLMUserAggregatorParams())
    speaker = Speaker()
    async with running([LLMTextProcessor(text_aggregator=FenceAggregator()), speaker, assistant, Marks()]) as pipeline:
        conversation = Conversation(pipeline, assistant, speaker)
        record_turns(user, assistant, conversation.recorded.append)
        yield conversation


async def test_a_line_hands_says_after_the_models_reply_is_written_as_heard_whole_once_it_is_said() -> None:
    """hands-readback-ddk, hands-readback-v6m: it was written only at the next key press, as cut off, however long
    before it had been heard."""
    async with conversing() as conversation:
        await conversation.said(LLMFullResponseStartFrame(), LLMTextFrame("Staging it now."), LLMFullResponseEndFrame(), TTSSpeakFrame("Draft for bananas: cherry."))

        assert await conversation.replies(2) == [Replied("Staging it now.", interrupted=False), Replied("Draft for bananas: cherry.", interrupted=False)]


async def test_a_question_hands_asks_in_the_middle_of_a_reply_is_written_between_the_two_halves_of_it() -> None:
    """The brain's stage ends the reply so far, has hands ask, and starts the reply again once the question is answered."""
    async with conversing() as conversation:
        await conversation.said(
            LLMFullResponseStartFrame(),
            LLMTextFrame("Deleting the branch."),
            LLMFullResponseEndFrame(),
            TTSSpeakFrame("May it run git push?"),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Done."),
            LLMFullResponseEndFrame(),
        )

        assert await conversation.replies(3) == [
            Replied("Deleting the branch.", interrupted=False),
            Replied("May it run git push?", interrupted=False),
            Replied("Done.", interrupted=False),
        ]


async def test_a_line_kept_out_of_the_context_is_no_reply() -> None:
    async with conversing() as conversation:
        await conversation.said(TTSSpeakFrame("api: reading files.", append_to_context=False), TTSSpeakFrame("The session api is gone."))

        assert await conversation.replies(1) == [Replied("The session api is gone.", interrupted=False)]


async def test_a_reply_started_again_after_a_barge_in_it_goes_on_through_is_written_whole_at_its_end() -> None:
    """The brain's stage starts the reply again where its turn goes on through a barge-in: each sentence said after one
    would otherwise be a turn of its own."""
    async with conversing() as conversation:
        await conversation.said(LLMFullResponseStartFrame())
        await conversation.barge_in()
        await conversation.said(LLMFullResponseStartFrame(), LLMTextFrame("It opened pull request 68. "), LLMTextFrame("Nothing else changed."), LLMFullResponseEndFrame())

        assert (await conversation.replies(2))[-1] == Replied("It opened pull request 68. Nothing else changed.", interrupted=False)

async def test_over_many_replies_the_assistant_side_keeps_none_of_them() -> None:
    """hands-brain-80y: every line hands spoke was appended to a message list nothing read, for as long as the daemon ran.
    The brain keeps its own transcript; the audit log is hands' record."""
    async with conversing() as conversation:
        for n in range(20):
            await conversation.said(LLMFullResponseStartFrame(), LLMTextFrame(f"Reply {n}."), LLMFullResponseEndFrame(), TTSSpeakFrame(f"Line {n}."))

        assert len(await conversation.replies(40)) == 40
        assert conversation.assistant.messages == []


async def test_a_turn_ended_not_to_be_asked_is_an_error_not_words_asked() -> None:
    """Pipecat ends a turn so only to confirm a speculative inference, which hands never runs: asking its words anyway
    would hand the brain them twice."""
    user, _ = turns(LLMUserAggregatorParams())
    with pytest.raises(ValueError, match="speculative"):
        await user._push_aggregation(run_llm=False)  # pyright: ignore[reportPrivateUsage]  (Pipecat's seam UserTurns overrides)
