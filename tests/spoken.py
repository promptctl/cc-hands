"""What leaves the brain's stage for TTS, played at once as the speaker and the markers behind it would play it, and the frame
that tells a test the speaker has all that was queued ahead of it: shared by the stage's tests (test_stage.py) and the rig the
pinned Claude Code runs against (pinnedbrain.py)."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core.playback import Bookmark, Line, Own
from hands.voice.player import LineFrame, Marker
from hands.voice.utterance import Uttered, Uttering


@dataclass
class Received(DataFrame):
    """Queued behind what the stage has handed on: `reached` is called once the rig's speaker has it, and so has every
    frame queued ahead of it, whether or not it is holding. A data frame reaches each processor in the order it was
    queued, and the stage passes on what it does not know."""

    reached: Callable[[], None]


class Spoken(FrameProcessor):
    """What leaves the stage for TTS, played at once, as the speaker and the markers behind it would play it, unless held."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        # How many frames have reached it: what a test's wait sees move while the stage has work it is getting through.
        self.received = 0
        self.frames: list[Frame] = []
        # Markers not yet played, while `holding`: what is ahead of them is still being said.
        self.holding = False
        self.markers: list[Marker] = []
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

    def next_barge_in(self) -> asyncio.Future[Bookmark]:
        cut = asyncio.get_running_loop().create_future()
        self.cutting.append(cut)
        return cut

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.received += 1
        match frame:
            case Received(reached=reached):
                reached()
            case Marker(played=played) if not self.holding:
                played()
            case Marker():
                self.markers.append(frame)
            case VADUserStoppedSpeakingFrame():
                self.releases += 1
                await self.push_frame(frame, direction)
            case UserStoppedSpeakingFrame():
                self.stops += 1
                await self.push_frame(frame, direction)
            case LLMFullResponseStartFrame() | LLMFullResponseEndFrame() | LLMTextFrame() | TTSSpeakFrame() | InterruptionFrame():
                if isinstance(frame, InterruptionFrame):
                    said = tuple(self.lines())
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

    def lines(self) -> list[Line]:
        """What was said, each line whose it is: a line of hands' by its frame, and any other text as the brain's."""
        return [frame.line if isinstance(frame, LineFrame) else Own(frame.text) for frame in self.frames if isinstance(frame, LLMTextFrame | TTSSpeakFrame)]

    def shape(self) -> list[str]:
        return [type(frame).__name__ for frame in self.frames]
