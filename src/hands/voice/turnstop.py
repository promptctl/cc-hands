"""The end of a user turn: once the clip owner is done with every clip the turn took in.

A turn is Pipecat's; a clip is the key's. They are not one to one: Pipecat keeps a turn open until it ends, so a press
while the last clip is still being transcribed joins that turn rather than opening one. The clip owner
(`hands.voice.clipowner`) numbers each clip as it opens, and says when it is done with it, after pushing whatever text the
clip had, or as soon as the clips ahead of it are done for a clip the key dropped. The turn ends when every clip it took
in is done, so no clip's words are left out of it, and a clip with no words ends it as promptly as one with.
"""

from dataclasses import dataclass

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UninterruptibleFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import BaseUserTurnStopStrategy

from hands.voice.clips import Clip


@dataclass(kw_only=True)
class Typed(DataFrame, UninterruptibleFrame):
    """Words the user typed and sent, queued into the pipeline for the clip owner to take as a clip of their own, already heard.

    The user's own, as Words are, so no interruption drops it: the page was told they were taken, and a turn's cut can
    land while they are still on their way to the clip owner, or while it is opening their clip.
    """

    text: str


@dataclass(kw_only=True)
class TurnOpened(VADUserStartedSpeakingFrame):
    """The user started speaking: the key went down, opening this clip."""

    clip: Clip


class ClipDiscarded(VADUserStoppedSpeakingFrame):
    """The user stopped speaking, and what the clip recorded is thrown away: nothing of it is transcribed or sent."""


class Words(TranscriptionFrame, UninterruptibleFrame):
    """What the transcriber heard said in a clip. The user's own, so no interruption drops it: one can land while the turn's words
    are still on their way to it, since a turn the voice opened cuts hands off only once they are heard."""


class InterimWords(InterimTranscriptionFrame, UninterruptibleFrame):
    """Words a listener heard in a clip still open, Pipecat's interim transcription: someone is speaking, so a turn the voice
    opened cuts hands off on them while the user speaks, not once they stop. Never sent: the clip's Words, heard whole as
    it closes, are what the turn says. The user's own, as Words are, so no interruption drops it."""


@dataclass(kw_only=True)
class TurnResolved(DataFrame, UninterruptibleFrame):
    """The clip owner is done with this clip: whatever text it had has been pushed ahead of this.

    Nothing else ends a turn, so the interruption a turn broadcasts cannot drop it on its way.
    """

    clip: Clip


class KeyTurnStop(BaseUserTurnStopStrategy):
    """Ends the user turn when every clip opened in it has been resolved."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # [LAW:one-source-of-truth] the clip owner opens and resolves the clips; this is the set it has opened and not yet
        # resolved. A set, because typed words are a clip that opens and ends at once, so it resolves ahead of a clip of the
        # key's that opened before it and is still open.
        self._open: set[Clip] = set()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        match frame:
            case TurnOpened(clip=clip):
                self._open.add(clip)
            case TurnResolved(clip=clip):
                # The clip owner opens a clip, a system frame, before it pushes anything that resolves it, and a system frame is
                # never overtaken; a clip resolved but never opened is the owner's bug, so it fails here.
                self._open.remove(clip)
                if not self._open:
                    # A turn whose clips had no text ends with nothing aggregated, so nothing is sent to the model.
                    await self.trigger_user_turn_stopped()
            case _:
                pass
        return ProcessFrameResult.CONTINUE
