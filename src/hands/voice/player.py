"""The player: what is on the speaker, read off the frames the TTS service and the output transport push, and what is
said again on request.

Pipecat's TTS hands each sentence on as an `AggregatedTextFrame` marked to be spoken, ahead of its audio, and a
`TTSTextFrame` once the sentence's audio is made (pocket-tts reports no word timings, so these are per sentence). The
output transport queues both with the audio and lets each go downstream only once the audio ahead of it is written to
the device (Pipecat 1.10.0, `BaseOutputTransport.handle_sync_frame`). So what the TTS service pushes is every sentence
the speaker was given, as soon as it is made; what the output transport pushes is a `TTSTextFrame` for a sentence that
has played to its end, and a barge-in after every sentence that finished before it. An observer sees each push as it
is made, in the order made, where a processor standing in the pipeline would take a barge-in ahead of a sentence's end
still waiting in its queue.

Every line hands says as written crosses `lines` as a `LineFrame`, which says whose it is, and is queued there: the
brain's own words reach the speaker as the model's text, never as one. The speaker hands each line on as a `LineText`,
still whose it is, or says it dropped it (`Unsaid`), so the bare text it hands is the brain's.

[LAW:single-enforcer] the one holder of `Playback`: its observer reports to it, `lines` queues to it, and the playback
tools act on it.
"""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pipecat.frames.frames import AggregatedTextFrame, DataFrame, Frame, InterruptionFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hands.core import playback
from hands.core.playback import Bookmark, Labelled, LastOne, Line, NothingBargedIn, NothingSaid, Played, Playback, Replay, hands_says
from hands.sessions.audit import BargeIn, Record

Act = Callable[[Playback], tuple[Playback, Played]]


@dataclass
class LineFrame(TTSSpeakFrame):
    """A line for the speaker, and whose it is: the one way anything of hands' reaches it, queued as it passes `lines`."""

    line: Line = field(kw_only=True)


def line_frame(line: Line, *, append_to_context: bool = True) -> LineFrame:
    """The frame that says `line`. [LAW:one-source-of-truth] its text is the line's."""
    return LineFrame(line.text, append_to_context=append_to_context, line=line)


@dataclass
class LineText(AggregatedTextFrame):
    """A line of hands' as the speaker hands it on to be said: the sentence Pipecat makes of a `LineFrame`, still whose it is."""

    line: Line = field(kw_only=True)


@dataclass
class Unsaid(DataFrame):
    """The speaker took up `line` and says it as nothing, so will never hand it on."""

    line: Line


class Player:
    """Where playback is, and the lines to say again, which `lines` hands the TTS service the moment they are asked for.

    The lines go straight to the speaker rather than through the model's stage: the call that asked for them is the
    whole reply, and a barge-in drops them from the pipeline like any sentence not yet played, while their reading holds
    them to go back to.
    """

    def __init__(self, record: Record) -> None:
        self._record = record
        self._playback = Playback()
        # Stood ahead of the TTS service, the one place the lines enter the pipeline; it passes everything else on.
        self.lines = Lines(self)
        # [LAW:no-shared-mutable-globals] each `heard` waiting on its line, by the marker behind it: settled by the output
        # passing the marker on, or by the next barge-in, whichever the output pushes first.
        self._hearing: dict[asyncio.Future[bool], Marker] = {}
        # [LAW:no-shared-mutable-globals] each `next_barge_in` waiting on the next barge-in to reach the output, which settles it.
        self._cutting: list[asyncio.Future[Bookmark]] = []

    @property
    def waiting(self) -> int:
        """How many readings cut off wait to be gone back to."""
        return len(self._playback.bookmarks)

    def watching(self, speaker: FrameProcessor, output: FrameProcessor) -> BaseObserver:
        """The observer reading the speaker off what `speaker`, the TTS service, and `output`, the output transport, push."""
        return _Watch(self, speaker, output)

    def handed(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame():
                # A sentence's audio made, which is a kind of AggregatedTextFrame too: it has played only once the
                # output transport pushes it.
                pass
            case LineText(line=line) if frame.will_be_spoken:
                self._playback = playback.handed(self._playback, line)
            case Unsaid(line=line):
                self._playback = playback.dropped(self._playback, line)
            case AggregatedTextFrame(text=text) if frame.will_be_spoken:
                self._playback = playback.streamed(self._playback, text)
            case _:
                pass

    def played(self, frame: Frame) -> None:
        match frame:
            case TTSTextFrame() if frame.will_be_spoken:
                self._playback = playback.finished(self._playback)
            case Marker():
                # [LAW:no-ambient-temporal-coupling] read off the output's push, in order with its barge-ins: a marker
                # waiting in Markers' queue is one a barge-in empties out, though its line played to the end.
                for hearing, marker in self._hearing.items():
                    if marker is frame:
                        _settle(hearing, True)
            case InterruptionFrame():
                for hearing in self._hearing:
                    _settle(hearing, False)
                was = self._playback
                cutting, self._cutting = self._cutting, []
                for cut in cutting:
                    if not cut.done():
                        cut.set_result(Bookmark((*was.reading, *was.coming), was.played))
                self._playback = playback.barge_in(was)
                # [LAW:nothing-unseen] every barge-in is a line, one that cut nothing off included.
                self._record(BargeIn(None if was.over else (*was.reading, *was.coming)[was.played].text, len(self._playback.bookmarks)))
            case _:
                pass

    def queued(self, frames: Sequence[Frame]) -> None:
        """Takes frames on their way to the speaker past `lines`: each line among them is one hands will hand it."""
        for frame in frames:
            match frame:
                case LineFrame(line=line):
                    self._playback = playback.queued(self._playback, line)
                case _:
                    pass

    def next_barge_in(self) -> asyncio.Future[Bookmark]:
        """The reading on the speaker when the next barge-in reaches it, and how many of its sentences had played to their
        end: all of them when it was cut off in none. Asked for before that barge-in is passed down, so it is the one."""
        cut = asyncio.get_running_loop().create_future()
        self._cutting.append(cut)
        return cut

    async def act(self, act: Act) -> tuple[Line, ...]:
        """Moves playback by `act`, and says what it says, which is always something: a reading said again, each line still
        whose it was, or why not. What it says is held whole from the start, so a barge-in on it loses none of it."""
        self._playback, played = act(self._playback)
        lines = said(played)
        # One line a frame, so each is a position.
        await self.lines.push_frames(tuple(line_frame(line) for line in lines))
        return lines

    async def heard(self, line: Labelled) -> bool:
        """Says `line`, and returns once the speaker has played it to its end, True, or a barge-in has cut it off, False.

        [LAW:no-ambient-temporal-coupling] what follows a line only once the user has heard it waits on this, never on how
        long the line takes to say.
        """
        hearing = asyncio.get_running_loop().create_future()
        # Settled as the output passes it on, which the observer sees (`played`); Markers only takes it out of the pipeline.
        marker = Marker(lambda: None)
        self._hearing[hearing] = marker
        try:
            await self.lines.push_frame(line_frame(line, append_to_context=False))
            await self.lines.push_frame(marker)
            return await hearing
        finally:
            del self._hearing[hearing]


def _settle(hearing: "asyncio.Future[bool]", played: bool) -> None:
    # The first of a line's marker and a barge-in says how it went; the output passes a marker on only if no barge-in cut it off.
    if not hearing.done():
        hearing.set_result(played)


def said(played: Played) -> tuple[Line, ...]:
    """What an act has hands say: the lines again, or in one line why there are none."""
    match played:
        case Replay(lines=lines):
            return lines
        case NothingBargedIn():
            return (hands_says("Nothing was cut off to go back to."),)
        case NothingSaid():
            return (hands_says("I haven't said anything yet."),)
        case LastOne():
            return (hands_says("That was the last of it."),)


@dataclass
class Marker(DataFrame):
    """A point in what is said: `played` is called once the speaker has played everything handed it ahead of the marker, and
    never if a barge-in cut any of that off. The TTS service keeps a data frame behind the audio of what was handed it
    first, and the output transport passes it on only once that audio is written, or drops it with that audio on a barge-in
    (Pipecat 1.10.0)."""

    played: Callable[[], None]


class Markers(FrameProcessor):
    """Stood right behind the output transport: calls each marker's `played` as it arrives, and passes the rest on."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Marker(played=played):
                played()
            case _:
                await self.push_frame(frame, direction)


class Lines(FrameProcessor):
    """Where every line of hands' enters the pipeline, ahead of the TTS service: what passes it downstream, pushed through
    it or by hands, is queued with the player before any of it is pushed on."""

    def __init__(self, player: Player) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._player = player

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        # Both ways in meet here: a frame passing through, and one hands pushes.
        await self.push_frames((frame,), direction)

    async def push_frames(self, frames: Sequence[Frame], direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        """Pushes `frames` on, in order, each queued before the first is pushed: a barge-in while they are pushed finds
        every one of them coming."""
        # [LAW:single-enforcer] the one place a line is queued.
        self._player.queued(frames if direction is FrameDirection.DOWNSTREAM else ())
        for frame in frames:
            await super().push_frame(frame, direction)


class _Watch(BaseObserver):
    """Shows the player what the speaker is handed and what it has played; every other push is one of those frames
    passing another processor, and tells it nothing."""

    def __init__(self, player: Player, speaker: FrameProcessor, output: FrameProcessor) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._player = player
        self._speaker = speaker
        self._output = output

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.source is self._speaker:
            self._player.handed(data.frame)
        elif data.source is self._output:
            self._player.played(data.frame)
