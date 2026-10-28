"""Talking over hands: a hold engaged conversation opens, where the desk's own detector opens it and with what the hold
owner kept from before it, cuts hands off on the first words the streaming recognizer hears in it, at the same place in
the audio on every machine; a cough and the keyboard cut nothing.

Every part is production's, on recorded audio: the desk's driver with Silero and Smart Turn, the push-to-talk gate it
moves, the hold owner the gate's frames are cut into holds by, and the real pair hearing them, which holds this module's
pair (conftest's `Pairs`). "Sooner than Whisper" is measured on the ticket, in wall-clock time, not here: master no
longer has Whisper's overhearing."""

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pipecat.audio.turn.smart_turn.base_smart_turn as base_smart_turn
import pipecat.audio.vad.silero as silero
import pytest
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import Frame, InputAudioRawFrame, InterimTranscriptionFrame
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import PAIRS, events, hearing_with, running
from test_streaming import NOT_SPEECH, READER, fixture_holds, hearings, pcm
from hands.sessions.audit import Entry
from hands.voice import engaged, streaming
from hands.voice.desk import Leveled, Made
from hands.voice.engaged import Ears
from hands.voice.hold import KeyEvent, Move
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import RATE, Fate, Hearings, Hold, Sound
from hands.voice.ptt import PushToTalk
from hands.voice.trigger import Choice

# The real load, bound before conftest stands a pair that hears nothing in for it.
from hands.voice.streaming import load as nemotron

FRAME = 320

# The first four utterances of the reader, then a cough, then the keyboard, each after 4 s of exact zeros, and 4 s after
# the last: longer than Smart Turn's stop_secs (3 s), so the desk ends every hold before the next sound, whatever Smart
# Turn judges.
SOUNDS = [*fixture_holds(READER)[:4], *NOT_SPEECH]
GAP = np.zeros(4 * RATE, np.float32)
TIMELINE = np.concatenate([*(part for sound in SOUNDS for part in (GAP, sound)), GAP])
# The frames the desk hears it in, the last one short.
FRAMES = -(-len(TIMELINE) // FRAME)
# Where each sound's first sample is on the timeline.
FIRST = [len(GAP) * (n + 1) + sum(map(len, SOUNDS[:n])) for n in range(len(SOUNDS))]
FIRST_NOT_SPEECH = FIRST[4]


class Heard:
    """The clock Silero and Smart Turn read, as the `time` module, set to how much audio the desk has heard.

    [LAW:no-ambient-temporal-coupling] both read the wall clock as if audio came at real-time pace: Silero resets its
    state every 5 s of it, and Smart Turn keeps the speech it judges by it. Here a frame takes as long as the run spends
    on it, so on the wall clock the resets would fall at other places in the audio on another machine, and the segments
    judged differ from a live turn's."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def time(self) -> float:
        return self.seconds

    def monotonic(self) -> float:
        return self.seconds

    def __getattr__(self, name: str) -> Any:
        # Everything else is the wall clock's: Smart Turn times its own inference by `perf_counter`.
        return getattr(time, name)


@dataclass
class Counted:
    """The recognizer, as the hold owner's listener, counting the hearings it returns and the sample of the timeline each
    was returned at: the owner runs them on one lane, in that order, so the k-th `hearing` event is the k-th returned."""

    recognizer: streaming.StreamingRecognizer
    captured: Callable[[], int]
    returned: list[int] = field(default_factory=list[int])
    name: str = streaming.StreamingRecognizer.name

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        return self._counted(self.recognizer.opened(hold, before))

    def heard(self, sound: Sound) -> Hearings:
        return self._counted(self.recognizer.heard(sound))

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        return self._counted(self.recognizer.ended(hold, fate))

    def _counted(self, hearings: Hearings) -> Hearings:
        self.returned += [self.captured()] * len(hearings)
        return hearings


class Recorded:
    """Every entry the run records, from whichever thread records it, each one waking what waits on them.

    [LAW:single-enforcer] a wait that never ends, loading or decoding on the recognizer's thread, which moves nothing
    the loop can see until it does, is ended by the suite's timeout."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.entries: list[Entry] = []
        self._loop = loop
        self._added = asyncio.Event()

    def __call__(self, entry: Entry) -> None:
        self.entries.append(entry)
        self._loop.call_soon_threadsafe(self._added.set)

    async def until(self, done: Callable[[], bool]) -> None:
        """Once `done` holds of what has been recorded."""
        while not done():
            self._added.clear()
            await self._added.wait()


class Downstream(FrameProcessor):
    """What reaches the frame after the hold owner: how many frames of audio, and how many cuts."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames = 0
        self.cuts = 0
        self._passed = asyncio.Event()

    async def reached(self, frames: int) -> None:
        """Once `frames` frames of audio have passed, woken by each as it passes."""
        while self.frames < frames:
            self._passed.clear()
            await self._passed.wait()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case InputAudioRawFrame():
                self.frames += 1
                self._passed.set()
            case InterimTranscriptionFrame():
                self.cuts += 1
            case _:
                pass
        await self.push_frame(frame, direction)


class Unheard:
    """A transcriber that says nothing of any hold: what the cut is, not what the hold held, is heard here."""

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        return None


class Microphone:
    """The desk's ears, as the microphone hands them each frame: captured first, tagged with the gate as the desk's moves
    left it and into the hold owner, the way the capture thread tags it, then heard by the models.

    [LAW:no-ambient-temporal-coupling] `drive` asks of frame n only once it has stepped every move frame n - 1 made, so
    each frame carries the gate those moves left; and a frame returns only once the owner has passed it on and every
    hearing returned so far has its event, so where a hold's words cut is the same on every machine."""

    def __init__(self, models: Ears, capture: Callable[[bytes], Any], clock: Heard) -> None:
        self._models = models
        self._capture = capture
        self._clock = clock

    async def detect(self, audio: bytes) -> VADState:
        await self._capture(audio)
        self._clock.seconds += len(audio) // 2 / RATE
        return await self._models.detect(audio)

    def heard(self, audio: bytes, speech: bool) -> bool:
        return self._models.heard(audio, speech)

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        return await self._models.judge()

    def clear(self) -> None:
        self._models.clear()

    def afresh(self) -> None:
        self._models.afresh()


async def unwoken(audio: bytes, begun: bool) -> bool:
    return False


@asynccontextmanager
async def untapped(into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
    yield


def sound_at(sample: int) -> int:
    """Which sound of the timeline a sample is in, or after: the last whose first sample it is at or past."""
    return max(n for n, first in enumerate(FIRST) if first <= sample)


async def test_talking_over_hands_cuts_at_the_first_words_heard_and_a_cough_or_the_keys_cut_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Heard()
    monkeypatch.setattr(silero, "time", clock)
    monkeypatch.setattr(base_smart_turn, "time", clock)
    record = Recorded(asyncio.get_running_loop())
    gate = PushToTalk(record)
    captured = 0
    moves: list[tuple[int, Move]] = []
    recognizer = streaming.StreamingRecognizer(lambda gating: PAIRS.loaded(lambda: nemotron(gating)), record)
    # The pair loads on the recognizer's thread before the pipeline starts, whose start a load holding the loop would
    # hold past its bound on CI's VMs; the first frame then waits for nothing it does.
    await record.until(lambda: bool(events(record.entries, "streaming.loaded")))
    [load] = events(record.entries, "streaming.loaded")
    assert load.outcome == "ok", load.error
    listener = Counted(recognizer, lambda: captured * FRAME)
    downstream = Downstream()
    async with running([HoldOwner(Unheard(), (listener,), record), downstream]) as run, engaged.loaded(RATE, record) as models:
        async def capture(audio: bytes) -> None:
            nonlocal captured
            captured += 1
            await run.worker.queue_frame(gate.gate.framed(audio, audio, RATE, 1, "desk"))
            # The owner tells its listeners of a frame before it passes it on, so once the frame has passed, every
            # hearing it returned is counted.
            await downstream.reached(captured)
            await record.until(lambda: len(hearings(record.entries)) == len(listener.returned))

        frames = pcm(TIMELINE)
        chosen: asyncio.Queue[Choice] = asyncio.Queue()
        engaging: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        chosen.put_nowait(Choice("engaged", engaging))

        @asynccontextmanager
        async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
            async def buffers() -> AsyncIterator[bytes]:
                # Engaged before the first frame is overheard, so the desk hears all of it: the choice is answered once
                # the desk has stepped it.
                await engaging
                for at in range(0, len(frames), 2 * FRAME):
                    yield frames[at : at + 2 * FRAME]
                await asyncio.Event().wait()

            yield buffers()

        async def on_move(made: Made) -> None:
            moves.append((captured, made.move))
            gate.move(made.move, made.by)

        async def on_level(leveled: Leveled) -> None:
            pass

        driving = asyncio.create_task(
            engaged.drive(untapped, overheard, hearing_with(Microphone(models, capture, clock), unwoken), chosen, on_move, on_level, record)
        )
        # Heard to its last frame, past the owner, or the drive failed on the way, which is said; then every hearing the
        # owner was given has its event.
        through = asyncio.ensure_future(downstream.reached(FRAMES))
        await asyncio.wait({driving, through}, return_when=asyncio.FIRST_COMPLETED)
        through.cancel()
        assert not driving.done(), driving
        await record.until(lambda: len(hearings(record.entries)) == len(listener.returned))
        # Nothing in the pipeline raised on the way, which a frame that still passed would not show.
        assert run.errors == [], run.errors
        driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driving
    heard = hearings(record.entries)
    # Every hearing heard its hold: a failed one says why here, not as a fact it lacks below.
    assert [event.error for event in heard if event.outcome != "ok"] == []
    # Each utterance opens one hold, where the desk's own detector opened it.
    opened = [sound_at(frame * FRAME) for frame, move in moves if move == "start"]
    assert opened[:4] == [0, 1, 2, 3], moves
    # Each hold's words cut while it was open, at this sample of the audio, in ms from its utterance's first sample: the
    # first `hearing` of the hold with `cut: true`, as the hold owner said it.
    cuts: dict[int, int] = {}
    for at, event in zip(listener.returned, heard, strict=True):
        match event.facts:
            case {"cut": True, "hold": int() as hold}:
                cuts.setdefault(hold, at)
            case _:
                pass
    assert {hold: (at - FIRST[sound_at(at)]) * 1000 // RATE for hold, at in cuts.items()} == {1: 940, 2: 810, 3: 910, 4: 890}, moves
    # Every cut the owner said reached the frame after it.
    assert downstream.cuts == sum(bool(event.facts["cut"]) for event in heard)
    # Nothing cut from the cough's first sample to the timeline's end.
    assert [at for at, event in zip(listener.returned, heard, strict=True) if event.facts["cut"] and at >= FIRST_NOT_SPEECH] == []
