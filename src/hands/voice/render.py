"""Sentences said by hands' TTS service and kept as audio, not played: what `scripts/record-double-talk.py` records its takes
from, so a take is speech made by the service hands speaks with (`voices.hands_tts`), its voice changed and its line sent
through the same frames the speaker is handed (`voices.say`).

[LAW:no-silent-failure] a sentence is rendered, or `NotRendered` says which and why: Pipecat reports a frame it failed to
process as a non-fatal error nobody downstream sees, and skips a service that is not usable without a word, so every way a
sentence can fail to come out is turned into one outcome the caller waits on.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import soxr
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from hands.clock import Clock
from hands.voice.clips import RATE, Samples, Synthesis, decoded
from hands.voice.player import Unsaid
from hands.voice.synthesis import Made, SpeakingIn
from hands.voice.voicechoice import Chosen
from hands.voice.voices import Fetch, fetched, say, source

# How long a sentence is waited for, from the line sent to the sound made. It bounds a TTS that says neither that it made the
# sentence nor why it did not, and a cold load of the model is the longest thing in it.
SENTENCE_TIMEOUT = 120.0


@dataclass(frozen=True)
class Saying:
    """What is to be said: `text`, in `voice`."""

    voice: Chosen
    text: str


@dataclass(frozen=True)
class Rendered:
    """What the TTS made of one saying: the `synthesis` it opened, and its sound at RATE."""

    synthesis: Synthesis
    audio: Samples


@dataclass(frozen=True)
class Failed:
    """What kept a saying from being rendered."""

    reason: str


class NotRendered(Exception):
    """hands' TTS did not make `saying`: `reason` is what it, or the pipeline around it, said."""

    def __init__(self, saying: Saying, reason: str) -> None:
        super().__init__(f"hands' TTS made nothing of {saying.text!r} in {saying.voice}: {reason}")


class _Heard(FrameProcessor):
    """Behind the TTS: each synthesis's sound, from its `SpeakingIn` to its `Made`, as the speaker reads it, or what kept it
    from being made."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.outcomes = asyncio.Queue[Rendered | Failed]()
        self._audio = bytearray()
        self._rates = set[int]()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case SpeakingIn():
                self._audio.clear()
                self._rates.clear()
            case TTSAudioRawFrame(audio=audio, sample_rate=rate):
                self._audio.extend(audio)
                self._rates.add(rate)
            case Made() as made:
                self.outcomes.put_nowait(self._rendered(made))
            case Unsaid(line=line):
                self.outcomes.put_nowait(Failed(f"it said {line.text!r} as nothing"))
            case _:
                pass
        await self.push_frame(frame, direction)

    def _rendered(self, made: Made) -> Rendered | Failed:
        # [LAW:one-source-of-truth] whether anything was made is what the TTS says it made, not what the bytes add up to.
        if made.failure is not None:
            return Failed(made.failure)
        if not made.made:
            return Failed("it made no sound")
        if len(self._rates) != 1:
            return Failed(f"its sound came at {sorted(self._rates)} Hz, and is cut at one rate")
        (rate,) = self._rates
        return Rendered(made.synthesis, soxr.resample(decoded(bytes(self._audio)), rate, RATE).astype(np.float32))


async def render(clock: Clock, tts: FrameProcessor, sayings: Sequence[Saying], fetch: Fetch = fetched, timeout: float = SENTENCE_TIMEOUT) -> list[Rendered]:
    """Each saying made into speech by `tts`, one after another, in order: the voices fetched first, as a hearing's are, then
    each saying's voice changed and its line sent through, and its sound taken as the speaker would be handed it. Raises
    `NotRendered` for the first that cannot be made."""
    await asyncio.gather(*(asyncio.to_thread(fetch, voice) for voice in dict.fromkeys(saying.voice for saying in sayings)))
    heard = _Heard()
    worker = PipelineWorker(Pipeline([tts, heard]), idle_timeout_secs=None)

    @worker.event_handler("on_pipeline_error")
    async def _failed(_worker: PipelineWorker, error: ErrorFrame) -> None:  # pyright: ignore[reportUnusedFunction]
        heard.outcomes.put_nowait(Failed(error.error))

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    running.add_done_callback(lambda ended: heard.outcomes.put_nowait(Failed("the pipeline ended" if ended.cancelled() or ended.exception() is None else f"the pipeline ended: {ended.exception()!r}")))
    rendered: list[Rendered] = []
    try:
        for saying in sayings:
            await say(worker.queue_frame, saying.voice, saying.text)
            rendered.append(_made(saying, await _outcome(clock, heard, saying, timeout)))
    finally:
        try:
            await worker.cancel()
        finally:
            await running
    return rendered


async def _outcome(clock: Clock, heard: _Heard, saying: Saying, timeout: float) -> Rendered | Failed:
    try:
        async with clock.timeout(timeout):
            return await heard.outcomes.get()
    except TimeoutError:
        raise NotRendered(saying, f"nothing came of it in {timeout:g} s") from None


def _made(saying: Saying, outcome: Rendered | Failed) -> Rendered:
    """The outcome of `saying` as what the TTS made of it, which is its words in its voice, or the failure it came to."""
    match outcome:
        case Failed(reason=reason):
            raise NotRendered(saying, reason)
        case Rendered(synthesis=synthesis) if (synthesis.voice, synthesis.text) != (source(saying.voice), saying.text):
            # [LAW:no-silent-failure] the filter in front of the TTS changed what was said, so what a take records is no longer
            # the line its script scores.
            raise NotRendered(saying, f"it said {synthesis.text!r} in {synthesis.voice}")
        case Rendered():
            return outcome
