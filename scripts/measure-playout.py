"""How hands' voice plays out on this machine's speaker, alone or while its listeners work: each synthesis's gaps, late
callbacks and real-time factor, read off the `speech` events hands emits for it.

hands' own TTS and speaker run in a Pipecat pipeline, as the daemon runs them, and say a reply out loud on the default
output device. Beside them in the same process run the listeners named, each working without a pause, as hands' run
while someone talks over it at length: `streaming`, the streaming recognizer and diarizer on MLX, fed speech 20 ms at a
time on a thread of their own; `whisper`, Whisper transcribing one clip after another. The speech they hear is made by
the TTS ahead of the run, in voices other than the one that speaks. With none named, hands speaks alone.

    uv run python scripts/measure-playout.py VOICE [streaming] [whisper]
"""

import asyncio
import sys
import threading
import time

import numpy as np
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.local.audio import LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from hands.clock import SYSTEM
from hands.core.playback import hands_says
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import streaming, voices
from hands.voice.clips import Clip, Samples
from hands.voice.microphone import KeyedAudioTransport
from hands.voice.phone import Phone
from hands.voice.player import line_frame
from hands.voice.ptt import PushToTalk
from hands.voice.spoken import SpokenForm
from hands.voice.voices import SaidInTTS
from hands.voice.whisper import Whisper


REPLY = (
    "I looked at the failing test, and the problem is in how the playout counts the frames it takes from the queue.",
    "The fix is small: the tally has to move to the device's thread, so nothing else touches it while the callback runs.",
    "I've pushed the change to the branch and opened a pull request.",
    "The review is running now, and I'll tell you what it finds.",
    "In the meantime, the unit tests are green, and the type checks are clean.",
    "One more thing: the second session is still waiting on your answer about the migration.",
)
# What the listeners hear: someone talking over hands, in voices other than the one that speaks.
OVERHEARD = (
    ("marius", "Wait, hold on, which branch did you push that to? I want to check it before the review starts."),
    ("javert", "Can you also look at the session that's been stuck since this morning, it never finished its build."),
)
# How long after the last speech event the reply is taken to have played out.
SETTLED_SECS = 4.0


def listening(stop: threading.Event, heard: list[Samples], fed: list[int]) -> None:
    """The streaming recognizer's work while someone talks: each clip fed 20 ms at a time, as fast as it is spoken, one
    after another until `stop`, the length of each clip fed kept in `fed`."""
    stream = streaming.Stream(streaming.load(streaming.GATING))
    start = time.monotonic()
    spoken = 0
    while not stop.is_set():
        for clip in heard:
            # As tests/listening.py's Live feeds a clip, paced by the clock: the listener is fed as the microphone hears.
            stream.opened(np.zeros(0, np.float32), spoken, ())
            for at in range(0, len(clip), 320):
                stream.heard(clip[at : at + 320], ())
                spoken += len(clip[at : at + 320])
                time.sleep(max(0.0, start + spoken / streaming.RATE - time.monotonic()))
            stream.said()
            silence = streaming.through(spoken) - spoken
            stream.silence(silence)
            spoken += silence
            stream.ended()
            fed.append(len(clip))
            if stop.is_set():
                return


async def transcribing(whisper: Whisper, heard: list[Samples], transcribed: list[str | None]) -> None:
    """Whisper's work at the end of each clip, as each clip ends, one after another until cancelled, what it heard kept
    in `transcribed`."""
    while True:
        for clip in heard:
            await asyncio.sleep(len(clip) / streaming.RATE)
            transcribed.append(await whisper.transcribed(Clip(len(transcribed) + 1, "typed"), (np.clip(clip, -1, 1) * 32767).astype(np.int16).tobytes()))


async def main() -> None:
    voice, listeners = sys.argv[1], set(sys.argv[2:])
    if listeners - {"streaming", "whisper"}:
        raise SystemExit(f"the listeners are streaming and whisper, not {sorted(listeners - {'streaming', 'whisper'})}")
    speech: list[WideEvent] = []

    def record(entry: Entry) -> None:
        match entry:
            case WideEvent(event="speech"):
                speech.append(entry)
            case _:
                pass

    stop = threading.Event()
    fed: list[int] = []
    transcribed: list[str | None] = []
    tasks: list[asyncio.Task[None]] = []
    listener: threading.Thread | None = None
    said = voices.pocket()
    heard = [said(speaker, text) for speaker, text in OVERHEARD]
    if "streaming" in listeners:
        listener = threading.Thread(target=listening, args=(stop, heard, fed), name="Streaming")
        listener.start()
    if "whisper" in listeners:

        async def no_prompt() -> str | None:
            return None

        tasks.append(asyncio.create_task(transcribing(Whisper(clock=SYSTEM, prompt=no_prompt, record=record), heard, transcribed)))
    # The models load before hands speaks, as they do at the daemon's start.
    await asyncio.sleep(15)

    key = PushToTalk(SYSTEM, record)
    params = PipelineParams()
    phone = Phone(SYSTEM, key, heard_rate=params.audio_in_sample_rate, played_rate=params.audio_out_sample_rate, record=record)
    transport = KeyedAudioTransport(SYSTEM, LocalAudioTransportParams(audio_in_enabled=False, audio_out_enabled=True), key, phone, record)
    tts = SaidInTTS(clock=SYSTEM, settings=SaidInTTS.Settings(voice=voice, language=voices.LANGUAGE), text_filters=[SpokenForm()])
    worker = PipelineWorker(Pipeline([tts, transport.output()]), params=params, idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    running = asyncio.create_task(runner.run(worker))
    await asyncio.sleep(2)
    for line in REPLY:
        await worker.queue_frame(line_frame(hands_says(line)))
    seen = -1
    while seen != len(speech) or not speech:
        seen = len(speech)
        await asyncio.sleep(SETTLED_SECS)
    await worker.queue_frame(EndFrame())
    await running
    # [LAW:no-silent-failure] a load that died partway measured a quieter machine than it says.
    if listener is not None and not listener.is_alive():
        raise SystemExit("the streaming recognizer stopped before hands finished speaking: this run was not loaded")
    if any(task.done() for task in tasks):
        raise SystemExit("Whisper stopped before hands finished speaking: this run was not loaded")
    stop.set()
    for task in tasks:
        task.cancel()
    if listener is not None:
        listener.join()

    print(f"{' and '.join(sorted(listeners)) or 'alone'}, voice {voice}: {len(speech)} syntheses; the streaming recognizer fed {sum(fed) / streaming.RATE:.1f} s of speech, Whisper transcribed {len(transcribed)} clips")
    for event in speech:
        facts = event.facts
        print(f"  played {facts['played_ms']:8.1f} ms  gaps {event.counts['gaps']} ({facts['gap_ms']:6.1f} ms)  late {event.counts['late']}  rtf {facts.get('rtf')}")
    print(
        f"  total: gaps {sum(event.counts['gaps'] for event in speech)}, gap_ms {sum(float(str(event.facts['gap_ms'])) for event in speech):.1f}, "
        f"late {sum(event.counts['late'] for event in speech)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
