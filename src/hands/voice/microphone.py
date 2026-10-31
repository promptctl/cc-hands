"""The Mac's own microphone and speakers, deciding what the pipeline hears at the moment sound is captured.

Two sounds must not reach the pipeline as the user's voice. The first is anything
captured while the key is up. Pipecat's input filter decides that when the event
loop gets to a frame, tens of milliseconds after capture, so a press passes audio
recorded before it; the key is read in the capture callback instead. The second is
the pipeline's own speech, which crosses the room from the speakers: an interruption
stops new audio reaching the speaker within a few milliseconds, but what was already
written still plays for about 185 ms (measured 2026-09-14, MacBook Pro speakers and
microphone), and Whisper made a word of it with nobody speaking. So everything the
speaker plays is given to the echo canceller (`hands.voice.echo`), and every buffer
the microphone captures is heard through it: the microphone stays open while the
speaker plays, and a press mid-reply keeps its first word.

Each side is on the device config.toml chooses for it (`hands.voice.devicechoice`): the system's default, one
device by name, or the default while that one is missing. When a device goes, a
headset unplugged, PortAudio says nothing: measured 2026-09-24 against an aggregate
device destroyed mid-stream, the microphone's callbacks simply stop and a write to
the speaker blocks for good. So the transport is reopened whole when where the sides should be changes, which it does
the moment the device under one disappears: macOS moves the defaults then, and a named device leaves the list.

The phone is the other place hands can be (`hands.voice.phone`). The gate says which: the pipeline hears only that
place's microphone, so the clip owner is handed one stream of frames, and the speaker plays to that place, so a reply goes
where the user is.
"""

import asyncio
import threading
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from functools import partial
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, Protocol, cast

import pyaudio
from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndWorkerFrame, Frame, OutputAudioRawFrame, StartFrame, TTSAudioRawFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from hands.clock import Clock
from hands.voice import coreaudio
from hands.voice.devicechoice import Auto, Choice, Choices, Hardware, Named, Placement, placed
from hands.voice.cues import Cue, sound
from hands.sessions.audit import Played, Record
from hands.sessions.wide import annotate, begun, count, unit
from hands.voice.echo import COUNTS, SAMPLE_BYTES, Cancelled, Echo, EchoCanceller, frame_bytes
from hands.voice.phone import Phone
from hands.voice.ptt import Key, PushToTalk
from hands.voice.turnstop import TurnOpened
from hands.voice.voices import SpeakingIn
from hands.threads import SerialThread, off_loop

Instant = float  # seconds on the clock's `now`

# The speaker stream's period: how much sound the device takes from the playout at each callback.
SPEAKER_PERIOD_SECS = 0.02

# How far written sound may run ahead of what the device takes. Every chunk reaches the playout through the event loop
# and the writer thread, so a stall longer than this is a gap in the sound, and a reply broken by one every chunk is
# crunchy. A blocking stream with a 20 ms period kept 138 ms ahead on a MacBook's speakers (measured, 24 kHz mono),
# and played without gaps. The device's own latency comes after it.
WRITE_AHEAD_SECS = 0.14

# What a playout counts over its life: 10 ms frames the device took; of those, frames it took with nothing written,
# played as silence; and callbacks PortAudio said came too late, the device having played a gap before them.
PLAYOUT_COUNTS = ("pulled", "unwritten", "underflowed")

# How long a reopen may take before the run gives up on reopening in place. Closing a microphone whose device is
# gone was measured at 2.7 to 3.7 s; a reopen that never finishes fails the run, which reads as down, and the next
# `hands run` opens on the new defaults.
REOPEN_DEADLINE = timedelta(seconds=10)


@dataclass(frozen=True)
class Overheard:
    """A buffer of the desk's microphone through the echo canceller, whatever the key: what it heard and what the
    speaker played over it; whether the desk is where hands is, the gate's say, the desk heard as silence where not; and
    the gate's key as it was captured, which says both the press and the level: up at the desk is the off level with no
    press, where nothing but the shortcut opens a clip."""

    cancelled: Cancelled
    desk: bool
    key: Key


class Stream(Protocol):
    """What is used of a PyAudio stream, which is untyped."""

    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


class Playback(Stream, Protocol):
    """A stream the speaker writes to."""

    def write(self, audio: bytes, voices: tuple[str, ...]) -> None: ...
    def counts(self) -> dict[str, int]: ...


Pull = Callable[[bytes | None, int, object, int], tuple[bytes, int]]


class Playout:
    """The speaker's stream, fed to the device from PortAudio's callback, which gives the echo canceller exactly the
    sound the device takes, silence included, on the device's clock.

    [LAW:one-source-of-truth] the canceller's reference is taken where the device takes it. Given to the canceller as it
    was written to a blocking stream instead, measured 2026-10-06 on MacBook Pro speakers and microphone, the reference
    moved 20 ms against its echo at the start of a reading after a silence, by where the first write fell between the
    device's buffers and the microphone's; AEC3 met each move as a new echo path, and let the reading's opening through.

    A write fills WRITE_AHEAD_SECS of waiting sound as the device makes room in it, as a blocking stream's write fills
    its buffer.
    """

    def __init__(self, opening: Callable[[Pull], Stream], echo: Echo, sample_rate: int, channels: int, failed: Callable[[Exception], None]) -> None:
        self._echo = echo
        self._rate = sample_rate
        self._channels = channels
        self._frame_bytes = frame_bytes(sample_rate, channels)
        self._ahead_bytes = round(sample_rate * WRITE_AHEAD_SECS) * channels * SAMPLE_BYTES
        self._failed = failed
        # [LAW:no-shared-mutable-globals] written by the speaker's writer thread, taken by the device's callback, under
        # the one condition, which wakes a waiting write as the device takes sound and as the stream stops.
        self._waiting = bytearray()
        # The voices of hands' TTS in what is waiting, in order: how many bytes of it each run of voices is.
        self._voiced: deque[tuple[int, tuple[str, ...]]] = deque()
        self._room = threading.Condition()
        self._stopped = False
        # [LAW:nothing-unseen] read onto the speaker's event as it is let go of; written by the device's callback alone.
        self._counts: dict[str, int] = dict.fromkeys(PLAYOUT_COUNTS, 0)
        self._stream = opening(self._pull)

    def write(self, audio: bytes, voices: tuple[str, ...]) -> None:
        """Queue sound behind what is waiting, as there is room for it, with the `voices` of hands' TTS in it. A stopped
        stream refuses it, as PortAudio's does."""
        with self._room:
            while audio:
                self._room.wait_for(lambda: self._stopped or len(self._waiting) < self._ahead_bytes)
                if self._stopped:
                    raise OSError("the speaker stream is stopped")
                room = self._ahead_bytes - len(self._waiting)
                self._waiting += audio[:room]
                self._voiced.append((len(audio[:room]), voices))
                audio = audio[room:]

    def counts(self) -> dict[str, int]:
        """What the device has taken so far, by the names in PLAYOUT_COUNTS."""
        return dict(self._counts)

    def start_stream(self) -> None:
        self._stream.start_stream()

    def stop_stream(self) -> None:
        # [LAW:no-ambient-temporal-coupling] a write waiting on a device that is gone is released here, before PortAudio
        # is asked to stop it, which on such a device fails.
        with self._room:
            self._stopped = True
            self._room.notify_all()
        self._stream.stop_stream()

    def close(self) -> None:
        self._stream.close()

    def _pull(self, in_data: bytes | None, frame_count: int, time_info: object, status: int) -> tuple[bytes, int]:
        # [LAW:dataflow-not-control-flow] the device takes a whole buffer every time: what is waiting, then silence.
        size = frame_count * self._channels * SAMPLE_BYTES
        with self._room:
            taken = bytes(self._waiting[:size])
            del self._waiting[:size]
            voices = self._taken_voices(len(taken))
            self._room.notify_all()
        played = taken + bytes(size - len(taken))
        try:
            self._echo.played(played, self._rate, self._channels, voices)
        except Exception as error:
            # [LAW:no-silent-failure] PyAudio would abort the stream and say so only on stderr, leaving hands mute.
            self._failed(error)
            return played, pyaudio.paAbort
        self._counts["pulled"] += size // self._frame_bytes
        self._counts["unwritten"] += (size - len(taken)) // self._frame_bytes
        self._counts["underflowed"] += bool(status & pyaudio.paOutputUnderflow)
        return played, pyaudio.paContinue

    def _taken_voices(self, taken: int) -> tuple[str, ...]:
        """The voices in the `taken` bytes just taken from what is waiting, let go of from its runs; under the condition."""
        voices = dict[str, None]()
        while taken:
            length, run = self._voiced.popleft()
            voices |= dict.fromkeys(run)
            match length > taken:
                case True:
                    self._voiced.appendleft((length - taken, run))
                    taken = 0
                case False:
                    taken -= length
        return tuple(voices)


class NoInput:
    """The microphone's stream when the system has no input device, a null object: nothing to start, stop, or close.

    [LAW:dataflow-not-control-flow] the transport opens, starts, stops, and closes the microphone the same way with
    or without one; only this value differs.
    """

    def start_stream(self) -> None:
        pass

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class Instead:
    """The default device, opened in place of the one config.toml names, which is missing; for a microphone, no name
    when there is no default either."""

    missing: str
    device: str | None


# What a side is on, by the names the system gives its devices: a device, or the default in place of one missing. A
# microphone may be on nothing at all, None, when the system has no input device.
PlayedOn = str | Instead
HeardOn = str | Instead | None


@dataclass(frozen=True)
class Output:
    """A speaker stream as opened, with the device it was opened on and the canceller told what it plays."""

    stream: Playback
    device: PlayedOn
    echo: Echo


@dataclass(frozen=True)
class Attached(Output):
    """A speaker stream as attached, with the writer of everything played on it.

    [LAW:one-source-of-truth] a stream and its writer are one value, so letting go of one ends the other whichever
    stream the speaker has attached by then.
    """

    writes: SerialThread


@dataclass(frozen=True)
class Input:
    """A microphone stream as opened, with its device, and the canceller its every buffer is heard through; a NoInput
    stream when there is no device to open."""

    stream: Stream
    device: HeardOn
    echo: Echo


class PortAudio(Protocol):
    """What is used of PyAudio itself, which is untyped."""

    def open(self, **settings: object) -> Stream: ...
    def get_format_from_width(self, width: int) -> int: ...
    def get_default_input_device_info(self) -> object: ...
    def get_default_output_device_info(self) -> object: ...
    def get_device_count(self) -> int: ...
    def get_device_info_by_index(self, device_index: int) -> object: ...
    def terminate(self) -> None: ...


class Speaker(LocalAudioOutputTransport):
    """The local speaker, whose stream gives the echo canceller everything the device plays, and which hands the sound
    to the phone instead while hands is there."""

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, key: PushToTalk, phone: Phone, echo: Echo, record: Record, clock: Clock, choice: Choice) -> None:
        super().__init__(py_audio, params)
        self._key = key
        self._phone = phone
        self._choice = choice
        # The canceller the first stream is opened with; each stream after it is opened with its own.
        self._echo = echo
        self._record = record
        self._clock = clock
        # When sound was last handed to the speaker, for the heartbeat; None before the first.
        self.sounded_at: Instant | None = None
        # What the attached stream was opened on; None until setup opens the first.
        self.opened: Attached | None = None
        # Set while a stream is attached; a write waits on it through a reopen.
        self._attached = asyncio.Event()
        # Who is speaking, as the frames passing out of the pipeline say: hands as its sender starts and stops, the user
        # as a turn opens and closes. Quiet is set while it is nobody, and a cue for silence waits on it.
        self._speaking: set[Literal["hands", "user"]] = set()
        self.quiet = asyncio.Event()
        self.quiet.set()
        # [LAW:one-source-of-truth] the voice hands' TTS said its last utterance in, as the TTS said so ahead of its
        # audio (`hands.voice.voices.SpeakingIn`), and the voices of the next chunk written: Pipecat's sender joins
        # utterances into its chunks, so the first chunk after a new one may hold the end of the one before.
        self._voice: str | None = None
        self._chunk_voices: tuple[str, ...] = ()

    @property
    def hands_speaking(self) -> bool:
        """Whether hands is speaking, between its sender's start and stop."""
        return "hands" in self._speaking

    async def setup(self, setup: FrameProcessorSetup) -> None:
        # [LAW:one-source-of-truth] Pipecat's local setup is only the base's and an open; the open is open_stream's, so
        # the stream is opened one way at setup and at every reopen.
        await BaseOutputTransport.setup(self, setup)
        py_audio = cast(PortAudio, self._py_audio)
        self.attach(py_audio, self.open_stream(py_audio, self._echo))

    def open_stream(self, py_audio: PortAudio, echo: Echo) -> Output:
        """A new stream on the output config.toml chooses, not yet the speaker's. Blocking: PortAudio talks to the device."""
        # [LAW:one-source-of-truth] the name is read from the same live PortAudio the stream is opened on, as it opens;
        # asked later, an instance that has since been ended answers "no device" whatever is plugged in.
        index, device = _chosen(self._choice, py_audio, "maxOutputChannels", lambda: _name(py_audio.get_default_output_device_info()))
        playout = Playout(
            lambda pull: py_audio.open(
                format=py_audio.get_format_from_width(2),
                channels=self._params.audio_out_channels,
                rate=self.sample_rate,
                frames_per_buffer=round(self.sample_rate * SPEAKER_PERIOD_SECS),
                output=True,
                output_device_index=self._params.output_device_index if index is None else index,
                stream_callback=pull,
            ),
            echo,
            self.sample_rate,
            self._params.audio_out_channels,
            self._failed,
        )
        return Output(playout, device, echo)

    def attach(self, py_audio: PortAudio, opened: Output) -> None:
        self._py_audio = cast(pyaudio.PyAudio, py_audio)
        # [LAW:single-enforcer] every write to the stream goes through one thread, in place of Pipecat's executor, whose
        # thread the interpreter waits for at exit: a write blocked on a lost device would hold the process open.
        self.opened = Attached(opened.stream, opened.device, opened.echo, SerialThread("speaker writes"))
        self._out_stream = opened.stream
        self._attached.set()

    def detach(self) -> Attached:
        """Take the stream away: a write from here on waits for the next one to be attached."""
        opened = self._held()
        assert opened is not None, "the speaker stream opens in setup"
        self._attached.clear()
        self._out_stream = None
        return opened

    def _held(self) -> Attached | None:
        # As for the microphone: `_out_stream` says a stream is attached; `opened` outlives a detach.
        return None if self._out_stream is None else self.opened

    async def let_go(self, opened: Attached) -> None:
        """Stop a detached stream, end its writer once the writes already given to it have run, then close it."""
        # [LAW:nothing-unseen] one event for each stream let go of, carrying what its device took over its life.
        with unit("speaker.let_go", self._record, begun(self._clock), PLAYOUT_COUNTS):
            annotate(device=opened.device)
            try:
                # [LAW:no-ambient-temporal-coupling] closing frees the stream, so it waits until no write is inside it.
                # Stopping is what releases a write blocked on a lost device, and every write handed over before the
                # detach has returned once the writer thread has ended.
                await off_loop(lambda: _letting_go(opened.stream.stop_stream), "stopping the speaker")
                await opened.writes.close()
                await off_loop(opened.stream.close, "closing the speaker")
            finally:
                count(**opened.stream.counts())

    def _failed(self, error: Exception) -> None:
        # On the device's thread, as the microphone's capture fails on its own.
        asyncio.run_coroutine_threadsafe(self._fail(error), self.get_event_loop())

    async def _fail(self, error: Exception) -> None:
        await self.push_error("the speaker stopped: its playout failed", exception=error, force_treat_as_permanent=True)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        await self.push_frame(EndWorkerFrame(reason="the speaker's playout failed"), FrameDirection.UPSTREAM)

    async def cleanup(self) -> None:
        # Pipecat's local cleanup stops and closes the stream on the loop, under a write that may still be running;
        # the stream is let go of here the one way a reopen does it, off the loop.
        await BaseOutputTransport.cleanup(self)
        await _let_go_at_cleanup(self._clock, self._held(), self.let_go, self._attached.clear)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        # [LAW:one-source-of-truth] the sender pushes both edges of hands' speech through here, an interruption stops it,
        # and the user's turn edges pass through on their way out.
        match frame:
            case BotStartedSpeakingFrame():
                self._speaking.add("hands")
            case BotStoppedSpeakingFrame():
                self._speaking.discard("hands")
            case TurnOpened():
                # The user is speaking from their turn's first clip, before it cuts anything off.
                self._speaking.add("user")
            case UserStoppedSpeakingFrame():
                self._speaking.discard("user")
            case SpeakingIn(voice=voice):
                self._chunk_voices = tuple(dict.fromkeys((*self._chunk_voices, voice)))
                self._voice = voice
            case _:
                pass
        if self._speaking:
            self.quiet.clear()
        else:
            self.quiet.set()
        await super().push_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        # [LAW:one-source-of-truth] the gate says where hands is, read as each frame is written: a reply carries on at
        # the place the user has moved to. The phone's earbuds keep its microphone clear, so the canceller is not told.
        voices = self._voiced(frame)
        match self._key.gate.place:
            case "phone":
                if frame.audio.count(0) != len(frame.audio):
                    self.sounded_at = self._clock.now()
                await self._phone.play(frame.audio)
                return True
            case "desk":
                return await self._write_here(frame, voices)

    def _voiced(self, frame: OutputAudioRawFrame) -> tuple[str, ...]:
        """The voices of hands' TTS in `frame`: those of the chunk for a chunk of TTS audio, none for the silence and
        padding the transport writes of its own."""
        match frame:
            case TTSAudioRawFrame():
                voices, self._chunk_voices = self._chunk_voices, () if self._voice is None else (self._voice,)
                return voices
            case _:
                return ()

    async def _write_here(self, frame: OutputAudioRawFrame, voices: tuple[str, ...]) -> bool:
        # [LAW:dataflow-not-control-flow] every frame is written; one that comes while the transport reopens waits for
        # the new stream, so a reply carries on over the move instead of losing its middle.
        await self._attached.wait()
        opened = cast(Attached, self.opened)

        def write() -> None:
            # [LAW:no-ambient-temporal-coupling] on the writer thread, behind a turn's cue queued ahead of it. An
            # interruption cancels the await, but the chunk already handed to that thread plays out all the same.
            if frame.audio.count(0) != len(frame.audio):
                self.sounded_at = self._clock.now()
            opened.stream.write(frame.audio, voices)

        await opened.writes.run(write)
        return True

    def cue(self, cue: Cue) -> Played:
        """Hand a turn's cue to the stream attached now, ahead of the pipeline's next chunk, and not wait for it to play.

        [LAW:no-ambient-temporal-coupling] the talk key's edge calls this, and the key's next move never waits on the
        speaker, which a reopen holds for seconds. A cue belongs to the moment of its edge: with no stream attached
        there is nothing to play it on, and a tone that fails to play loses only the tone.

        It moves `sounded_at`, since it is sound given to the speaker, and the canceller is given it as the device plays
        it, like all the speaker's sound, so a word said over the tone that opens a turn reaches Whisper without the tone.
        """
        match self._key.gate.place, self._attached.is_set():
            case "phone", _:
                self.sounded_at = self._clock.now()
                # Not waited for, as a cue given to the desk's writer thread is not.
                self._phone.play(sound(cue, self.sample_rate, 1))
                return "phone"
            case "desk", False:
                logger.warning(f"no speaker is attached; the tone for {cue.line!r} is not played")
                return "unattached"
            case "desk", True:
                opened = cast(Attached, self.opened)
                audio = sound(cue, self.sample_rate, self._params.audio_out_channels)

                def write() -> None:
                    # A stream a reopen has stopped refuses the write; the writer thread says so and carries on.
                    self.sounded_at = self._clock.now()
                    opened.stream.write(audio, ())

                opened.writes.give(write)
                return "desk"


class KeyedMicrophone(LocalAudioInputTransport):
    """The local microphone, heard through the echo canceller and silent while the key is up; and the phone's, which
    the pipeline hears in its place while hands is at the phone."""

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams, key: PushToTalk, phone: Phone, echo: Echo, record: Record, clock: Clock, choice: Choice) -> None:
        super().__init__(py_audio, params)
        self._clock = clock
        self._key = key
        self._phone = phone
        self._choice = choice
        # The canceller the first stream is opened with; each stream after it is opened with its own.
        self._echo = echo
        self._record = record
        self._listening: asyncio.Task[None] | None = None
        # What the attached stream was opened on; None until setup opens the first.
        self.opened: Input | None = None
        # [LAW:no-shared-mutable-globals] who hears the desk whatever the gate says, each with the loop it hears on:
        # replaced whole on the event loop, read whole on the capture thread.
        self._overhearing: tuple[asyncio.Queue[Overheard], ...] = ()

    async def setup(self, setup: FrameProcessorSetup) -> None:
        # As for the speaker: the base's setup, then the one way the stream is opened.
        await BaseInputTransport.setup(self, setup)
        py_audio = cast(PortAudio, self._py_audio)
        self.attach(py_audio, self.open_stream(py_audio, self._echo))

    def open_stream(self, py_audio: PortAudio, echo: Echo) -> Input:
        """A new stream on the input config.toml chooses, not yet the microphone's; a NoInput where there is no device
        to open. Blocking: PortAudio talks to the device."""
        index, device = _chosen(self._choice, py_audio, "maxInputChannels", lambda: default_input(py_audio))
        match device:
            case None | Instead(device=None):
                return Input(NoInput(), device, echo)
            case _:
                return Input(self._open(py_audio, echo, index), device, echo)

    def _open(self, py_audio: PortAudio, echo: Echo, index: int | None) -> Stream:
        return py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=int(self._sample_rate / 100) * 2,  # 20 ms, as Pipecat opens it
            # [LAW:single-enforcer] a stream's buffers are heard through its own canceller alone: one still calling
            # back after a reopen has attached the next never touches the new one from a second thread.
            stream_callback=partial(self._captured, echo),
            input=True,
            input_device_index=self._params.input_device_index if index is None else index,
        )

    def attach(self, py_audio: PortAudio, opened: Input) -> None:
        self._py_audio = cast(pyaudio.PyAudio, py_audio)
        self.opened = opened
        self._in_stream = opened.stream

    def detach(self) -> Input:
        opened = self._held()
        assert opened is not None, "the microphone stream opens in setup"
        self._in_stream = None
        return opened

    def _held(self) -> Input | None:
        # Pipecat's start reads `_in_stream`, so it is what says a stream is attached; `opened` outlives a detach.
        return None if self._in_stream is None else self.opened

    async def let_go(self, opened: Input) -> None:
        """Stop and close a detached stream, then its canceller. Measured at 2.7 to 3.7 s when its device is gone."""

        def stop_and_close() -> None:
            _letting_go(opened.stream.stop_stream)
            opened.stream.close()
            # [LAW:no-ambient-temporal-coupling] on the thread that closed the stream, once it can call back no more:
            # a stream that never finishes closing keeps its canceller.
            opened.echo.close()

        # [LAW:nothing-unseen] one event for each stream let go of, carrying what its canceller did over its life.
        with unit("microphone.let_go", self._record, begun(self._clock), COUNTS):
            annotate(device=opened.device)
            try:
                await off_loop(stop_and_close, "closing the microphone")
            finally:
                count(**opened.echo.counts())

    @asynccontextmanager
    async def overheard(self) -> AsyncGenerator[AsyncIterator[bytes]]:
        """Every buffer the desk's microphone captures while open, heard through the echo canceller whatever the key
        says, and silent while hands is at the phone: what a trigger listens to for when to open and close the turn."""
        async with self.echoes() as echoes:

            async def buffers() -> AsyncIterator[bytes]:
                async for each in echoes:
                    yield each.cancelled.heard if each.desk else bytes(len(each.cancelled.heard))

            yield buffers()

    @asynccontextmanager
    async def echoes(self) -> AsyncGenerator[AsyncIterator[Overheard]]:
        """Every buffer the desk's microphone captures while open, as `overheard`, with what the speaker played over it
        and where the gate stood as it was captured."""
        heard: asyncio.Queue[Overheard] = asyncio.Queue()
        self._overhearing = (*self._overhearing, heard)

        async def buffers() -> AsyncIterator[Overheard]:
            while True:
                yield await heard.get()

        try:
            yield buffers()
        finally:
            self._overhearing = tuple(each for each in self._overhearing if each is not heard)

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._listening = self.create_task(self._hear_the_phone(), "the phone's microphone")  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

    async def _hear_the_phone(self) -> None:
        while True:
            # Keyed by the phone as each arrived, in order with its button.
            await self.push_audio_frame(await self._phone.heard.get())

    async def cleanup(self) -> None:
        if self._listening is not None:
            await self.cancel_task(self._listening)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        # As for the speaker: off the loop, where closing a microphone whose device is gone takes seconds.
        await BaseInputTransport.cleanup(self)
        await _let_go_at_cleanup(self._clock, self._held(), self.let_go, lambda: None)

    def _captured(self, echo: Echo, in_data: bytes, frame_count: int, time_info: object, status: int) -> tuple[None, int]:
        # [LAW:no-ambient-temporal-coupling] decided on the audio thread, not when the event loop gets to the frame.
        # [LAW:dataflow-not-control-flow] every buffer goes through the canceller, key up or down and at either place:
        # it keeps learning the room, and takes the speaker's sound in step with it.
        try:
            cancelled = echo.heard(in_data, self._sample_rate, self._params.audio_in_channels)
        except Exception as error:
            # [LAW:no-silent-failure] PyAudio would abort the stream on a raise and say so only on stderr, leaving
            # hands running deaf; the pipeline is ended instead, and the run with it.
            asyncio.run_coroutine_threadsafe(self._fail(error), self.get_event_loop())
            return None, pyaudio.paAbort
        loop = self.get_event_loop()
        # One read of the gate: what the frame holds and the key it says it was captured under always agree.
        gate = self._key.gate
        # [LAW:single-enforcer] the gate says which place is heard, by a trigger as by the clip owner: while hands is at the
        # phone, the desk's buffers reach a trigger as silence of the same length, so the room there opens nothing.
        overheard = Overheard(cancelled, gate.hears("desk"), gate.key)
        for overhearing in self._overhearing:
            loop.call_soon_threadsafe(overhearing.put_nowait, overheard)
        # [LAW:single-enforcer] the gate alone says which place's microphone reaches the clip owner; while hands is at the
        # phone, the desk's frames are not the pipeline's at all, so its stream is the phone's unbroken.
        if not gate.hears("desk"):
            return None, pyaudio.paContinue
        frame = gate.framed(cancelled.heard, in_data, cancelled.played, self._sample_rate, self._params.audio_in_channels, "desk")
        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), loop)
        return None, pyaudio.paContinue

    async def _fail(self, error: Exception) -> None:
        await self.push_error("the microphone stopped: its capture failed", exception=error, force_treat_as_permanent=True)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        await self.push_frame(EndWorkerFrame(reason="the microphone's capture failed"), FrameDirection.UPSTREAM)


@dataclass(frozen=True)
class Devices:
    """The devices the transport is on, by the names the system gives them."""

    input: HeardOn
    output: PlayedOn


class KeyedAudioTransport(LocalAudioTransport):
    """`LocalAudioTransport` with the keyed microphone and the speaker, joined by the echo canceller of the streams open."""

    def __init__(
        self,
        clock: Clock,
        params: LocalAudioTransportParams,
        key: PushToTalk,
        phone: Phone,
        record: Record,
        portaudio: Callable[[], PortAudio] = lambda: cast(PortAudio, pyaudio.PyAudio()),
        hardware: Callable[[], Hardware] = coreaudio.hardware,
        echo: Callable[[], Echo] = EchoCanceller,
        choices: Choices = Choices(),
    ) -> None:
        # [LAW:one-source-of-truth] where the streams are is read where PortAudio lists the devices, just before it
        # does, here and at every reopen; a change after the read is one the follower sees. Read any later, it would
        # miss an unplug during the model load between this and the pipeline's start.
        self._choices = choices
        self._hardware = hardware
        self.opened_on = self.wanted()
        # [LAW:single-enforcer] PortAudio is started by the one factory, here and at every reopen. LocalAudioTransport's
        # own init would start an instance of its own, and while two are alive, ending one lists no devices anew.
        BaseTransport.__init__(self)
        self._clock = clock
        self._params = params
        self._pyaudio = cast(pyaudio.PyAudio, portaudio())
        # [LAW:one-source-of-truth] one canceller for the streams opened together: what the speaker plays is what it
        # takes out of what the microphone hears. A reopen opens the next pair with a new one, which learns the new
        # devices' room from nothing and holds none of the old one's sound.
        cancelling = echo()
        self._echo = echo
        self._speaker = Speaker(self._pyaudio, params, key, phone, cancelling, record, clock, choices.output)
        self._microphone = KeyedMicrophone(self._pyaudio, params, key, phone, cancelling, record, clock, choices.input)
        self._portaudio = portaudio

    def input(self) -> KeyedMicrophone:
        return self._microphone

    def output(self) -> Speaker:
        return self._speaker

    def wanted(self) -> Placement:
        """Where config.toml puts the streams on the devices there are now. Blocking: CoreAudio is asked."""
        return placed(self._choices, self._hardware())

    async def reopen(self) -> Devices:
        """Close both streams and PortAudio itself, and open them again where config.toml puts them on the devices as they are now."""
        # [LAW:no-ambient-temporal-coupling] the order is the mechanism. Detached first, so a write that comes
        # meanwhile finds no stream rather than the lost one. The speaker is let go of before the microphone,
        # because stopping it is what releases a write blocked on a lost device. PortAudio is ended before it is
        # started again, because it lists the devices only when it starts, and while one instance lives a new one
        # still sees the device that is gone. Every step that talks to a device runs off the loop, inside the
        # deadline, so a device that hangs stops neither the heartbeat nor the run's own stop.
        speaker, microphone, ending = self._speaker.detach(), self._microphone.detach(), self._audio_now()
        async with self._clock.timeout(REOPEN_DEADLINE.total_seconds()):
            await self._speaker.let_go(speaker)
            await self._microphone.let_go(microphone)
            started = await off_loop(lambda: self._start_again(ending), "starting PortAudio again")
        self.opened_on = started.placement
        # [LAW:one-source-of-truth] one PortAudio handle, shared by the transport and both sides; the ended one is dropped everywhere.
        self._pyaudio = cast(pyaudio.PyAudio, started.portaudio)
        self._speaker.attach(started.portaudio, started.speaker)
        self._microphone.attach(started.portaudio, started.microphone)
        # A write that timed out on the lost device cost the speaker its usability; the new device has it back.
        await self._speaker.set_usable(True)
        return self.devices

    @property
    def devices(self) -> Devices:
        """The devices the streams are open on, as each was named when it was opened."""
        match (self._microphone.opened, self._speaker.opened):
            case (Input(device=heard), Output(device=played)):
                return Devices(heard, played)
            case _:
                raise AssertionError("the devices are known once the pipeline has set up its streams")

    @property
    def deaf(self) -> bool:
        """Whether there is no device for the microphone to open, so a press to talk hears nothing; never before setup
        opens it."""
        # [LAW:one-source-of-truth] the device the spoken warning names, read where `devices` reads it.
        match self._microphone.opened:
            case Input(device=None | Instead(device=None)):
                return True
            case _:
                return False

    def _audio_now(self) -> PortAudio:
        return cast(PortAudio, self._pyaudio)

    def _start_again(self, ending: PortAudio) -> "_Started":
        ending.terminate()
        placement = self.wanted()
        portaudio = self._portaudio()
        cancelling = self._echo()
        opened: list[Stream] = []
        try:
            speaker = self._speaker.open_stream(portaudio, cancelling)
            opened.append(speaker.stream)
            microphone = self._microphone.open_stream(portaudio, cancelling)
            opened.append(microphone.stream)
            speaker.stream.start_stream()
            microphone.stream.start_stream()
        except BaseException:
            # [LAW:no-ambient-temporal-coupling] PyAudio starts a callback stream as it opens it, so each stream opened
            # here calls back into the canceller; each is stopped and closed before the canceller goes, and no let_go
            # will reach any of them.
            for stream in opened:
                _letting_go(stream.stop_stream)
                stream.close()
            cancelling.close()
            raise
        return _Started(placement, portaudio, speaker, microphone)


@dataclass(frozen=True)
class _Started:
    """PortAudio started again, with both streams open and running on it, not yet the sides'."""

    placement: Placement
    portaudio: PortAudio
    speaker: Output
    microphone: Input


def _letting_go(step: Callable[[], None]) -> None:
    try:
        step()
    except OSError as error:
        # A stream on a device that is gone refuses to stop cleanly; it is let go of all the same, and said.
        logger.warning(f"letting go of an audio stream whose device is gone: {error}")


async def _let_go_at_cleanup[S](clock: Clock, held: S | None, let_go: Callable[[S], Awaitable[None]], detached: Callable[[], None]) -> None:
    match held:
        case None:
            # Nothing is held: a reopen that failed part way through let go of the old stream and attached no new one.
            pass
        case stream:
            detached()
            async with clock.timeout(REOPEN_DEADLINE.total_seconds()):
                await let_go(stream)


def default_input(py_audio: PortAudio) -> str | None:
    """The default input device's name, or None when the system has none: a Mac with no built-in microphone, its headset unplugged."""
    # [LAW:parse-dont-validate] PortAudio says there is no default input by raising; here that becomes a typed absence.
    try:
        info = py_audio.get_default_input_device_info()
    except OSError as error:
        # Asked only of a PortAudio that is running: an ended one says the same words whatever is plugged in.
        if error.args != (_NO_DEFAULT_INPUT,):
            raise
        return None
    return _name(info)


def _chosen[D: str | None](choice: Choice, py_audio: PortAudio, channels: Literal["maxInputChannels", "maxOutputChannels"], default: Callable[[], D]) -> tuple[int | None, D | str | Instead]:
    """Where a side opens on the devices `py_audio` lists: the index of the device `choice` names, None for the default,
    and the device as the side is on it."""
    match choice:
        case Auto():
            return None, default()
        case Named(name=name):
            match device_index(py_audio, name, channels):
                case None:
                    # [LAW:no-silent-failure] the default, said to be in place of the one named.
                    return None, Instead(name, default())
                case index:
                    return index, name


def device_index(py_audio: PortAudio, name: str, channels: Literal["maxInputChannels", "maxOutputChannels"]) -> int | None:
    """The index `py_audio` lists the device called `name` at, among those with `channels`; None where it lists none."""
    for index in range(py_audio.get_device_count()):
        # PyAudio's device info is an untyped mapping.
        info = cast(Mapping[str, object], py_audio.get_device_info_by_index(index))
        match info.get("name"), info.get(channels):
            case str(listed), int(count) if listed == name and count > 0:
                return index
            case _:
                pass
    return None


# PyAudio's words when PortAudio lists no default input, raised with no errno to tell it by.
_NO_DEFAULT_INPUT = "No Default Input Device Available"


def _name(info: object) -> str:
    # PyAudio's device info is an untyped mapping; the name is the one field read from it.
    match info:
        case {"name": str(name)}:
            return name
        case _:
            raise AssertionError(f"PortAudio described a device without a name: {info!r}")

