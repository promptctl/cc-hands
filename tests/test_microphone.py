"""What the keyed microphone lets the pipeline hear, decided at capture, with no audio device."""

import asyncio
import itertools
import threading
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast
from collections.abc import AsyncGenerator, Callable

import pyaudio
import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndWorkerFrame, Frame, InputAudioRawFrame, OutputAudioRawFrame, TTSAudioRawFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.local.audio import LocalAudioInputTransport, LocalAudioOutputTransport, LocalAudioTransportParams

from conftest import speaker_writers

from hands.virtualclock import VirtualClock
from hands.clock import SYSTEM
from hands.sessions.audit import Entry, Record
from hands.sessions.wide import WideEvent
from hands.voice.cues import OPENED, sound
from hands.voice.echo import Cancelled
from hands.voice.clips import Played, Reference
from hands.voice.devicechoice import Auto, Choices, Closed, DefaultDevices, Hardware, Instead, Missing, Named, Off, OnDefault, OnNamed, Placement
from hands.voice.microphone import WRITE_AHEAD_SECS, Attached, Devices, Input, KeyedAudioTransport, NoInput, NoOutput, Output, Playout, PortAudio, Pull
from hands.voice.system import AudioMoved, system_text
from hands.voice.phone import Phone
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.voice.clips import Clip
from hands.voice.turnstop import TurnOpened
from hands.voice.voices import SpeakingIn

LOUD = b"\x7f\x7f" * 320
QUIET = bytes(len(LOUD))
CLEANED = b"\x01\x00" * 320  # what the room's canceller makes of anything the microphone hears
PLAYED = (Reference(b"\x02\x00" * 480, 48_000, 1, ("alba",)),) * 2  # what it matched each buffer's two frames with


def _phone() -> Phone:
    return Phone(SYSTEM, PushToTalk(SYSTEM, lambda _: None), heard_rate=16000, played_rate=16000, record=lambda _: None)


class Room:
    """An echo canceller that keeps what the speaker played and cleans every microphone buffer to CLEANED."""

    def __init__(self, log: list[str] | None = None, name: str = "canceller") -> None:
        self.plays: list[tuple[bytes, int, int, tuple[str, ...]]] = []
        self.hears: list[bytes] = []
        self.log = [] if log is None else log
        self.name = name

    def played(self, audio: bytes, sample_rate: int, channels: int, voices: tuple[str, ...]) -> None:
        self.plays.append((audio, sample_rate, channels, voices))

    def heard(self, audio: bytes, sample_rate: int, channels: int) -> Cancelled:
        self.hears.append(audio)
        return Cancelled(CLEANED, PLAYED)

    def counts(self) -> dict[str, int]:
        return {"heard": len(self.hears), "unplayed": 0, "dropped": 0}

    def close(self) -> None:
        self.log.append(f"close {self.name}")


class Rig:
    """The transport with its PyAudio calls replaced: the clock is set by hand and pushed audio is kept."""

    def __init__(self) -> None:
        self.clock = VirtualClock()
        self.key = PushToTalk(SYSTEM, lambda _: None)
        self.phone = Phone(SYSTEM, self.key, heard_rate=16000, played_rate=16000, record=lambda _: None)
        self.pushed: list[bytes] = []
        # Each pushed frame's sound as the microphone captured it, before the canceller, and what was played over it.
        self.captured: list[bytes] = []
        self.played: list[Played] = []
        self.room = Room()
        params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
        self.transport = KeyedAudioTransport(self.clock, params, self.key, self.phone, lambda _: None, echo=lambda: self.room)
        self.speaker = self.transport.output()
        self.microphone = self.transport.input()
        self.microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
        self.stream = SimpleStream()
        self.speaker.attach(cast(PortAudio, SimpleNamespace()), Output(self.stream, "MacBook Pro Speakers", self.room))  # as setup attaches the stream it opened

        async def push_audio_frame(frame: InputAudioRawFrame) -> None:
            assert isinstance(frame, KeyedAudio)
            self.pushed.append(frame.audio)
            self.captured.append(frame.captured)
            self.played.append(frame.played)

        self.microphone.push_audio_frame = push_audio_frame
        self.microphone.get_event_loop = asyncio.get_running_loop
        self.speaker.get_event_loop = asyncio.get_running_loop

    async def play(self, audio: bytes, at: float) -> None:
        self.clock.at = at
        await self.speaker.write_audio_frame(OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1))

    async def tts(self, audio: bytes, at: float) -> None:
        """A chunk of hands' TTS, as Pipecat's sender writes it."""
        self.clock.at = at
        await self.speaker.write_audio_frame(TTSAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1, context_id="reply"))

    async def written(self) -> None:
        """Once the speaker's writer has done everything given to it."""
        await cast(Attached, self.speaker.opened).writes.run(lambda: None)

    async def capture(self, at: float) -> None:
        self.clock.at = at
        self.microphone._captured(self.room, LOUD, 320, {}, 0)  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0.01)


@pytest.fixture
async def devices() -> AsyncGenerator[Rig, None]:
    """The rig, its speaker cleaned up as the test ends, which ends its stream's writer."""
    rig = Rig()
    yield rig
    await rig.speaker.cleanup()


class SimpleStream:
    def __init__(self) -> None:
        self.blocking = False
        self.written: list[bytes] = []
        self.voiced: list[tuple[str, ...]] = []

    def write(self, audio: bytes, voices: tuple[str, ...]) -> None:
        # A blocking write returns when the device has taken the chunk, about when it has played.
        while self.blocking:
            time.sleep(0.001)
        self.written.append(audio)
        self.voiced.append(voices)

    def get_output_latency(self) -> float:
        return 0.0

    def counts(self) -> dict[str, int]:
        return {"pulled": len(self.written), "unwritten": 0}

    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


async def test_the_microphone_is_heard_through_the_canceller_and_only_while_the_key_is_down(devices: Rig) -> None:
    await devices.capture(at=1.0)  # key up
    devices.key.move("start", "shortcut")
    await devices.capture(at=1.02)
    assert devices.pushed == [QUIET, CLEANED]
    # Each frame carries what the microphone captured too, muted by the same key: a clip's level before the canceller.
    assert devices.captured == [QUIET, LOUD]
    # And what the speaker played over it, the canceller's reference, which no key mutes: what hands said over a clip.
    assert devices.played == [PLAYED, PLAYED]
    assert devices.room.hears == [LOUD, LOUD]  # key up too: the canceller learns the room from every buffer


async def test_the_canceller_hears_the_desk_microphone_while_hands_is_at_the_phone(devices: Rig) -> None:
    devices.key.go("phone")
    await devices.capture(at=1.0)
    assert devices.pushed == []  # the phone's microphone is the pipeline's
    assert devices.room.hears == [LOUD]  # and the canceller takes the speaker's sound in step all the same


async def test_a_trigger_overhears_the_desk_cleaned_whatever_the_key_says_and_only_while_it_listens(devices: Rig) -> None:
    async with devices.microphone.overheard() as heard:
        await devices.capture(at=1.0)  # key up: the pipeline hears silence, the trigger hears the room
        devices.key.go("phone")
        await devices.capture(at=1.02)  # hands at the phone: the room at the desk is silence to the trigger
        assert [await anext(heard), await anext(heard)] == [CLEANED, bytes(len(CLEANED))]
    await devices.capture(at=1.04)
    assert devices.pushed == [QUIET]


class Refusing(Room):
    def heard(self, audio: bytes, sample_rate: int, channels: int) -> Cancelled:
        raise RuntimeError("the audio processing module refused the frame")


async def test_a_capture_the_canceller_fails_ends_the_pipeline_rather_than_leaving_hands_deaf(devices: Rig) -> None:
    errors: list[str] = []
    pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_error(error_msg: str, exception: Exception | None = None, fatal: bool = False, category: object = None, force_treat_as_permanent: bool = False) -> None:
        errors.append(f"{error_msg}: {exception}")

    async def push_frame(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append((frame, direction))

    setattr(devices.microphone, "push_error", push_error)
    setattr(devices.microphone, "push_frame", push_frame)
    assert devices.microphone._captured(Refusing(), LOUD, 320, {}, 0) == (None, pyaudio.paAbort)  # pyright: ignore[reportPrivateUsage]
    await asyncio.sleep(0.01)
    assert errors == ["the microphone stopped: its capture failed: the audio processing module refused the frame"]
    assert [(type(frame), direction) for frame, direction in pushed] == [(EndWorkerFrame, FrameDirection.UPSTREAM)]
    assert devices.pushed == []


async def test_a_press_while_the_reply_plays_is_heard_at_once(devices: Rig) -> None:
    await devices.play(LOUD, at=1.0)
    devices.key.move("start", "shortcut")
    await devices.capture(at=1.0)  # the reply still in the room, and a word over it
    assert devices.pushed == [CLEANED]


async def test_silence_given_to_the_speaker_is_no_sound_for_the_heartbeat(devices: Rig) -> None:
    await devices.play(QUIET, at=1.0)
    assert devices.stream.written == [QUIET]
    assert devices.speaker.sounded_at is None  # silence padding is no sound for the heartbeat


async def test_a_chunk_whose_write_an_interruption_cancels_is_still_played(devices: Rig) -> None:
    devices.stream.blocking = True
    devices.clock.at = 1.0
    writing = asyncio.create_task(devices.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    writing.cancel()  # the barge-in; PortAudio's thread plays the chunk out regardless
    devices.stream.blocking = False
    await devices.written()
    assert devices.stream.written == [LOUD]


class LostStream:
    """A stream on a device that is gone, as PortAudio leaves it: a write blocks until the stream is stopped, and stopping fails."""

    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name
        self.stopped = threading.Event()

    def write(self, audio: bytes, voices: tuple[str, ...]) -> None:
        self.stopped.wait()
        self.log.append(f"write returned from {self.name}")
        raise OSError(-9986, "Internal PortAudio error")

    def start_stream(self) -> None:
        self.log.append(f"start {self.name}")

    def get_output_latency(self) -> float:
        return 0.2  # the new speaker's buffer, longer than the one it replaced

    def counts(self) -> dict[str, int]:
        return {"pulled": 0, "unwritten": 0}

    def stop_stream(self) -> None:
        self.log.append(f"stop {self.name}")
        self.stopped.set()
        raise OSError(-9986, "Internal PortAudio error")

    def close(self) -> None:
        self.log.append(f"close {self.name}")


# The devices PortAudio lists, by the fields of its device info hands reads.
BUILT_IN: tuple[Mapping[str, object], ...] = (
    {"name": "MacBook Pro Microphone", "maxInputChannels": 1, "maxOutputChannels": 0},
    {"name": "MacBook Pro Speakers", "maxInputChannels": 0, "maxOutputChannels": 2},
)
DISPLAY: tuple[Mapping[str, object], ...] = (
    {"name": "Studio Display Microphone", "maxInputChannels": 1, "maxOutputChannels": 0},
    {"name": "Studio Display Speakers", "maxInputChannels": 0, "maxOutputChannels": 2},
)


class FreshPortAudio:
    """PortAudio started again: it opens streams on the defaults it lists now, among the `listed` devices."""

    def __init__(self, log: list[str], listed: tuple[Mapping[str, object], ...] = BUILT_IN) -> None:
        self.log = log
        self.listed = listed
        self.opened: list[dict[str, object]] = []
        log.append("start portaudio")

    def get_device_count(self) -> int:
        return len(self.listed)

    def get_device_info_by_index(self, device_index: int) -> object:
        return self.listed[device_index]

    def open(self, **settings: object) -> LostStream:
        side = "microphone" if settings.get("input") else "speaker"
        self.log.append(f"open {side}")
        self.opened.append(settings)
        return LostStream(self.log, f"new {side}")

    def get_format_from_width(self, width: int) -> int:
        return 8

    def get_default_input_device_info(self) -> object:
        return {"name": "MacBook Pro Microphone"}

    def get_default_output_device_info(self) -> object:
        return {"name": "MacBook Pro Speakers"}

    def terminate(self) -> None:
        self.log.append("end portaudio")


def lost_transport(log: list[str], events: list[Entry] | None = None, choices: Choices = Choices()) -> KeyedAudioTransport:
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    hardware = iter([Hardware(DefaultDevices(input=1, output=1)), Hardware(DefaultDevices(input=2, output=2))])
    cancellers = iter(Room(log, f"canceller {n}") for n in itertools.count(1))
    record: Record = (lambda _: None) if events is None else events.append
    transport = KeyedAudioTransport(SYSTEM, 
        params, PushToTalk(SYSTEM, lambda _: None), _phone(), record, portaudio=lambda: FreshPortAudio(log), hardware=lambda: next(hardware), echo=lambda: next(cancellers),
        choices=choices,
    )
    speaker, microphone = transport.output(), transport.input()
    speaker.get_event_loop = asyncio.get_running_loop
    microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    speaker._sample_rate = 24000  # pyright: ignore[reportPrivateUsage]
    old = Room(log, "old canceller")
    speaker.attach(cast(PortAudio, SimpleNamespace()), Output(LostStream(log, "old speaker"), "headset", old))
    microphone.attach(cast(PortAudio, SimpleNamespace()), Input(LostStream(log, "old microphone"), "headset", old))
    log.clear()  # the first PortAudio, started by the same factory as every later one
    return transport


Lost = Callable[..., KeyedAudioTransport]


@pytest.fixture
async def lost() -> AsyncGenerator[Lost, None]:
    """`lost_transport`, each one built cleaned up as the test ends, which ends its stream's writer."""
    built: list[KeyedAudioTransport] = []

    def building(log: list[str], events: list[Entry] | None = None, choices: Choices = Choices()) -> KeyedAudioTransport:
        built.append(lost_transport(log, events, choices))
        return built[-1]

    yield building
    for transport in built:
        await transport.output().cleanup()


async def test_reopening_lets_go_of_the_lost_devices_and_opens_on_the_defaults_as_they_are_now(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log)
    speaker = transport.output()
    frame = OutputAudioRawFrame(audio=LOUD, sample_rate=24000, num_channels=1)
    stuck = asyncio.create_task(speaker.write_audio_frame(frame))  # the reply playing when the headset went
    await asyncio.sleep(0.01)
    await speaker.set_usable(False)  # as Pipecat's write timeout leaves it

    devices = await transport.reopen()

    assert devices == Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
    assert log == [
        "stop old speaker", "write returned from old speaker", "close old speaker",  # closed only once no write is inside it
        "stop old microphone", "close old microphone", "close old canceller", "end portaudio",
        "start portaudio", "open speaker", "open microphone", "start new speaker", "start new microphone",
    ]  # fmt: skip
    assert transport.opened_on == Placement(OnDefault(2), OnDefault(2))  # read again, as PortAudio listed them anew
    with pytest.raises(OSError):
        await stuck
    assert speaker.is_usable


async def test_a_reopen_opens_both_streams_on_one_new_canceller_and_reports_the_old_streams_lives(lost: Lost) -> None:
    log: list[str] = []
    events: list[Entry] = []
    transport = lost(log, events)
    old = transport.input().opened
    assert old is not None
    old.echo.heard(LOUD, 16000, 1)  # a buffer heard on the old devices

    await transport.reopen()

    speaker, microphone = transport.output().opened, transport.input().opened
    assert speaker is not None and microphone is not None
    assert speaker.echo is microphone.echo is not old.echo
    assert "close canceller 2" not in log
    assert all(isinstance(event, WideEvent) for event in events)
    assert [(event.event, event.outcome, event.facts, dict(event.counts)) for event in cast(list[WideEvent], events)] == [
        ("speaker.let_go", "ok", {"device": "headset"}, {"pulled": 0, "unwritten": 0, "underflowed": 0}),
        ("microphone.let_go", "ok", {"device": "headset"}, {"heard": 1, "unplayed": 0, "dropped": 0, "slipped": 0}),
    ]


class RefusingPortAudio(FreshPortAudio):
    """PortAudio started again on a device that refuses to open."""

    def open(self, **settings: object) -> LostStream:
        raise OSError(-9996, "Invalid input device")


async def test_a_reopen_whose_streams_fail_to_open_lets_go_of_the_canceller_it_made_for_them(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log)
    setattr(transport, "_portaudio", lambda: RefusingPortAudio(log))
    with pytest.raises(OSError, match="Invalid input device"):
        await transport.reopen()
    assert "close canceller 2" in log  # the run fails, and exits without LiveKit's assertion over a canceller left open


class MicrophoneRefusingPortAudio(FreshPortAudio):
    """PortAudio started again with a speaker that opens, and so starts, and a microphone that refuses to open."""

    def open(self, **settings: object) -> LostStream:
        if settings.get("input"):
            raise OSError(-9996, "Invalid input device")
        return super().open(**settings)


async def test_a_reopen_whose_microphone_fails_to_open_stops_the_speaker_it_started_before_letting_go_of_the_canceller(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log)
    setattr(transport, "_portaudio", lambda: MicrophoneRefusingPortAudio(log))
    with pytest.raises(OSError, match="Invalid input device"):
        await transport.reopen()
    assert log[log.index("open speaker") :] == ["open speaker", "stop new speaker", "close new speaker", "close canceller 2"]


async def test_a_frame_given_while_the_transport_reopens_waits_and_plays_on_the_new_stream(devices: Rig) -> None:
    old = devices.speaker.detach()
    devices.clock.at = 1.0
    writing = asyncio.create_task(devices.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    assert not writing.done()
    assert devices.speaker.sounded_at is None  # nothing has sounded yet
    reopened = SimpleStream()
    devices.speaker.attach(cast(PortAudio, SimpleNamespace()), Output(reopened, "AirPods", devices.room))
    assert await writing is True
    assert reopened.written == [LOUD]
    assert devices.speaker.sounded_at == 1.0
    await devices.speaker.let_go(old)


async def test_a_reopen_ends_the_writer_of_the_stream_it_left_and_starts_the_new_ones(lost: Lost) -> None:
    transport = lost([])
    left = speaker_writers()
    await transport.reopen()
    assert len(left) == 1 and speaker_writers() != left and len(speaker_writers()) == 1


async def test_a_reopen_after_the_pipeline_cleaned_up_lets_go_of_the_same_stream_again_and_the_new_stream_plays(lost: Lost) -> None:
    transport = lost([])
    speaker = transport.output()
    speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    await speaker.cleanup()
    await transport.reopen()
    assert speaker.cue(OPENED) == "desk"


async def test_a_cleanup_still_letting_go_as_a_reopen_attaches_leaves_the_new_streams_writer_open(lost: Lost) -> None:
    transport = lost([])
    speaker = transport.output()
    speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    entered, released = threading.Event(), threading.Event()
    stream = cast(LostStream, cast(Attached, speaker.opened).stream)
    stop = stream.stop_stream

    def stalled() -> None:
        # The first stop stands still; the reopen's own stop of the same stream, behind it, does not.
        stream.stop_stream = stop
        entered.set()
        released.wait()
        stop()

    stream.stop_stream = stalled
    cleaning = asyncio.create_task(speaker.cleanup())
    while not entered.is_set():
        await asyncio.sleep(0)
    await transport.reopen()
    released.set()
    await cleaning
    assert speaker.cue(OPENED) == "desk"


async def test_the_streams_are_opened_as_pipecat_opens_them() -> None:
    """The sides open their own streams so they can open them again; this is what says a Pipecat upgrade changed how."""
    ours, pipecats = FreshPortAudio([]), FreshPortAudio([])
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=cast(Any, None), audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    speaker = KeyedAudioTransport(SYSTEM, params, PushToTalk(SYSTEM, lambda _: None), _phone(), lambda _: None).output()
    for portaudio, output, input_ in (
        (ours, speaker, KeyedAudioTransport(SYSTEM, params, PushToTalk(SYSTEM, lambda _: None), _phone(), lambda _: None).input()),
        (pipecats, LocalAudioOutputTransport(cast(pyaudio.PyAudio, pipecats), params), LocalAudioInputTransport(cast(pyaudio.PyAudio, pipecats), params)),
    ):
        setattr(output, "_py_audio", portaudio)
        setattr(input_, "_py_audio", portaudio)
        await output.setup(setup)
        await input_.setup(setup)
    await speaker.cleanup()
    assert len(ours.opened) == 2
    # The one difference is chosen: the speaker's period, which Pipecat leaves to the device's low-latency default.
    speaker, microphone = ours.opened
    assert speaker.pop("frames_per_buffer") == 480
    assert [{k: v for k, v in opened.items() if k != "stream_callback"} for opened in (speaker, microphone)] == [
        {k: v for k, v in opened.items() if k != "stream_callback"} for opened in pipecats.opened
    ]


class DeafPortAudio(FreshPortAudio):
    """PortAudio on a Mac whose last input device is gone: it lists no default input."""

    def get_default_input_device_info(self) -> object:
        raise OSError("No Default Input Device Available")  # as PyAudio raises it: the words alone, no errno


async def test_with_no_microphone_left_the_transport_reopens_to_speak_and_says_it_cannot_hear(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log)
    setattr(transport, "_portaudio", lambda: DeafPortAudio(log))
    assert not transport.deaf

    devices = await transport.reopen()

    assert devices == Devices(input=None, output="MacBook Pro Speakers")
    assert transport.deaf and transport.microphone_lost  # what the heartbeat carries to the menu bar
    assert "open microphone" not in log  # nothing to open: the microphone holds a stream of nothing
    assert "start new speaker" in log
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]

    # A microphone plugged in again is opened by the next reopen, the same way as any other.
    setattr(transport, "_portaudio", lambda: FreshPortAudio(log))
    setattr(transport, "_hardware", lambda: Hardware(DefaultDevices(input=3, output=2)))
    log.clear()
    assert (await transport.reopen()).input == "MacBook Pro Microphone"
    assert "open microphone" in log
    assert not transport.deaf


async def test_a_daemon_started_with_no_microphone_runs_rather_than_failing_its_setup() -> None:
    params = LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    transport = KeyedAudioTransport(SYSTEM, params, PushToTalk(SYSTEM, lambda _: None), _phone(), lambda _: None, portaudio=lambda: DeafPortAudio([]), hardware=lambda: Hardware(DefaultDevices(0, 1)), echo=Room)
    assert not transport.deaf  # nothing is said of hearing before setup opens the microphone
    setup = FrameProcessorSetup(clock=SystemClock(), task_manager=TaskManager(), pipeline_worker=cast(Any, None), audio_in_sample_rate=16000, audio_out_sample_rate=24000)
    await transport.input().setup(setup)
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]
    opened = transport.input().opened
    assert opened is not None and opened.device is None and isinstance(opened.stream, NoInput)
    assert transport.deaf
    await transport.input().cleanup()  # and lets go of nothing, without complaint


async def test_a_turns_cue_is_played_at_once_and_a_word_over_it_is_heard(devices: Rig) -> None:
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]  # as setup sets it
    devices.key.move("start", "shortcut")
    devices.clock.at = 1.0
    devices.speaker.cue(OPENED)
    await devices.capture(at=1.0)  # a word said over the cue
    await devices.written()
    assert devices.stream.written == [sound(OPENED, 16000, 1)]
    assert devices.pushed == [CLEANED]  # the word said over it, through the canceller
    assert devices.speaker.sounded_at == 1.0


async def test_a_cue_never_holds_the_talk_key_on_the_speaker(devices: Rig) -> None:
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    devices.stream.blocking = True  # a write stuck on a device that is going
    assert devices.speaker.cue(OPENED) == "desk"
    detached = devices.speaker.detach()  # a reopen under way
    assert devices.speaker.cue(OPENED) == "unattached"  # said, so a cue for silence is never recorded as heard
    devices.stream.blocking = False
    await devices.speaker.let_go(detached)
    assert devices.stream.written == [sound(OPENED, 16000, 1)]


async def test_a_tone_that_fails_to_play_loses_only_the_tone(devices: Rig) -> None:
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    written = devices.stream.write

    def stopped(audio: bytes, voices: tuple[str, ...]) -> None:
        raise OSError("Stream is stopped")

    devices.stream.write = stopped
    devices.speaker.cue(OPENED)
    await devices.written()
    devices.stream.write = written
    await devices.play(LOUD, at=1.0)
    assert devices.stream.written == [LOUD]


async def test_each_chunk_is_written_with_the_voices_hands_tts_made_it_in_and_a_cue_with_none(monkeypatch: pytest.MonkeyPatch, devices: Rig) -> None:
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]

    async def passed(_self: object, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pass

    monkeypatch.setattr(LocalAudioOutputTransport, "push_frame", passed)
    await devices.tts(LOUD, at=1.0)  # before the TTS says any voice: none
    await devices.speaker.push_frame(SpeakingIn("alba"))
    await devices.tts(LOUD, at=1.02)
    await devices.play(LOUD, at=1.03)  # the transport's own silence or padding: no voice, and alba's next chunk keeps hers
    await devices.tts(LOUD, at=1.04)
    await devices.speaker.push_frame(SpeakingIn("mary"))
    # Pipecat's sender joins utterances into its chunks: the first after the change may end alba's last.
    await devices.tts(LOUD, at=1.06)
    await devices.tts(LOUD, at=1.08)
    devices.speaker.cue(OPENED)
    await devices.written()
    assert devices.stream.voiced == [(), ("alba",), (), ("alba",), ("alba", "mary"), ("mary",), ()]


def test_the_canceller_is_given_the_voices_in_each_buffer_the_device_takes() -> None:
    room, device = Room(), Device()
    playout = Playout(device.open, room, 16000, 1, _unfailing)
    playout.write(LOUD[:400], ("alba",))
    playout.write(LOUD[:400], ("mary",))
    device.take()  # 640 bytes: all of alba's, and 240 of mary's
    device.take()  # the rest of mary's, then silence
    device.take()  # silence
    assert [voices for *_, voices in room.plays] == [("alba", "mary"), ("mary",), ()]


async def test_the_desk_is_overheard_with_what_was_played_and_where_the_gate_stood(devices: Rig) -> None:
    async with devices.microphone.echoes() as echoes:
        await devices.capture(at=1.0)  # the off level, no press
        devices.key.move("listen", "wake word")
        await devices.capture(at=1.02)  # the wake word's level, where a voice can open a clip
        devices.key.move("start", "shortcut")
        await devices.capture(at=1.04)  # a clip open
        devices.key.go("phone")
        await devices.capture(at=1.06)  # hands at the phone
        heard = [await anext(echoes) for _ in range(4)]
    assert [(each.cancelled, each.desk, each.key) for each in heard] == [
        (Cancelled(CLEANED, PLAYED), True, "up"),
        (Cancelled(CLEANED, PLAYED), True, "listening"),
        (Cancelled(CLEANED, PLAYED), True, "down"),
        (Cancelled(CLEANED, PLAYED), False, "up"),
    ]


async def test_a_chunk_queued_behind_a_cue_is_played_after_the_cue(devices: Rig) -> None:
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    devices.stream.blocking = True
    devices.speaker.cue(OPENED)  # the barge-in's tone, still going to the device
    writing = asyncio.create_task(devices.speaker.write_audio_frame(OutputAudioRawFrame(audio=LOUD, sample_rate=16000, num_channels=1)))
    await asyncio.sleep(0.01)
    devices.stream.blocking = False
    await writing
    assert devices.stream.written == [sound(OPENED, 16000, 1), LOUD]  # the order the room hears them, and the canceller with it


async def test_the_speaker_is_quiet_while_neither_hands_nor_the_user_is_speaking(monkeypatch: pytest.MonkeyPatch, devices: Rig) -> None:
    pushed: list[Frame] = []

    async def passed(_self: object, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append(frame)

    # Pipecat's sender pushes each edge of hands' speech through the transport, downstream and up.
    monkeypatch.setattr(LocalAudioOutputTransport, "push_frame", passed)
    assert devices.speaker.quiet.is_set()
    await devices.speaker.push_frame(BotStartedSpeakingFrame())
    assert not devices.speaker.quiet.is_set()
    await devices.speaker.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await devices.speaker.push_frame(BotStoppedSpeakingFrame())
    assert devices.speaker.quiet.is_set()
    # The user's turn is speech too, from its first clip's opening, whether or not it has cut hands off: quiet waits for
    # both sides, whichever stops last.
    await devices.speaker.push_frame(TurnOpened(clip=Clip(1, "engaged conversation")))
    await devices.speaker.push_frame(BotStartedSpeakingFrame())
    await devices.speaker.push_frame(BotStoppedSpeakingFrame())
    assert not devices.speaker.quiet.is_set()
    await devices.speaker.push_frame(UserStoppedSpeakingFrame())
    assert devices.speaker.quiet.is_set()
    assert len(pushed) == 7  # each passed on


class Device:
    """A speaker device as PortAudio drives a callback stream: it takes a buffer whenever the test says it does."""

    def __init__(self) -> None:
        self.pull: Pull | None = None
        self.log: list[str] = []

    def open(self, pull: Pull) -> "Device":
        self.pull = pull
        return self

    def take(self, frames: int = 320) -> tuple[bytes, int]:
        assert self.pull is not None, "the playout opens its stream as it is made"
        return self.pull(None, frames, {}, 0)

    def start_stream(self) -> None: ...

    def stop_stream(self) -> None:
        self.log.append("stop")

    def close(self) -> None: ...


def _unfailing(error: Exception) -> None:
    raise AssertionError(f"the playout failed: {error}")


def test_the_canceller_is_given_what_the_device_takes_as_it_takes_it_silence_included() -> None:
    """The reference keeps the device's time through a silence, so a reading's opening meets its echo where the last
    reading's did: written ahead instead, it moved against its echo by where the first write fell."""
    room, device = Room(), Device()
    playout = Playout(device.open, room, 16000, 1, _unfailing)
    taken = [device.take()]  # nothing written yet: the device plays silence, and the canceller is given it
    playout.write(LOUD[:400], ())  # a reading's first chunk, ending partway through the device's next buffer
    taken += [device.take(), device.take()]
    assert [audio for audio, _ in taken] == [QUIET, LOUD[:400] + bytes(240), QUIET]
    assert room.plays == [(audio, 16000, 1, ()) for audio, _ in taken]
    # Three 20 ms buffers, two 10 ms frames each; the frames wholly of silence are the ones nothing was written for.
    assert playout.counts() == {"pulled": 6, "unwritten": 4, "underflowed": 0}


def test_a_callback_portaudio_says_came_too_late_is_counted() -> None:
    device = Device()
    playout = Playout(device.open, Room(), 16000, 1, _unfailing)
    assert device.pull is not None
    device.pull(None, 320, {}, pyaudio.paOutputUnderflow)
    device.pull(None, 320, {}, 0)
    assert playout.counts() == {"pulled": 4, "unwritten": 4, "underflowed": 1}


def test_a_write_waits_until_the_write_ahead_has_room_for_all_of_it_and_goes_on_as_the_device_takes_sound() -> None:
    device = Device()
    playout = Playout(device.open, Room(), 16000, 1, _unfailing)
    playout.write(bytes(round(16000 * WRITE_AHEAD_SECS) * 2 - 2), ())  # a write-ahead full but for one sample
    writing = threading.Thread(target=playout.write, args=(LOUD, ()))
    writing.start()
    writing.join(0.05)
    assert writing.is_alive()  # waits, as a blocking stream's write does on a buffer without room for it
    device.take()
    writing.join(1)
    assert not writing.is_alive()


def test_stopping_the_stream_releases_a_waiting_write_and_refuses_it() -> None:
    """What lets a reopen go on when the device is gone and takes nothing more."""
    device = Device()
    playout = Playout(device.open, Room(), 16000, 1, _unfailing)
    playout.write(bytes(round(16000 * WRITE_AHEAD_SECS) * 2), ())
    refused: list[OSError] = []

    def write() -> None:
        try:
            playout.write(LOUD, ())
        except OSError as error:
            refused.append(error)

    writing = threading.Thread(target=write)
    writing.start()
    writing.join(0.05)
    playout.stop_stream()
    writing.join(1)
    assert [str(error) for error in refused] == ["the speaker stream is stopped"]
    assert device.log == ["stop"]


def test_writes_to_no_output_run_no_further_ahead_of_the_sound_than_a_device_lets_them_and_stopping_it_releases_and_refuses_one() -> None:
    """The pipeline's hearing of its own speech is paced by the write, as a device's playout paces it."""
    nothing = NoOutput(SYSTEM, 1000, 1)
    began = time.monotonic()
    nothing.write(bytes(2 * 300), ())  # 300 ms of 16-bit mono at 1 kHz, of which WRITE_AHEAD_SECS is let go ahead
    nothing.write(bytes(2 * 300), ())
    assert time.monotonic() - began >= 0.6 - WRITE_AHEAD_SECS
    refused: list[OSError] = []

    def write() -> None:
        try:
            nothing.write(bytes(2 * 60_000), ())  # a minute of it
        except OSError as error:
            refused.append(error)

    writing = threading.Thread(target=write)
    writing.start()
    writing.join(0.05)
    nothing.stop_stream()
    writing.join(1)
    assert not writing.is_alive()
    assert [str(error) for error in refused] == ["the speaker stream is stopped"]


class Deafened(Room):
    def played(self, audio: bytes, sample_rate: int, channels: int, voices: tuple[str, ...]) -> None:
        raise RuntimeError("the audio processing module refused the frame")


async def test_a_playout_the_canceller_fails_ends_the_pipeline_rather_than_leaving_hands_mute(devices: Rig) -> None:
    devices.speaker._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]
    errors: list[str] = []
    pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_error(error_msg: str, exception: Exception | None = None, fatal: bool = False, category: object = None, force_treat_as_permanent: bool = False) -> None:
        errors.append(f"{error_msg}: {exception}")

    async def push_frame(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        pushed.append((frame, direction))

    setattr(devices.speaker, "push_error", push_error)
    setattr(devices.speaker, "push_frame", push_frame)
    portaudio = FreshPortAudio([])
    devices.speaker.open_stream(cast(PortAudio, portaudio), Deafened())
    pull = cast(Pull, portaudio.opened[-1]["stream_callback"])
    assert pull(None, 320, {}, 0)[1] == pyaudio.paAbort
    await asyncio.sleep(0.01)
    assert errors == ["the speaker stopped: its playout failed: the audio processing module refused the frame"]
    assert [(type(frame), direction) for frame, direction in pushed] == [(EndWorkerFrame, FrameDirection.UPSTREAM)]


def opened(portaudio: FreshPortAudio) -> dict[str, object]:
    """The device index each stream `portaudio` opened was opened on, None for the default."""
    return {"speaker" if settings.get("output") else "microphone": settings.get("output_device_index", settings.get("input_device_index")) for settings in portaudio.opened}


async def test_a_side_on_auto_opens_on_the_default_whatever_else_is_plugged_in(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log)
    portaudio = FreshPortAudio(log, BUILT_IN + DISPLAY)
    setattr(transport, "_portaudio", lambda: portaudio)
    assert await transport.reopen() == Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
    assert opened(portaudio) == {"speaker": None, "microphone": None}


async def test_a_named_device_is_opened_whatever_the_default_and_kept_to_while_the_default_moves(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log, choices=Choices(Named("Studio Display Microphone"), Named("Studio Display Speakers")))
    portaudio = FreshPortAudio(log, BUILT_IN + DISPLAY)
    setattr(transport, "_portaudio", lambda: portaudio)
    there = Hardware(DefaultDevices(1, 1), {"MacBook Pro Microphone": 10, "Studio Display Microphone": 11}.get, {"MacBook Pro Speakers": 20, "Studio Display Speakers": 21}.get)
    setattr(transport, "_hardware", lambda: there)

    assert await transport.reopen() == Devices(input="Studio Display Microphone", output="Studio Display Speakers")
    assert opened(portaudio) == {"speaker": 3, "microphone": 2}  # by where PortAudio lists them
    assert transport.opened_on == Placement(OnNamed("Studio Display Microphone", 11), OnNamed("Studio Display Speakers", 21))
    # The default moving changes nothing of where they are wanted, so the follower reopens nothing.
    setattr(transport, "_hardware", lambda: Hardware(DefaultDevices(5, 5), there.inputs, there.outputs))
    assert transport.wanted() == transport.opened_on


async def test_a_named_device_that_is_missing_falls_back_to_the_default_and_says_so(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log, choices=Choices(Named("Studio Display Microphone"), Named("Studio Display Speakers")))
    portaudio = FreshPortAudio(log, BUILT_IN)
    setattr(transport, "_portaudio", lambda: portaudio)

    devices = await transport.reopen()

    assert devices == Devices(input=Instead("Studio Display Microphone", "MacBook Pro Microphone"), output=Instead("Studio Display Speakers", "MacBook Pro Speakers"))
    assert opened(portaudio) == {"speaker": None, "microphone": None}
    assert not transport.deaf
    assert transport.opened_on == Placement(OnDefault(2), OnDefault(2))  # on the default, followed as it moves, until the display is back
    assert system_text(AudioMoved(devices)) == "Audio moved: listening on MacBook Pro Microphone in place of the missing Studio Display Microphone, speaking on MacBook Pro Speakers in place of the missing Studio Display Speakers."

    # With no default input either, the microphone is on nothing, and says which device it missed.
    setattr(transport, "_portaudio", lambda: DeafPortAudio(log, BUILT_IN[1:]))
    setattr(transport, "_hardware", lambda: Hardware(DefaultDevices(3, 2)))
    assert (await transport.reopen()).input == Missing("Studio Display Microphone")
    assert transport.deaf
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]


async def test_a_side_left_off_opens_no_device_and_the_other_opens_as_chosen(lost: Lost) -> None:
    log: list[str] = []
    transport = lost(log, choices=Choices(Off(), Off()))
    portaudio = FreshPortAudio(log, BUILT_IN + DISPLAY)
    setattr(transport, "_portaudio", lambda: portaudio)

    assert await transport.reopen() == Devices(input=Off(), output=Off())

    assert portaudio.opened == []
    # Turned off by config.toml, the microphone hears nothing and is not lost: the heartbeat is not told of a fault.
    assert transport.deaf and not transport.microphone_lost
    assert transport.opened_on == Placement(Closed(), Closed())
    assert isinstance(transport.input()._in_stream, NoInput)  # pyright: ignore[reportPrivateUsage]
    # What hands says goes nowhere, and the speaker writes it as it writes to a device.
    attached = transport.output().opened
    assert attached is not None and isinstance(attached.stream, NoOutput)
    attached.stream.write(LOUD, ())
    assert attached.stream.counts() == {"pulled": 0, "unwritten": 0, "underflowed": 0}

    # Only the speaker off: the microphone is on the default, and hears.
    transport = lost(log, choices=Choices(Auto(), Off()))
    portaudio = FreshPortAudio(log, BUILT_IN)
    setattr(transport, "_portaudio", lambda: portaudio)
    assert await transport.reopen() == Devices(input="MacBook Pro Microphone", output=Off())
    assert not transport.deaf
    assert opened(portaudio) == {"microphone": None}
