"""Acoustic echo cancellation: the speaker's sound taken back out of what the Mac's microphone hears.

With speakers on, the room carries every reply to the microphone. WebRTC's echo canceller (AEC3, through LiveKit's
binding of its audio processing module) is told what the speaker plays, the far-end or reference signal, and subtracts
the echo of it from every microphone buffer, the near-end signal, estimating the delay and the room's response as it
goes. Measured 2026-10-03 on MacBook Pro speakers and microphone, through hands' own transport: about 27 dB of the
echo removed; a press mid-reply with nobody speaking left no word of the reply in 8 clips of 8, where the raw
microphone made one of it ("Run.", "in the") in every clip; a voice starting 50 ms after the press kept its first word
in 8 of 8, where muting the microphone until the reply's sound had died away lost it in 8 of 8.
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from livekit import rtc

from hands.voice.clips import Reference, Playhead

# The canceller works on 10 ms frames, in both directions.
_FRAMES_PER_SECOND = 100

SAMPLE_BYTES = 2  # 16-bit, as the transport opens both streams

# The most reference held for the microphone to take, in 10 ms frames: a ring buffer's capacity, for a microphone that
# has stopped taking frames, as on a Mac with no input device.
_HELD_FRAMES = 100

# How often the reference held is trimmed back to what the microphone needs, in microphone frames: once a second.
_SLIP_WINDOW = 100

# Reference kept spare at the tightest moment of a window, in frames: one 20 ms buffer of either device, for a callback
# that comes late.
_SPARE_FRAMES = 2


# The stretch the echo path is read over, in 10 ms frames, and how often: 2 s, once a second. Over one 2 s stretch hands
# played in, the reading below was 42.9 to 81.6 at the desk with its speaker on (tests/fixtures/hands_cancelled_and_another.flac)
# and 4.2 to 7.8 in headphones (hands_no_echo_and_another.flac); over 1 s, the speaker's dipped to 9.1.
_PATH_FRAMES = 200
_PATH_HOP = 100

# The reading at least, where hands' voice reached the microphone: the GCC-PHAT of what the microphone captured against
# what the speaker played, its peak over its RMS. Between the two takes' 7.8 and 42.9; with the echo 30 dB down, hands
# alone still read 49 to 73. Fitted on one room, 13 stretches of each.
_ECHO_PATH = 20.0

# Where the echo path is looked for, in Hz, where speech carries; and how far apart in time, in seconds, the reference
# and its echo may be, either way: 68 ms at the desk, and at a reopen the reference can come as late as all of it the
# canceller holds (`_HELD_FRAMES`).
_PATH_BAND = (200.0, 4000.0)
_PATH_LAG = 1.0

# How loud the microphone is, at most, over a stretch whose low reading says no echo reached it: hands alone in
# headphones -60 to -61.5 dBFS, the room alone -63, a person speaking -26 to -34. Under double talk an echo 20 dB down
# reads 7 to 9, as low as none, so a stretch a person spoke in says nothing either way.
_QUIET = -45.0

# How long, in 10 ms frames, hands plays nothing before the echo path is taken to be there again, as it is when the
# canceller is made: 3 s, so a speaker unmuted or turned up between replies is read afresh, not taken for headphones.
_FORGET_FRAMES = 300

# What a canceller counts over its life: microphone frames heard through it; of those, the ones heard with nothing
# playing, matched with silence; frames the speaker's device took, silence included, dropped unheard because the
# microphone had stopped taking them (how much of the device's sound was silence is the speaker's own `unwritten`);
# frames slipped, held through a whole window that never needed them; and frames passed on as captured, with no echo
# path to take anything out of.
COUNTS = ("heard", "unplayed", "dropped", "slipped", "unreached")


def frame_bytes(sample_rate: int, channels: int) -> int:
    """The size of one 10 ms frame, the canceller's unit in both directions."""
    return sample_rate // _FRAMES_PER_SECOND * channels * SAMPLE_BYTES


def seconds(length: int, sample_rate: int, channels: int) -> float:
    """How long `length` bytes of 16-bit sound at `sample_rate` over `channels` play."""
    return length / (sample_rate * channels * SAMPLE_BYTES)


@dataclass(frozen=True)
class Cancelled:
    """A microphone buffer through the canceller: `heard`, with the speaker's echo taken out; and `played`, the reference
    each of its 10 ms frames was matched with, as the canceller was given it, so the two line up frame for frame."""

    heard: bytes
    played: tuple[Reference | None, ...]


class Echo(Protocol):
    """What the transport asks of an echo canceller: told what the speaker plays, it cleans what the microphone hears."""

    def played(self, audio: bytes, sample_rate: int, channels: int, playheads: tuple[Playhead, ...]) -> None: ...
    def heard(self, audio: bytes, sample_rate: int, channels: int) -> Cancelled: ...
    def counts(self) -> dict[str, int]: ...
    def close(self) -> None: ...


def referenced(reference: rtc.AudioFrame, playheads: tuple[Playhead, ...], reached: bool) -> Reference | None:
    """The frame of reference as hands' TTS, kept as given, where it carries TTS audio: a synthesis it is of, and sound;
    None for silence, a cue, or the speaker's own padding. Only a frame of TTS is copied, on the capture thread."""
    match playheads:
        case ():
            return None
        case _:
            audio = bytes(reference.data.cast("B"))
            return None if audio.count(0) == len(audio) else Reference(audio, reference.sample_rate, reference.num_channels, playheads, reached)


def mono(audio: bytes, channels: int) -> np.ndarray:
    """16-bit audio over `channels` as one channel of samples from -1 to 1."""
    return np.frombuffer(audio, np.int16).reshape(-1, channels).mean(axis=1) / 32768


def echo_path(captured: np.ndarray, rate: int, played: np.ndarray) -> float:
    """How far above chance the echo of `played` is in `captured`, at `rate`, the same stretch of time each at its own
    rate: the GCC-PHAT of the two in `_PATH_BAND`, within `_PATH_LAG` either way, its peak over its RMS. The phase
    transform weighs every frequency alike, so the room, which spreads an echo over many lags, still leaves one peak.
    Zero-padded to twice the stretch, so no lag wraps round onto another, each side's bin k is the same frequency at
    either rate, and the two spectra are multiplied bin by bin with no resampling."""
    size = 2 * len(captured)
    heard, said = np.fft.rfft(captured, size), np.fft.rfft(played, 2 * len(played))
    bins = min(len(heard), len(said))
    cross = np.zeros(len(heard), complex)
    cross[:bins] = heard[:bins] * np.conj(said[:bins])
    frequency = np.fft.rfftfreq(size, 1 / rate)
    band = (frequency >= _PATH_BAND[0]) & (frequency <= _PATH_BAND[1]) & (np.abs(cross) > 0)
    gcc = np.fft.irfft(np.where(band, cross / np.where(band, np.abs(cross), 1), 0), size)
    lag = round(_PATH_LAG * rate)
    lags = np.concatenate((gcc[: lag + 1], gcc[size - lag :]))
    spread = float(np.sqrt(np.mean(lags**2)))
    # Nothing in the band on either side, as digital silence captured: no higher above chance than nothing.
    return float(np.abs(lags).max()) / spread if spread else 0.0


class EchoPath:
    """Whether hands' voice reaches the microphone, read off the last `_PATH_FRAMES` of what it captured and what the
    speaker played, every `_PATH_HOP`: a fact of the speaker and the room, not of any clip, so it is read wherever hands
    plays, not only in the stretch a clip holds.

    An echo path is taken to be there until a stretch shows none, the canceller's way. The evidence is asymmetric as the
    costs are: a path taken for none sends hands' own words as a person's, one taken for there costs a person's words.
    So any stretch hands played in that reads `_ECHO_PATH` or more says there is one, and only hands alone says there
    is none: a stretch hands played at least half of reading under it, the microphone under `_QUIET` over it. Any other,
    a person speaking over hands, says nothing. `_FORGET_FRAMES` with hands playing nothing and it is taken to be there
    again. A path that comes while it is read as none, a speaker unmuted mid-reply, is read at the next hop, up to
    `_PATH_HOP` of hands heard as captured. The capture thread alone touches it."""

    def __init__(self) -> None:
        self.reached = True
        self._captured: deque[np.ndarray] = deque(maxlen=_PATH_FRAMES)
        # Each frame the speaker played TTS in; None where it played none, made silence at the TTS's rate when read.
        self._played: deque[np.ndarray | None] = deque(maxlen=_PATH_FRAMES)
        self._frames = 0
        self._unplayed = 0

    def heard(self, captured: np.ndarray, played: np.ndarray | None) -> None:
        """Take one 10 ms frame of the microphone, mono, and what the speaker played of hands' TTS with it, mono, or None
        where it played none."""
        self._captured.append(captured)
        self._played.append(played)
        self._frames += 1
        self._unplayed = 0 if played is not None else self._unplayed + 1
        # [LAW:dataflow-not-control-flow] every hop is read; a stretch hands played none of reads as saying nothing.
        if self._frames % _PATH_HOP == 0 and len(self._captured) == _PATH_FRAMES:
            self.reached = _verdict(self.reached, *self._read())
        self.reached = self.reached or self._unplayed >= _FORGET_FRAMES

    def _read(self) -> tuple[float, bool]:
        """The stretch's reading, nothing above chance where hands played none of it, and whether it was hands alone:
        hands played at least half of it, and the microphone was quiet over it."""
        captured = np.concatenate(self._captured)
        tts = [frame for frame in self._played if frame is not None]
        alone = 2 * len(tts) >= _PATH_FRAMES and float(np.mean(captured**2)) < 10 ** (_QUIET / 10)
        if not tts:
            return 0.0, alone
        silence = np.zeros(len(tts[0]))
        played = np.concatenate([silence if frame is None else frame for frame in self._played])
        return echo_path(captured, len(self._captured[0]) * _FRAMES_PER_SECOND, played), alone


def _verdict(reached: bool, reading: float, alone: bool) -> bool:
    """Whether there is an echo path, given whether there was and a stretch's reading: one at `_ECHO_PATH` says there
    is; one under it over hands alone says there is not; any other leaves it as it was."""
    return reading >= _ECHO_PATH or (reached and not alone)


class EchoCanceller:
    """WebRTC's AEC3, with its reference paced by the microphone's clock.

    AEC3 expects one frame of reference for every frame of microphone, as a device that plays and records at once
    gives them. The speaker's device gives its every frame, silence included, as it takes it
    (`hands.voice.microphone.Playout`), on a clock of its own, so what it plays is held, and each microphone frame takes
    the next frame of it, or silence when there is none.

    How much is held is how far the reference lags its echo, and past the two devices' latencies it reaches AEC3 after
    the echo it would cancel. The speaker starts before the microphone at a reopen, and its clock may run fast against
    the microphone's, and either leaves frames held that no microphone frame needs. Held through a whole window, they
    are surplus, and are slipped: dropped, as a jitter buffer drops samples between two clocks.
    """

    def __init__(self) -> None:
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True)
        # [LAW:single-enforcer] the one reading of whether hands' voice reaches the microphone; each reference says it.
        self._path = EchoPath()
        # The speaker's audio short of a whole frame, carried to its next buffer, with the syntheses in it. The speaker's
        # device callback alone touches them.
        self._unplayed = b""
        self._unplayed_playheads: tuple[Playhead, ...] = ()
        # [LAW:no-shared-mutable-globals] written by the speaker's device callback, taken by the microphone's capture
        # thread, under the one lock; a ring buffer, so the oldest goes first when nothing takes it.
        self._held: deque[tuple[rtc.AudioFrame, tuple[Playhead, ...]]] = deque(maxlen=_HELD_FRAMES)
        self._holding = threading.Lock()
        # The fewest frames left held after a microphone frame took one, over the window so far, and the frames taken
        # in it. The capture thread alone touches them, under the lock.
        self._least = _HELD_FRAMES
        self._taken = 0
        # The reference's format, from the speaker's first buffer; silence is made in it until then.
        self._format = (16000, 1)
        # [LAW:nothing-unseen] the canceller's own decisions, read onto the microphone's event as it is let go of.
        # Each is written by one thread: `heard` and `unplayed` by the capture thread, `dropped` and `slipped` under the lock.
        self._counts: dict[str, int] = dict.fromkeys(COUNTS, 0)

    def played(self, audio: bytes, sample_rate: int, channels: int, playheads: tuple[Playhead, ...]) -> None:
        """Hold sound as the speaker's device takes it, silence included, for the microphone frames that hear its echo,
        each frame with the `playheads` of hands' TTS in the buffer it came from, and the first with those carried into
        it: each synthesis, where in it the buffer began, so a frame is placed in its synthesis to within one buffer."""
        size = frame_bytes(sample_rate, channels)
        pending = self._unplayed + audio
        whole = len(pending) - len(pending) % size
        carried = tuple(dict.fromkeys((*self._unplayed_playheads, *playheads))) if self._unplayed else playheads
        frames = [
            (rtc.AudioFrame(pending[start : start + size], sample_rate, channels, sample_rate // _FRAMES_PER_SECOND), carried if start == 0 else playheads)
            for start in range(0, whole, size)
        ]
        self._unplayed = pending[whole:]
        self._unplayed_playheads = carried if whole == 0 else playheads
        with self._holding:
            self._counts["dropped"] += max(0, len(self._held) + len(frames) - _HELD_FRAMES)
            self._held.extend(frames)
            self._format = (sample_rate, channels)

    def heard(self, audio: bytes, sample_rate: int, channels: int) -> Cancelled:
        """A microphone buffer with the speaker's echo taken out, and what it was matched with; the same length, of whole
        10 ms frames."""
        size = frame_bytes(sample_rate, channels)
        # [LAW:parse-dont-validate] the microphone is opened on 20 ms buffers; any other length is a stream opened wrong.
        if len(audio) % size:
            raise ValueError(f"a microphone buffer of {len(audio)} bytes is not whole 10 ms frames at {sample_rate} Hz")
        cleaned, played = bytearray(), list[Reference | None]()
        for start in range(0, len(audio), size):
            # [LAW:dataflow-not-control-flow] every microphone frame is matched with one frame of reference, and goes
            # through AEC3, which keeps learning the room either way; the echo path says which of the two is heard.
            reference, playheads = self._next_played()
            self._apm.process_reverse_stream(reference)
            captured = audio[start : start + size]
            # The echo path as the frame came: read once a second, it changes at a frame's edge, never within one.
            reached = self._path.reached
            tts = referenced(reference, playheads, reached)
            self._path.heard(mono(captured, channels), None if tts is None else mono(tts.audio, tts.channels))
            played.append(tts)
            frame = rtc.AudioFrame(captured, sample_rate, channels, sample_rate // _FRAMES_PER_SECOND)
            self._apm.process_stream(frame)  # in place
            # With no echo path there is nothing to take out, and AEC3 still ducks a person's voice under hands', 30 to
            # 50 dB on hands_no_echo_and_another.flac: what the microphone captured is heard as it was.
            cleaned += frame.data.cast("B") if reached else captured
            self._counts["unreached"] += not reached
        self._counts["heard"] += len(audio) // size
        return Cancelled(bytes(cleaned), tuple(played))

    def counts(self) -> dict[str, int]:
        """What the canceller has done so far, by the names in COUNTS."""
        return dict(self._counts)

    def close(self) -> None:
        """Let go of the native canceller, once the microphone has stopped capturing.

        Left to the garbage collector, it is let go of after LiveKit's own exit handler has shut its runtime down, and
        LiveKit fails an assertion on every exit saying so.
        """
        self._apm._ffi_handle.dispose()  # pyright: ignore[reportPrivateUsage]  (LiveKit gives the module no close of its own)

    def _next_played(self) -> tuple[rtc.AudioFrame, tuple[Playhead, ...]]:
        with self._holding:
            taken = self._held.popleft() if self._held else None
            self._slip()
            rate, channels = self._format
        match taken:
            case (rtc.AudioFrame(), _):
                return taken
            case None:
                self._counts["unplayed"] += 1
                return rtc.AudioFrame.create(rate, channels, rate // _FRAMES_PER_SECOND), ()

    def _slip(self) -> None:
        # [LAW:dataflow-not-control-flow] every window ends in a slip; with nothing surplus, it slips no frames.
        self._least = min(self._least, len(self._held))
        self._taken += 1
        if self._taken == _SLIP_WINDOW:
            surplus = max(0, self._least - _SPARE_FRAMES)
            for _ in range(surplus):
                self._held.popleft()
            self._counts["slipped"] += surplus
            self._least, self._taken = _HELD_FRAMES, 0
