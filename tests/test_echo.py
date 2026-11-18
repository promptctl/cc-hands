"""The echo canceller, run for real on DTLN-aec against a room made of arithmetic."""

import numpy as np
import pytest

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.clips import RATE, SLOT, Reference, Playhead, Sound, Synthesis
from hands.voice.dtln import Dtln
from hands.voice.echo import Cancelled, EchoCanceller, EchoPath, Reading, echo_path
from echoes import CANCELLED, NO_ECHO, OPENING, cut_in, replayed, take

HEARD_RATE, PLAYED_RATE = 16000, 24000

# Two sentences of hands', in two voices, as the speaker played some of each.
ALBA = Playhead(Synthesis("alba", "Hands here.", begun(VirtualClock())), 0.0)
MARY = Playhead(Synthesis("mary", "Turn ended.", begun(VirtualClock())), 0.5)


def _power_db(audio: bytes) -> float:
    samples = np.frombuffer(audio, np.int16).astype(np.float64)
    return float(10 * np.log10(np.mean(samples**2) + 1e-9))


def _noise(seconds: float, rate: int, seed: int) -> np.ndarray:
    """Noise held below about 4 kHz, as speech through a speaker is: none of it folds over on the way to 16 kHz."""
    white = np.random.default_rng(seed).normal(0, 6000, int(seconds * rate))
    return np.convolve(white, np.ones(6) / 6, mode="same").clip(-32768, 32767).astype(np.int16)


def test_the_echo_of_what_the_speaker_played_is_taken_out_of_what_the_microphone_hears() -> None:
    canceller = EchoCanceller()
    played = _noise(6.0, PLAYED_RATE, seed=1)
    # The room: the speaker's sound 120 ms later at half strength, at the microphone's rate.
    at_microphone = np.interp(np.arange(0, len(played), PLAYED_RATE / HEARD_RATE), np.arange(len(played)), played) * 0.5
    echo = np.concatenate([np.zeros(int(0.12 * HEARD_RATE)), at_microphone])[: len(at_microphone)].astype(np.int16)
    heard_raw, heard_clean = b"", b""
    # 20 ms buffers on both sides, as the transport writes and captures; the speaker's chunks straddle its frames,
    # 3 samples over and then 3 under, as a cue of any length leaves them.
    played_bytes, chunk = played.tobytes(), 2 * int(0.02 * PLAYED_RATE)
    for index in range(len(echo) // 320):
        start, end = index * chunk + (6 if index % 2 else 0), (index + 1) * chunk + (0 if index % 2 else 6)
        # hands' TTS, whose echo the path is read for: the canceller's output is heard once it is read there.
        canceller.played(played_bytes[start:end], PLAYED_RATE, 1, (ALBA,))
        buffer = echo[index * 320 : (index + 1) * 320].tobytes()
        cleaned = canceller.heard(buffer, HEARD_RATE, 1).heard
        assert len(cleaned) == len(buffer)
        if index >= 100:  # after two seconds to learn the room
            heard_raw, heard_clean = heard_raw + buffer, heard_clean + cleaned
    assert _power_db(heard_raw) - _power_db(heard_clean) > 15  # nothing taken out is 0


def test_a_voice_with_nothing_playing_is_heard_as_it_was_said() -> None:
    canceller = EchoCanceller()
    voice = _noise(2.0, HEARD_RATE, seed=2).tobytes()
    cleaned = b"".join(canceller.heard(voice[start : start + 640], HEARD_RATE, 1).heard for start in range(0, len(voice), 640))
    assert abs(_power_db(cleaned[16000:]) - _power_db(voice[16000:])) < 3


def test_a_voice_after_the_speaker_stops_being_written_to_is_heard_as_it_was_said() -> None:
    """The pipeline writes nothing once a reply is interrupted; the canceller's reference goes on, as silence."""
    canceller = EchoCanceller()
    played = _noise(3.0, HEARD_RATE, seed=1)
    for index in range(len(played) // 320):  # the reply, and its echo straight back
        buffer = played[index * 320 : (index + 1) * 320].tobytes()
        canceller.played(buffer, HEARD_RATE, 1, ())
        canceller.heard(buffer, HEARD_RATE, 1)
    voice = _noise(2.0, HEARD_RATE, seed=2).tobytes()  # then the user, with nothing more written
    cleaned = b"".join(canceller.heard(voice[start : start + 640], HEARD_RATE, 1).heard for start in range(0, len(voice), 640))
    assert abs(_power_db(cleaned[16000:]) - _power_db(voice[16000:])) < 3


def _delayed(captured: bytes) -> bytes:
    """What the canceller hears of `captured` where it takes nothing out: the same sound, `Dtln.LATENCY` samples later."""
    return (bytes(Dtln.LATENCY * 2) + captured)[: len(captured)]


def test_a_microphone_buffer_of_part_of_a_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="not whole 10 ms frames"):
        EchoCanceller().heard(bytes(330), HEARD_RATE, 1)


def test_a_stereo_microphone_is_heard_in_frames_of_both_channels() -> None:
    canceller = EchoCanceller()
    voice = np.repeat(_noise(1.0, HEARD_RATE, seed=3), 2).tobytes()  # each sample on both channels
    cleaned = b"".join(canceller.heard(voice[start : start + 1280], HEARD_RATE, 2).heard for start in range(0, len(voice), 1280))
    assert len(cleaned) == len(voice)
    assert canceller.counts()["heard"] == 100  # one second of 10 ms frames, not two
    assert abs(_power_db(cleaned[32000:]) - _power_db(voice[32000:])) < 3


def test_the_canceller_counts_frames_heard_with_nothing_playing_and_sound_it_had_to_drop() -> None:
    canceller = EchoCanceller()
    canceller.heard(bytes(640), HEARD_RATE, 1)  # two frames, nothing played
    canceller.played(bytes(320 * 102), HEARD_RATE, 1, ())  # 102 frames played to a microphone that takes none
    canceller.heard(bytes(320), HEARD_RATE, 1)
    assert canceller.counts() == {"heard": 3, "unplayed": 2, "dropped": 2, "slipped": 0, "unreached": 0, "unread": 0}


def test_reference_held_through_a_whole_second_that_never_needed_it_is_slipped_but_for_one_buffer() -> None:
    """As a speaker started before the microphone leaves it: 30 frames ahead, then one played for every one heard."""
    canceller = EchoCanceller()
    canceller.played(bytes(320 * 30), HEARD_RATE, 1, ())
    for _ in range(50):  # one second of 20 ms buffers
        canceller.played(bytes(640), HEARD_RATE, 1, ())
        canceller.heard(bytes(640), HEARD_RATE, 1)
    assert canceller.counts() == {"heard": 100, "unplayed": 0, "dropped": 0, "slipped": 28, "unreached": 0, "unread": 0}
    for _ in range(50):  # the next second holds only the spare buffer, and slips nothing more
        canceller.played(bytes(640), HEARD_RATE, 1, ())
        canceller.heard(bytes(640), HEARD_RATE, 1)
    assert canceller.counts() == {"heard": 200, "unplayed": 0, "dropped": 0, "slipped": 28, "unreached": 0, "unread": 0}


def test_each_microphone_buffer_comes_with_what_the_speaker_played_against_each_of_its_frames_as_it_was_played() -> None:
    """What hands keeps of what it said over a clip: the reference frame each 10 ms of the microphone was matched with,
    in the speaker's own format, untouched on the capture thread, where it was hands' TTS; None for anything else."""
    canceller = EchoCanceller()
    tone = (8000 * np.sin(2 * np.pi * 440 * np.arange(int(0.02 * PLAYED_RATE)) / PLAYED_RATE)).astype(np.int16).tobytes()
    canceller.played(tone, PLAYED_RATE, 1, (ALBA,))
    cancelled = canceller.heard(bytes(640), HEARD_RATE, 1)
    half = len(tone) // 2
    assert cancelled.played == (Reference(tone[:half], PLAYED_RATE, 1, (ALBA,), True), Reference(tone[half:], PLAYED_RATE, 1, (ALBA,), True))
    # Nothing more played: no TTS. Nor is a cue, nor silence the TTS's voice is on: none of it is hands speaking.
    assert canceller.heard(bytes(320), HEARD_RATE, 1).played == (None,)
    canceller.played(tone[:half], PLAYED_RATE, 1, ())
    canceller.played(bytes(half), PLAYED_RATE, 1, (ALBA,))
    assert canceller.heard(bytes(640), HEARD_RATE, 1).played == (None, None)


def test_a_frame_made_of_two_buffers_carries_the_syntheses_of_both() -> None:
    canceller = EchoCanceller()
    canceller.played(b"\x01\x00" * 150, HEARD_RATE, 1, (ALBA,))  # short of a frame: carried
    canceller.played(b"\x01\x00" * 170, HEARD_RATE, 1, (MARY,))  # completes it, and the next
    played = canceller.heard(bytes(640), HEARD_RATE, 1).played
    assert [None if reference is None else reference.playheads for reference in played] == [(ALBA, MARY), (MARY,)]
    assert [None if reference is None else reference.voices for reference in played] == [("alba", "mary"), ("mary",)]


def _room(played: np.ndarray, gain: float) -> np.ndarray:
    """What of `played`, at PLAYED_RATE, the microphone hears at HEARD_RATE: 68 ms later, spread over the room's taps."""
    at_microphone = np.interp(np.arange(0, len(played), PLAYED_RATE / HEARD_RATE), np.arange(len(played)), played)
    taps = np.zeros(800)
    taps[[0, 90, 230, 500]] = [1.0, 0.6, -0.4, 0.25]
    room = np.convolve(at_microphone, taps)[: len(at_microphone)]
    return np.concatenate([np.zeros(int(0.068 * HEARD_RATE)), room])[: len(at_microphone)] * gain


def _dbfs(level: float, seconds: float, seed: int) -> np.ndarray:
    """Sound at the microphone that is no echo of hands', at `level` dBFS: the room, or a person."""
    noise = _noise(seconds, HEARD_RATE, seed).astype(np.float64)
    return noise / np.sqrt(np.mean(noise**2)) * 32768 * 10 ** (level / 20)


def _path_over(path: EchoPath, captured: np.ndarray, played: np.ndarray | None) -> list[Reading]:
    """Each 10 ms of `captured` at HEARD_RATE and `played` at PLAYED_RATE, None for no TTS, through `path`, and what it
    read of the echo path after each second. A frame of `played` that is all silence is no TTS, as `referenced` has it."""
    after: list[Reading] = []
    for frame in range(len(captured) // 160):
        tts = None if played is None else played[frame * 240 : (frame + 1) * 240] / 32768
        path.heard(captured[frame * 160 : (frame + 1) * 160] / 32768, None if tts is None or not tts.any() else tts)
        if frame % 100 == 99:
            after.append(path.reading)
    return after


def test_an_echo_of_what_hands_played_reads_as_one_at_another_rate_with_no_resampling() -> None:
    played = _noise(2.0, PLAYED_RATE, seed=3).astype(np.float64)
    assert echo_path(_room(played, 0.1) / 32768, HEARD_RATE, played / 32768).score > 40
    assert echo_path(_dbfs(-50, 2.0, seed=4) / 32768, HEARD_RATE, played / 32768).score < 10


def test_hands_heard_through_the_room_is_an_echo_path_even_quiet_and_a_person_over_it_leaves_it_unread() -> None:
    played = _noise(6.0, PLAYED_RATE, seed=3).astype(np.float64)
    echo = _room(played, 10 ** (-40 / 20))  # the speaker turned well down
    assert _path_over(EchoPath(), echo, played) == ["there"] * 6
    # A person 25 dB over it hides it (a reading of 7, as low as none): hands' voice is taken to be maybe in what is
    # heard, never none.
    assert _path_over(EchoPath(), echo + _dbfs(-30, 6.0, seed=5), played) == ["unread"] * 6


def test_hands_heard_by_no_microphone_is_no_echo_path_once_two_seconds_of_it_are_read() -> None:
    # In headphones: the microphone hears the room alone while hands plays. A second of it is too short to say so.
    played = _noise(4.0, PLAYED_RATE, seed=3).astype(np.float64)
    assert _path_over(EchoPath(), _dbfs(-62, 4.0, seed=6), played) == ["unread", "none", "none", "none"]


def test_a_person_speaking_over_hands_in_headphones_says_nothing_of_the_echo_path() -> None:
    # Under double talk an echo well down reads as low as none: a loud stretch with a low reading leaves the path as it was.
    played = _noise(4.0, PLAYED_RATE, seed=3).astype(np.float64)
    assert _path_over(EchoPath(), _dbfs(-30, 4.0, seed=7), played) == ["unread"] * 4
    path = EchoPath()
    _path_over(path, _dbfs(-62, 3.0, seed=6), played[: 3 * PLAYED_RATE])
    assert _path_over(path, _dbfs(-30, 4.0, seed=7), played) == ["none"] * 4


def test_the_echo_path_is_read_again_when_hands_is_heard_through_the_room_again() -> None:
    played = _noise(6.0, PLAYED_RATE, seed=3).astype(np.float64)
    path = EchoPath()
    _path_over(path, _dbfs(-62, 3.0, seed=6), played[: 3 * PLAYED_RATE])
    assert path.reading == "none"
    # The headphones come off mid-reply: the first stretch the room carries hands in says so.
    assert _path_over(path, _room(played, 0.3), played) == ["there"] * 6


def test_short_replies_through_the_room_say_there_is_an_echo_path_though_too_short_to_say_there_is_none() -> None:
    played = _noise(6.0, PLAYED_RATE, seed=3).astype(np.float64)
    path = EchoPath()
    _path_over(path, _dbfs(-62, 3.0, seed=6), played[: 3 * PLAYED_RATE])
    assert path.reading == "none"
    # The speaker unmuted, and hands says the last half second of each two, never three seconds apart: no stretch is
    # half hands, but each carries it through the room.
    short = np.where(np.arange(len(played)) % (2 * PLAYED_RATE) >= 3 * PLAYED_RATE // 2, played, 0)
    assert _path_over(path, _room(short, 0.3), short) == ["none"] + ["there"] * 5


def test_the_echo_path_is_unread_again_after_hands_plays_nothing_for_three_seconds() -> None:
    played = _noise(3.0, PLAYED_RATE, seed=3).astype(np.float64)
    path = EchoPath()
    _path_over(path, _dbfs(-62, 3.0, seed=6), played)
    assert _path_over(path, _dbfs(-62, 4.0, seed=6), None) == ["none", "none", "unread", "unread"]


def test_with_no_echo_path_read_the_microphone_is_heard_as_it_captured_and_each_reference_says_whether_it_reached() -> None:
    canceller = EchoCanceller()
    played = _noise(4.0, PLAYED_RATE, seed=3)
    room = _dbfs(-62, 4.0, seed=6).astype(np.int16)
    voice = (_dbfs(-30, 4.0, seed=8)).astype(np.int16)  # a person, over hands, in headphones
    heard: list[tuple[bytes, Cancelled]] = []
    for index in range(len(room) // 320):
        canceller.played(played[index * 480 : (index + 1) * 480].tobytes(), PLAYED_RATE, 1, (ALBA,))
        # Hands alone for 3 s, then the person speaks.
        captured = (room if index < 150 else voice)[index * 320 : (index + 1) * 320].tobytes()
        heard.append((captured, canceller.heard(captured, HEARD_RATE, 1)))
    reached = [reference.reached for _, cancelled in heard for reference in cancelled.played if reference is not None]
    # Unread, hands' voice may be in the microphone; read none, a whole stretch in, it is not.
    assert reached[:200] == [True] * 200 and not any(reached[200:])
    # Unread or read none, the person is heard as the microphone captured them, not ducked under hands.
    assert b"".join(cancelled.heard for _, cancelled in heard) == _delayed(b"".join(captured for captured, _ in heard))
    assert (canceller.counts()["unread"], canceller.counts()["unreached"]) == (200, 200)


def _unreached(sound: Sound) -> int:
    return sum(reference is not None and not reference.reached for reference in sound.played or ())


def test_hands_through_the_desks_speaker_never_reads_as_reaching_nothing_and_is_cancelled_within_a_reply_s_opening() -> None:
    sound = replayed(*take(CANCELLED))
    assert _unreached(sound) == 0
    # Unread until a stretch says there is an echo path: hands' first sentence, from 1.24 s, is heard uncancelled for
    # 0.26 s.
    assert _uncancelled_from(sound, 1.24, 4.0) <= 0.3


def test_a_reply_at_the_desk_after_a_pause_is_cancelled_within_its_opening() -> None:
    # Unread again over the 4 s pause, the path is read there 0.21 s into the next reply, at 9.54 s, a person over it.
    sound = replayed(*take(OPENING))
    assert _uncancelled_from(sound, cut_in(OPENING), len(sound.heard) / 2 / RATE) <= 0.3


def _uncancelled_from(sound: Sound, onset: float, end: float) -> float:
    """How long `sound` is heard as the microphone captured it from `onset`, in seconds, where from then to `end` it is
    heard through the canceller."""
    buffer, per_second = 4 * SLOT, RATE // (2 * SLOT)
    delayed = _delayed(sound.captured)
    cancelled = [sound.heard[at : at + buffer] != delayed[at : at + buffer] for at in range(0, len(sound.heard), buffer)]
    after = cancelled[round(onset * per_second) : round(end * per_second)]
    opened = after.index(True)
    assert all(after[opened:])
    return opened / per_second


def test_hands_echo_20_db_down_under_a_person_never_reads_as_reaching_nothing() -> None:
    # Under double talk an echo well down reads as low as none: the person is loud, so it says nothing of the path.
    echo, reference, playheads = take(CANCELLED)
    person, _, _ = take(NO_ECHO)
    length = min(len(echo), len(person))
    mixed = (echo[:length].astype(np.float64) / 10 + person[:length]).clip(-32768, 32767).astype(np.int16)
    assert _unreached(replayed(mixed, reference, playheads)) == 0


def test_hands_in_headphones_is_heard_as_captured_and_reaches_nothing_once_a_stretch_of_it_is_read() -> None:
    sound = replayed(*take(NO_ECHO))
    reached = [reference.reached for reference in sound.played or () if reference is not None]
    # The microphone is heard as captured throughout, the person's lines over hands with it; hands' voice may be in it
    # until a whole stretch shows no echo path, 1.01 s of hands in, and is not from then on.
    assert sound.heard == _delayed(sound.captured)
    assert all(reached[:100]) and not any(reached[102:])
