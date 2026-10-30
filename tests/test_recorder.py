"""The recorder: each hold kept on disk as hands heard it, beside what hands was playing over it, in the hold's own
format, and what the hold was, written off the event loop once it ends; a hold that cannot be written fails its own
hearing with why."""

import contextlib
import json
import wave
from pathlib import Path

import numpy as np

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import RATE, Hold, Reference, Sound
from hands.voice.recorder import Recorder, reformatted


class Unused:
    """A transcriber no test here gives a hold to."""

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        raise AssertionError("nothing here transcribes")


def tone(level: int, frames: int = 1) -> bytes:
    """`frames` of 10 ms every sample of which is `level`."""
    return level.to_bytes(2, "little", signed=True) * 160 * frames


def read(path: Path) -> tuple[int, int, bytes]:
    with wave.open(str(path), "rb") as kept:
        return kept.getnchannels(), kept.getframerate(), kept.readframes(kept.getnframes())


async def kept(recorder: Recorder, hold: Hold, sound: Sound, recorded: list[Entry]) -> WideEvent:
    """The hold told to the recorder as the owner tells it, whole as it ends, and its writing run as the owner runs a
    hearing."""
    owner = HoldOwner(Unused(), (recorder,), recorded.append)
    assert recorder.opened(hold, Sound(b"", b"", ())) == () and recorder.heard(sound) == ()
    [write] = recorder.ended(hold, "sent", sound)
    # As the owner's hearing lane runs it: one that fails is said on its event, and the lane hears on.
    with contextlib.suppress(Exception):
        await owner._heard(recorder, hold, write, begun())  # pyright: ignore[reportPrivateUsage]
    [event] = [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hearing"]
    return event


async def test_a_hold_at_the_desk_is_kept_as_heard_with_what_hands_played_over_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    directory = tmp_path / "holds" / "run"
    hold = Hold(7, "engaged conversation")
    played = (Reference(tone(-1), RATE, 1), Reference(tone(-2), RATE, 1), Reference(tone(-3), RATE, 1))
    event = await kept(Recorder(directory), hold, Sound(tone(1) + tone(2, 2), tone(9, 3), played), recorded)
    assert event.outcome == "ok"
    # What it heard, and what the speaker played against each 10 ms of it, sample for sample.
    assert read(directory / "7.heard.wav") == (1, RATE, tone(1) + tone(2, 2))
    assert read(directory / "7.played.wav") == (1, RATE, tone(-1) + tone(-2) + tone(-3))
    assert json.loads((directory / "7.json").read_text()) == {"hold": 7, "opener": "engaged conversation", "fate": "sent", "seconds": 0.03, "played": True}
    # [LAW:nothing-unseen] where it went, how long, and whether what was played went with it.
    assert (event.facts["kept"], event.facts["seconds"], event.facts["played"]) == (directory / "7.heard.wav", 0.03, True)


def test_what_the_speaker_played_is_put_in_the_holds_format_mono_at_its_rate() -> None:
    played_rate = 24_000
    wave_ = (8000 * np.sin(2 * np.pi * 440 * np.arange(played_rate // 100) / played_rate)).astype(np.int16)
    stereo = np.repeat(wave_, 2).tobytes()
    held = np.frombuffer(reformatted(Reference(stereo, played_rate, 2)), np.int16)
    # The same 440 Hz tone, at the hold's rate: each sample where the tone is at that instant, give or take one step.
    expected = 8000 * np.sin(2 * np.pi * 440 * np.arange(RATE // 100) / RATE)
    assert len(held) == RATE // 100 and np.max(np.abs(held - expected)) < 8000 * 2 * np.pi * 440 / played_rate


async def test_a_hold_at_the_phone_is_kept_with_nothing_said_to_have_played_over_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    event = await kept(Recorder(tmp_path), Hold(1, "phone button"), Sound(tone(3), tone(3), None), recorded)
    assert event.outcome == "ok" and event.facts["played"] is False
    assert read(tmp_path / "1.heard.wav") == (1, RATE, tone(3))
    assert not (tmp_path / "1.played.wav").exists()


async def test_a_hold_that_cannot_be_written_fails_its_hearing_saying_why(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("in the way")
    recorded: list[Entry] = []
    event = await kept(Recorder(blocked / "holds"), Hold(1, "held key"), Sound(tone(3), tone(3), ()), recorded)
    assert (event.outcome, event.error) == ("failed", f"NotADirectoryError: [Errno 20] Not a directory: '{blocked / 'holds'}'")
