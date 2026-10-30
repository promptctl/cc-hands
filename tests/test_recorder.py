"""The recorder: each hold kept on disk as hands heard it, beside what hands was playing over it and what the hold was,
written off the event loop once it ends, and a hold that cannot be written failing its own hearing with why."""

import contextlib
import json
import wave
from pathlib import Path

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import RATE, Hold, Sound
from hands.voice.recorder import Recorder


class Unused:
    """A transcriber no test here gives a hold to."""

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        raise AssertionError("nothing here transcribes")


def tone(level: int, frames: int = 1) -> bytes:
    return level.to_bytes(2, "little", signed=True) * 320 * frames


def read(path: Path) -> tuple[int, int, bytes]:
    with wave.open(str(path), "rb") as kept:
        return kept.getnchannels(), kept.getframerate(), kept.readframes(kept.getnframes())


async def kept(recorder: Recorder, hold: Hold, before: Sound, sounds: list[Sound], recorded: list[Entry]) -> WideEvent:
    """The hold told to the recorder as the owner tells it, and its writing run as the owner runs a hearing."""
    owner = HoldOwner(Unused(), (recorder,), recorded.append)
    assert recorder.opened(hold, before) == ()
    assert all(recorder.heard(sound) == () for sound in sounds)
    [write] = recorder.ended(hold, "sent")
    # As the owner's hearing lane runs it: one that fails is said on its event, and the lane hears on.
    with contextlib.suppress(Exception):
        await owner._heard(recorder, hold, write, begun())  # pyright: ignore[reportPrivateUsage]
    [event] = [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hearing"]
    return event


async def test_a_hold_at_the_desk_is_kept_as_heard_with_what_hands_played_over_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    directory = tmp_path / "holds" / "run"
    hold = Hold(7, "engaged conversation")
    event = await kept(Recorder(directory), hold, Sound(tone(1), tone(9), tone(-1)), [Sound(tone(2, 2), tone(9, 2), tone(-2, 2))], recorded)
    assert event.outcome == "ok"
    # What it heard from the second before it opened on, and what the speaker played against each frame, sample for sample.
    assert read(directory / "7.heard.wav") == (1, RATE, tone(1) + tone(2, 2))
    assert read(directory / "7.played.wav") == (1, RATE, tone(-1) + tone(-2, 2))
    assert json.loads((directory / "7.json").read_text()) == {"hold": 7, "opener": "engaged conversation", "fate": "sent", "seconds": 0.06, "played": True}
    # [LAW:nothing-unseen] where it went, how long, and whether what was played went with it.
    assert (event.facts["kept"], event.facts["seconds"], event.facts["played"]) == (directory / "7.heard.wav", 0.06, True)


async def test_a_hold_at_the_phone_is_kept_with_nothing_said_to_have_played_over_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    event = await kept(Recorder(tmp_path), Hold(1, "phone button"), Sound(b"", b"", None), [Sound(tone(3), tone(3), None)], recorded)
    assert event.outcome == "ok" and event.facts["played"] is False
    assert read(tmp_path / "1.heard.wav") == (1, RATE, tone(3))
    assert not (tmp_path / "1.played.wav").exists()


async def test_a_hold_that_cannot_be_written_fails_its_hearing_saying_why(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("in the way")
    recorded: list[Entry] = []
    recorder = Recorder(blocked / "holds")
    event = await kept(recorder, Hold(1, "held key"), Sound(b"", b"", b""), [Sound(tone(3), tone(3), tone(0))], recorded)
    assert (event.outcome, event.error) == ("failed", f"NotADirectoryError: [Errno 20] Not a directory: '{blocked / 'holds'}'")
