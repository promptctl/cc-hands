"""The recorder: each clip kept on disk as hands heard it, beside what hands was playing over it and in which voices, in
the clip's own format, in a directory of the run's own, written off the event loop once it ends; the store kept under its
cap by whole runs, oldest first; and anything that cannot be written failing its own unit with why."""

import contextlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from hands.virtualclock import VirtualClock
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice.clipowner import ClipOwner
from hands.voice.clips import RATE, Clip, Reference, Sound, Playhead, Synthesis
from hands.voice import recorder as recorders
from hands.voice.recorder import Recorder, track


class Unused:
    """A transcriber no test here gives a clip to."""

    async def transcribed(self, clip: Clip, sound: Sound) -> str | None:
        raise AssertionError("nothing here transcribes")


# Two sentences of hands', in two voices: the first from its start, the second 1.5 s into it.
ALBA = Playhead(Synthesis("alba", "Hands here.", begun(VirtualClock())), 0.0)
MARY = Playhead(Synthesis("mary", "Turn ended.", begun(VirtualClock())), 1.5)


def tone(level: int, frames: int = 1) -> bytes:
    """`frames` of 10 ms every sample of which is `level`."""
    return level.to_bytes(2, "little", signed=True) * 160 * frames


def read(path: Path) -> tuple[int, int, bytes]:
    with wave.open(str(path), "rb") as kept:
        return kept.getnchannels(), kept.getframerate(), kept.readframes(kept.getnframes())


def events(recorded: list[Entry], name: str) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == name]


async def kept(clock: VirtualClock, recorder: Recorder, clip: Clip, sound: Sound, recorded: list[Entry]) -> WideEvent:
    """The clip told to the recorder as the owner tells it, whole as it ends, and its writing run as the owner runs a
    hearing."""
    owner = ClipOwner(clock, Unused(), (recorder,), recorded.append)
    assert recorder.opened(clip, Sound(b"", b"", ())) == () and recorder.heard(sound) == ()
    [write] = recorder.ended(clip, "sent", sound)
    # As the owner's hearing lane runs it: one that fails is said on its event, and the lane hears on.
    with contextlib.suppress(Exception):
        await owner._heard(recorder, clip, write, begun(clock))  # pyright: ignore[reportPrivateUsage]
    return events(recorded, "hearing")[-1]


async def test_a_clip_at_the_desk_is_kept_as_heard_with_what_hands_played_over_it_and_in_which_voices(tmp_path: Path) -> None:
    clock = VirtualClock()
    recorded: list[Entry] = []
    directory = tmp_path / "clips" / "run"
    clip = Clip(7, "engaged conversation")
    played = (Reference(tone(-1), RATE, 1, (ALBA,), True), Reference(tone(-2), RATE, 1, (ALBA, MARY), False), None)
    event = await kept(clock, Recorder(clock, tmp_path / "clips", "run", recorded.append), clip, Sound(tone(1) + tone(2, 2), tone(9, 3), played), recorded)
    assert event.outcome == "ok"
    # What it heard, and what the speaker played against each 10 ms of it, sample for sample: silence where no TTS.
    assert read(directory / "7.heard.wav") == (1, RATE, tone(1) + tone(2, 2))
    assert read(directory / "7.played.wav")[:2] == (1, RATE)
    assert len(read(directory / "7.played.wav")[2]) == len(tone(1, 3)) and read(directory / "7.played.wav")[2][-320:] == tone(0)
    assert json.loads((directory / "7.json").read_text()) == {
        "clip": 7, "opener": "engaged conversation", "fate": "sent", "seconds": 0.03, "played": True, "voices": ["alba"], "unreached": 0.01,
        # Each sentence hands said over it, from the 10 ms it began playing in, as far into it as it was, to the end of
        # the last.
        "said": [
            {"voice": "alba", "text": "Hands here.", "start": 0.0, "into": 0.0, "stop": 0.02},
            {"voice": "mary", "text": "Turn ended.", "start": 0.01, "into": 1.5, "stop": 0.02},
        ],
    }
    # [LAW:nothing-unseen] where it went, how long, whether what was played went with it, and the voices of hands' that
    # may be in it: not mary's, which reached nothing.
    assert (event.facts["kept"], event.facts["seconds"], event.facts["played"], event.facts["voices"]) == (directory / "7.heard.wav", 0.03, True, ("alba",))


def test_what_the_speaker_played_is_put_in_the_clips_format_a_run_at_a_time_slot_for_slot() -> None:
    played_rate = 24_000
    # Half a second of a 440 Hz tone, in stereo at the speaker's rate, in 10 ms references, then 10 ms of no TTS.
    wave_ = (8000 * np.sin(2 * np.pi * 440 * np.arange(played_rate // 2) / played_rate)).astype(np.int16)
    stereo = np.repeat(wave_, 2).tobytes()
    size = 4 * played_rate // 100
    references = tuple(Reference(stereo[at : at + size], played_rate, 2, (ALBA,), True) for at in range(0, len(stereo), size))
    played = np.frombuffer(track((*references, None)), np.int16).astype(np.float64)
    # One slot for each reference, the last silent; and the same tone at the clip's rate, past the resampler's edges.
    assert len(played) == (len(references) + 1) * RATE // 100 and not played[-RATE // 100 :].any()
    expected = 8000 * np.sin(2 * np.pi * 440 * np.arange(len(played)) / RATE)
    middle = slice(RATE // 20, RATE // 2 - RATE // 20)
    assert np.max(np.abs(played[middle] - expected[middle])) < 100


async def test_a_clip_at_the_phone_is_kept_with_nothing_said_to_have_played_over_it(tmp_path: Path) -> None:
    clock = VirtualClock()
    recorded: list[Entry] = []
    event = await kept(clock, Recorder(clock, tmp_path, "run", recorded.append), Clip(1, "phone button"), Sound(tone(3), tone(3), None), recorded)
    assert event.outcome == "ok" and event.facts["played"] is False
    assert read(tmp_path / "run" / "1.heard.wav") == (1, RATE, tone(3))
    assert not (tmp_path / "run" / "1.played.wav").exists()


async def test_two_runs_named_alike_keep_their_clips_apart(tmp_path: Path) -> None:
    clock = VirtualClock()
    recorded: list[Entry] = []
    for level in (1, 2):
        await kept(clock, Recorder(clock, tmp_path, "2026-10-30T10-00-00Z", recorded.append), Clip(1, "shortcut"), Sound(tone(level), tone(level), ()), recorded)
    assert read(tmp_path / "2026-10-30T10-00-00Z" / "1.heard.wav")[2] == tone(1)
    assert read(tmp_path / "2026-10-30T10-00-00Z-2" / "1.heard.wav")[2] == tone(2)


async def test_a_clip_that_cannot_be_written_fails_its_hearing_saying_why(tmp_path: Path) -> None:
    clock = VirtualClock()
    blocked = tmp_path / "file"
    blocked.write_text("in the way")
    recorded: list[Entry] = []
    event = await kept(clock, Recorder(clock, blocked / "clips", "run", recorded.append), Clip(1, "shortcut"), Sound(tone(3), tone(3), ()), recorded)
    assert (event.outcome, event.error) == ("failed", f"NotADirectoryError: [Errno 20] Not a directory: '{blocked / 'clips'}'")


def older(root: Path, name: str, size: int) -> None:
    """A run before this one, of `size` bytes of clips."""
    (root / name).mkdir(parents=True)
    (root / name / "1.heard.wav").write_bytes(bytes(size))


async def test_a_store_taken_over_its_cap_deletes_whole_runs_oldest_first_and_never_the_runs_own(tmp_path: Path) -> None:
    clock = VirtualClock()
    recorded: list[Entry] = []
    older(tmp_path, "2026-10-01T00-00-00Z", 10_000)
    older(tmp_path, "2026-10-02T00-00-00Z", 10_000)
    # This run's clip is about 3.3 KB: with both older runs the store is over 21 KB, under 14 KB without the oldest.
    recorder = Recorder(clock, tmp_path, "2026-10-03T00-00-00Z", recorded.append, cap=14_000)
    await kept(clock, recorder, Clip(1, "shortcut"), Sound(tone(1, 10), tone(1, 10), ()), recorded)
    [pruned] = events(recorded, "clips.pruned")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["2026-10-02T00-00-00Z", "2026-10-03T00-00-00Z"]
    kept_bytes = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert (pruned.outcome, pruned.facts["runs"], pruned.facts["freed"], pruned.facts["kept"]) == ("ok", ("2026-10-01T00-00-00Z",), 10_000, kept_bytes)
    # A cap this run alone is over deletes every other run, and keeps this one whole.
    recorder = Recorder(clock, tmp_path, "2026-10-04T00-00-00Z", recorded.append, cap=100)
    await kept(clock, recorder, Clip(1, "shortcut"), Sound(tone(1, 10), tone(1, 10), ()), recorded)
    assert [path.name for path in tmp_path.iterdir()] == ["2026-10-04T00-00-00Z"]
    assert (tmp_path / "2026-10-04T00-00-00Z" / "1.heard.wav").exists()


async def test_a_store_under_its_cap_deletes_nothing(tmp_path: Path) -> None:
    clock = VirtualClock()
    recorded: list[Entry] = []
    older(tmp_path, "2026-10-01T00-00-00Z", 10_000)
    await kept(clock, Recorder(clock, tmp_path, "2026-10-03T00-00-00Z", recorded.append, cap=14_000), Clip(1, "shortcut"), Sound(tone(1), tone(1), ()), recorded)
    assert events(recorded, "clips.pruned") == []
    assert len(list(tmp_path.iterdir())) == 2


async def test_a_run_alone_over_the_cap_prunes_nothing_and_never_walks_the_store_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = VirtualClock()
    recorded: list[Entry] = []
    walked: list[Path] = []
    measured = recorders.size

    def size(directory: Path) -> int:
        walked.append(directory)
        return measured(directory)

    monkeypatch.setattr(recorders, "size", size)
    recorder = Recorder(clock, tmp_path, "run", recorded.append, cap=100)
    for number in (1, 2, 3):
        await kept(clock, recorder, Clip(number, "shortcut"), Sound(tone(1, 10), tone(1, 10), ()), recorded)
    assert events(recorded, "clips.pruned") == [] and walked == []
    assert len(list((tmp_path / "run").glob("*.heard.wav"))) == 3
