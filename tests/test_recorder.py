"""The recorder: each hold kept on disk as hands heard it, beside what hands was playing over it and in which voices, in
the hold's own format, in a directory of the run's own, written off the event loop once it ends; hands' own echo kept
and identified as spans while it speaks with no hold open; the store kept under its cap by whole runs, oldest first;
and anything that cannot be written failing its own unit with why."""

import contextlib
import json
import wave
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.voice.echo import Cancelled
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import RATE, Hold, Reference, Sound
from hands.voice import recorder as recorders
from hands.voice.recorder import Recorder, track
from hands.voice.voiceprints import Voiced


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


def events(recorded: list[Entry], name: str) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == name]


def unidentified(span: int, voiced: Voiced) -> None:
    """No span identified: what a test of the store measures is the store."""


async def kept(recorder: Recorder, hold: Hold, sound: Sound, recorded: list[Entry]) -> WideEvent:
    """The hold told to the recorder as the owner tells it, whole as it ends, and its writing run as the owner runs a
    hearing."""
    owner = HoldOwner(Unused(), (recorder,), recorded.append)
    assert recorder.opened(hold, Sound(b"", b"", ())) == () and recorder.heard(sound) == ()
    [write] = recorder.ended(hold, "sent", sound)
    # As the owner's hearing lane runs it: one that fails is said on its event, and the lane hears on.
    with contextlib.suppress(Exception):
        await owner._heard(recorder, hold, write, begun())  # pyright: ignore[reportPrivateUsage]
    return events(recorded, "hearing")[-1]


async def test_a_hold_at_the_desk_is_kept_as_heard_with_what_hands_played_over_it_and_in_which_voices(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    directory = tmp_path / "holds" / "run"
    hold = Hold(7, "engaged conversation")
    played = (Reference(tone(-1), RATE, 1, ("alba",)), Reference(tone(-2), RATE, 1, ("alba", "mary")), None)
    event = await kept(Recorder(tmp_path / "holds", "run", unidentified, recorded.append), hold, Sound(tone(1) + tone(2, 2), tone(9, 3), played), recorded)
    assert event.outcome == "ok"
    # What it heard, and what the speaker played against each 10 ms of it, sample for sample: silence where no TTS.
    assert read(directory / "7.heard.wav") == (1, RATE, tone(1) + tone(2, 2))
    assert read(directory / "7.played.wav")[:2] == (1, RATE)
    assert len(read(directory / "7.played.wav")[2]) == len(tone(1, 3)) and read(directory / "7.played.wav")[2][-320:] == tone(0)
    assert json.loads((directory / "7.json").read_text()) == {
        "hold": 7, "opener": "engaged conversation", "fate": "sent", "seconds": 0.03, "played": True, "voices": ["alba", "mary"],
    }
    # [LAW:nothing-unseen] where it went, how long, whether what was played went with it, and in which voices.
    assert (event.facts["kept"], event.facts["seconds"], event.facts["played"], event.facts["voices"]) == (directory / "7.heard.wav", 0.03, True, ("alba", "mary"))


def test_what_the_speaker_played_is_put_in_the_holds_format_a_run_at_a_time_slot_for_slot() -> None:
    played_rate = 24_000
    # Half a second of a 440 Hz tone, in stereo at the speaker's rate, in 10 ms references, then 10 ms of no TTS.
    wave_ = (8000 * np.sin(2 * np.pi * 440 * np.arange(played_rate // 2) / played_rate)).astype(np.int16)
    stereo = np.repeat(wave_, 2).tobytes()
    size = 4 * played_rate // 100
    references = tuple(Reference(stereo[at : at + size], played_rate, 2, ("alba",)) for at in range(0, len(stereo), size))
    held = np.frombuffer(track((*references, None)), np.int16).astype(np.float64)
    # One slot for each reference, the last silent; and the same tone at the hold's rate, past the resampler's edges.
    assert len(held) == (len(references) + 1) * RATE // 100 and not held[-RATE // 100 :].any()
    expected = 8000 * np.sin(2 * np.pi * 440 * np.arange(len(held)) / RATE)
    middle = slice(RATE // 20, RATE // 2 - RATE // 20)
    assert np.max(np.abs(held[middle] - expected[middle])) < 100


async def test_a_hold_at_the_phone_is_kept_with_nothing_said_to_have_played_over_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    event = await kept(Recorder(tmp_path, "run", unidentified, recorded.append), Hold(1, "phone button"), Sound(tone(3), tone(3), None), recorded)
    assert event.outcome == "ok" and event.facts["played"] is False
    assert read(tmp_path / "run" / "1.heard.wav") == (1, RATE, tone(3))
    assert not (tmp_path / "run" / "1.played.wav").exists()


async def test_two_runs_named_alike_keep_their_holds_apart(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    for level in (1, 2):
        await kept(Recorder(tmp_path, "2026-10-30T10-00-00Z", unidentified, recorded.append), Hold(1, "held key"), Sound(tone(level), tone(level), ()), recorded)
    assert read(tmp_path / "2026-10-30T10-00-00Z" / "1.heard.wav")[2] == tone(1)
    assert read(tmp_path / "2026-10-30T10-00-00Z-2" / "1.heard.wav")[2] == tone(2)


async def test_a_hold_that_cannot_be_written_fails_its_hearing_saying_why(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("in the way")
    recorded: list[Entry] = []
    event = await kept(Recorder(blocked / "holds", "run", unidentified, recorded.append), Hold(1, "held key"), Sound(tone(3), tone(3), ()), recorded)
    assert (event.outcome, event.error) == ("failed", f"NotADirectoryError: [Errno 20] Not a directory: '{blocked / 'holds'}'")


def older(root: Path, name: str, size: int) -> None:
    """A run before this one, of `size` bytes of holds."""
    (root / name).mkdir(parents=True)
    (root / name / "1.heard.wav").write_bytes(bytes(size))


async def test_a_store_taken_over_its_cap_deletes_whole_runs_oldest_first_and_never_the_runs_own(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    older(tmp_path, "2026-10-01T00-00-00Z", 10_000)
    older(tmp_path, "2026-10-02T00-00-00Z", 10_000)
    # This run's hold is about 3.3 KB: with both older runs the store is over 21 KB, under 14 KB without the oldest.
    recorder = Recorder(tmp_path, "2026-10-03T00-00-00Z", unidentified, recorded.append, cap=14_000)
    await kept(recorder, Hold(1, "held key"), Sound(tone(1, 10), tone(1, 10), ()), recorded)
    [pruned] = events(recorded, "holds.pruned")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["2026-10-02T00-00-00Z", "2026-10-03T00-00-00Z"]
    kept_bytes = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert (pruned.outcome, pruned.facts["runs"], pruned.facts["freed"], pruned.facts["kept"]) == ("ok", ("2026-10-01T00-00-00Z",), 10_000, kept_bytes)
    # A cap this run alone is over deletes every other run, and keeps this one whole.
    recorder = Recorder(tmp_path, "2026-10-04T00-00-00Z", unidentified, recorded.append, cap=100)
    await kept(recorder, Hold(1, "held key"), Sound(tone(1, 10), tone(1, 10), ()), recorded)
    assert [path.name for path in tmp_path.iterdir()] == ["2026-10-04T00-00-00Z"]
    assert (tmp_path / "2026-10-04T00-00-00Z" / "1.heard.wav").exists()


async def test_a_store_under_its_cap_deletes_nothing(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    older(tmp_path, "2026-10-01T00-00-00Z", 10_000)
    await kept(Recorder(tmp_path, "2026-10-03T00-00-00Z", unidentified, recorded.append, cap=14_000), Hold(1, "held key"), Sound(tone(1), tone(1), ()), recorded)
    assert events(recorded, "holds.pruned") == []
    assert len(list(tmp_path.iterdir())) == 2


@dataclass(frozen=True)
class Heard:
    """A buffer of the desk's microphone as the microphone hands it on (`hands.voice.microphone.Overheard`)."""

    cancelled: Cancelled
    desk: bool
    key: str


def buffer(level: int, voices: tuple[str, ...], key: str = "up", desk: bool = True) -> Heard:
    """20 ms of the desk, heard at `level`, over 20 ms of hands' TTS in `voices`, or none for no voice."""
    reference = Reference(tone(-level), RATE, 1, voices) if voices else None
    return Heard(Cancelled(tone(level, 2), (reference, reference)), desk, key)


def overheard(buffers: list[Heard]) -> Callable[[], AbstractAsyncContextManager[AsyncIterator[Heard]]]:
    """The desk's buffers, as the microphone hands them on."""

    @asynccontextmanager
    async def echoes() -> AsyncGenerator[AsyncIterator[Heard]]:
        async def each() -> AsyncIterator[Heard]:
            for heard in buffers:
                yield heard

        yield each()

    return echoes


async def test_hands_echo_at_the_off_level_with_no_press_is_kept_and_identified_span_by_span(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    identified: list[tuple[int, Voiced]] = []
    recorder = Recorder(tmp_path, "run", lambda span, voiced: identified.append((span, voiced)), recorded.append)
    await recorder.overhear(
        overheard([
            buffer(1, ()),  # no TTS played: hands not speaking, no span
            buffer(2, ("alba",)),  # a span opens
            buffer(3, ("alba", "mary")),
            buffer(4, ("mary",), key="arming"),  # a press: the span ends there, and the press is the hold's
            buffer(5, ("mary",)),  # a second span
            buffer(6, ("mary",), key="listening"),  # the wake word's level, where a voice can open a hold: it ends
            buffer(7, ("mary",)),  # a third
            buffer(8, ("mary",), desk=False),  # hands at the phone: it ends
        ])
    )
    assert read(tmp_path / "run" / "echo-1.heard.wav")[2] == tone(2, 2) + tone(3, 2)
    assert read(tmp_path / "run" / "echo-1.played.wav")[2] == tone(-2, 2) + tone(-3, 2)
    assert json.loads((tmp_path / "run" / "echo-1.json").read_text()) == {
        "span": 1, "hold": None, "echo": True, "seconds": 0.04, "played": True, "voices": ["alba", "mary"],
    }
    assert read(tmp_path / "run" / "echo-2.heard.wav")[2] == tone(5, 2)
    assert read(tmp_path / "run" / "echo-3.heard.wav")[2] == tone(7, 2)
    assert [(event.outcome, event.facts["span"], event.facts["voices"]) for event in events(recorded, "echo.kept")] == [
        ("ok", 1, ("alba", "mary")), ("ok", 2, ("mary",)), ("ok", 3, ("mary",)),
    ]
    # Each span is one voice, hands', identified against its prints once it has ended, with every voice it played.
    assert [(span, voiced.speaker, round(voiced.seconds, 3), voiced.played) for span, voiced in identified] == [
        (1, 0, 0.04, ("alba", "mary")), (2, 0, 0.02, ("mary",)), (3, 0, 0.02, ("mary",)),
    ]
    assert np.array_equal(identified[1][1].voice, np.full(320, 5 / 32768, np.float32))


async def test_echo_spans_count_toward_the_stores_cap(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    older(tmp_path, "2026-10-01T00-00-00Z", 10_000)
    recorder = Recorder(tmp_path, "2026-10-03T00-00-00Z", unidentified, recorded.append, cap=10_500)
    await recorder.overhear(overheard([buffer(1, ("alba",)), buffer(1, ())]))
    [pruned] = events(recorded, "holds.pruned")
    assert pruned.facts["runs"] == ("2026-10-01T00-00-00Z",)


async def test_a_run_alone_over_the_cap_prunes_nothing_and_never_walks_the_store_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[Entry] = []
    walked: list[Path] = []
    measured = recorders.size

    def size(directory: Path) -> int:
        walked.append(directory)
        return measured(directory)

    monkeypatch.setattr(recorders, "size", size)
    recorder = Recorder(tmp_path, "run", unidentified, recorded.append, cap=100)
    for number in (1, 2, 3):
        await kept(recorder, Hold(number, "held key"), Sound(tone(1, 10), tone(1, 10), ()), recorded)
    assert events(recorded, "holds.pruned") == [] and walked == []
    assert len(list((tmp_path / "run").glob("*.heard.wav"))) == 3
