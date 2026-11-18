"""The recorder: each clip of the key's kept on disk as hands heard it, after the echo canceller, beside what hands was
playing over it, so real use is kept as labelled data; nothing here loads Pipecat, and nothing here plays a sound.

It is one listener on the clip owner (`hands.voice.clipowner`), given each clip whole as it ends, and writes it off the
event loop into its run's own directory: `<clip>.heard.wav`, mono 16-bit at RATE; `<clip>.played.wav`, what the desk's
speaker played over it, the echo canceller's reference, put in the clip's format so the two line up sample for sample,
which the phone has none of; and `<clip>.json`, what opened it, what the gate did with it, and every voice of hands'
own TTS that may be in it: played over it, its echo path not read none.
"""

import asyncio
import contextlib
import json
import shutil
import threading
import wave
from itertools import groupby
from pathlib import Path

import numpy as np
import soxr

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, begun, unit
from hands.voice.clips import RATE, SLOT, Clip, Fate, Heard, Hearings, Reference, Sound, plays, voices

# The most the clips store keeps, about nine hours of clips at 64 KB a second, heard and played, kept as the most recent
# real use: past this, whole runs go, oldest first.
CLIPS_BYTES = 2 * 1024**3

def track(played: tuple[Reference | None, ...]) -> bytes:
    """What the speaker played, one 10 ms slot for each reference, in a recording's format: each run of references of
    one format joined, mixed down to mono and resampled once, and silence where it played no TTS."""
    out = list[bytes]()
    for form, run in groupby(played, lambda reference: None if reference is None else (reference.rate, reference.channels)):
        references = list(run)
        wanted = len(references) * SLOT
        match form:
            case None:
                out.append(bytes(2 * wanted))
            case (rate, channels):
                joined = b"".join(reference.audio for reference in references if reference is not None)
                samples = np.frombuffer(joined, np.int16).reshape(-1, channels).mean(axis=1).astype(np.float32)
                # The resampler's length is the run's to a sample or two: fitted to it, so every slot lines up.
                fitted = np.zeros(wanted, np.float32)
                resampled = soxr.resample(samples, rate, RATE)[:wanted]
                fitted[: len(resampled)] = resampled
                out.append(np.round(fitted).clip(-32768, 32767).astype(np.int16).tobytes())
    return b"".join(out)


def written(path: Path, audio: bytes) -> int:
    """`audio`, mono 16-bit at RATE, as a WAV file at `path`; how many bytes the file is."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(audio)
    return path.stat().st_size


def recording(directory: Path, name: str, sound: Sound, facts: dict[str, object]) -> int:
    """[LAW:single-enforcer] one recording, a clip's, in its run's `directory`: `<name>.heard.wav`,
    `<name>.played.wav` where what the speaker played is known, and `<name>.json`, `facts` with how long it was, every
    voice of hands' TTS that may be in it, each sentence it said over it, with when, so whose words were whose can be told from
    the recording alone, and how much of hands' TTS over it, in seconds, played with no echo path to the microphone; how
    many bytes they are."""
    size = written(directory / f"{name}.heard.wav", sound.heard)
    match sound.played:
        case tuple() as played:
            size += written(directory / f"{name}.played.wav", track(played))
        case None:
            pass
    described = directory / f"{name}.json"
    unreached = sum(reference is not None and not reference.reached for reference in sound.played or ()) * SLOT / RATE
    described.write_text(
        json.dumps({**facts, "seconds": round(sound.seconds, 3), "played": sound.played is not None, "voices": voices(sound), "said": said(sound), "unreached": unreached}) + "\n"
    )
    return size + described.stat().st_size


def said(sound: Sound) -> list[dict[str, object]]:
    """Each sentence hands said over `sound`, in the order it began to play: its voice, its text, from when to when it
    played, in seconds from the sound's start, and how many seconds of it had played before."""
    return [
        dict(voice=play.synthesis.voice, text=play.synthesis.text, start=round(play.start * SLOT / RATE, 3), into=round(play.into, 3), stop=round(play.stop * SLOT / RATE, 3))
        for play in plays(sound)
    ]


def claimed(root: Path, named: str) -> Path:
    """A directory of the run's own under `root`, named `named`, or `named-2` and on where an earlier run of the same
    second has it: made here, so no two runs ever share one."""
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        directory = root / (named if attempt == 1 else f"{named}-{attempt}")
        with contextlib.suppress(FileExistsError):
            directory.mkdir()
            return directory
    raise FileExistsError(f"every run directory named {named} under {root} is taken")


def size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def pruned(runs: dict[Path, int], current: int, cap: int) -> dict[Path, int]:
    """The runs left once whole ones are deleted, oldest first, until the store, `current` bytes of the run's own with
    them, is under `cap`, or none is left to delete: never the run's own.

    [LAW:nothing-unseen] the runs deleted, the bytes freed, and the bytes kept."""
    left, deleted = dict(sorted(runs.items())), list[Path]()
    for run in sorted(runs):
        if current + sum(left.values()) <= cap:
            break
        shutil.rmtree(run)
        del left[run]
        deleted.append(run)
    annotate(runs=tuple(run.name for run in deleted), freed=sum(runs[run] for run in deleted), kept=current + sum(left.values()))
    return left


class Recorder:
    """Keeps every clip of the key's in a directory of the run's own under `root`, named `named`, and the store under `cap`.

    [LAW:no-silent-failure] a clip that cannot be written fails its hearing, the unit it is written in, with why, and a
    prune that fails its own `clips.pruned` unit; the next is written as if none had.
    """

    name = "recorder"

    def __init__(self, clock: Clock, root: Path, named: str, record: Record, cap: int = CLIPS_BYTES) -> None:
        self._root = root
        self._named = named
        self._clock = clock
        self._record = record
        self._cap = cap
        # [LAW:no-shared-mutable-globals] the run's directory once claimed, the bytes it holds, and each earlier run and
        # its bytes, counted once as the directory is claimed and kept up to date by each write and prune since: written
        # by each recording's writing thread under the one lock.
        self._writing = threading.Lock()
        self._directory: Path | None = None
        self._current = 0
        self._runs: dict[Path, int] = {}

    def opened(self, clip: Clip, before: Sound) -> Hearings:
        return ()

    def heard(self, sound: Sound) -> Hearings:
        return ()

    def ended(self, clip: Clip, fate: Fate, sound: Sound) -> Hearings:
        async def keep() -> Heard:
            # [LAW:nothing-unseen] how long the clip was, whether what hands played over it went with it, and in which voices.
            annotate(seconds=round(sound.seconds, 3), played=sound.played is not None, voices=voices(sound))
            facts: dict[str, object] = dict(clip=clip.number, opener=clip.opener, fate=fate)
            directory = await asyncio.to_thread(self._written, str(clip.number), sound, facts)
            annotate(kept=directory / f"{clip.number}.heard.wav")
            return Heard(None, None)

        return (keep,)

    def _written(self, name: str, sound: Sound, facts: dict[str, object]) -> Path:
        """One recording in the run's directory, claimed by the first, and the store pruned back under the cap after it
        where an earlier run is left to delete; the run's directory."""
        with self._writing:
            match self._directory:
                case None:
                    directory = self._directory = claimed(self._root, self._named)
                    self._runs = {run: size(run) for run in self._root.iterdir() if run.is_dir() and run != directory}
                case Path() as directory:
                    pass
            self._current += recording(directory, name, sound, facts)
            if self._runs and self._current + sum(self._runs.values()) > self._cap:
                # Its own unit: what was written is kept whether or not the prune after it fails, which its event says.
                with contextlib.suppress(Exception), unit("clips.pruned", self._record, begun(self._clock)):
                    self._runs = pruned(self._runs, self._current, self._cap)
            return directory
