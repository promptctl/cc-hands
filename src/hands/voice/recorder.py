"""The recorder: each hold of the key's kept on disk as hands heard it, after the echo canceller, beside what hands was
playing over it, so real use becomes labelled data to fit the voiceprint thresholds on (`hands.voice.voiceprints`);
nothing here loads Pipecat, and nothing here plays a sound.

It is one listener on the hold owner (`hands.voice.holdowner`), given each hold whole as it ends, and writes it off the
event loop into its run's own directory: `<hold>.heard.wav`, mono 16-bit at RATE; `<hold>.played.wav`, what the desk's
speaker played over it, the echo canceller's reference, put in the hold's format so the two line up sample for sample,
which the phone has none of; and `<hold>.json`, what opened it, what the gate did with it, and every voice of hands'
own TTS played over it.

It keeps hands' own echo too, labelled. While the desk is at the off level, where only a held key opens a hold, with
no press arming or held, and the speaker is playing hands' TTS, what the desk's microphone hears is nobody's but
hands', whatever is left of it after the canceller: the user is on the held key alone until promptctl-voice-i15t.8vg.idh.
Each stretch of it is an echo span, kept as a hold is, `echo-<span>.*`, its `.json` saying no hold was open, and
identified as one speaker at its end, so the speaker check is fitted on held-key use (promptctl-voice-i15t.8vg.xvr).
"""

import asyncio
import contextlib
import json
import shutil
import threading
import wave
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from itertools import groupby
from pathlib import Path
from typing import Protocol

import numpy as np
import soxr

from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.voice.echo import Cancelled
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Reference, Sound, decoded, voices
from hands.voice.voiceprints import WINDOW, Voiced

# The most the holds store keeps, about nine hours of holds at 64 KB a second, heard and played. The recordings are the
# dataset the speaker check is fitted and checked on (promptctl-voice-i15t.8vg.xvr), kept as the most recent real use:
# past this, whole runs go, oldest first.
HOLDS_BYTES = 2 * 1024**3

# A 10 ms slot of a recording, in samples at RATE: what the speaker's reference comes in.
SLOT = RATE // 100


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
    """[LAW:single-enforcer] one recording, a hold's or an echo span's, in its run's `directory`: `<name>.heard.wav`,
    `<name>.played.wav` where what the speaker played is known, and `<name>.json`, `facts` with how long it was and
    every voice of hands' TTS over it; how many bytes they are."""
    size = written(directory / f"{name}.heard.wav", sound.heard)
    match sound.played:
        case tuple() as played:
            size += written(directory / f"{name}.played.wav", track(played))
        case None:
            pass
    described = directory / f"{name}.json"
    described.write_text(json.dumps({**facts, "seconds": round(sound.seconds, 3), "played": sound.played is not None, "voices": voices(sound)}) + "\n")
    return size + described.stat().st_size


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


class Overheard(Protocol):
    """A buffer of the desk's microphone through the canceller, whatever the key (`hands.voice.microphone.Overheard`)."""

    @property
    def cancelled(self) -> Cancelled: ...
    @property
    def desk(self) -> bool: ...
    @property
    def key(self) -> str: ...


def echoed(each: Overheard) -> bool:
    """Whether a buffer is hands' echo and nobody else's: heard at the desk at the off level with no press, as it was
    captured, while the speaker played hands' TTS."""
    return each.desk and each.key == "up" and any(reference is not None for reference in each.cancelled.played)


class Recorder:
    """Keeps every hold of the key's, and each echo span, in a directory of the run's own under `root`, named `named`,
    and the store under `cap`.

    [LAW:no-silent-failure] a hold that cannot be written fails its hearing, the unit it is written in, with why, a
    span its `echo.kept` unit, and a prune that fails its own `holds.pruned` unit; the next is written as if none had.
    """

    name = "recorder"

    def __init__(self, root: Path, named: str, identify: Callable[[int, Voiced], None], record: Record, cap: int = HOLDS_BYTES) -> None:
        self._root = root
        self._named = named
        self._identify = identify
        self._record = record
        self._cap = cap
        self._spans = 0
        # [LAW:no-shared-mutable-globals] the run's directory once claimed, the bytes it holds, and each earlier run and
        # its bytes, counted once as the directory is claimed and kept up to date by each write and prune since: written
        # by each recording's writing thread under the one lock.
        self._writing = threading.Lock()
        self._directory: Path | None = None
        self._current = 0
        self._runs: dict[Path, int] = {}

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        return ()

    def heard(self, sound: Sound) -> Hearings:
        return ()

    def ended(self, hold: Hold, fate: Fate, sound: Sound) -> Hearings:
        async def keep() -> Heard:
            # [LAW:nothing-unseen] how long the hold was, whether what hands played over it went with it, and in which voices.
            annotate(seconds=round(sound.seconds, 3), played=sound.played is not None, voices=voices(sound))
            facts: dict[str, object] = dict(hold=hold.number, opener=hold.opener, fate=fate)
            directory = await asyncio.to_thread(self._written, str(hold.number), sound, facts)
            annotate(kept=directory / f"{hold.number}.heard.wav")
            return Heard(None)

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
                with contextlib.suppress(Exception), unit("holds.pruned", self._record):
                    self._runs = pruned(self._runs, self._current, self._cap)
            return directory

    async def overhear(self, echoes: Callable[[], AbstractAsyncContextManager[AsyncIterator[Overheard]]]) -> None:
        """Keep each echo span the desk overhears, until cancelled: a span runs while each buffer is hands' echo, as it
        was captured (`echoed`), and ends at the first that is not, a press or a level that would let a voice open a
        hold included."""
        heard, played = bytearray(), list[Reference | None]()
        async with echoes() as overheard:
            async for each in overheard:
                match echoed(each):
                    case True:
                        heard += each.cancelled.heard
                        played += each.cancelled.played
                    case False if heard:
                        await self._span(Sound(bytes(heard), bytes(heard), tuple(played)))
                        heard, played = bytearray(), list[Reference | None]()
                    case False:
                        pass

    async def _span(self, sound: Sound) -> None:
        """One echo span kept, a unit of its own, and identified as one speaker, hands, against hands' own prints.

        [LAW:no-silent-failure] a span that cannot be written fails its `echo.kept` unit with why; the next span is
        kept as if it had not."""
        self._spans += 1
        span = self._spans
        with contextlib.suppress(Exception), unit("echo.kept", self._record):
            annotate(span=span, seconds=round(sound.seconds, 3), voices=voices(sound))
            facts: dict[str, object] = dict(span=span, hold=None, echo=True)
            directory = await asyncio.to_thread(self._written, f"echo-{span}", sound, facts)
            annotate(kept=directory / f"echo-{span}.heard.wav")
        voice = decoded(sound.heard)
        self._identify(span, Voiced(0, voice[-round(WINDOW * RATE) :], len(voice) / RATE, 0.0, voices(sound)))
