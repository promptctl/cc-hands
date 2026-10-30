"""The recorder: each hold of the key's kept on disk as hands heard it, after the echo canceller, beside what hands was
playing over it, so real use becomes labelled data to fit the voiceprint thresholds on (`hands.voice.voiceprints`);
nothing here loads Pipecat, and nothing here plays a sound.

It is one listener on the hold owner (`hands.voice.holdowner`), given each hold whole as it ends, and writes it off the
event loop into its run's own directory: `<hold>.heard.wav`, mono 16-bit at RATE; `<hold>.played.wav`, what the desk's
speaker played over it, the echo canceller's reference, put in the hold's format so the two line up sample for sample,
which the phone has none of; and `<hold>.json`, what opened it, what the gate did with it, and every voice of hands'
own TTS played over it.

It keeps hands' own echo too, labelled: while hands is speaking at the desk with no press of the key arming or held,
what the desk's microphone hears is nobody's but hands', whatever is left of it after the canceller. Each stretch of it
is an echo span, kept as a hold is, `echo-<span>.*`, its `.json` saying no hold was open, and identified as one
speaker at its end, so the speaker check is fitted on held-key use (promptctl-voice-i15t.8vg.xvr) before engaged
conversation is.
"""

import asyncio
import contextlib
import json
import shutil
import threading
import wave
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Protocol

import numpy as np

from hands.sessions.audit import Record
from hands.sessions.wide import annotate, unit
from hands.voice.echo import Cancelled
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Reference, Sound, decoded
from hands.voice.voiceprints import WINDOW, Voiced

# The most the holds store keeps, about nine hours of holds at 64 KB a second, heard and played. The recordings are the
# dataset the speaker check is fitted and checked on (promptctl-voice-i15t.8vg.xvr), kept as the most recent real use:
# past this, whole runs go, oldest first.
HOLDS_BYTES = 2 * 1024**3


def reformatted(reference: Reference) -> bytes:
    """10 ms of what the speaker played in a hold's format: mixed down to mono and linearly interpolated to RATE."""
    samples = np.frombuffer(reference.audio, np.int16).reshape(-1, reference.channels).mean(axis=1)
    wanted = RATE // 100
    resampled = np.interp(np.arange(wanted) * len(samples) / wanted, np.arange(len(samples)), samples)
    return np.round(resampled).astype(np.int16).tobytes()


def written(path: Path, audio: bytes) -> int:
    """`audio`, mono 16-bit at RATE, as a WAV file at `path`; how many bytes the file is."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(audio)
    return path.stat().st_size


class Overheard(Protocol):
    """A buffer of the desk's microphone through the canceller, whatever the key (`hands.voice.microphone.Overheard`)."""

    @property
    def cancelled(self) -> Cancelled: ...
    @property
    def desk(self) -> bool: ...
    @property
    def held(self) -> bool: ...


def voices(sound: Sound) -> tuple[str, ...]:
    """Every voice of hands' TTS played over `sound`, in the order first played."""
    return tuple(dict.fromkeys(voice for reference in sound.played or () for voice in reference.voices))


def kept(directory: Path, hold: Hold, fate: Fate, sound: Sound) -> int:
    """One hold's files in its run's `directory`; how many bytes they are."""
    size = written(directory / f"{hold.number}.heard.wav", sound.heard)
    match sound.played:
        case tuple() as played:
            size += written(directory / f"{hold.number}.played.wav", b"".join(map(reformatted, played)))
        case None:
            pass
    facts = dict(hold=hold.number, opener=hold.opener, fate=fate, seconds=round(sound.seconds, 3), played=sound.played is not None, voices=voices(sound))
    described = directory / f"{hold.number}.json"
    described.write_text(json.dumps(facts) + "\n")
    return size + described.stat().st_size


def echoed(directory: Path, span: int, sound: Sound) -> int:
    """One echo span's files in its run's `directory`; how many bytes they are."""
    played = sound.played or ()
    size = written(directory / f"echo-{span}.heard.wav", sound.heard) + written(directory / f"echo-{span}.played.wav", b"".join(map(reformatted, played)))
    facts = dict(span=span, hold=None, echo=True, seconds=round(sound.seconds, 3), played=True, voices=voices(sound))
    described = directory / f"echo-{span}.json"
    described.write_text(json.dumps(facts) + "\n")
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


def pruned(root: Path, current: Path, cap: int) -> int:
    """Whole runs under `root` deleted, oldest first and never `current`, until the store is under `cap`; what it keeps.

    [LAW:nothing-unseen] the runs deleted, the bytes freed, and the bytes kept."""
    runs = {run: size(run) for run in sorted(root.iterdir()) if run.is_dir()}
    store, deleted = sum(runs.values()), list[str]()
    for run, bytes_ in runs.items():
        if store <= cap:
            break
        if run == current:
            continue
        shutil.rmtree(run)
        store -= bytes_
        deleted.append(run.name)
    annotate(runs=tuple(deleted), freed=sum(runs.values()) - store, kept=store)
    return store


class Recorder:
    """Keeps every hold of the key's in a directory of the run's own under `root`, named `named`, and the store under
    `cap`.

    [LAW:no-silent-failure] a hold that cannot be written fails its hearing, the unit it is written in, with why, and a
    prune that fails fails its own `holds.pruned` unit; the next hold is written as if neither had.
    """

    name = "recorder"

    def __init__(self, root: Path, named: str, identify: Callable[[int, Voiced], None], record: Record, cap: int = HOLDS_BYTES) -> None:
        self._root = root
        self._named = named
        self._identify = identify
        self._record = record
        self._cap = cap
        self._spans = 0
        # [LAW:no-shared-mutable-globals] the run's directory once claimed, and how much the store holds, written by each
        # hold's writing thread under the one lock.
        self._writing = threading.Lock()
        self._directory: Path | None = None
        self._store = 0

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        return ()

    def heard(self, sound: Sound) -> Hearings:
        return ()

    def ended(self, hold: Hold, fate: Fate, sound: Sound) -> Hearings:
        async def keep() -> Heard:
            # [LAW:nothing-unseen] how long the hold was, and whether what hands played over it went with it, and in which voices.
            annotate(seconds=round(sound.seconds, 3), played=sound.played is not None, voices=voices(sound))
            annotate(kept=await asyncio.to_thread(self._kept, hold, fate, sound))
            return Heard(None)

        return (keep,)

    def _kept(self, hold: Hold, fate: Fate, sound: Sound) -> Path:
        return self._written(lambda directory: kept(directory, hold, fate, sound)) / f"{hold.number}.heard.wav"

    def _written(self, write: Callable[[Path], int]) -> Path:
        """`write` into the run's directory, claimed by the first, and the store pruned back under the cap after it;
        the run's directory."""
        with self._writing:
            match self._directory:
                case None:
                    directory = self._directory = claimed(self._root, self._named)
                    self._store = size(self._root)
                case Path() as directory:
                    pass
            self._store += write(directory)
            if self._store > self._cap:
                # Its own unit: what was written is kept whether or not the prune after it fails, which its event says.
                with contextlib.suppress(Exception), unit("holds.pruned", self._record):
                    self._store = pruned(self._root, directory, self._cap)
            return directory

    async def overhear(self, echoes: Callable[[], AbstractAsyncContextManager[AsyncIterator[Overheard]]], speaking: Callable[[], bool]) -> None:
        """Keep each echo span the desk hears, until cancelled: a span runs while hands is `speaking` at the desk with no
        press of the key arming or held, and ends at the first buffer that is not, a press that opens a hold included.

        [LAW:one-source-of-truth] whether hands is speaking is the speaker's say, read as each buffer is taken, and
        whether a press is open is the gate's, read as each buffer was captured."""
        heard, played = bytearray(), list[Reference]()
        async with echoes() as overheard:
            async for each in overheard:
                match each.desk and not each.held and speaking():
                    case True:
                        heard += each.cancelled.heard
                        played += each.cancelled.played
                    case False if heard:
                        await self._span(Sound(bytes(heard), bytes(heard), tuple(played)))
                        heard, played = bytearray(), list[Reference]()
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
            directory = await asyncio.to_thread(self._written, lambda directory: echoed(directory, span, sound))
            annotate(kept=directory / f"echo-{span}.heard.wav")
        voice = decoded(sound.heard)
        self._identify(span, Voiced(0, voice[-round(WINDOW * RATE) :], len(voice) / RATE, 0.0, voices(sound)))
