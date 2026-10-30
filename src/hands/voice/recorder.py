"""The recorder: each hold of the key's kept on disk as hands heard it, after the echo canceller, beside what hands was
playing over it, so real use becomes labelled data to fit the voiceprint thresholds on (`hands.voice.voiceprints`);
nothing here loads Pipecat, and nothing here plays a sound.

It is one listener on the hold owner (`hands.voice.holdowner`), given each hold whole as it ends, and writes it off the
event loop: `<hold>.heard.wav`, mono 16-bit at RATE; `<hold>.played.wav`, what the desk's speaker played over it, the
echo canceller's reference, put in the hold's format so the two line up sample for sample, which the phone has none of;
and `<hold>.json`, what opened it and what the gate did with it.
"""

import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from hands.sessions.wide import annotate
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Reference, Sound


def reformatted(reference: Reference) -> bytes:
    """10 ms of what the speaker played in a hold's format: mixed down to mono and linearly interpolated to RATE."""
    samples = np.frombuffer(reference.audio, np.int16).reshape(-1, reference.channels).mean(axis=1)
    wanted = RATE // 100
    resampled = np.interp(np.arange(wanted) * len(samples) / wanted, np.arange(len(samples)), samples)
    return np.round(resampled).astype(np.int16).tobytes()


def written(path: Path, audio: bytes) -> None:
    """`audio`, mono 16-bit at RATE, as a WAV file at `path`."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(audio)


def kept(directory: Path, hold: Hold, fate: Fate, sound: Sound) -> None:
    """One hold's files in `directory`, made if it is not there yet."""
    directory.mkdir(parents=True, exist_ok=True)
    written(directory / f"{hold.number}.heard.wav", sound.heard)
    match sound.played:
        case tuple() as played:
            written(directory / f"{hold.number}.played.wav", b"".join(map(reformatted, played)))
        case None:
            pass
    facts = dict(hold=hold.number, opener=hold.opener, fate=fate, seconds=round(sound.seconds, 3), played=sound.played is not None)
    (directory / f"{hold.number}.json").write_text(json.dumps(facts) + "\n")


class Recorder:
    """Keeps every hold of the key's in `directory`, one run's.

    [LAW:no-silent-failure] a hold that cannot be written fails its hearing, the unit it is written in, with why; the
    next hold is written as if it had not.
    """

    name = "recorder"

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        return ()

    def heard(self, sound: Sound) -> Hearings:
        return ()

    def ended(self, hold: Hold, fate: Fate, sound: Sound) -> Hearings:
        directory = self._directory

        async def keep() -> Heard:
            # [LAW:nothing-unseen] where the hold went, how long, and whether what hands played over it went with it.
            annotate(kept=directory / f"{hold.number}.heard.wav", seconds=round(sound.seconds, 3), played=sound.played is not None)
            await asyncio.to_thread(kept, directory, hold, fate, sound)
            return Heard(None)

        return (keep,)
