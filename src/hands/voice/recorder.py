"""The recorder: each hold of the key's kept on disk as hands heard it, after the echo canceller, beside what hands was
playing over it, so real use becomes labelled data to fit the voiceprint thresholds on (`hands.voice.voiceprints`);
nothing here loads Pipecat, and nothing here plays a sound.

It is one listener on the hold owner (`hands.voice.holdowner`), given every hold's audio as the other listeners are, and
writes each hold, off the event loop, once it ends: `<hold>.heard.wav`, mono 16-bit at RATE; `<hold>.played.wav`, what
the desk's speaker played over it, sample for sample, the echo canceller's reference, which the phone has none of; and
`<hold>.json`, what opened it and what the gate did with it.
"""

import asyncio
import json
import wave
from pathlib import Path

from hands.sessions.wide import annotate
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Recording, Sound


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
        case bytes() as played:
            written(directory / f"{hold.number}.played.wav", played)
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
        self._recording = Recording()

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        self._recording.clear()
        self._recording.add(before, gathering=True)
        return ()

    def heard(self, sound: Sound) -> Hearings:
        self._recording.add(sound, gathering=True)
        return ()

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        sound, directory = self._recording.sound, self._directory

        async def keep() -> Heard:
            # [LAW:nothing-unseen] where the hold went, how long, and whether what hands played over it went with it.
            annotate(kept=directory / f"{hold.number}.heard.wav", seconds=round(sound.seconds, 3), played=sound.played is not None)
            await asyncio.to_thread(kept, directory, hold, fate, sound)
            return Heard(None, ())

        return (keep,)
