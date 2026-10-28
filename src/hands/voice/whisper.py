"""Whisper, the transcriber on the hold owner (`hands.voice.holdowner`): it transcribes each hold the key sent, whole, as it
ends (hands.voice.transcription), and those are the words a turn sends. A hold still open is heard by the streaming
recognizer (`hands.voice.streaming`), whose words only cut."""

import threading
from collections.abc import Awaitable, Callable

from hands.sessions.audit import HoldHeard, Record, Unsaid
from hands.sessions.wide import annotate
from hands.voice import loads
from hands.threads import SerialThread
from hands.voice import transcription
from hands.voice.holds import SECOND, Hold

# Whisper's own compression_ratio_threshold: above it, the decoder was repeating itself, as it does over noise primed
# with a vocabulary, sure of every word ("and turnstop, and turnstop, ..." read 17.1 to 23.8; speech no more than 0.92,
# hands-dictation-2bs.1xq).
_REPEATING = 2.4

# The average log probability below which a segment is Whisper guessing. Primed noise came back as "and slow-talking."
# at -2.8, which the compression ratio (0.68) does not catch; "okay" said quietly at -25 dBFS read
# -0.89 to -1.11 (hands-dictation-d7i).
_GUESSED = -1.5

# Silence after a hold's last sample: a segment that ends right where the speech stopped has its last word dropped or
# garbled, so Whisper hears half a second of it past the end, as Pipecat padded every segment it gave a transcriber.
_TRAILING_SILENCE = bytes(SECOND // 2)


class Whisper:
    """Whisper large-v3-turbo on MLX: the hold owner's transcriber."""

    def __init__(self, *, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with loads.loading("whisper.loaded", record):
            transcription.load()
            annotate(model=transcription.MODEL)
        # The vocabulary each hold is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # [LAW:single-enforcer] the one thread the model runs on: a transcription given up on is still running, and the
        # next must not run beside it.
        self._model = SerialThread("Whisper")

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        """The words said in a hold of the key's, primed with the vocabulary as it is now; recorded, so "I spoke and nothing
        happened" can be looked into."""
        heard = await self._heard(hold.number, audio, await self._prompt())
        self._record(heard)
        return heard.said

    async def _heard(self, hold: int, audio: bytes, prompt: str | None) -> HoldHeard:
        """What was said in a hold's audio, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, the vocabulary.

        A segment with no word in it is dropped, as are one the decoder repeated itself in and one Whisper only guessed
        at: that is what a primed Whisper makes of noise. A hold with nothing said in it comes back with no segment.
        """
        # Set once nobody waits for this hold's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()
        try:
            # Off the loop: the model runs for a third of a second a hold, and the speaker and the key go on meanwhile.
            scored = await self._model.run(lambda: [] if given_up.is_set() else transcription.segments(audio + _TRAILING_SILENCE, prompt))
        finally:
            given_up.set()
        said: list[str] = []
        dropped: list[Unsaid] = []
        for segment in scored:
            worded = any(character.isalnum() for character in segment.text)
            if worded and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                said.append(segment.text)
            else:
                dropped.append(segment)
        return HoldHeard(hold, " ".join(said).strip() or None, tuple(dropped))
