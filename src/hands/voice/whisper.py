"""Whisper, the transcriber on the clip owner (`hands.voice.clipowner`): it transcribes each clip the key sent, whole, as it
ends (hands.voice.transcription), and those are the words a turn sends, none of them hands' own. A clip still open is
heard by the streaming recognizer (`hands.voice.streaming`), whose words only cut."""

import threading
from collections.abc import Awaitable, Callable, Sequence
from importlib.resources import files

import numpy as np
from pipecat.audio.vad.silero import SileroOnnxModel

from hands.clock import Clock
from hands.sessions.audit import ClipHeard, Record, Unsaid
from hands.sessions.wide import annotate
from hands.voice import loads
from hands.threads import SerialThread
from hands.voice import transcription
from hands.voice.clips import RATE, SECOND, Clip, Echo, Identified, decoded

# Whisper's own compression_ratio_threshold: above it, the decoder was repeating itself, as it does over noise primed
# with a vocabulary, sure of every word ("and turnstop, and turnstop, ..." read 17.1 to 23.8; speech no more than 0.92,
# hands-dictation-2bs.1xq).
_REPEATING = 2.4

# The average log probability below which a segment is Whisper guessing. Primed noise came back as "and slow-talking."
# at -2.8, which the compression ratio (0.68) does not catch; "okay" said quietly at -25 dBFS read
# -0.89 to -1.11 (hands-dictation-d7i).
_GUESSED = -1.5

# Silero's voice activity detector as Pipecat ships it, and the samples it scores at a time: 32 ms at RATE.
_SILERO = files("pipecat.audio.vad.data") / "silero_vad.onnx"
_FRAME = 512

# The least confidence Silero has in its surest frame of a segment anyone spoke in. Whisper hears words where nobody spoke,
# sure of them, whatever it is primed with: unprimed, each of the user's 26 silent holds came back "Thank you." at -0.21 to
# -0.61, and primed with file names, the last of the names read back; a hold spoken in for 2 s and held silent for 35 s more
# came back with "Thank you." heard from 30 s to 60 s, a stretch Silero was 0.024 sure of. Over the user's holds, every one
# Silero was at most 0.23 sure of came back as a stock phrase ("Thank you.", "you", "Okay.", "I'm going to go ahead and do
# it.") or nothing, and speech was 0.31 or more: distant voices 0.31 to 0.49, the quiet LibriVox reader 0.98, `say` 0.996
# at any level down to -55 dBFS (promptctl-voice-i15t.a5s.x48).
_VOICED = 0.25

# Silence after a clip's last sample: a segment that ends right where the speech stopped has its last word dropped or
# garbled, so Whisper hears half a second of it past the end, as Pipecat padded every segment it gave a transcriber.
_TRAILING_SILENCE = bytes(SECOND // 2)


class Whisper:
    """Whisper large-v3-turbo on MLX: the clip owner's transcriber."""

    def __init__(self, *, clock: Clock, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with loads.loading(clock, "whisper.loaded", record):
            transcription.load()
            # Silero, which says how sure it is that anyone spoke in a segment; run on its own thread alone.
            self._silero = SileroOnnxModel(str(_SILERO))
            annotate(model=transcription.MODEL, revision=transcription.REVISION)
        # The vocabulary each clip is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # [LAW:single-enforcer] the one thread the model runs on: a transcription given up on is still running, and the
        # next must not run beside it.
        self._model = SerialThread("Whisper")
        # Silero's pass over a clip, a few milliseconds, on a thread of its own: it waits on the clip's identification,
        # and must not then wait behind the next clip's decode.
        self._voicing = SerialThread("Silero")

    async def transcribed(self, clip: Clip, audio: bytes, identified: Awaitable[Identified]) -> str | None:
        """The words a person said in a clip of the key's, primed with the vocabulary as it is now; recorded, so "I spoke
        and nothing happened" can be looked into."""
        heard = await self._heard(clip.number, audio, await self._prompt(), identified)
        self._record(heard)
        return heard.said

    async def _heard(self, clip: int, audio: bytes, prompt: str | None, identified: Awaitable[Identified]) -> ClipHeard:
        """What a person said in a clip's audio, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, the
        vocabulary.

        A segment with no word in it is dropped, as are one the decoder repeated itself in, one Whisper only guessed at,
        and one heard where Silero heard nobody speak: that is what Whisper makes of silence and noise. Silero hears the
        clip with hands' own echo silenced, as the clip's last identification found it, so a segment heard only in
        hands' voice is nobody's, and is dropped with them. A clip with nothing said in it comes back with no segment.
        """
        # Set once nobody waits for this clip's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()
        try:
            # Off the loop, the clip's decoding with it: the model runs for a third of a second a clip, and the speaker and
            # the key go on meanwhile.
            # [LAW:no-ambient-temporal-coupling] Whisper hears the clip while its speakers are still being identified;
            # only Silero's pass, a few milliseconds, waits on who they were.
            samples, segments = await self._model.run(lambda: _UNHEARD if given_up.is_set() else _segments(audio, prompt))
            echo = (await identified).echo
            scored = await self._voicing.run(lambda: [] if given_up.is_set() else self._scored(segments, _silenced(samples, echo)))
        finally:
            given_up.set()
        said: list[str] = []
        dropped: list[Unsaid] = []
        for segment in scored:
            worded = any(character.isalnum() for character in segment.text)
            if worded and segment.voiced >= _VOICED and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                said.append(segment.text)
            else:
                dropped.append(segment)
        return ClipHeard(clip, " ".join(said).strip() or None, tuple(dropped))

    def _scored(self, segments: Sequence[transcription.Segment], people: np.ndarray) -> list[Unsaid]:
        """Each segment Whisper heard with the scores it is dropped for, Silero's over the stretch of `people`, the clip
        with hands' echo silenced, that Whisper heard it in."""
        return [
            Unsaid(segment.text, segment.compression_ratio, segment.avg_logprob, self._voiced(people[round(segment.start * RATE) : round(segment.end * RATE)]))
            for segment in segments
        ]

    def _voiced(self, samples: np.ndarray) -> float:
        """How sure Silero is that anyone spoke in `samples`: its confidence in the surest of its 32 ms frames, from a fresh
        state, read only until one is `_VOICED` sure, all a segment is kept for. Speech is that sure within a few frames,
        so a segment someone spoke in costs next to nothing; one nobody did is read through, at 2.4 ms a second of it."""
        self._silero.reset_states()
        surest = 0.0
        for start in range(0, len(samples) - _FRAME + 1, _FRAME):
            surest = max(surest, float(self._silero(samples[start : start + _FRAME], RATE)[0, 0]))
            if surest >= _VOICED:
                break
        return surest


# A clip given up on before the model reached it: no samples, and no segment heard in them.
_UNHEARD: tuple[np.ndarray, list[transcription.Segment]] = (np.zeros(0, np.float32), [])


def _segments(audio: bytes, prompt: str | None) -> tuple[np.ndarray, list[transcription.Segment]]:
    """A clip's audio as samples, heard with `_TRAILING_SILENCE` after it, and each segment Whisper hears in them, primed
    with `prompt`. It blocks while the model runs."""
    samples = decoded(audio + _TRAILING_SILENCE)
    return samples, transcription.segments(samples, prompt)


def _silenced(samples: np.ndarray, echo: Echo) -> np.ndarray:
    """`samples` with each stretch of `echo`, where only hands' own voice was heard, made silence: what people said."""
    people = samples.copy()
    for at in echo:
        people[at.start : at.stop] = 0
    return people
