"""Whisper, the transcriber on the clip owner (`hands.voice.clipowner`): it transcribes each clip the key sent, whole, as it
ends (hands.voice.transcription), and those are the words a turn sends, none of them hands' own. A clip still open is
heard by the streaming recognizer (`hands.voice.streaming`), whose words only cut.

What Whisper hears is sent whole: hands' own voice is taken out of the microphone's sound by the echo canceller
(`hands.voice.echo`) before anything hears it, so every word Whisper writes is a person's, hands' own words said at the same
moment included, and nothing here decides whose a word is by its text or its time. A segment is dropped only for being
noise (`_heard`), and no quiet stretch longer than half a second is heard (`_without_long_quiet`).
"""

import re
import threading
from collections.abc import Awaitable, Callable, Iterator
from importlib.resources import files
from itertools import groupby

import numpy as np
from pipecat.audio.vad.silero import SileroOnnxModel

from hands.clock import Clock
from hands.sessions.audit import ClipHeard, Record, Unsaid
from hands.sessions.wide import annotate
from hands.voice import loads, mlxthread
from hands.voice import transcription
from hands.voice.clips import RATE, SECOND, Clip, Sound, decoded
from hands.voice.mlxthread import OnMLX

# Whisper's own compression_ratio_threshold: above it, the decoder was repeating itself, as it does over noise primed
# with a vocabulary, sure of every word ("and turnstop, and turnstop, ..." read 17.1 to 23.8; speech no more than 0.92,
# hands-dictation-2bs.1xq). Primed, it loops over speech too: "the build is green", 1.2 s of it through the desk's
# canceller, read "the build is green" three times over 4.5 s at 20.8, and unprimed, "The build is green." at 0.7.
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

# A word, as Whisper writes one: letters and digits, with an apostrophe inside ("won't"); an underscore is no letter, so
# "auth_middleware" is two words and "_" none.
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")

# Silence after a clip's last sample: a segment that ends right where the speech stopped has its last word dropped or
# garbled, so Whisper hears half a second of it past the end, as Pipecat padded every segment it gave a transcriber.
_TRAILING_SILENCE = bytes(SECOND // 2)

# The longest quiet stretch Whisper hears, in Silero frames: a longer one is cut to it. Primed with the vocabulary, Whisper
# answers a clip with a long quiet lead with the vocabulary's own words, sure of them (avg_logprob -0.9, so no gate drops
# them), as one segment from 0.0 to 30.0 s, and the person's words are lost: the takes `hands_over_an_opening` (the person
# speaks at 9.5 s) and `hands_no_echo_in_two_voices` (8.1 s); from 1 to 3 s before the person speaks it reads them too
# (promptctl-voice-i15t.65f.qem). Cut to half a second, the same as is heard after the clip, all twelve recorded takes are
# read whole, primed, with every word the person said and none of the vocabulary's.
_QUIET = (RATE // 2) // _FRAME


class Whisper:
    """Whisper large-v3-turbo on MLX: the clip owner's transcriber."""

    def __init__(self, *, clock: Clock, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with loads.loading(clock, "whisper.loaded", record):
            mlxthread.call(transcription.load)
            # Silero, which says how sure it is that anyone spoke in a segment.
            self._silero = SileroOnnxModel(str(_SILERO))
            annotate(model=transcription.MODEL, revision=transcription.REVISION)
        # The vocabulary each clip is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record

    async def transcribed(self, clip: Clip, sound: Sound) -> str | None:
        """The words a person said in a clip of the key's, primed with the vocabulary as it is now; recorded, so "I spoke
        and nothing happened" can be looked into."""
        heard = await self._heard(clip.number, sound, await self._prompt())
        self._record(heard)
        return heard.said

    async def _heard(self, clip: int, sound: Sound, prompt: str | None) -> ClipHeard:
        """What a person said in a clip's sound, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, the
        vocabulary.

        A segment with no word in it is dropped, as are one the decoder repeated itself in, one Whisper only guessed at,
        and one heard where Silero heard nobody speak: that is what Whisper makes of silence and noise. The rest are said,
        every word of them. A clip with nothing said in it comes back with no segment.
        """
        # Set once nobody waits for this clip's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()

        def scored_unless_given_up(on: OnMLX) -> list[tuple[transcription.Segment, float]]:
            return [] if given_up.is_set() else self._scored(on, sound, prompt)

        try:
            # [LAW:single-enforcer] on the MLX thread, the clip's decoding and Silero's pass after it: the model runs for a
            # third of a second a clip, and the speaker and the key go on meanwhile; a transcription given up on is still
            # running, and the next waits behind it there.
            scored = await mlxthread.run(scored_unless_given_up)
        finally:
            given_up.set()
        kept: list[transcription.Segment] = []
        dropped: list[Unsaid] = []
        for segment, voiced in scored:
            if _words(segment.text) and voiced >= _VOICED and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                kept.append(segment)
            else:
                dropped.append(Unsaid(segment.text, segment.compression_ratio, segment.avg_logprob, voiced))
        return ClipHeard(clip, " ".join(segment.text for segment in kept) or None, tuple(dropped))

    def _scored(self, on: OnMLX, sound: Sound, prompt: str | None) -> list[tuple[transcription.Segment, float]]:
        """Each segment Whisper hears in a clip, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, and how sure
        Silero is that anyone spoke in the stretch it was heard in."""
        samples = self._without_long_quiet(decoded(sound.heard + _TRAILING_SILENCE))
        segments = transcription.segments(on, samples, prompt)
        # A decode primed with the vocabulary that loops is heard again unprimed (`_REPEATING`): the looped segments still go
        # to the gates, which drop and record them, and the unprimed decode's take the clip's place. The whole clip is heard
        # again, not the looped stretch: a loop's own times are not to be trusted (one ran 0.0 to 30.0 s, promptctl-voice-i15t.65f.qem).
        looping = [segment for segment in segments if segment.compression_ratio > _REPEATING]
        heard = transcription.segments(on, samples, None) if looping else segments
        return [(segment, self._voiced(samples[round(segment.start * RATE) : round(segment.end * RATE)])) for segment in [*looping, *heard]]

    def _without_long_quiet(self, samples: np.ndarray) -> np.ndarray:
        """`samples` with every stretch Silero is under `_VOICED` sure of in each of its 32 ms frames cut to `_QUIET` frames,
        half of it from either end of the stretch, so a voice keeps the lead and the trail it had. What Whisper writes of a
        clip is then timed in these samples, not the clip's."""
        voiced = [confidence >= _VOICED for confidence in self._confidences(samples)]
        kept: list[np.ndarray] = []
        first = 0
        for is_voiced, run in groupby(voiced):
            end = first + len(list(run))
            if is_voiced or end - first <= _QUIET:
                kept.append(samples[first * _FRAME : end * _FRAME])
            else:
                # A quiet stretch too long: its first and last frames, half of `_QUIET` from either end.
                kept += [samples[first * _FRAME : (first + _QUIET // 2) * _FRAME], samples[(end - (_QUIET - _QUIET // 2)) * _FRAME : end * _FRAME]]
            first = end
        # The few samples after the last whole frame.
        return np.concatenate([*kept, samples[len(voiced) * _FRAME :]])

    def _confidences(self, samples: np.ndarray) -> Iterator[float]:
        """How sure Silero is that anyone spoke in each 32 ms frame of `samples`, from a fresh state, read as it is asked
        for."""
        self._silero.reset_states()
        for start in range(0, len(samples) - _FRAME + 1, _FRAME):
            yield float(self._silero(samples[start : start + _FRAME], RATE)[0, 0])

    def _voiced(self, samples: np.ndarray) -> float:
        """How sure Silero is that anyone spoke in `samples`: its confidence in the surest of its 32 ms frames, from a fresh
        state, read only until one is `_VOICED` sure, all a segment is kept for. Speech is that sure within a few frames,
        so a segment someone spoke in costs next to nothing; one nobody did is read through, at 2.4 ms a second of it."""
        surest = 0.0
        for confidence in self._confidences(samples):
            surest = max(surest, confidence)
            if surest >= _VOICED:
                break
        return surest


def _words(text: str) -> list[str]:
    """The words of `text`, in lower case."""
    return _WORD.findall(text.lower())
