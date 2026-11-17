"""Whisper, the transcriber on the clip owner (`hands.voice.clipowner`): it transcribes each clip the key sent, whole, as it
ends (hands.voice.transcription), and those are the words a turn sends, none of them hands' own. A clip still open is
heard by the streaming recognizer (`hands.voice.streaming`), whose words only cut.

hands' own words are known by their text. Whisper hears hands' voice come back through the microphone as whole segments
of what hands said, a clause or a sentence ("Turn ended," a segment of its own), and on zoj's recordings never in one
segment with a person's (promptctl-voice-i15t.8vg.06g): so a segment, not a word, is hands' or a person's.
"""

import re
import threading
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from importlib.resources import files

import numpy as np
from pipecat.audio.vad.silero import SileroOnnxModel

from hands.clock import Clock
from hands.sessions.audit import ClipHeard, Record, Unsaid
from hands.sessions.wide import annotate
from hands.voice import loads
from hands.threads import SerialThread
from hands.voice import transcription
from hands.voice.clips import RATE, SECOND, SLOT, Clip, Play, Sound, decoded, plays

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

# How far from a segment, in seconds, the words of hands' heard in it may have played for it to be hands' echo. A word of
# a sentence placed by its characters (`_played`) lands within 0.05 to 0.14 s of where Whisper's word timestamps put it,
# 0.9 s at most, on both of zoj's recordings; the timestamps themselves would double the decode the user waits on.
_ECHO_SECONDS = 1.0

# The least likeness of a word Whisper heard to a word hands said, by difflib's ratio, at which it is that word spelled
# Whisper's own way: "diariser" for "diarizer" is 0.875.
_RESPELLED = 0.75

# A word, as Whisper and hands' TTS both write one: letters and digits, with an apostrophe inside ("won't").
_WORD = re.compile(r"\w+(?:'\w+)*")

# Silence after a clip's last sample: a segment that ends right where the speech stopped has its last word dropped or
# garbled, so Whisper hears half a second of it past the end, as Pipecat padded every segment it gave a transcriber.
_TRAILING_SILENCE = bytes(SECOND // 2)


class Whisper:
    """Whisper large-v3-turbo on MLX: the clip owner's transcriber."""

    def __init__(self, *, clock: Clock, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with loads.loading(clock, "whisper.loaded", record):
            transcription.load()
            # Silero, which says how sure it is that anyone spoke in a segment.
            self._silero = SileroOnnxModel(str(_SILERO))
            annotate(model=transcription.MODEL, revision=transcription.REVISION)
        # The vocabulary each clip is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # [LAW:single-enforcer] the one thread the model runs on, Silero's pass after it: a transcription given up on is
        # still running, and the next must not run beside it.
        self._model = SerialThread("Whisper")

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
        and one heard where Silero heard nobody speak: that is what Whisper makes of silence and noise. Of the rest, one
        that is hands' own voice come back (`_echo`) goes too. A clip with nothing said in it comes back with no segment.
        """
        # Set once nobody waits for this clip's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()
        try:
            # Off the loop, the clip's decoding with it: the model runs for a third of a second a clip, and the speaker and
            # the key go on meanwhile.
            scored = await self._model.run(lambda: [] if given_up.is_set() else self._scored(sound.heard, prompt))
        finally:
            given_up.set()
        played = _played(plays(sound))
        said: list[str] = []
        echo: list[str] = []
        dropped: list[Unsaid] = []
        for segment, voiced in scored:
            heard = _words(segment.text)
            if not heard or voiced < _VOICED or segment.compression_ratio > _REPEATING or segment.avg_logprob < _GUESSED:
                dropped.append(Unsaid(segment.text, segment.compression_ratio, segment.avg_logprob, voiced))
            elif _echo(heard, segment, played):
                echo.append(segment.text)
            else:
                said.append(segment.text)
        return ClipHeard(clip, " ".join(said).strip() or None, tuple(dropped), tuple(echo))

    def _scored(self, audio: bytes, prompt: str | None) -> list[tuple[transcription.Segment, float]]:
        """Each segment Whisper hears in a clip's audio, heard with `_TRAILING_SILENCE` after it, primed with `prompt`,
        and how sure Silero is that anyone spoke in the stretch it was heard in. It blocks while the model runs."""
        samples = decoded(audio + _TRAILING_SILENCE)
        return [(segment, self._voiced(samples[round(segment.start * RATE) : round(segment.end * RATE)])) for segment in transcription.segments(samples, prompt)]

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


def _words(text: str) -> list[str]:
    """The words of `text`, in lower case."""
    return _WORD.findall(text.lower())


# Each word hands said over a clip, with when it played, in seconds from the clip's start, earliest first.
type Played = tuple[tuple[str, float], ...]


def _played(plays: tuple[Play, ...]) -> Played:
    """Each word hands said over a clip, placed by its characters between its play's start and stop: no time is known
    for a word of a sentence played, only for the sentence."""
    placed = (
        (word.group(), (play.start + (play.stop - play.start) * (word.start() + word.end()) / 2 / len(text)) * SLOT / RATE)
        for play in plays
        for text in (play.synthesis.text.lower(),)
        for word in _WORD.finditer(text)
    )
    return tuple(sorted(placed, key=lambda each: each[1]))


def _echo(heard: list[str], segment: transcription.Segment, played: Played) -> bool:
    """Whether a segment is hands' own voice come back through the microphone: every word `heard` in it is a word hands
    played within `_ECHO_SECONDS` of it, in order, said or spelled Whisper's own way.

    [LAW:single-enforcer] the one rule of whose words are hands'. One word that is not hands' makes the segment a
    person's, whole: "Yes, run the tests." after hands asked "Should I run the tests?" is sent. A word of hands' the
    canceller took out is a gap in hands' words, never in those heard. A person saying only words of hands', within a
    second of hands saying them, is taken for hands: by text, the two are the same."""
    near = iter(word for word, at in played if segment.start - _ECHO_SECONDS <= at <= segment.end + _ECHO_SECONDS)
    # Each word heard takes the first of hands' after the last one taken that it is: matched greedily, in order, which
    # finds a match wherever any that keeps the order exists.
    return all(any(_respelled(word, theirs) for theirs in near) for word in heard)


def _respelled(heard: str, said: str) -> bool:
    """Whether a word Whisper `heard` is a word hands `said`, as said or spelled Whisper's own way."""
    return heard == said or SequenceMatcher(None, heard, said, autojunk=False).ratio() >= _RESPELLED
