"""Whisper, the transcriber on the clip owner (`hands.voice.clipowner`): it transcribes each clip the key sent, whole, as it
ends (hands.voice.transcription), and those are the words a turn sends, none of them hands' own. A clip still open is
heard by the streaming recognizer (`hands.voice.streaming`), whose words only cut."""

import math
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
from difflib import SequenceMatcher
from itertools import groupby
from importlib.resources import files

import numpy as np
from pipecat.audio.vad.silero import SileroOnnxModel

from hands.clock import Clock
from hands.sessions.audit import ClipHeard, Record, Unsaid
from hands.sessions.wide import annotate
from hands.threads import SerialThread
from hands.voice import loads, transcription
from hands.voice.clips import RATE, SECOND, Clip, Sentence, Sound, decoded, said

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


# How long before a segment hands' sentence may have been played and still be heard in it: hands' echo reaches the
# microphone up to about 0.2 s after its TTS slot (zoj, promptctl-voice-i15t.99s.zoj), and Whisper places a segment's
# start up to about half a second off the words in it.
_ECHO_SECONDS = 1.0

# The fewest words in a row of hands' sentences that make a run of a segment hands' own, short of all hands said: one
# word alone ("the", "yes") is as likely a person's.
_RUN = 2

# How alike, letter for letter, words Whisper heard in place of hands' must be to hands' to be them, spelled its own way:
# "diariser" for "diarizer" (0.88), "Macalbox" for "macklebox" (0.82), "Phosphine" for "Phosphene" (0.89). A person's
# words over hands' are nothing like the words they cover.
_RESPELLED = 0.75


class Whisper:
    """Whisper large-v3-turbo on MLX: the clip owner's transcriber."""

    def __init__(self, *, clock: Clock, prompt: Callable[[], Awaitable[str | None]], record: Record) -> None:
        # [LAW:nothing-unseen] the load is a unit of work of its own: how long the start waited on it, and on what model.
        with loads.loading(clock, "whisper.loaded", record):
            transcription.load()
            # Silero, which says how sure it is that anyone spoke in a segment; run on the model's thread, after it.
            self._silero = SileroOnnxModel(str(_SILERO))
            annotate(model=transcription.MODEL, revision=transcription.REVISION)
        # The vocabulary each clip is transcribed with, read as it is: see hands.voice.vocabulary.
        self._prompt = prompt
        self._record = record
        # [LAW:single-enforcer] the one thread the model runs on: a transcription given up on is still running, and the
        # next must not run beside it.
        self._model = SerialThread("Whisper")

    async def transcribed(self, clip: Clip, sound: Sound) -> str | None:
        """The words a person said in a clip of the key's, primed with the vocabulary as it is now; recorded, so "I spoke
        and nothing happened" can be looked into."""
        heard = await self._heard(clip.number, sound, await self._prompt())
        self._record(heard)
        return heard.said

    async def _heard(self, clip: int, sound: Sound, prompt: str | None) -> ClipHeard:
        """What a person said in a clip's audio, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, the
        vocabulary.

        A segment with no word in it is dropped, as are one the decoder repeated itself in, one Whisper only guessed at,
        and one heard where Silero heard nobody speak: that is what Whisper makes of silence and noise. Of the rest, the
        words hands' own sentences say (`_theirs`) are hands' echo, and are dropped too. A clip with nothing said in it
        comes back with no segment.
        """
        # Set once nobody waits for this clip's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()
        try:
            # Off the loop: the model runs for a third of a second a clip, and the speaker and the key go on meanwhile.
            scored = await self._model.run(lambda: [] if given_up.is_set() else self._scored(decoded(sound.heard + _TRAILING_SILENCE), prompt))
        finally:
            given_up.set()
        sentences = said(sound)
        spoken: list[str] = []
        dropped: list[Unsaid] = []
        echo: list[str] = []
        for segment, unsaid in scored:
            worded = any(character.isalnum() for character in segment.text)
            if worded and unsaid.voiced >= _VOICED and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                theirs, hands = _theirs(segment.text, _played_over(sentences, segment))
                spoken += [theirs] if theirs else []
                echo += [hands] if hands else []
            else:
                dropped.append(unsaid)
        return ClipHeard(clip, " ".join(spoken).strip() or None, tuple(dropped), tuple(echo))

    def _scored(self, samples: np.ndarray, prompt: str | None) -> list[tuple[transcription.Segment, Unsaid]]:
        """Each segment Whisper hears in `samples`, primed with `prompt`, with the scores it is dropped for, Silero's over
        the stretch it heard it in. It blocks while the models run."""
        return [
            (segment, Unsaid(segment.text, segment.compression_ratio, segment.avg_logprob, self._voiced(samples[round(segment.start * RATE) : round(segment.end * RATE)])))
            for segment in transcription.segments(samples, prompt)
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


def _played_over(sentences: Sequence[tuple[Sentence, ...]], segment: transcription.Segment) -> tuple[Sentence, ...]:
    """Each sentence hands played over a segment, or in the `_ECHO_SECONDS` before it, from `sentences`, those of each
    10 ms of the clip; in the order first played."""
    start, end = max(0, math.floor((segment.start - _ECHO_SECONDS) * 100)), math.ceil(segment.end * 100)
    return tuple(dict.fromkeys(sentence for slot in sentences[start:end] for sentence in slot))


def _words(text: str) -> list[tuple[int, str]]:
    """Each word of `text` to match, with the place of the whitespace-separated token it is in: lower case, its letters,
    digits and apostrophes, so "Hold," and "hold" are one word, "houdini-mcp" two, as "Houdini MCP" is, and a dash none."""
    return [(at, word.lower()) for at, token in enumerate(text.split()) for word in re.findall(r"[\w']+", token)]


def _theirs(text: str, played: tuple[Sentence, ...]) -> tuple[str, str]:
    """`text`, a segment Whisper heard, split into what a person said in it and what was hands' own echo.

    [LAW:single-enforcer] the one rule of which words are hands'. hands knows what it said over the segment, `played`,
    so its words are those of the segment that say it: every run of `_RUN` or more in a row, in order, that the sentences
    say, or one that is all the sentences say, as a one-word "Done." is. A word the alignment of the two puts in place of
    one of hands' says it where it is spelled as Whisper spells what it heard (`_RESPELLED`). What is left is a person's,
    so a person speaking over the tail of hands' sentence keeps their words, whichever speaker the diarizer took either
    for."""
    tokens = text.split()
    words = _words(text)
    owners, heard = [owner for owner, _ in words], [word for _, word in words]
    spoken = [word for _, word in _words(" ".join(sentence.text for sentence in played))]
    # [LAW:domain-language] the alignment word error rate is counted on: words equal, substituted, inserted or deleted.
    saying = {
        at
        for kind, start, end, said_from, said_to in SequenceMatcher(None, heard, spoken, autojunk=False).get_opcodes()
        if kind == "equal" or kind == "replace" and SequenceMatcher(None, " ".join(heard[start:end]), " ".join(spoken[said_from:said_to])).ratio() >= _RESPELLED
        for at in range(start, end)
    }
    stretches = [list(stretch) for says, stretch in groupby(range(len(heard)), key=saying.__contains__) if says]
    hands = {at for stretch in stretches if len(stretch) >= min(_RUN, len(spoken)) for at in stretch}
    # A token is hands' where every word in it is; punctuation alone is nobody's, and goes with the person's.
    echo = {owner for owner in owners if all(at in hands for at, other in enumerate(owners) if other == owner)}
    theirs = " ".join(token for at, token in enumerate(tokens) if at not in echo)
    return theirs if _words(theirs) else "", " ".join(tokens[at] for at in sorted(echo))
