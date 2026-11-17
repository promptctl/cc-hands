"""Whisper, the transcriber on the clip owner (`hands.voice.clipowner`): it transcribes each clip the key sent, whole, as it
ends (hands.voice.transcription), and those are the words a turn sends, none of them hands' own. A clip still open is
heard by the streaming recognizer (`hands.voice.streaming`), whose words only cut.

hands' own words are known by their text and their time: each word Whisper heard is hands' where it is a word
hands played near when it was heard. A Whisper segment is no voice boundary: primed with hands' vocabulary, Whisper
heard a person's lines and hands' sentences in one segment (tests/fixtures/hands_and_another.flac, 8.2 s to 24 s), so
only a word, timed, is one voice's. Text and time alone cannot tell a person repeating hands' words from hands' echo,
so a word of hands' counts only where hands' voice is in what Whisper hears: its echo reached the microphone, and the
echo canceller left it in.
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
from hands.voice.clips import RATE, SECOND, SLOT, Clip, Play, Sound, decoded, plays, track

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

# How far from when Whisper heard a word, in seconds, the word of hands' it is may have played: no time is known for a
# word of a sentence played, so it is placed by its characters (`_placed`).
_ECHO_SECONDS = 1.5

# How many characters of a sentence hands' TTS says a second, Pocket TTS's speaking rate: on
# tests/fixtures/hands_echo_uncancelled.flac, every word of hands' Whisper heard was within 0.8 s of its characters at
# this rate from its sentence's start, later where the sentence pauses at its punctuation.
_CHARACTERS_PER_SECOND = 16.0

# The echo return loss enhancement (ERLE), in dB, at which the echo canceller is taking hands' voice out: how far below
# what the microphone captured it brings what Whisper hears. On tests/fixtures/hands_cancelled_and_another.flac, through
# hands' canceller at the desk, 29 to 47 dB in every 0.2 s hands spoke alone, and 0 to 10 dB where a second voice spoke
# over it; heard through no canceller, as on the uncancelled recordings, 0.
_CANCELLING = 20.0

# The stretch, in 10 ms slots, the canceller's ERLE is taken over: 0.2 s, as `_CANCELLING` was measured. A single slot's
# is too unsteady to read, its power down to a phoneme.
_ERLE_SLOTS = 20

# How far above chance the echo path's peak is, at least, where hands' voice reached the microphone: the GCC-PHAT of what
# the microphone captured against what the speaker played (`_echo_path`). Over the stretch hands played, at the desk with
# its speaker on (tests/fixtures/hands_cancelled_and_another.flac) 58, and in every 1 s of it 16 to 58; with hands'
# voice in headphones (hands_no_echo_and_another.flac) 4.8, chance, and in no 0.5 s of it above 13.8.
_ECHO_PATH = 15.0

# The band the echo path is looked for in, in Hz, where speech carries: above 4 kHz, two voices of hands' own TTS
# correlate with nobody's echo between them, 12.5 over 100 Hz to 7 kHz with no echo at all.
_ECHO_BAND = (200.0, 4000.0)

# How far apart in time, in seconds, the speaker's reference and its echo are looked for, either way: 68 ms at the
# desk, and the reference can reach the canceller after its echo (hands.voice.echo).
_ECHO_LAG = 0.5

# How far either side of where a word of hands' is placed, in seconds, the canceller is looked for taking hands' voice
# out: a word is placed up to 0.8 s from where it played (`_CHARACTERS_PER_SECOND`), and while a person speaks over
# hands only the slots between their words show the canceller at work, as the person's own words come through it at
# 0 to 10 dB.
_CANCELLED_SECONDS = 1.0

# The least likeness of a word Whisper heard to a word hands said, by difflib's ratio, at which it is that word spelled
# Whisper's own way: "diariser" for "diarizer" and "maclbox" for "macklebox" are 0.875, while "hear" and "here", two
# words, are 0.75.
_RESPELLED = 0.8

# The fewest letters of a word Whisper spells its own way: in a shorter one, a letter off is another word ("they" for
# "the", "its" for "it", "your" for "you").
_SPELLED = 5

# A word, as Whisper and hands' TTS both write one: letters and digits, with an apostrophe inside ("won't"); an
# underscore is no letter, so "auth_middleware" is two words and "_" none.
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")

# Silence after a clip's last sample: a segment that ends right where the speech stopped has its last word dropped or
# garbled, so Whisper hears half a second of it past the end, as Pipecat padded every segment it gave a transcriber.
_TRAILING_SILENCE = bytes(SECOND // 2)

# Words hands said over a clip, in lower case, each with when it played, in seconds from the clip's start.
type Placed = tuple[tuple[str, float], ...]


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
        and one heard where Silero heard nobody speak: that is what Whisper makes of silence and noise. Of the rest, each
        word that is hands' own voice come back (`_parted`) goes too. A clip with nothing said in it comes back with no
        segment. A clip hands' voice was left in is decoded with its words timed, which one hands played nothing over, one
        no echo of hands' reached, or one the canceller took all of hands' voice out of, has no need of, and does not
        wait on.
        """
        played = _placed(plays(sound))
        # Set once nobody waits for this clip's transcription any more: the model is not run on it if it has yet to be.
        given_up = threading.Event()

        def scored_unless_given_up() -> tuple[float | None, Placed, list[tuple[transcription.Segment, float]]]:
            return (None, (), []) if given_up.is_set() else self._scored(sound, prompt, played)

        try:
            # Off the loop, the clip's decoding and its levels with it: the model runs for a third of a second a clip, and
            # the speaker and the key go on meanwhile.
            echo_path, placed, scored = await self._model.run(scored_unless_given_up)
        finally:
            given_up.set()
        kept: list[transcription.Segment] = []
        dropped: list[Unsaid] = []
        for segment, voiced in scored:
            if _words(segment.text) and voiced >= _VOICED and segment.compression_ratio <= _REPEATING and segment.avg_logprob >= _GUESSED:
                kept.append(segment)
            else:
                dropped.append(Unsaid(segment.text, segment.compression_ratio, segment.avg_logprob, voiced))
        said, echo = _parted(kept, placed)
        return ClipHeard(clip, " ".join(said) or None, tuple(dropped), echo, len(played) - len(placed), echo_path)

    def _scored(self, sound: Sound, prompt: str | None, played: Placed) -> tuple[float | None, Placed, list[tuple[transcription.Segment, float]]]:
        """The clip's echo path (`_echo_path`), the words of hands' `played` over it that are in what Whisper hears, and
        each segment Whisper hears in it, heard with `_TRAILING_SILENCE` after it, primed with `prompt`, its words timed
        where any of hands' were left, and how sure Silero is that anyone spoke in the stretch it was heard in. It blocks
        while the model runs.

        Where hands played and no echo of it reached the microphone, an echo path short of `_ECHO_PATH` (headphones, or
        the speaker muted), none of hands' words is in it, and Whisper hears the clip as the microphone captured it: the
        canceller had nothing to take out, and still ducks a person's voice under hands', 30 to 50 dB under "the build
        is" on tests/fixtures/hands_no_echo_and_another.flac, which Whisper then never hears. Elsewhere it hears the
        clip through the canceller, and of hands' words, those the canceller left in (`_left`)."""
        echo_path = _echo_path(sound)
        unechoed = echo_path is not None and echo_path < _ECHO_PATH
        heard, placed = (sound.captured, ()) if unechoed else (sound.heard, _left(played, sound))
        samples = decoded(heard + _TRAILING_SILENCE)
        segments = transcription.segments(samples, prompt, words=bool(placed))
        return echo_path, placed, [(segment, self._voiced(samples[round(segment.start * RATE) : round(segment.end * RATE)])) for segment in segments]

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


def _placed(plays: tuple[Play, ...]) -> Placed:
    """Each word of the sentences hands played over a clip, placed in its sentence by its characters, at
    `_CHARACTERS_PER_SECOND`, and in the clip by where the speaker was in the sentence as its play began, so a sentence
    begun before the clip opened has its first words before the clip's start. A word placed further than
    `_ECHO_SECONDS` past where its play stopped was never played over the clip, as the rest of a sentence a barge-in
    cut off, and is not placed."""
    return tuple(
        (word.group(), play.start * SLOT / RATE + into - play.into)
        for play in plays
        for word in _WORD.finditer(play.synthesis.text.lower())
        for into in ((word.start() + word.end()) / 2 / _CHARACTERS_PER_SECOND,)
        if into - play.into <= (play.stop - play.start) * SLOT / RATE + _ECHO_SECONDS
    )


def _echo_path(sound: Sound) -> float | None:
    """How far above chance the echo of what the speaker played over `sound` is in what the microphone captured: the
    GCC-PHAT of the two over the stretch hands played, in `_ECHO_BAND`, within `_ECHO_LAG` either way, its peak over its
    RMS. None where hands played nothing over it, or what it played is not known, as at the phone. The phase transform
    weighs every frequency alike, so the room's response, which spreads an echo over many lags, still leaves one peak."""
    reference = decoded(track(sound.played or ()))
    playing = np.flatnonzero(reference)
    if not playing.size:
        return None
    span = slice(playing[0], playing[-1] + 1)
    length = int(span.stop - span.start)
    # No further than the span reaches: a lag past it pairs no sample with another.
    lag = min(round(_ECHO_LAG * RATE), length - 1)
    # Twice the span, and the correlation is linear: no lag wraps round onto another.
    size = 1 << (2 * length - 1).bit_length()
    cross = np.fft.rfft(decoded(sound.captured)[span], size) * np.conj(np.fft.rfft(reference[span], size))
    frequency = np.fft.rfftfreq(size, 1 / RATE)
    band = (frequency >= _ECHO_BAND[0]) & (frequency <= _ECHO_BAND[1]) & (np.abs(cross) > 0)
    gcc = np.fft.irfft(np.where(band, cross / np.where(band, np.abs(cross), 1), 0), size)
    lags = np.concatenate((gcc[: lag + 1], gcc[size - lag :]))
    spread = float(np.sqrt(np.mean(lags**2)))
    # Captured in digital silence while hands played holds no echo of it at all: no higher above chance than nothing.
    return round(float(np.abs(lags).max()) / spread, 1) if spread else 0.0


def _left(placed: Placed, sound: Sound) -> Placed:
    """The words of hands' placed over `sound` that the echo canceller left in what Whisper hears: none placed within
    `_CANCELLED_SECONDS` of a 10 ms slot it took hands' voice out of, where, while hands played, what it heard over the
    `_ERLE_SLOTS` about it was more than `_CANCELLING` dB below what the microphone captured. A person's word heard there
    is theirs, whatever hands said."""
    slots = len(sound.heard) // (2 * SLOT)
    # The energy of each side over the `_ERLE_SLOTS` centred on each slot, as far as the clip reaches: a moving sum.
    about = np.arange(slots)
    # Summed in float64: a running sum in float32 over a minute of speech has no digits left for a quiet slot after it.
    totals = [np.concatenate(([0.0], np.cumsum((decoded(audio[: 2 * SLOT * slots]).reshape(slots, SLOT) ** 2).sum(axis=1), dtype=np.float64))) for audio in (sound.captured, sound.heard)]
    energy = [total[np.minimum(about + _ERLE_SLOTS // 2, slots)] - total[np.maximum(about - _ERLE_SLOTS // 2, 0)] for total in totals]
    # At the phone, where what was played is not known, nothing was taken out, as nothing of hands' is placed.
    playing = np.array([reference is not None for reference in sound.played or (None,) * slots][:slots], dtype=bool)
    # [LAW:types-are-the-program] a ratio of powers, not of levels: digital silence heard has no level (clips.dbfs), and
    # where both sides are silent nothing was taken out.
    taking = playing & (energy[0] > energy[1] * 10 ** (_CANCELLING / 10))
    reach = round(_CANCELLED_SECONDS * RATE / SLOT)
    # Each word's slots, clamped to the clip's: a word placed before the clip opened looks only at what of its reach
    # the clip holds.
    return tuple((word, when) for word, when in placed if not taking[max(round(when * RATE / SLOT) - reach, 0) : max(round(when * RATE / SLOT) + reach + 1, 0)].any())


def _parted(segments: list[transcription.Segment], placed: Placed) -> tuple[list[str], tuple[str, ...]]:
    """The text of each segment as a person said it, none of a segment with no word of a person's left in it, and each
    word of hands' own echo in them, in order.

    [LAW:single-enforcer] the one rule of whose words are hands', of the words of hands' `placed`: those the canceller
    left in (`_left`), which is all of them heard through no canceller. Each word Whisper heard is hands' where each word
    of it takes a word of hands' not yet taken that it is, said or respelled, played within `_ECHO_SECONDS` of it, the
    nearest in time. A person's word that hands said within that time is taken for hands': by text and time, the two are
    the same. A segment none of whose words is hands' is sent as Whisper wrote it, as is every segment of a clip hands
    played nothing over, whose words were never timed."""
    free = placed
    said: list[str] = []
    echo: list[str] = []
    for segment in segments:
        kept: list[str] = []
        for word in segment.words:
            match _taking(_words(word.text), (word.start + word.end) / 2, free):
                case None:
                    kept.append(word.text)
                case left:
                    free = left
                    echo.append(word.text.strip())
        said.append(segment.text if len(kept) == len(segment.words) else "".join(kept).strip())
    return [text for text in said if _words(text)], tuple(echo)


def _taking(heard: list[str], at: float, free: Placed) -> Placed | None:
    """What is left of `free` once `heard`, one word Whisper heard at `at` s, is hands' own: each of its words takes the
    nearest in time of `free` that it is, played within `_ECHO_SECONDS` of `at`; None where any takes none, or it has
    none."""
    for each in heard:
        near = [index for index, (said, when) in enumerate(free) if abs(when - at) <= _ECHO_SECONDS and _respelled(each, said)]
        match min(near, key=lambda index: abs(free[index][1] - at), default=None):
            case None:
                return None
            case index:
                free = free[:index] + free[index + 1 :]
    return free if heard else None


def _respelled(heard: str, said: str) -> bool:
    """Whether a word Whisper `heard` is a word hands `said`, as said or, in a word long enough, spelled Whisper's own
    way."""
    return heard == said or min(len(heard), len(said)) >= _SPELLED and SequenceMatcher(None, heard, said, autojunk=False).ratio() >= _RESPELLED
