"""An echo-cancelling chain scored on recorded takes by two numbers: hands' words that leak, and a person's words that are
lost (promptctl-voice-i15t.65f.1w5). Every ticket of promptctl-voice-i15t.65f picks or proves a chain by these.

A take is what scripts/record-double-talk.py recorded: what the microphone captured, the reference hands played, which of
hands' sentences played in each 10 ms, and the person's lines as said, each with when it began. A chain is the function
from a take to the sound the recognizer hears; the recognizer is the real pair, and what it decoded is dated by mlx-audio's
own alignment (`listening.Timed`). The score is a measurement and not product code, so it may judge whose a word is by its
text and its time, which the product may not.

What a score counts, over one take:

- lost: a word of the person's lines the recognizer did not decode as said, among the words dated from the first line's
  start on. Misheard counts as lost: the person's words not decoded. A word the person and hands say at once, as the
  scripts have the person do, is credited to the person: text and time cannot tell the two voices apart there.
- leaked: any other decoded word, dated where hands played in the stretch it was spoken in. mlx-audio dates a word late by
  up to a chunk (`LATE`), never early, so a word dated before the person's first line began is none of theirs.
- stray: any other decoded word, with hands silent: the room.
"""

import difflib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import echoes
import numpy as np
from listening import Live, tapped

from hands.voice.clips import RATE, SLOT, Playhead, decoded
from hands.voice.streaming import CHUNK, MEL_HOP, Said

# How late the pair dates a word, at most: the chunk the encoder was pushed. A word spoken at t is dated in [t, t + LATE].
LATE = CHUNK * MEL_HOP / RATE


@dataclass(frozen=True)
class Line:
    """A line the person said, `text` as the TTS was given it, from `start` seconds into the take."""

    text: str
    start: float


@dataclass(frozen=True)
class Take:
    """One recording: `captured` and `reference` mono 16-bit samples, as the canceller is handed them, the playheads of
    hands' sentences over each 10 ms of them, and the person's lines in the order said."""

    name: str
    captured: np.ndarray
    reference: np.ndarray
    playheads: tuple[tuple[Playhead, ...], ...]
    person: tuple[Line, ...]

    def played(self, first: float, last: float) -> bool:
        """Whether hands played anything in the stretch from `first` to `last` seconds into the take."""
        slots = self.playheads[max(0, int(first * RATE / SLOT)) : int(last * RATE / SLOT) + 1]
        return any(slots)

    @property
    def cut_in(self) -> float:
        """When the person first speaks, in seconds: infinity for a take where they never do."""
        return min((line.start for line in self.person), default=float("inf"))


def recorded(fixture: Path) -> Take:
    """The take of scripts/record-double-talk.py at `fixture`, as it was recorded."""
    captured, reference, playheads = echoes.take(fixture)
    lines = json.loads(fixture.with_suffix(".json").read_text())["other"]
    return Take(fixture.name, captured, reference, playheads, tuple(Line(line["text"], line["start"]) for line in lines))


# The takes recorded so far, each with hands out of the speaker as at the desk and out of none as in headphones.
TAKES = (echoes.CANCELLED, echoes.NO_ECHO, echoes.OPENING, echoes.NO_ECHO_OPENING)

# A chain: what the recognizer hears of a take, as mono samples in [-1, 1) from the take's start.
type Chain = Callable[[Take], np.ndarray]


def aec3(take: Take) -> np.ndarray:
    """hands' own canceller (hands.voice.echo), as the desk runs it."""
    return decoded(echoes.replayed(take.captured, take.reference, take.playheads).heard)


def raw(take: Take) -> np.ndarray:
    """No canceller: the microphone as it captured, the control a chain's leak is read against."""
    return decoded(take.captured.tobytes())


CHAINS: dict[str, Chain] = {"aec3": aec3, "raw": raw}


@dataclass(frozen=True)
class Word:
    """A word the recognizer decoded, lower case with no punctuation, dated in seconds into the take."""

    text: str
    start: float


def normal(text: str) -> list[str]:
    """The words in `text`, as they are scored: lower case, letters, digits and apostrophes only."""
    return re.findall(r"[\w']+", text.lower())


def words_of(tokens: Iterable[tuple[int, float, str]], closing: Iterable[Said], closed_at: float) -> tuple[Word, ...]:
    """The words the pair's tokens spell, in order. A token that begins with a space begins a word of its speaker's and any
    other continues it, so a word is dated by its first piece; the words decoded as the clip closed carry no date of their
    own and are dated `closed_at`."""
    spelled = list[tuple[float, str]]()
    last = dict[int, int]()
    for speaker, start, piece in tokens:
        if piece[:1].isspace() or speaker not in last:
            last[speaker] = len(spelled)
            spelled.append((start, piece))
        else:
            spelled[last[speaker]] = (spelled[last[speaker]][0], spelled[last[speaker]][1] + piece)
    dated = [(start, text) for start, spelling in spelled for text in normal(spelling)]
    return tuple(Word(text, start) for start, text in sorted(dated + [(closed_at, text) for said in closing for text in normal(said.text)], key=lambda pair: pair[0]))


def heard_words(pair: Any, samples: np.ndarray) -> tuple[Word, ...]:
    """What the pair, begun afresh, decodes from `samples` fed as one clip, as hands' listener feeds a clip, each word dated."""
    pair.reset()
    with tapped(pair) as tap:
        _, ended = Live(pair).spoken(samples.astype(np.float32))
    closed_at = max((start for _, start, _ in tap.tokens), default=ended.seconds)
    return words_of(tap.tokens, ended.closed.words, closed_at)


@dataclass(frozen=True)
class Score:
    """One take through one chain: the person's words not decoded, hands' words decoded, the room's, and how many words the
    person said, with how many dB the chain took off where hands played alone."""

    take: str
    said: int
    lost: tuple[str, ...]
    leaked: tuple[Word, ...]
    stray: tuple[Word, ...]
    erle: float | None


def erle(take: Take, cleaned: np.ndarray) -> float | None:
    """The echo return loss enhancement, in dB, over the 10 ms slots before the person first speaks where hands played: the
    captured sound's energy over the cleaned sound's. None where hands played nowhere there to be measured over."""
    slots = [slot for slot, playing in enumerate(take.playheads) if playing and (slot + 1) * SLOT / RATE <= take.cut_in and (slot + 1) * SLOT <= len(cleaned)]
    if not slots:
        return None

    def energy(samples: np.ndarray) -> float:
        return sum(float(np.sum(np.square(samples[slot * SLOT : (slot + 1) * SLOT], dtype=np.float64))) for slot in slots)

    return float(10 * np.log10(energy(decoded(take.captured.tobytes())) / max(energy(cleaned), 1e-20)))


def judged(take: Take, heard: tuple[Word, ...]) -> tuple[tuple[str, ...], tuple[Word, ...], tuple[Word, ...]]:
    """What `heard` of a take leaves of the person's words lost, hands' leaked, and the room's stray."""
    mine = [word for word in heard if word.start >= take.cut_in]
    said = [word for line in take.person for word in normal(line.text)]
    lost, others = list[str](), [word for word in heard if word.start < take.cut_in]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, said, [word.text for word in mine], autojunk=False).get_opcodes():
        # A word heard in place of one said is that one misheard: lost, and none of hands'. What it is longer by is not theirs.
        lost += said[i1:i2] if tag != "equal" else []
        others += mine[j1 + (i2 - i1) if tag == "replace" else j1 : j2] if tag in ("replace", "insert") else []
    leaked = tuple(word for word in others if take.played(word.start - LATE, word.start))
    return tuple(lost), leaked, tuple(word for word in others if word not in leaked)


def scored(take: Take, cleaned: np.ndarray, heard: tuple[Word, ...]) -> Score:
    """A take's `heard` words, as the recognizer decoded them from `cleaned`, against what was said and played."""
    lost, leaked, stray = judged(take, heard)
    return Score(take.name, sum(len(normal(line.text)) for line in take.person), lost, leaked, stray, erle(take, cleaned))


def summary(score: Score) -> str:
    """One take's numbers on one line, and the words behind them."""
    level = "n/a" if score.erle is None else f"{score.erle:5.1f} dB"
    return (
        f"{score.take:36} leaked {len(score.leaked):2}  lost {len(score.lost):2} of {score.said:2}  stray {len(score.stray):2}  ERLE {level}"
        f"   leaked: {' '.join(word.text for word in score.leaked) or '-'};  lost: {' '.join(score.lost) or '-'}"
    )
