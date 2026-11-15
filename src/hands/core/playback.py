"""Where playback is: the reading on the speaker, how far it has played, and where earlier readings were cut off.

The user cuts in often, to ask which file or to answer something else, so every interrupted reading must be resumable.
Where it stopped lives here, in the daemon, never in the model's memory: a model asked to go back paraphrases what it
remembers, and the user hears a different reading than the one they cut off.

A reading is the run of sentences handed to the speaker since it last fell quiet: a reply, a line said as written, or
both back to back. A position is what the TTS service hands the speaker as one: a sentence of a reply, or a line said as
written, whole. pocket-tts reports no word timings, so there is nothing finer, and "where it stopped" is the one that was
playing, said again from its start.

Each position knows whose it is: a sentence of the brain's own words, or a line hands says as written, which the brain
is told of by its label and never by its text. [LAW:types-are-the-program] whose a line is is its type, so nothing that
tells the brain of a line can quote one it did not write.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

# How many cut-off readings are kept to go back to. Each barge-in that is never resumed leaves one, so the stack is
# bounded by the user's own cutting in, and past this the oldest goes: nobody goes back nine readings by voice.
BOOKMARKS = 8


@dataclass(frozen=True)
class Own:
    """A sentence of the brain's own words, as the TTS service made it of the model's text."""

    text: str


@dataclass(frozen=True)
class Labelled:
    """A line hands says as written, and what the brain is told it by: `label`, never `text`, which the brain did not write."""

    text: str
    label: str


Line = Own | Labelled


def hands_says(text: str) -> Labelled:
    """A short line in hands' own words, with nothing in it to keep from the brain: told by what it says."""
    return Labelled(text, f'hands saying "{text}"')


def as_told(lines: Iterable[Line]) -> str:
    """The lines as the brain is told them: each run of its own words quoted, and each line hands said as written by its
    label. [LAW:single-enforcer] the one place a line is put into words for the brain."""
    parts: list[str] = []
    run: list[str] = []
    for line in lines:
        match line:
            case Own(text=text):
                run.append(text)
            case Labelled(label=label):
                parts.extend(_quoted(run))
                run = []
                parts.append(label)
    parts.extend(_quoted(run))
    return ", then ".join(parts)


def _quoted(run: list[str]) -> list[str]:
    # Words that are all space say nothing, so are told as nothing.
    words = " ".join(" ".join(text.split()) for text in run if text.strip())
    return [f'"{words}"'] if words else []


@dataclass(frozen=True)
class Bookmark:
    """A reading that was cut off, and the line that was playing when it was."""

    lines: tuple[Line, ...]
    at: int


@dataclass(frozen=True)
class Playback:
    """The latest reading, how many of its lines have finished playing, the lines of it hands has yet to hand
    the speaker, and the readings cut off.

    A reading every line of which has played is over, and the next line handed starts a new one. `pressed_on`
    is the reading the user's latest barge-in found, cut off or played to its end: what "say that again" says, whatever
    the model said before asking for it. `stopped` is that reading when the barge-in cut it off, kept apart from the
    earlier ones because going back is going back from it, to the reading before.
    """

    reading: tuple[Line, ...] = ()
    played: int = 0
    coming: tuple[Line, ...] = ()
    pressed_on: tuple[Line, ...] = ()
    stopped: Bookmark | None = None
    interrupted: tuple[Bookmark, ...] = ()

    @property
    def over(self) -> bool:
        return self.played == len(self.reading) and not self.coming

    @property
    def bookmarks(self) -> tuple[Bookmark, ...]:
        """Every reading cut off, most recent last."""
        return (*self.interrupted, *_held(self.stopped))


def begun(playback: Playback) -> Playback:
    """A new reading starts; the reading the latest barge-in cut off joins the earlier ones."""
    return Playback(pressed_on=playback.pressed_on, interrupted=playback.bookmarks[-BOOKMARKS:])


def queued(playback: Playback, line: Line) -> Playback:
    """Hands will hand the speaker this line, and knows it before it does: the rest of the reading playing, or a new
    one, as for each line handed."""
    start = begun(playback) if playback.over else playback
    return replace(start, coming=(*start.coming, line))


def handed(playback: Playback, line: Line) -> Playback:
    """A line is on its way to the speaker: the next of the reading playing, or the first of a new one, and no longer
    one hands has yet to hand. A word of the model's handed meanwhile takes its own place in the reading ahead of the
    lines still coming."""
    start = begun(playback) if playback.over else playback
    return replace(start, reading=(*start.reading, line), coming=_without(start.coming, line))


def dropped(playback: Playback, line: Line) -> Playback:
    """The speaker took up a line hands queued and says it as nothing, so will never hand it on: it is not coming."""
    return replace(playback, coming=_without(playback.coming, line))


def _without(coming: tuple[Line, ...], line: Line) -> tuple[Line, ...]:
    """`coming` with the first line equal to `line` taken off it."""
    at = coming.index(line) if line in coming else len(coming)
    return (*coming[:at], *coming[at + 1 :])


def finished(playback: Playback) -> Playback:
    """The next sentence of the reading has played to its end. The speaker plays them in the order they were handed, so
    the one that finished is the first that had not; one reported after its reading was cut off changes nothing."""
    return replace(playback, played=min(playback.played + 1, len(playback.reading)))


def barge_in(playback: Playback) -> Playback:
    """The user barged in: the reading stops, with what hands had yet to hand of it, and the sentence it stopped on is
    bookmarked to go back to. A reading that had played to its end is not cut off, and leaves nothing to go back to."""
    whole = (*playback.reading, *playback.coming)
    pressed = replace(playback, pressed_on=whole)
    return pressed if playback.over else replace(pressed, reading=whole, played=len(whole), coming=(), stopped=Bookmark(whole, playback.played))


@dataclass(frozen=True)
class Replay:
    """Lines to say again, in order, as they were said, each still whose it was."""

    lines: tuple[Line, ...]


@dataclass(frozen=True)
class NothingBargedIn:
    """No reading was cut off, so there is nowhere to go back to."""


@dataclass(frozen=True)
class NothingSaid:
    """Nothing has been said yet, so there is nothing to say again."""


@dataclass(frozen=True)
class LastOne:
    """The sentence skipped was the last of its reading, so nothing of it is left to say."""


Played = Replay | NothingBargedIn | NothingSaid | LastOne


def resume(playback: Playback) -> tuple[Playback, Played]:
    """"Go back to what you were talking about": the latest reading cut off before the one the user just cut in on, from
    the start of the sentence it stopped on. The one cut in on is what they are going back from, and is left; with none
    before it, it is the one gone back to."""
    return _popped(playback, playback.interrupted or _held(playback.stopped), 0)


def skip(playback: Playback) -> tuple[Playback, Played]:
    """"Skip that": the latest reading cut off, from the sentence after the one it stopped on."""
    return _popped(playback, playback.bookmarks, 1)


def repeat(playback: Playback) -> tuple[Playback, Played]:
    """"Say that again": the reading the user's latest barge-in found, whole, whether it was cut off or played to its end.
    Said whole, it no longer waits to be gone back to."""
    return replace(playback, stopped=None), Replay(playback.pressed_on) if playback.pressed_on else NothingSaid()


def _popped(playback: Playback, stack: tuple[Bookmark, ...], past: int) -> tuple[Playback, Played]:
    """The latest bookmark of `stack` taken off, the rest left as the readings cut off, and its reading from `past`
    lines after the one it stopped on."""
    if not stack:
        return playback, NothingBargedIn()
    *earlier, latest = stack
    rest = latest.lines[latest.at + past :]
    return replace(playback, stopped=None, interrupted=tuple(earlier)), Replay(rest) if rest else LastOne()


def _held(bookmark: Bookmark | None) -> tuple[Bookmark, ...]:
    return () if bookmark is None else (bookmark,)
