"""The rules of a turn the voice opened: how it ends, whoever owns it.

A turn the voice opens, the wake word's or engaged conversation's, ends by Smart Turn's verdict on the speech that stopped,
by the silence that runs on past it, by the limit, or, for a wake that nothing was asked after, by being thrown away. The
owner of the turn, the desk's phase at the Mac or the gate for anything that hears a person speak elsewhere, holds which
turn is open and steps it with what it is told. This module is the step, which is pure: a turn and what it was told make
the turn after it, or its end. Every end and every verdict, limit and ask names the turn it is for, so the owner applies
one only to that turn. The work around the models that goes with the step is `hands.voice.turnlife`.
"""

from dataclasses import dataclass
from typing import Literal, NewType

from hands.voice import shortcut
from hands.voice.shortcut import Instant

# How long after the wake word what is asked may take to start: past it, the wake is taken for a word misheard.
ASK_SECONDS = 8.0


@dataclass(frozen=True)
class SpeechStarting:
    """The detector heard what may be speech, and is making sure."""


@dataclass(frozen=True)
class SpeechStarted:
    """The detector is sure: the user is speaking, since `at`."""

    at: Instant


@dataclass(frozen=True)
class SpeechStopped:
    """The detector hears no speech: a start that came to nothing, or speech that has stopped."""


# A turn is named by a number its owner mints as it opens the turn, so what names one is for it and for no turn opened after
# it, however close in time.
TurnId = NewType("TurnId", int)


@dataclass(frozen=True)
class TurnEnded:
    """Turn `of` is over, `by` Smart Turn judging the speech complete, or by the silence after it running past
    `stop_secs`."""

    by: Literal["verdict", "silence"]
    of: TurnId


@dataclass(frozen=True)
class TurnTooLong:
    """`shortcut.TURN_LIMIT_SECONDS` have passed since turn `of` opened."""

    of: TurnId


@dataclass(frozen=True)
class NothingAsked:
    """`ASK_SECONDS` have passed since the wake word opened turn `of`."""

    of: TurnId


# What a turn the voice has open is told: the speech, its end, and the deadlines it set itself.
Told = SpeechStarting | SpeechStarted | SpeechStopped | TurnEnded | TurnTooLong | NothingAsked


@dataclass(frozen=True)
class Waiting:
    """The wake word opened turn `id`, at `since`, and what is asked has yet to start: a stop now is the end of the wake
    word, not of the turn."""

    id: TurnId
    since: Instant


@dataclass(frozen=True)
class Speaking:
    """Turn `id`, opened at `since`, with speech in it: said after the wake word, or opened by speech, engaged."""

    id: TurnId
    since: Instant


Turn = Waiting | Speaking

# How a turn ended: sent, sent as it stood at the limit, or thrown away.
How = Literal["stop", "timeout", "drop"]


@dataclass(frozen=True)
class Going:
    """The turn goes on as `turn`."""

    turn: Turn


@dataclass(frozen=True)
class Judging:
    """The speech in `turn` has stopped, and it goes on while Smart Turn is asked whether that ends it."""

    turn: Speaking


@dataclass(frozen=True)
class Ended:
    """Turn `of` is over, `how`."""

    how: How
    of: TurnId


def step(turn: Turn, told: Told) -> Going | Judging | Ended:
    """The turn after `told`, or how it ended."""
    match turn, told:
        case Waiting(id=id, since=since), SpeechStarted():
            return Going(Speaking(id, since))
        # [LAW:no-ambient-temporal-coupling] a wake that nothing asked after, a word misheard in a quiet room, is thrown
        # away, counted only for the wake it was set for; a limit and a verdict count only for the turn they name.
        case Waiting(id=id), NothingAsked(of=of) if of == id:
            return Ended("drop", id)
        case Speaking() as speaking, SpeechStopped():
            return Judging(speaking)
        case Speaking(id=id), TurnEnded(of=of) if of == id:
            return Ended("stop", id)
        case Waiting(id=id) | Speaking(id=id), TurnTooLong(of=of) if of == id:
            return Ended("timeout", id)
        # [LAW:types-are-the-program] every other pair is named, so a new turn or a new thing told is a pair the checker
        # reports until it is decided. The pause after the wake word is no end of the turn: Smart Turn judges the wake
        # word complete, so only once what is asked has started does a stop go to it. A verdict, a limit or an ask that
        # names a turn that is not this one leaves this one as it is.
        case Waiting(), SpeechStarting() | SpeechStopped() | TurnEnded() | TurnTooLong() | NothingAsked():
            return Going(turn)
        case Speaking(), SpeechStarting() | SpeechStarted() | TurnEnded() | TurnTooLong() | NothingAsked():
            return Going(turn)


def opened(was: Turn | None, now: Turn | None) -> Turn | None:
    """The turn that has just opened as the owner's turn went from `was` to `now`: not one that went on, a wake's from
    waiting to speaking."""
    return None if now is None or (was is not None and was.id == now.id) else now


def closed(was: Turn | None, now: Turn | None) -> Turn | None:
    """The turn that has just closed as the owner's turn went from `was` to `now`, however it came to: ended by its own
    rules, or taken by its owner's gate."""
    return None if was is None or (now is not None and now.id == was.id) else was


def deadlines(turn: Turn) -> tuple[tuple[Instant, TurnTooLong | NothingAsked], ...]:
    """What a turn that has just opened asks to be told later: once it has been open `TURN_LIMIT_SECONDS`, and a wake's
    once `ASK_SECONDS` have passed with nothing asked."""
    # [LAW:one-source-of-truth] the limit is read from the shortcut's own name at each use, the one number a held key's
    # turn and a voice turn share.
    limit = (turn.since + shortcut.TURN_LIMIT_SECONDS, TurnTooLong(turn.id))
    match turn:
        case Waiting(id=id, since=since):
            return limit, (since + ASK_SECONDS, NothingAsked(id))
        case Speaking():
            return (limit,)


def asked(was: Turn | None, now: Turn | None) -> tuple[tuple[Instant, TurnTooLong | NothingAsked], ...]:
    """What the owner's turn going from `was` to `now` asks to be told later: the deadlines of the turn it opened, if it did."""
    return () if (turn := opened(was, now)) is None else deadlines(turn)
