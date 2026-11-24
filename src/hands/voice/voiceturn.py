"""The life of a turn the voice opened: how it ends, whoever runs it.

A turn the voice opens, the wake word's or engaged conversation's, ends by Smart Turn's verdict on the speech that stopped,
by the silence that runs on past it, by the limit, or, for a wake that nothing was asked after, by being thrown away. The
desk (`hands.voice.desk`) runs this life on its own ears, and so does anything else that hears a person speak.

This module is the step, which is pure: a turn and what was heard make the turn after it, or its end. The work around the
models that goes with it, hearing afresh, feeding Smart Turn, judging, clearing as a turn ends, is `Life`, and the owner of
the one turn open for a runner with no desk is `Runner`.
"""

from dataclasses import dataclass
from typing import Literal

from hands.voice import shortcut

Instant = shortcut.Instant

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


@dataclass(frozen=True)
class TurnEnded:
    """The turn opened at `of` is over, `by` Smart Turn judging the speech complete, or by the silence after it running
    past `stop_secs`."""

    by: Literal["verdict", "silence"]
    of: Instant


@dataclass(frozen=True)
class TurnTooLong:
    """`shortcut.TURN_LIMIT_SECONDS` have passed since the turn opened at `opened_at`."""

    opened_at: Instant


@dataclass(frozen=True)
class NothingAsked:
    """`ASK_SECONDS` have passed since the wake word opened the turn at `woke_at`."""

    woke_at: Instant


# What a turn the voice has open is told: the speech, its end, and the deadlines it set itself.
Told = SpeechStarted | SpeechStopped | TurnEnded | TurnTooLong | NothingAsked


@dataclass(frozen=True)
class Waiting:
    """The wake word opened the turn, at `since`, and what is asked has yet to start: a stop now is the end of the wake
    word, not of the turn."""

    since: Instant


@dataclass(frozen=True)
class Speaking:
    """A turn with speech in it, since `since`: said after the wake word, or opened by speech, engaged."""

    since: Instant


Turn = Waiting | Speaking

# How a turn ended: sent, sent as it stood at the limit, or thrown away.
How = Literal["stop", "timeout", "drop"]


@dataclass(frozen=True)
class Going:
    """The turn goes on as `turn`, and `judge` says Smart Turn is to be asked whether the speech that stopped ends it."""

    turn: Turn
    judge: bool = False


@dataclass(frozen=True)
class Ended:
    """The turn is over, `how`."""

    how: How


def step(turn: Turn, told: Told) -> Going | Ended:
    """The turn after `told`, or how it ended."""
    match turn, told:
        case Waiting(since=since), SpeechStarted():
            return Going(Speaking(since))
        # [LAW:types-are-the-program] the pause after the wake word is no end of the turn: Smart Turn judges the wake word
        # complete, so only once what is asked has started does a stop go to it.
        # [LAW:no-ambient-temporal-coupling] a wake that nothing asked after, a word misheard in a quiet room, is thrown
        # away, counted only for the wake it was set for; a limit and a verdict count only for the turn they name.
        case Waiting(since=since), NothingAsked(woke_at=woke_at) if woke_at == since:
            return Ended("drop")
        case Speaking(), SpeechStopped():
            return Going(turn, judge=True)
        case Speaking(since=since), TurnEnded(of=of) if of == since:
            return Ended("stop")
        case Waiting(since=since) | Speaking(since=since), TurnTooLong(opened_at=opened_at) if opened_at == since:
            return Ended("timeout")
        case _:
            return Going(turn)


def opened(was: Turn | None, now: Turn | None) -> Turn | None:
    """The turn that has just opened as the turn went from `was` to `now`: not one that went on, a wake's from waiting to
    speaking."""
    return None if now is None or (was is not None and was.since == now.since) else now


def deadlines(turn: Turn) -> tuple[tuple[Instant, TurnTooLong | NothingAsked], ...]:
    """What a turn that has just opened asks to be told later: once it has been open `TURN_LIMIT_SECONDS`, and a wake's
    once `ASK_SECONDS` have passed with nothing asked."""
    # [LAW:one-source-of-truth] the limit is read from the shortcut's own name at each use, the one number a held key's
    # turn and a voice turn share.
    limit = (turn.since + shortcut.TURN_LIMIT_SECONDS, TurnTooLong(turn.since))
    match turn:
        case Waiting(since=since):
            return limit, (since + ASK_SECONDS, NothingAsked(since))
        case Speaking():
            return (limit,)
