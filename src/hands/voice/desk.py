"""The desk: how a turn of the user's is opened at the Mac, by the talk key at every level, and by the voice at two.

The desk listens at one of three levels. Off, the start's, the microphone is closed between turns. Under the wake word,
saying it opens a turn. Engaged, speech alone opens each turn. At every level Right Shift held alone opens a turn and its
release sends it (`hands.voice.hold`). A triple tap of Right Command toggles the wake word, and one of Right Shift
toggles engaged conversation (`hands.voice.taps`); either switches straight from the other, and the brain's tools choose
a level as a tap does. A turn the voice opened is closed by end-of-turn detection (`hands.voice.engaged`).

[LAW:types-are-the-program] the desk is one value whose variants carry the level, so whether it listens, which edge has
the turn, and whether the key is down are each said once: a key turn and a voice turn are never open together, and the
level never says the desk is deaf while a variant says it listens.
"""

from dataclasses import dataclass, replace
from typing import Literal

from hands.voice.hold import Arming as KeyArming, Idle, Instant, KeyEvent, Move, Talking as KeyTalking, TURN_LIMIT_SECONDS, Typing, step as hold_step
from hands.voice.holds import Edge
from hands.voice.taps import TapKey, Taps, counted

# The levels the desk listens at between turns; it is deaf under "off" alone.
Listens = Literal["wake word", "engaged"]
# One flat Literal, so a tool argument of it has a schema: "off" and every level of Listens.
Level = Literal["off", "wake word", "engaged"]
LEVELS: tuple[Level, ...] = ("off", "wake word", "engaged")

# Who chose a level: a triple tap, or the brain's tool, asked by voice.
Chooser = Literal["tap", "tool"]


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
    """The turn is over, `by` Smart Turn judging the speech complete, or by the silence after it running past
    `stop_secs`."""

    by: Literal["verdict", "silence"]


@dataclass(frozen=True)
class Woken:
    """The wake word was heard, at `at`."""

    at: Instant


@dataclass(frozen=True)
class TurnTooLong:
    """`TURN_LIMIT_SECONDS` have passed since the turn opened at `opened_at`."""

    opened_at: Instant


@dataclass(frozen=True)
class Chosen:
    """The brain's tool chose `level`."""

    level: Level


Heard = SpeechStarting | SpeechStarted | SpeechStopped | Woken | TurnEnded | TurnTooLong
Event = KeyEvent | Heard | Chosen


@dataclass(frozen=True)
class Resting:
    """No turn open and the key up: deaf under "off", listening under the other two."""

    level: Level


@dataclass(frozen=True)
class Starting:
    """Engaged, with what may be speech starting: the microphone is armed and the turn is not open."""


@dataclass(frozen=True)
class Woke:
    """The wake word opened a turn, since `since`, and what is asked has yet to start: a stop now is the end of the wake
    word, not of the turn."""

    since: Instant


@dataclass(frozen=True)
class Voiced:
    """A turn the voice opened, since `since`: said after the wake word, or opened by speech, engaged."""

    level: Listens
    since: Instant


KeyDown = KeyArming | KeyTalking | Typing


@dataclass(frozen=True)
class Keyed:
    """Right Shift is down, at `level`: arming, a turn of the key's open, or Shift typing."""

    level: Level
    hold: KeyDown


Phase = Resting | Starting | Woke | Voiced | Keyed


@dataclass(frozen=True)
class Desk:
    """Where the desk is, and the presses a triple tap has counted so far."""

    phase: Phase = Resting("off")
    taps: Taps = Taps()

    @property
    def level(self) -> Level:
        return level_of(self.phase)


def level_of(phase: Phase) -> Level:
    match phase:
        case Resting(level=level) | Voiced(level=level) | Keyed(level=level):
            return level
        case Starting():
            return "engaged"
        case Woke():
            return "wake word"


@dataclass(frozen=True)
class Made:
    """A move of the gate, and the edge that made it."""

    move: Move
    by: Edge


@dataclass(frozen=True)
class Leveled:
    """The level changed, from `was` to `now`, chosen `by` a tap or the tool."""

    was: Level
    now: Level
    by: Chooser


# What the desk does: a move of the gate, a change of level, or asking Smart Turn whether the speech that just stopped
# ends the turn.
Act = Made | Leveled | Literal["judge"]


def step(desk: Desk, event: Event) -> tuple[Desk, tuple[Act, ...]]:
    """The desk after `event`, and what it does."""
    match event:
        case Chosen(level=level):
            phase, acts = _leveled(desk.phase, level, "tool")
            return replace(desk, phase=phase), acts
        case SpeechStarting() | SpeechStarted() | SpeechStopped() | Woken() | TurnEnded() | TurnTooLong():
            phase, acts = _heard(desk.phase, event)
            return replace(desk, phase=phase), acts
        case _:
            taps, tapped = counted(desk.taps, event)
            phase, acts = _keyed(desk.phase, event)
            # [LAW:one-source-of-truth] a press held to a turn is no tap: the hold that opened it is the one judge.
            if any(act == Made("start", "held key") for act in acts):
                taps = Taps()
            if tapped is not None:
                phase, more = _leveled(phase, toggled(level_of(phase), tapped), "tap")
                acts = (*acts, *more)
            return Desk(phase, taps), acts


def toggled(level: Level, key: TapKey) -> Level:
    """The level a triple tap of `key` sets from `level`: its own, or off from its own."""
    own: Listens = "wake word" if key == "right command" else "engaged"
    return "off" if level == own else own


def _keyed(phase: Phase, event: KeyEvent) -> tuple[Phase, tuple[Act, ...]]:
    level = level_of(phase)
    hold, moves = hold_step(phase.hold if isinstance(phase, Keyed) else Idle(), event)
    # A press over what the voice has open takes the desk for the key: what the voice armed is disarmed, and a turn
    # it opened is sent, before the press arms.
    left: tuple[Act, ...]
    match phase:
        case Starting() if "arm" in moves:
            left = (Made("disarm", "engaged conversation"),)
        case Woke() | Voiced() if "arm" in moves:
            left = (Made("stop", _voice(phase)),)
        case Starting() | Woke() | Voiced():
            return phase, ()
        case Resting() | Keyed():
            left = ()
    made = tuple(Made(move, "held key") for move in moves)
    match hold:
        case Idle():
            return Resting(level), (*left, *made)
        case KeyArming() | KeyTalking() | Typing():
            return Keyed(level, hold), (*left, *made)


def _leveled(phase: Phase, now: Level, by: Chooser) -> tuple[Phase, tuple[Act, ...]]:
    was = level_of(phase)
    if now == was:
        return phase, (Leveled(was, now, by),)
    listening: tuple[Act, ...]
    match was, now:
        case "off", ("wake word" | "engaged") as heard:
            listening = (Made("listen", _voice_of(heard)),)
        case ("wake word" | "engaged") as heard, "off":
            listening = (Made("deafen", _voice_of(heard)),)
        case _:
            listening = ()
    match phase:
        case Resting():
            return Resting(now), (*listening, Leveled(was, now, by))
        # What the voice had open is the level's: leaving it, the armed start is disarmed and the turn is sent.
        case Starting():
            return Resting(now), (Made("disarm", "engaged conversation"), *listening, Leveled(was, now, by))
        case Woke() | Voiced():
            return Resting(now), (Made("stop", _voice(phase)), *listening, Leveled(was, now, by))
        # A turn of the key's goes on as it is, at the new level.
        case Keyed(hold=hold):
            return Keyed(now, hold), (*listening, Leveled(was, now, by))


def _heard(phase: Phase, heard: Heard) -> tuple[Phase, tuple[Act, ...]]:
    match phase, heard:
        case Resting(level="wake word"), Woken(at=at):
            return Woke(at), (Made("arm", "wake word"), Made("start", "wake word"))
        case Resting(level="engaged"), SpeechStarting():
            return Starting(), (Made("arm", "engaged conversation"),)
        case Resting(level="engaged"), SpeechStarted(at=at):
            return Voiced("engaged", at), (Made("arm", "engaged conversation"), Made("start", "engaged conversation"))
        case Starting(), SpeechStarted(at=at):
            return Voiced("engaged", at), (Made("start", "engaged conversation"),)
        case Starting(), SpeechStopped():
            return Resting("engaged"), (Made("disarm", "engaged conversation"),)
        # [LAW:types-are-the-program] the pause after the wake word is no end of the turn: Smart Turn judges the wake
        # word complete, so only once what is asked has started does a stop go to it.
        case Woke(since=since), SpeechStarted():
            return Voiced("wake word", since), ()
        case Woke() as turn, TurnTooLong():
            return _in_turn(turn, heard)
        case Voiced() as turn, SpeechStopped() | TurnEnded() | TurnTooLong():
            return _in_turn(turn, heard)
        case _:
            return phase, ()


def _in_turn(turn: Woke | Voiced, heard: SpeechStopped | TurnEnded | TurnTooLong) -> tuple[Phase, tuple[Act, ...]]:
    """How a turn the voice opened ends: Smart Turn's verdict on a stop, the silence after it, or the limit, which sends
    it as it stands."""
    rest = Resting(level_of(turn))
    match heard:
        case SpeechStopped():
            return turn, ("judge",)
        case TurnEnded():
            return rest, (Made("stop", _voice(turn)),)
        # [LAW:no-ambient-temporal-coupling] a limit counts only for the turn it was set for.
        case TurnTooLong(opened_at=opened_at) if opened_at == turn.since:
            return rest, (Made("timeout", _voice(turn)),)
        case TurnTooLong():
            return turn, ()


def _voice(turn: Woke | Voiced) -> Edge:
    """The edge that opened a turn the voice has open."""
    match turn:
        case Woke():
            return "wake word"
        case Voiced(level=level):
            return _voice_of(level)


def _voice_of(level: Listens) -> Edge:
    """The edge the desk's listening at `level` is: the wake word's or engaged conversation's."""
    match level:
        case "wake word":
            return "wake word"
        case "engaged":
            return "engaged conversation"


def deadline(was: Phase, now: Phase) -> tuple[tuple[Instant, TurnTooLong], ...]:
    """What the desk moving from `was` to `now` asks to be told later: a turn the voice just opened, once it has been
    open `TURN_LIMIT_SECONDS`. A turn of the key's has the hold's own limit."""
    match now, was:
        case Woke(since=since) | Voiced(since=since), Woke(since=opened) | Voiced(since=opened) if since == opened:
            return ()
        case Woke(since=since) | Voiced(since=since), _:
            return ((since + TURN_LIMIT_SECONDS, TurnTooLong(since)),)
        case _:
            return ()


def listens(level: Level) -> bool:
    """Whether the desk hears the room between turns at `level`."""
    return level != "off"
