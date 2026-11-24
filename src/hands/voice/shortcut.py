"""The talk shortcut: Right Shift held by itself for `SHORTCUT_SECONDS`. A pure machine from what the keyboard did to what
the turn does.

Right Shift is also Shift, so the key cannot mean talk the moment it goes down. It means talk only once it has been held
alone that long; a quick tap is nothing, and a key pressed before then is typing: no turn starts. A turn once started
is the user's: a key, a click or a scroll while it is open leaves it open, and only its release, or the limit, ends it.
Right Shift itself is never swallowed, so Shift keeps working as Shift.
The microphone opens on the press all the same, so words said before the shortcut means talk are kept: they reach a turn
only if the shortcut opens one, and are thrown away if it turns out to be Shift. A turn still open `TURN_LIMIT_SECONDS`
after its press is sent as it stands: a key stuck down, or a release macOS never reported, would otherwise record the room
forever, and what the turn recorded is the user's all the same, so it reaches the brain rather than being thrown away.
"""

from dataclasses import dataclass
from typing import Literal

# 0.3 s opened turns while typing: a rest on Right Shift before a capital, which Whisper filled with "you" or "Okay.".
SHORTCUT_SECONDS = 0.6
# Five minutes, as the user asked (promptctl-voice-je8.rqr): dictation runs past two, and a turn the limit ends is sent.
TURN_LIMIT_SECONDS = 300.0

Instant = float  # seconds on the monotonic clock


@dataclass(frozen=True)
class Pressed:
    """Right Shift went down."""

    at: Instant


@dataclass(frozen=True)
class Released:
    """Right Shift went up."""


@dataclass(frozen=True)
class Typed:
    """Anything else the hand did: another key or modifier, a click, a scroll; or Right Shift pressed in a chord."""


@dataclass(frozen=True)
class Ripe:
    """`SHORTCUT_SECONDS` have passed since the press made at `pressed_at`, whatever happened in between."""

    pressed_at: Instant


@dataclass(frozen=True)
class Overlong:
    """`TURN_LIMIT_SECONDS` have passed since the press made at `pressed_at`, whatever happened in between."""

    pressed_at: Instant


@dataclass(frozen=True)
class CommandPressed:
    """Right Command went down by itself: to the shortcut, a modifier like any other; counted, a tap (`hands.voice.taps`)."""

    at: Instant


@dataclass(frozen=True)
class CommandReleased:
    """Right Command went up: nothing to the shortcut, which its press has already made Shift, and no break in a tap's count."""


KeyEvent = Pressed | Released | Typed | Ripe | Overlong | CommandPressed | CommandReleased


@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Arming:
    """Right Shift is down alone, pressed at `since`, and not yet held long enough to mean talk."""

    since: Instant


@dataclass(frozen=True)
class Talking:
    """A turn is open, from the press made at `since`."""

    since: Instant


@dataclass(frozen=True)
class Typing:
    """Right Shift is down as Shift: its release ends nothing."""


Shortcut = Idle | Arming | Talking | Typing

# What an edge of the shortcut does: opens the microphone on a press, closes it on a press that was Shift after all, and
# to the turn: opens it, closes it and sends it, or closes it and sends it for being open too long. It throws no turn away.
ShortcutMove = Literal["arm", "disarm", "start", "stop", "timeout"]
# The moves of the gate that begin something: a press, an open clip, or a place's listening, and take them back.
Beginning = Literal["arm", "disarm", "start", "listen", "deafen"]
# The moves that end a clip: sent as it stands, thrown away, or sent for being open too long.
Ending = Literal["stop", "drop", "timeout"]
# Every move of the gate: the shortcut's, the wake word's throwing away of a wake that asked nothing, and the desk's
# (`hands.voice.desk`) having the microphone listen between turns, and stop.
Move = Beginning | Ending


def asked(event: KeyEvent) -> tuple[tuple[Instant, Ripe | Overlong], ...]:
    """What `event` asks to be told later, and at what instant: a press, once it has been held `SHORTCUT_SECONDS` and once
    `TURN_LIMIT_SECONDS`, counted from the press itself."""
    match event:
        case Pressed(at=at):
            return (at + SHORTCUT_SECONDS, Ripe(at)), (at + TURN_LIMIT_SECONDS, Overlong(at))
        case _:
            return ()


def step(shortcut: Shortcut, event: KeyEvent) -> tuple[Shortcut, tuple[ShortcutMove, ...]]:
    """The shortcut after `event`, and what it does: most keystrokes do nothing."""
    match shortcut, event:
        # A press is a fresh start whatever came before: one that finds the key already down proves a release went by
        # unseen, and a turn still open is sent, as that release would have sent it.
        case Talking(), Pressed(at=at):
            return Arming(at), ("stop", "arm")
        case _, Pressed(at=at):
            return Arming(at), ("arm",)
        # [LAW:no-ambient-temporal-coupling] a Ripe counts only for the press it was scheduled for: one left over
        # from an earlier press, released and pressed again since, names another instant and changes nothing.
        case Arming(since=since), Ripe(pressed_at=pressed_at) if pressed_at == since:
            return Talking(since), ("start",)
        # The key may still be down: its release, when it comes, ends nothing.
        case Talking(since=since), Overlong(pressed_at=pressed_at) if pressed_at == since:
            return Typing(), ("timeout",)
        case Arming(), Typed():
            return Typing(), ("disarm",)
        case Talking(), Released():
            return Idle(), ("stop",)
        case Arming(), Released():
            return Idle(), ("disarm",)
        case Typing(), Released():
            return Idle(), ()
        case _:
            return shortcut, ()
