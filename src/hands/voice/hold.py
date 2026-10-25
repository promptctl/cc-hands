"""The talk key: Right Shift held by itself. A pure machine from what the keyboard did to what the turn does.

Right Shift is also Shift, so the key cannot mean talk the moment it goes down. It means talk once it has been held
alone for `HOLD_SECONDS`; a quick tap is nothing, and a key pressed while it is held is typing: no turn starts, and a
turn already started is dropped, never sent. Right Shift itself is never swallowed, so Shift keeps working as Shift.
The microphone opens on the press all the same, so words said before the hold means talk are kept: they reach a turn
only if the hold opens one, and are thrown away if it turns out to be Shift. A turn still open `TURN_LIMIT_SECONDS`
after its press is sent as it stands: a key stuck down, or a release macOS never reported, would otherwise record the room
forever, and what the turn recorded is the user's all the same, so it reaches the brain rather than being thrown away.
"""

from dataclasses import dataclass
from typing import Literal

# 0.3 s opened turns while typing: a rest on Right Shift before a capital, which Whisper filled with "you" or "Okay.".
HOLD_SECONDS = 0.6
# Ten minutes: dictation runs past two (a 120 s limit threw the user's away, promptctl-voice-je8.d20), and Whisper hears
# ten minutes in about 17 s (TRANSCRIBING_SECONDS bounds it).
TURN_LIMIT_SECONDS = 600.0

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
    """`HOLD_SECONDS` have passed since the press made at `pressed_at`, whatever happened in between."""

    pressed_at: Instant


@dataclass(frozen=True)
class Overlong:
    """`TURN_LIMIT_SECONDS` have passed since the press made at `pressed_at`, whatever happened in between."""

    pressed_at: Instant


KeyEvent = Pressed | Released | Typed | Ripe | Overlong


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


Hold = Idle | Arming | Talking | Typing

# What an edge does: opens the microphone on a press, closes it on a press that was Shift after all, and to the turn:
# opens it, closes it and sends it, closes it and throws it away, or closes it and sends it for being open too long. The
# hold makes only those; engaged conversation (`hands.voice.engaged`) also has the desk listen between turns, and stop.
TurnMove = Literal["arm", "disarm", "start", "stop", "drop", "timeout"]
Move = TurnMove | Literal["listen", "deafen"]


def asked(event: KeyEvent) -> tuple[tuple[Instant, Ripe | Overlong], ...]:
    """What `event` asks to be told later, and at what instant: a press, once it has been held `HOLD_SECONDS` and once
    `TURN_LIMIT_SECONDS`, counted from the press itself."""
    match event:
        case Pressed(at=at):
            return (at + HOLD_SECONDS, Ripe(at)), (at + TURN_LIMIT_SECONDS, Overlong(at))
        case _:
            return ()


def step(hold: Hold, event: KeyEvent) -> tuple[Hold, tuple[Move, ...]]:
    """The hold after `event`, and what it does: most keystrokes do nothing."""
    match hold, event:
        # A press is a fresh start whatever came before: one that finds the key already down proves a release went by
        # unseen, while macOS had the tap switched off, and a turn still open is sent, as that release would have sent it.
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
        case Talking(), Typed():
            return Typing(), ("drop",)
        case Talking(), Released():
            return Idle(), ("stop",)
        case Arming(), Released():
            return Idle(), ("disarm",)
        case Typing(), Released():
            return Idle(), ()
        case _:
            return hold, ()
