"""The desk: how a turn of the user's is opened at the Mac, by the talk key at every level, and by the voice at two.

The desk listens at one of three levels (`hands.voice.listening`). Off, the microphone is closed between turns; the desk
rests there until the run's start names the level it listens at, off on a cold start and, on a restart, the level the run
before it ended at. Under the wake word, saying it opens a turn. Engaged, speech alone opens each turn. At every level
Right Shift held alone opens a turn and its release sends it (`hands.voice.shortcut`). A triple tap of Right Command
toggles the wake word, and one of Right Shift toggles engaged conversation (`hands.voice.taps`); either switches straight
from the other, and the brain's tools and the pages' level buttons choose a level as a tap does. A turn the voice opened
is closed by end-of-turn detection, by the rules of any voice turn's end (`hands.voice.voiceturn`).

[LAW:types-are-the-program] the desk is one value whose variants carry the level, in a field or, for the two that exist at
one level alone (Starting, engaged; Woke, the wake word), in the variant itself, so whether it listens, which trigger has
the turn, and whether the key is down are each said once: a key turn and a voice turn are never open together, and the
level never says the desk is deaf while a variant says it listens.
"""

from dataclasses import dataclass, replace
from typing import Literal

from hands.voice import voiceturn
from hands.voice.clips import Trigger
from hands.voice.listening import Asker, Level, Listens
from hands.voice.shortcut import Arming as KeyArming, Idle, Instant, KeyEvent, Move, Talking as KeyTalking, Typing, step as shortcut_step
from hands.voice.taps import TapKey, Taps, counted
from hands.voice.voiceturn import Ended, Going, Judging, NothingAsked, SpeechStarted, SpeechStarting, SpeechStopped, Speaking, Told, Turn, TurnEnded, TurnId, TurnTooLong, Waiting

# Who chose a level: a triple tap, toggling it; the start, naming the level the run starts at; or whoever asked for it by
# name.
Chooser = Literal["tap", "start"] | Asker


@dataclass(frozen=True)
class Woken:
    """The wake word was heard, at `at`."""

    at: Instant


@dataclass(frozen=True)
class Chosen:
    """`by` chose `level`."""

    level: Level
    by: Literal["start"] | Asker


Heard = Woken | Told
Event = KeyEvent | Heard | Chosen


@dataclass(frozen=True)
class Resting:
    """No turn open and the key up: deaf under "off", listening under the other two."""

    level: Level


# Right Shift under a turn the voice has open: up, down and not yet held long enough to mean talk, or down as Shift.
VoiceKey = Idle | KeyArming | Typing


@dataclass(frozen=True)
class Starting:
    """Engaged, with what may be speech starting: the microphone is armed and the turn is not open."""

    key: VoiceKey = Idle()


@dataclass(frozen=True)
class Woke:
    """The wake word opened a turn, which waits for what is asked to start (`hands.voice.voiceturn`)."""

    turn: Waiting
    key: VoiceKey = Idle()


@dataclass(frozen=True)
class Voiced:
    """A turn the voice opened, with speech in it: said after the wake word, or opened by speech, engaged."""

    level: Listens
    turn: Speaking
    key: VoiceKey = Idle()


KeyDown = KeyArming | KeyTalking | Typing


@dataclass(frozen=True)
class Keyed:
    """Right Shift is down, at `level`: arming, a turn of the key's open, or Shift typing."""

    level: Level
    shortcut: KeyDown


Phase = Resting | Starting | Woke | Voiced | Keyed


@dataclass(frozen=True)
class Desk:
    """Where the desk is, the presses a triple tap has counted so far, and how many turns the voice has opened, which names
    the next."""

    phase: Phase = Resting("off")
    taps: Taps = Taps()
    turns: TurnId = TurnId(0)

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
    """A move of the gate, and the trigger that made it."""

    move: Move
    by: Trigger


@dataclass(frozen=True)
class Leveled:
    """The level changed, from `was` to `now`, chosen `by`."""

    was: Level
    now: Level
    by: Chooser


# What the desk does: a move of the gate, a change of level, or asking Smart Turn whether the speech that just stopped
# ends the turn.
Act = Made | Leveled | Judging


def step(desk: Desk, event: Event) -> tuple[Desk, tuple[Act, ...]]:
    """The desk after `event`, and what it does."""
    match event:
        case Chosen(level=level, by=by):
            phase, acts = _leveled(desk.phase, level, by)
            return replace(desk, phase=phase), acts
        case SpeechStarting() | SpeechStarted() | SpeechStopped() | Woken() | TurnEnded() | TurnTooLong() | NothingAsked():
            phase, acts = _heard(desk.phase, event, TurnId(desk.turns + 1))
            # [LAW:one-source-of-truth] the number spent is the one the turn the phase holds now carries, so the count
            # follows the turns and not a second decision of whether one opened.
            turn = turn_of(phase)
            return replace(desk, phase=phase, turns=desk.turns if turn is None else turn.id), acts
        case _:
            taps, tapped = counted(desk.taps, event)
            phase, acts = _keyed(desk.phase, event)
            # [LAW:one-source-of-truth] a press held to a turn is no tap: the shortcut that opened it is the one judge.
            if any(act == Made("start", "shortcut") for act in acts):
                taps = Taps()
            if tapped is not None:
                phase, more = _leveled(phase, toggled(level_of(phase), tapped), "tap")
                acts = (*acts, *more)
            return replace(desk, phase=phase, taps=taps), acts


def toggled(level: Level, key: TapKey) -> Level:
    """The level a triple tap of `key` sets from `level`: its own, or off from its own."""
    own: Listens = "wake word" if key == "right command" else "engaged"
    return "off" if level == own else own


def _keyed(phase: Phase, event: KeyEvent) -> tuple[Phase, tuple[Act, ...]]:
    level = level_of(phase)
    match phase:
        case Starting() | Woke() | Voiced():
            return _over_voice(phase, event)
        case Resting():
            shortcut, moves = shortcut_step(Idle(), event)
        case Keyed(shortcut=down):
            shortcut, moves = shortcut_step(down, event)
    made = tuple(Made(move, "shortcut") for move in moves)
    match shortcut:
        case Idle():
            return Resting(level), made
        case KeyArming() | KeyTalking() | Typing():
            return Keyed(level, shortcut), made


def _over_voice(phase: Starting | Woke | Voiced, event: KeyEvent) -> tuple[Phase, tuple[Act, ...]]:
    """Right Shift under what the voice has open: nothing until it is held long enough to mean talk, so Shift typing a
    capital leaves the voice's turn as it is; held so long, it takes the desk for the key, what the voice armed disarmed
    and a turn it opened sent."""
    shortcut, _ = shortcut_step(phase.key, event)
    match shortcut:
        case KeyTalking():
            left = Made("disarm", "engaged conversation") if isinstance(phase, Starting) else Made("stop", _voice(phase))
            return Keyed(level_of(phase), shortcut), (left, Made("arm", "shortcut"), Made("start", "shortcut"))
        case Idle() | KeyArming() | Typing():
            return replace(phase, key=shortcut), ()


def _rest(level: Level, key: VoiceKey, left: Act) -> tuple[Resting | Keyed, tuple[Act, ...]]:
    """Where the desk is once what the voice had open is `left`: resting, or with Right Shift down, which a press still
    arming arms for itself, so the shortcut goes on to open its turn."""
    match key:
        case Idle():
            return Resting(level), (left,)
        case KeyArming():
            return Keyed(level, key), (left, Made("arm", "shortcut"))
        case Typing():
            return Keyed(level, key), (left,)


def _leveled(phase: Phase, now: Level, by: Chooser) -> tuple[Phase, tuple[Act, ...]]:
    was = level_of(phase)
    if now == was:
        return phase, ()
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
        case Starting(key=key):
            rest, left = _rest(now, key, Made("disarm", "engaged conversation"))
            return rest, (*left, *listening, Leveled(was, now, by))
        case Woke(key=key) | Voiced(key=key):
            rest, left = _rest(now, key, Made("stop", _voice(phase)))
            return rest, (*left, *listening, Leveled(was, now, by))
        # A turn of the key's goes on as it is, at the new level.
        case Keyed(shortcut=shortcut):
            return Keyed(now, shortcut), (*listening, Leveled(was, now, by))


def _heard(phase: Phase, heard: Heard, next_turn: TurnId) -> tuple[Phase, tuple[Act, ...]]:
    """The phase after `heard`, and what the desk does: a turn it opens is named `next_turn`."""
    match phase, heard:
        case Resting(level="wake word"), Woken(at=at):
            return Woke(Waiting(next_turn, at)), (Made("arm", "wake word"), Made("start", "wake word"))
        case Resting(level="engaged"), SpeechStarting():
            return Starting(), (Made("arm", "engaged conversation"),)
        case Resting(level="engaged"), SpeechStarted(at=at):
            return Voiced("engaged", Speaking(next_turn, at)), (Made("arm", "engaged conversation"), Made("start", "engaged conversation"))
        case Starting(key=key), SpeechStarted(at=at):
            return Voiced("engaged", Speaking(next_turn, at), key), (Made("start", "engaged conversation"),)
        case Starting(key=key), SpeechStopped():
            return _rest("engaged", key, Made("disarm", "engaged conversation"))
        case Woke() | Voiced(), SpeechStarting() | SpeechStarted() | SpeechStopped() | TurnEnded() | TurnTooLong() | NothingAsked():
            return _in_turn(phase, heard)
        case _:
            return phase, ()


def _in_turn(phase: Woke | Voiced, told: Told) -> tuple[Phase, tuple[Act, ...]]:
    """What a turn the voice opened is told, by the rules of its end (`voiceturn.step`): what they decide the desk makes
    of it, ending it by the move that names how, with the trigger that opened it."""
    level = _listens(phase)
    match voiceturn.step(phase.turn, told):
        # The phase is the turn the rules made of it, whole.
        case Judging(turn=speaking) as judging:
            return _holding(level, phase.key, speaking), (judging,)
        case Going(turn=turn):
            return _holding(level, phase.key, turn), ()
        case Ended(how=how):
            return _rest(level, phase.key, Made(how, _voice_of(level)))


def _holding(level: Listens, key: VoiceKey, turn: Turn) -> Woke | Voiced:
    """The phase of a turn the voice has open: a wake's, waiting for what is asked, or one with speech in it."""
    match turn:
        case Waiting():
            return Woke(turn, key)
        case Speaking():
            return Voiced(level, turn, key)


def turn_of(phase: Phase) -> Turn | None:
    """The turn the voice has open at `phase`, if it has one."""
    match phase:
        case Woke(turn=turn) | Voiced(turn=turn):
            return turn
        case _:
            return None


def _listens(phase: Woke | Voiced) -> Listens:
    """The level a turn the voice has open was opened at."""
    match phase:
        case Woke():
            return "wake word"
        case Voiced(level=level):
            return level


def _voice(phase: Woke | Voiced) -> Trigger:
    """The trigger that opened a turn the voice has open."""
    return _voice_of(_listens(phase))


def _voice_of(level: Listens) -> Trigger:
    """The trigger the desk's listening at `level` is: the wake word's or engaged conversation's."""
    match level:
        case "wake word":
            return "wake word"
        case "engaged":
            return "engaged conversation"


def listens(level: Level) -> bool:
    """Whether the desk hears the room between turns at `level`."""
    return level != "off"
