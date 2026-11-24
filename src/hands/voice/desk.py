"""The desk: how a turn of the user's is opened at the Mac, by the talk key at every level, and by the voice at two.

The desk listens at one of three levels (`hands.voice.listening`). Off, the microphone is closed between turns; the desk
listens at off until the models it hears with are loaded, and then at the level last chosen, first by the run's start:
off on a cold start and, on a restart, the level the run before it was at. Under the wake word, saying it opens a turn.
Engaged, speech alone opens each turn. At every level Right Shift held alone opens a turn and its release sends it
(`hands.voice.shortcut`). A triple tap of Right Command toggles the wake word, and one of Right Shift toggles engaged
conversation (`hands.voice.taps`); either switches straight from the other, and the brain's tools and the pages' level
buttons choose a level as a tap does. A turn the voice opened is closed by end-of-turn detection, by the rules of any voice
turn's end (`hands.voice.voiceturn`).

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
from hands.voice.shortcut import Arming as KeyArming, Idle, Instant, KeyEvent, Move, Shortcut, Talking as KeyTalking, Typing, step as shortcut_step
from hands.voice.taps import TapKey, Taps, counted
from hands.voice.voiceturn import Ended, Going, Judging, NothingAsked, SpeechStarted, SpeechStarting, SpeechStopped, Speaking, Told, Turn, TurnEnded, TurnId, TurnTooLong, Waiting

# Who chose a level: the start, the level the run starts at; a triple tap, toggling it; or whoever asked for it by name.
Chooser = Literal["start", "tap"] | Asker


@dataclass(frozen=True)
class Woken:
    """The wake word was heard, at `at`: `over_hands` where hands' voice may be in what the turn would keep from before
    it, and `built_in` where the desk is on the Mac's own microphone and speakers, which hear each other."""

    at: Instant
    over_hands: bool
    built_in: bool


@dataclass(frozen=True)
class Chosen:
    """`by` chose `level`."""

    level: Level
    by: Asker


@dataclass(frozen=True)
class ModelsLoaded:
    """The models every level hears with are loaded, and the room is overheard from now."""


Heard = Woken | Told
Event = KeyEvent | Heard | Chosen | ModelsLoaded


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
class Loading:
    """The models every level hears with are loading, so the desk listens at off whatever is chosen, and only the key is
    read: `chosen` is the level it listens at once they are loaded, chosen `by`, and `key` where Right Shift is."""

    chosen: Level
    by: Chooser
    key: Shortcut = Idle()


@dataclass(frozen=True)
class Loaded:
    """The models are loaded: the desk listens at the level its phase is at."""

    phase: Phase


@dataclass(frozen=True)
class Desk:
    """Whether the models the desk hears with are loaded, and where it is; the presses a triple tap has counted so far;
    and how many turns the voice has opened, which names the next.

    [LAW:types-are-the-program] a desk whose models load holds no phase of its own but the key's, at off, so it cannot be
    at a level that listens before anything can hear; and it is made from the level its run starts at.
    """

    models: Loading | Loaded
    taps: Taps = Taps()
    turns: TurnId = TurnId(0)

    @classmethod
    def starting(cls, level: Level) -> "Desk":
        """A run's desk, its models still to load, to listen at `level` once they are: the start's choice."""
        return cls(Loading(level, "start"))

    @property
    def phase(self) -> Phase:
        match self.models:
            case Loading(key=key):
                match key:
                    case Idle():
                        return Resting("off")
                    case KeyArming() | KeyTalking() | Typing():
                        return Keyed("off", key)
            case Loaded(phase=phase):
                return phase

    @property
    def level(self) -> Level:
        """The level the desk listens at."""
        return level_of(self.phase)

    @property
    def chosen(self) -> Level:
        """The level last chosen: the one the desk listens at, or will once its models are loaded."""
        match self.models:
            case Loading(chosen=chosen):
                return chosen
            case Loaded():
                return self.level


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
    match event, desk.models:
        case Chosen(level=level, by=by), _:
            return _chosen(desk, level, by)
        case ModelsLoaded(), Loading(chosen=chosen, by=by):
            # [LAW:no-ambient-temporal-coupling] the level chosen while they loaded is listened at from now, its listen
            # move made and its change said, as any choice's is once they are loaded.
            phase, acts = _leveled(desk.phase, chosen, by)
            return replace(desk, models=Loaded(phase)), acts
        case ModelsLoaded(), Loaded():
            # [LAW:no-silent-failure] the models load once a run: told so twice, the desk was stepped wrongly.
            raise RuntimeError("the desk was told its models were loaded once they already were")
        case SpeechStarting() | SpeechStarted() | SpeechStopped() | Woken() | TurnEnded() | TurnTooLong() | NothingAsked(), Loading():
            # At off, as a desk whose models load is, the room opens nothing.
            return desk, ()
        case SpeechStarting() | SpeechStarted() | SpeechStopped() | Woken() | TurnEnded() | TurnTooLong() | NothingAsked(), Loaded(phase=phase):
            phase, acts = _heard(phase, event, TurnId(desk.turns + 1))
            # [LAW:one-source-of-truth] the number spent is the one the turn the phase holds now carries, so the count
            # follows the turns and not a second decision of whether one opened.
            turn = turn_of(phase)
            return replace(desk, models=Loaded(phase), turns=desk.turns if turn is None else turn.id), acts
        case _, models:
            taps, tapped = counted(desk.taps, event)
            match models:
                case Loading(key=key):
                    down, moves = shortcut_step(key, event)
                    keyed, acts = replace(models, key=down), tuple[Act, ...](Made(move, "shortcut") for move in moves)
                case Loaded(phase=phase):
                    phase, acts = _keyed(phase, event)
                    keyed = Loaded(phase)
            # [LAW:one-source-of-truth] a press held to a turn is no tap: the shortcut that opened it is the one judge.
            if any(act == Made("start", "shortcut") for act in acts):
                taps = Taps()
            pressed = replace(desk, models=keyed, taps=taps)
            if tapped is None:
                return pressed, acts
            leveled, more = _chosen(pressed, toggled(pressed.chosen, tapped), "tap")
            return leveled, (*acts, *more)


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
    capital leaves the voice's turn as it is; held so long, it takes the desk for the key. [LAW:single-enforcer] the gate
    sends the turn the voice opened and takes the arming it made, so the desk only arms and starts."""
    shortcut, _ = shortcut_step(phase.key, event)
    match shortcut:
        case KeyTalking():
            return Keyed(level_of(phase), shortcut), (Made("arm", "shortcut"), Made("start", "shortcut"))
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


def _chosen(desk: Desk, now: Level, by: Chooser) -> tuple[Desk, tuple[Act, ...]]:
    """The desk once `by` chose `now`, and what it does: while its models load, nothing, the choice held until they are
    loaded, since nothing hears the room before then."""
    match desk.models:
        case Loading() as loading:
            return replace(desk, models=replace(loading, chosen=now, by=by)), ()
        case Loaded(phase=phase):
            leveled, acts = _leveled(phase, now, by)
            return replace(desk, models=Loaded(leveled)), acts


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
        case Resting(level="wake word"), Woken(over_hands=True, built_in=True):
            # Hands' own voice, heard back through the room, may have said it: a turn opened on it would cut the reply off
            # on hands' own word and send the brain hands' voice as the user's.
            return phase, ()
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
