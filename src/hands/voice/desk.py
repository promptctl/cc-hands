"""The desk: how a turn of the user's is opened at the Mac, by the talk key at every level, and by the voice at two and in
the reply window after hands spoke.

The desk is at one of three levels (`hands.voice.listening`). Off, it is deaf to the room between turns; it is at off
until the models it hears with are loaded, and then at the level last chosen, first by the run's start: off on a cold
start and, on a restart, the level the run before it was at. Under the wake word, saying it opens a turn. Engaged, speech
alone opens each turn. At every level Right Shift held alone opens a turn and its release sends it
(`hands.voice.shortcut`). A triple tap of Right Command toggles the wake word, and one of Right Shift toggles engaged
conversation (`hands.voice.taps`); either switches straight from the other, and the brain's tools and the pages' level
buttons choose a level as a tap does.

Off and under the wake word, hands falling quiet at the desk, with a microphone there to hear an answer, opens the reply
window, cued: until `REPLY_WINDOW_SECONDS` have passed, speech alone opens a turn, so the user can answer with no key and
no wake word. Hands speaking again closes it, and so does its time; speech that started in it goes on past either, to the
turn it opens or to nothing. A turn the voice opened is closed by end-of-turn detection, by the rules of any voice turn's
end (`hands.voice.voiceturn`).

[LAW:types-are-the-program] the desk is one value whose variants carry the level, in a field, in what let the voice open
what it has open, or, for Woke, which exists under the wake word alone, in the variant itself, so whether it hears the
room (`hears`), which trigger has the turn, and whether the key is down are each said once: a key turn and a voice turn
are never open together, and the gate is told the desk hears exactly while it does (`step`).
"""

from dataclasses import dataclass, replace
from typing import Literal

from hands.voice import voiceturn
from hands.voice.clips import Trigger
from hands.voice.handsvoice import HandsQuiet, HandsSpeaking, HandsVoice
from hands.voice.listening import Asker, Level, Listens
from hands.voice.shortcut import Arming as KeyArming, Idle, Instant, KeyEvent, Move, Shortcut, Talking as KeyTalking, Typing, step as shortcut_step
from hands.voice.taps import TapKey, Taps, counted
from hands.voice.voiceturn import Ended, Going, Judging, NothingAsked, SpeechStarted, SpeechStarting, SpeechStopped, Speaking, Told, Turn, TurnEnded, TurnId, TurnTooLong, Waiting

# Who chose a level: the start, the level the run starts at; a triple tap, toggling it; or whoever asked for it by name.
Chooser = Literal["start", "tap"] | Asker

# How long after hands falls quiet the user may start to answer with no key and no wake word.
REPLY_WINDOW_SECONDS = 5.0


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


@dataclass(frozen=True)
class ReplyWindowOver:
    """The reply window open until `until` is due to close."""

    until: Instant


# What the desk hears and is told beside the key and the levels: the user's speech and the wake word, the deadlines a turn
# the voice opened set itself, hands' own voice, and the reply window's end.
Heard = Woken | Told | HandsVoice | ReplyWindowOver
Event = KeyEvent | Heard | Chosen | ModelsLoaded


@dataclass(frozen=True)
class Resting:
    """No turn open and the key up: deaf under "off", listening under the other two."""

    level: Level


# Right Shift under what the voice has open: up, down and not yet held long enough to mean talk, or down as Shift.
VoiceKey = Idle | KeyArming | Typing


@dataclass(frozen=True)
class ReplyWindow:
    """The window after hands fell quiet at the desk, at `level`, where speech alone opens no turn between turns: in it,
    speech alone opens one, until `until`."""

    level: Literal["off", "wake word"]
    until: Instant


@dataclass(frozen=True)
class Reply:
    """The user's reply to hands, at `level`, begun in the reply window: it goes on once the window has closed."""

    level: Literal["off", "wake word"]


# What lets the voice open a turn: the level's own listening, the reply window while it is open, and the reply begun in it.
Opener = Listens | ReplyWindow | Reply


@dataclass(frozen=True)
class InWindow:
    """The reply window, `window`, open with nothing yet said in it."""

    window: ReplyWindow
    key: VoiceKey = Idle()


@dataclass(frozen=True)
class Starting:
    """What may be speech has started where speech alone opens a turn, `by` engaged conversation, the reply window, or a
    reply begun in it: the microphone is armed and the turn is not open."""

    by: Literal["engaged"] | ReplyWindow | Reply
    key: VoiceKey = Idle()


@dataclass(frozen=True)
class Woke:
    """The wake word opened a turn, which waits for what is asked to start (`hands.voice.voiceturn`)."""

    turn: Waiting
    key: VoiceKey = Idle()

    @property
    def by(self) -> Literal["wake word"]:
        return "wake word"


@dataclass(frozen=True)
class Voiced:
    """A turn the voice opened, with speech in it, `by` the level's own listening, said after the wake word or opened by
    speech, engaged, or by the user's reply to hands."""

    by: Listens | Reply
    turn: Speaking
    key: VoiceKey = Idle()


KeyDown = KeyArming | KeyTalking | Typing


@dataclass(frozen=True)
class Keyed:
    """Right Shift is down, at `level`: arming, a turn of the key's open, or Shift typing."""

    level: Level
    shortcut: KeyDown


Phase = Resting | InWindow | Starting | Woke | Voiced | Keyed


@dataclass(frozen=True)
class Loading:
    """The models every level hears with are loading, so the desk is at off whatever is chosen, and only the key is read:
    `chosen` is the level it is at once they are loaded, chosen `by`, and `key` where Right Shift is."""

    chosen: Level
    by: Chooser
    key: Shortcut = Idle()


@dataclass(frozen=True)
class Loaded:
    """The models are loaded: the desk is where its phase is."""

    phase: Phase


@dataclass(frozen=True)
class Desk:
    """Whether the models the desk hears with are loaded, and where it is; the presses a triple tap has counted so far;
    and how many turns the voice has opened, which names the next.

    [LAW:types-are-the-program] a desk whose models load holds no phase of its own but the key's, at off, so it cannot
    hear the room before anything can; and it is made from the level its run starts at.
    """

    models: Loading | Loaded
    taps: Taps = Taps()
    turns: TurnId = TurnId(0)

    @classmethod
    def starting(cls, level: Level) -> "Desk":
        """A run's desk, its models still to load, to be at `level` once they are: the start's choice."""
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
        """The level the desk is at, which it hears the room at between turns."""
        return level_of(self.phase)

    @property
    def chosen(self) -> Level:
        """The level last chosen: the one the desk is at, or will be once its models are loaded."""
        match self.models:
            case Loading(chosen=chosen):
                return chosen
            case Loaded():
                return self.level


def level_of(phase: Phase) -> Level:
    match phase:
        case Resting(level=level) | Keyed(level=level) | InWindow(window=ReplyWindow(level=level)):
            return level
        case Starting(by=by) | Woke(by=by) | Voiced(by=by):
            return _level(by)


def _level(by: Opener) -> Level:
    """The level what the voice has open `by` was opened at, and returns to."""
    match by:
        case ReplyWindow(level=level) | Reply(level=level):
            return level
        case "wake word" | "engaged" as level:
            return level


def _trigger(by: Opener) -> Trigger:
    """The trigger what the voice has open `by` is opened by."""
    match by:
        case ReplyWindow() | Reply():
            return "reply window"
        case "wake word":
            return "wake word"
        case "engaged":
            return "engaged conversation"


def hears(phase: Phase) -> bool:
    """Whether the desk hears the room: between turns, at a level that listens; and through all the voice has open, the
    reply window and what it opened among it."""
    match phase:
        case Resting(level=level) | Keyed(level=level):
            return listens(level)
        case InWindow() | Starting() | Woke() | Voiced():
            return True


def _hearing(phase: Phase) -> Trigger:
    """What the desk hears the room for, where it does at `phase`: its level, where that listens, and off, the reply window."""
    match level_of(phase):
        case "off":
            return "reply window"
        case ("wake word" | "engaged") as level:
            return _trigger(level)


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


@dataclass(frozen=True)
class ReplyWindowOpened:
    """The reply window opened, to close at `until`: its cue plays, and its end is told when it is due."""

    until: Instant


# What the desk does: a move of the gate, a change of level, opening the reply window, or asking Smart Turn whether the
# speech that just stopped ends the turn.
Act = Made | Leveled | ReplyWindowOpened | Judging


def step(desk: Desk, event: Event) -> tuple[Desk, tuple[Act, ...]]:
    """The desk after `event`, and what it does."""
    stepped, acts = _stepped(desk, event)
    return stepped, _gated(desk.phase, stepped.phase, acts)


def _gated(was: Phase, now: Phase, acts: tuple[Act, ...]) -> tuple[Act, ...]:
    """`acts`, with the gate told the desk hears the room as it starts to, ahead of anything it arms, and that it no longer
    does as it stops, after whatever ended what it heard for.

    [LAW:single-enforcer] the one rule that keeps the gate hearing the desk's room exactly while the desk does, however its
    phase moved: a level chosen, the reply window opening or closing, the key, or a turn's end."""
    match hears(was), hears(now):
        case False, True:
            return (Made("listen", _hearing(now)), *acts)
        case True, False:
            return (*acts, Made("deafen", _hearing(was)))
        case _:
            return acts


def _stepped(desk: Desk, event: Event) -> tuple[Desk, tuple[Act, ...]]:
    if isinstance(event, Heard):
        return _told(desk, event)
    match event, desk.models:
        case Chosen(level=level, by=by), _:
            return _chosen(desk, level, by)
        case ModelsLoaded(), Loading(chosen=chosen, by=by):
            # [LAW:no-ambient-temporal-coupling] the level chosen while they loaded is the desk's from now, its change
            # said, as any choice's is once they are loaded.
            phase, acts = _leveled(desk.phase, chosen, by)
            return replace(desk, models=Loaded(phase)), acts
        case ModelsLoaded(), Loaded():
            # [LAW:no-silent-failure] the models load once a run: told so twice, the desk was stepped wrongly.
            raise RuntimeError("the desk was told its models were loaded once they already were")
        case _, models:
            taps, tapped = counted(desk.taps, event)
            match models:
                case Loading(key=key):
                    down, moves = shortcut_step(key, event)
                    keyed, acts = replace(models, key=down), tuple[Act, ...](Made(move, "shortcut") for move in moves)
                case Loaded(phase=was):
                    phase, acts = _keyed(was, event)
                    keyed = Loaded(phase)
            # [LAW:one-source-of-truth] a press held to a turn is no tap: the shortcut that opened it is the one judge.
            if any(act == Made("start", "shortcut") for act in acts):
                taps = Taps()
            pressed = replace(desk, models=keyed, taps=taps)
            if tapped is None:
                return pressed, acts
            leveled, more = _chosen(pressed, toggled(pressed.chosen, tapped), "tap")
            return leveled, (*acts, *more)


def _told(desk: Desk, heard: Heard) -> tuple[Desk, tuple[Act, ...]]:
    match desk.models:
        case Loading():
            # At off, as a desk whose models load is, the room opens nothing, and nothing hears an answer to hands.
            return desk, ()
        case Loaded(phase=was):
            phase, acts = _heard(was, heard, TurnId(desk.turns + 1))
            # [LAW:one-source-of-truth] the number spent is the one the turn the phase holds now carries, so the count
            # follows the turns and not a second decision of whether one opened.
            turn = turn_of(phase)
            return replace(desk, models=Loaded(phase), turns=desk.turns if turn is None else turn.id), acts


def toggled(level: Level, key: TapKey) -> Level:
    """The level a triple tap of `key` sets from `level`: its own, or off from its own."""
    own: Listens = "wake word" if key == "right command" else "engaged"
    return "off" if level == own else own


def _keyed(phase: Phase, event: KeyEvent) -> tuple[Phase, tuple[Act, ...]]:
    level = level_of(phase)
    match phase:
        case InWindow() | Starting() | Woke() | Voiced():
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


def _over_voice(phase: InWindow | Starting | Woke | Voiced, event: KeyEvent) -> tuple[Phase, tuple[Act, ...]]:
    """Right Shift under what the voice has open: nothing until it is held long enough to mean talk, so Shift typing a
    capital leaves the voice's turn as it is; held so long, it takes the desk for the key. [LAW:single-enforcer] the gate
    sends the turn the voice opened and takes the arming it made, so the desk only arms and starts."""
    shortcut, _ = shortcut_step(phase.key, event)
    match shortcut:
        case KeyTalking():
            return Keyed(level_of(phase), shortcut), (Made("arm", "shortcut"), Made("start", "shortcut"))
        case Idle() | KeyArming() | Typing():
            return replace(phase, key=shortcut), ()


def _rest(level: Level, key: VoiceKey, *left: Act) -> tuple[Resting | Keyed, tuple[Act, ...]]:
    """Where the desk is once what the voice had open is `left`: resting, or with Right Shift down, which a press still
    arming arms for itself, so the shortcut goes on to open its turn."""
    match key:
        case Idle():
            return Resting(level), left
        case KeyArming():
            return Keyed(level, key), (*left, Made("arm", "shortcut"))
        case Typing():
            return Keyed(level, key), left


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
    leveled = Leveled(was, now, by)
    match phase:
        case Resting():
            return Resting(now), (leveled,)
        # A turn of the key's goes on as it is, at the new level.
        case Keyed(shortcut=shortcut):
            return Keyed(now, shortcut), (leveled,)
        # What the voice had open is the level's: leaving it, the window closes, an armed start is disarmed and a turn is sent.
        case InWindow(key=key):
            rest, left = _rest(now, key)
            return rest, (*left, leveled)
        case Starting(by=opener, key=key):
            rest, left = _rest(now, key, Made("disarm", _trigger(opener)))
            return rest, (*left, leveled)
        case Woke(key=key) | Voiced(key=key):
            rest, left = _rest(now, key, Made("stop", _trigger(phase.by)))
            return rest, (*left, leveled)


def _heard(phase: Phase, heard: Heard, next_turn: TurnId) -> tuple[Phase, tuple[Act, ...]]:
    """The phase after `heard`, and what the desk does: a turn it opens is named `next_turn`."""
    match phase, heard:
        case _, Woken(over_hands=True, built_in=True):
            # Hands' own voice, heard back through the room, may have said it: a turn opened on it would cut the reply off
            # on hands' own word and send the brain hands' voice as the user's.
            return phase, ()
        case Resting(level="wake word"), Woken(at=at):
            return Woke(Waiting(next_turn, at)), (Made("arm", "wake word"), Made("start", "wake word"))
        case InWindow(window=ReplyWindow(level="wake word"), key=key), Woken(at=at):
            return Woke(Waiting(next_turn, at), key), (Made("arm", "wake word"), Made("start", "wake word"))
        case Starting(by=ReplyWindow(level="wake word") | Reply(level="wake word"), key=key), Woken(at=at):
            # What started in the window was the wake word, whose turn it is.
            return Woke(Waiting(next_turn, at), key), (Made("disarm", "reply window"), Made("arm", "wake word"), Made("start", "wake word"))
        case Resting(level="engaged"), SpeechStarting():
            return Starting("engaged"), (Made("arm", "engaged conversation"),)
        case Resting(level="engaged"), SpeechStarted(at=at):
            return Voiced("engaged", Speaking(next_turn, at)), (Made("arm", "engaged conversation"), Made("start", "engaged conversation"))
        case Resting(level="off" | "wake word" as level), HandsQuiet(at=at, place="desk", deaf=False):
            window = ReplyWindow(level, at + REPLY_WINDOW_SECONDS)
            return InWindow(window), (ReplyWindowOpened(window.until),)
        case InWindow(window=window, key=key), SpeechStarting():
            return Starting(window, key), (Made("arm", "reply window"),)
        case InWindow(window=ReplyWindow(level=level), key=key), SpeechStarted(at=at):
            return Voiced(Reply(level), Speaking(next_turn, at), key), (Made("arm", "reply window"), Made("start", "reply window"))
        case InWindow(window=window, key=key), HandsSpeaking():
            return _rest(window.level, key)
        case InWindow(window=window, key=key), ReplyWindowOver(until=until) if until == window.until:
            return _rest(window.level, key)
        # Speech that started in the window goes on once it closes, to the turn it opens or to nothing.
        case Starting(by=ReplyWindow() as window, key=key), HandsSpeaking():
            return Starting(Reply(window.level), key), ()
        case Starting(by=ReplyWindow() as window, key=key), ReplyWindowOver(until=until) if until == window.until:
            return Starting(Reply(window.level), key), ()
        case Starting(by="engaged", key=key), SpeechStarted(at=at):
            return Voiced("engaged", Speaking(next_turn, at), key), (Made("start", "engaged conversation"),)
        case Starting(by=ReplyWindow(level=level) | Reply(level=level), key=key), SpeechStarted(at=at):
            return Voiced(Reply(level), Speaking(next_turn, at), key), (Made("start", "reply window"),)
        case Starting(by=ReplyWindow() as window, key=key), SpeechStopped():
            # A start that came to nothing, a cough or the room: the window goes on until it closes.
            return InWindow(window, key), (Made("disarm", "reply window"),)
        case Starting(by="engaged" | Reply() as by, key=key), SpeechStopped():
            return _rest(_level(by), key, Made("disarm", _trigger(by)))
        case Woke() | Voiced(), SpeechStarting() | SpeechStarted() | SpeechStopped() | TurnEnded() | TurnTooLong() | NothingAsked():
            return _in_turn(phase, heard)
        case _:
            return phase, ()


def _in_turn(phase: Woke | Voiced, told: Told) -> tuple[Phase, tuple[Act, ...]]:
    """What a turn the voice opened is told, by the rules of its end (`voiceturn.step`): what they decide the desk makes
    of it, ending it by the move that names how, with the trigger that opened it."""
    match voiceturn.step(phase.turn, told):
        # The phase is the turn the rules made of it, whole.
        case Judging(turn=speaking) as judging:
            return _holding(phase.by, phase.key, speaking), (judging,)
        case Going(turn=turn):
            return _holding(phase.by, phase.key, turn), ()
        case Ended(how=how):
            return _rest(level_of(phase), phase.key, Made(how, _trigger(phase.by)))


def _holding(by: Listens | Reply, key: VoiceKey, turn: Turn) -> Woke | Voiced:
    """The phase of a turn the voice has open: a wake's, waiting for what is asked, or one with speech in it."""
    match turn:
        case Waiting():
            return Woke(turn, key)
        case Speaking():
            return Voiced(by, turn, key)


def turn_of(phase: Phase) -> Turn | None:
    """The turn the voice has open at `phase`, if it has one."""
    match phase:
        case Woke(turn=turn) | Voiced(turn=turn):
            return turn
        case _:
            return None


def listens(level: Level) -> bool:
    """Whether the desk hears the room between turns at `level`."""
    return level != "off"
