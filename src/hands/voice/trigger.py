"""The trigger: the way a turn of the user's is opened, at the desk or the phone, and the level the desk listens at.

The desk's level (`hands.voice.desk`, owned by `hands.voice.listening`) is the one choice: the talk key opens turns at
every level, and the wake word or the voice alone at their own.
"""

from typing import Literal

from hands.core.place import Place
from hands.sessions.audit import TurnStart
from hands.voice.clips import Opener, Trigger
from hands.voice.listening import Level
from hands.voice.wakeword import Word


def place_of(trigger: Trigger) -> Place:
    """Where the user is when `trigger` moves the gate."""
    match trigger:
        case "shortcut" | "engaged conversation" | "wake word" | "answer window":
            return "desk"
        case "phone button" | "phone wake word":
            return "phone"


def by_hand(trigger: Trigger) -> bool:
    """Whether `trigger` is the user's hand on a key or a button, which only the user moves, rather than a voice the desk
    heard, which can be anyone's or nobody's."""
    match trigger:
        case "shortcut" | "phone button":
            return True
        case "engaged conversation" | "wake word" | "answer window" | "phone wake word":
            return False


def turn_start(opener: Opener) -> TurnStart:
    """When a turn `opener` opens cuts off what hands is saying.

    Sound the desk heard can be nobody: a cough, a door, the room. So a turn the voice opens, engaged conversation's or a
    wake word's, cuts only once speech is heard in it, hands' own voice having been taken out of the sound
    (`hands.voice.echo`), so speech left in it is a person's. One the user's hand opens cuts at once,
    since holding a key or a button and letting go with nothing said is how the user stops hands. Words typed and sent are
    as deliberate as a press, and are words already.
    """
    match opener:
        case "typed":
            return "on the clip"
        case trigger:
            return "on the clip" if by_hand(trigger) else "on a person"


def described(level: Level, word: Word) -> str:
    """What hands says of a level: its name, and how to talk under it, `word` being the wake word."""
    held = "Hold Right Shift to talk, and let go to send."
    match level:
        case "off":
            return f"The microphone is off between turns, but for a few seconds after each reply, so you can answer. {held} Triple tap Right Command for the wake word, or Right Shift for engaged conversation."
        case "wake word":
            return f"The wake word: say {word.phrase}, then what you want; hands answers when you finish, and just after a reply you can answer without it. {held} Triple tap Right Command to turn the microphone off."
        case "engaged":
            return f"Engaged conversation: just talk; hands answers when you finish, and listens again. {held} Triple tap Right Shift to turn the microphone off."


# The trigger of a turn the phone's page opens by its detector hearing the wake word.
PHONE_WAKE: Trigger = "phone wake word"

# The levels the phone listens at between turns, which its page tells: it is deaf under "off" alone, and hears for its wake
# word otherwise.
PhoneLevel = Literal["off", "wake word"]

# The name the brain is told the phone's page button has, spelled once for every readback that says it.
PHONE_WAKE_BUTTON = "The page's Phone wake word button"


def described_at_phone(level: PhoneLevel) -> str:
    """What hands says of the phone's level while hands is at the phone: its name, and that the page's own button sets it."""
    held = "Hold the page's button to talk, and let go to send."
    match level:
        case "off":
            return f"The phone is not listening between turns. {held} {PHONE_WAKE_BUTTON} turns on listening for the wake word."
        case "wake word":
            return f"The phone listens for the wake word: say it, then what you want; hands answers when you finish. {held} {PHONE_WAKE_BUTTON} turns it off."
