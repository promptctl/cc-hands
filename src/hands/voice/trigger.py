"""The trigger: the way a turn of the user's is opened, at the desk or the phone, and the level the desk listens at.

The desk's level (`hands.voice.desk`) is the one choice: the talk key opens turns at every level, and the wake word or
the voice alone at their own.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from hands.core.place import Place
from hands.sessions.audit import TurnStart
from hands.voice.clips import Opener, Trigger
from hands.voice.desk import Level, Leveled
from hands.voice.wakeword import Word


def place_of(trigger: Trigger) -> Place:
    """Where the user is when `trigger` moves the gate."""
    match trigger:
        case "shortcut" | "engaged conversation" | "wake word":
            return "desk"
        case "phone button" | "phone wake word":
            return "phone"


def by_hand(trigger: Trigger) -> bool:
    """Whether `trigger` is the user's hand on a key or a button, which only the user moves, rather than a voice, which
    can be anyone's or nobody's."""
    match trigger:
        case "shortcut" | "phone button":
            return True
        case "engaged conversation" | "wake word" | "phone wake word":
            return False


def turn_start(opener: Opener) -> TurnStart:
    """When a turn `opener` opens cuts off what hands is saying.

    Sound the desk heard can be nobody: a cough, a door, the room. So a turn the voice opens, engaged conversation's or the
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


@dataclass(frozen=True)
class Choice:
    """A level the brain's tool chose, and the level the desk was at as the choice was stepped, answered once it is."""

    level: Level
    was: asyncio.Future[Level]


class Levels:
    """The one owner of the level the desk listens at, as the brain's tools read it and choose it: a choice goes to the
    desk's driver, which steps it in order with the taps and answers with the level it found."""

    # [LAW:no-shared-mutable-globals] the driver writes the level and the tools read it, both on the event loop; the
    # tools write only the queue the driver reads.
    def __init__(self) -> None:
        self._level: Level = "off"
        self.chosen: asyncio.Queue[Choice] = asyncio.Queue()

    async def choose(self, level: Level) -> Level:
        """Ask the desk for `level`, and get back the level it was at as the driver stepped the choice."""
        was: asyncio.Future[Level] = asyncio.get_running_loop().create_future()
        self.chosen.put_nowait(Choice(level, was))
        return await was

    def leveled(self, leveled: Leveled) -> None:
        """The driver's report of the level the desk is at now."""
        self._level = leveled.now

    @property
    def level(self) -> Level:
        return self._level


# What the phone's page listens at, for the call: the phone's own two levels, with the desk's names.
PhoneLevel = Literal["off", "wake word"]

# The trigger of a turn the wake word its page heard opens at the phone: the one name every reader of it spells.
PHONE_WAKE: Trigger = "phone wake word"
# The page's only control of the phone's level, as hands names it to the user at the start of a sentence.
PHONE_BUTTON = "The page's Phone wake word button"


def described_at_phone(level: PhoneLevel, word: Word) -> str:
    """What hands says of the phone's level, `word` being the wake word. The page's own button is what sets it."""
    held = "Hold the page's button to talk, and let go to send."
    match level:
        case "off":
            return f"The phone's wake word is off. {held} {PHONE_BUTTON} turns it on."
        case "wake word":
            return f"The phone listens for the wake word: say {word.phrase}, then what you want; hands answers when you finish. {held} {PHONE_BUTTON} turns it off."


def described(level: Level, word: Word) -> str:
    """What hands says of a level: its name, and how to talk under it, `word` being the wake word."""
    held = "Hold Right Shift to talk, and let go to send."
    match level:
        case "off":
            return f"The microphone is off between turns. {held} Triple tap Right Command for the wake word, or Right Shift for engaged conversation."
        case "wake word":
            return f"The wake word: say {word.phrase}, then what you want; hands answers when you finish. {held} Triple tap Right Command to turn the microphone off."
        case "engaged":
            return f"Engaged conversation: just talk; hands answers when you finish, and listens again. {held} Triple tap Right Shift to turn the microphone off."
