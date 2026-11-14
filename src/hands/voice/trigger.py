"""The trigger: the way a turn of the user's is opened, at the desk or the phone, and the level the desk listens at.

The desk's level (`hands.voice.desk`) is the one choice: the talk key opens turns at every level, and the wake word or
the voice alone at their own.
"""

import asyncio
from dataclasses import dataclass

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
        case "phone button":
            return "phone"


def turn_start(opener: Opener) -> TurnStart:
    """When a turn `opener` opens cuts off what hands is saying.

    Sound the desk heard can be nobody, or hands itself: its own reply left over from the echo canceller, its own saying of
    the wake phrase, a cough, a door. So a turn the voice opens, engaged conversation's or the wake word's, cuts only once
    a speaker who is not hands is heard in it (`hands.voice.voiceprints.cleared`). One the user's hand opens cuts at once,
    since holding a key or a button and letting go with nothing said is how the user stops hands. Words typed and sent are
    as deliberate as a press, and are words already.
    """
    match opener:
        case "engaged conversation" | "wake word":
            return "on a person"
        case "shortcut" | "phone button" | "typed":
            return "on the clip"


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
