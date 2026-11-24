"""The levels the desk listens at between the user's turns, and their one owner, named apart from the desk
(`hands.voice.desk`) so the CLI, which holds the owner across the run and carries its level across a restart, names them
with nothing of the desk's audio loaded."""

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from hands.sessions.payload import one_of

# The levels the desk listens at between turns; it is deaf under "off" alone.
Listens = Literal["wake word", "engaged"]
# One flat Literal, so a tool argument of it has a schema: "off" and every level of Listens.
Level = Literal["off", "wake word", "engaged"]
LEVELS: tuple[Level, ...] = get_args(Level)

# Who asks the desk for a level by naming it: the brain's tool, asked by voice, or a page's level button, the phone's or
# the gallery's.
Asker = Literal["tool", "page"]


def parse_level(text: str) -> Level:
    """The level `text` names; raises Rejected naming the levels where it names none."""
    return one_of(text, LEVELS, "hands listens at")


@dataclass(frozen=True)
class Choice:
    """A level `by` asked for, and the level the desk was at as the choice was stepped, answered once it is."""

    level: Level
    by: Asker
    was: asyncio.Future[Level]


class Listening(Protocol):
    """The level the desk listens at as the tools and the pages see it: its level now, and a choice of another."""

    @property
    def level(self) -> Level: ...

    async def choose(self, level: Level, by: Asker) -> Level: ...


class Stepped(Protocol):
    """The desk as its driver stepped it (`hands.voice.desk.Desk`): the level it listens at, and the level last chosen,
    which it listens at once its models are loaded."""

    @property
    def level(self) -> Level: ...

    @property
    def chosen(self) -> Level: ...


class Levels:
    """The one owner of the level the desk listens at. The run makes it with the level its start chooses, off or the one a
    restart carried; the desk's driver steps that choice first, and tells it the desk it stepped after every step; the
    tools and the pages read it, and ask for another level, which the driver takes and answers in order with the taps.

    [LAW:no-shared-mutable-globals] the desk's driver is the one caller of `stepped` and `answer`; the run's start, the
    run and the desk's server pass it on to the driver unread, and the tools and the pages hold it as Listening, which
    reads and asks. All of them run on the event loop.
    """

    def __init__(self, start: Level) -> None:
        # Off, as the desk listens before its models are loaded, with the start's choice to be stepped first.
        self._level: Level = "off"
        self._chosen: Level = start
        self._asked = 0
        self._answered = 0
        self.chosen: asyncio.Queue[Choice] = asyncio.Queue()

    async def choose(self, level: Level, by: Asker) -> Level:
        """Ask the desk for `level`, and get back the level it listened at as the driver took the choice."""
        was: asyncio.Future[Level] = asyncio.get_running_loop().create_future()
        self._asked += 1
        self.chosen.put_nowait(Choice(level, by, was))
        return await was

    def answer(self, choice: Choice) -> None:
        """The driver's answer to `choice`, as it takes it to step: the level it found."""
        self._answered += 1
        # A tool call cancelled while it waited, the user cutting the brain off, is answered by no one; its choice is the
        # user's all the same.
        if not choice.was.cancelled():
            choice.was.set_result(self._level)

    def stepped(self, desk: Stepped) -> None:
        """The driver's report of the desk as it stepped it."""
        self._level, self._chosen = desk.level, desk.chosen

    @property
    def level(self) -> Level:
        """The level the desk listens at now."""
        return self._level

    @property
    def last_chosen(self) -> Level:
        """The level last chosen, which a restart starts at: the one the desk listens at, or will once its models load."""
        return self._chosen

    @property
    def unanswered(self) -> int:
        """How many choices were asked for that the driver is yet to take: each, at the run's end, a choice never made."""
        return self._asked - self._answered
