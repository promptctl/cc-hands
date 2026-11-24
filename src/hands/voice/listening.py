"""The levels the desk listens at between the user's turns, and their one owner, named apart from the desk
(`hands.voice.desk`) so the CLI, which holds the owner across the run and carries its level across a restart, names them
with nothing of the desk's audio loaded."""

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from hands.sessions.payload import Rejected

# The levels the desk listens at between turns; it is deaf under "off" alone.
Listens = Literal["wake word", "engaged"]
# One flat Literal, so a tool argument of it has a schema: "off" and every level of Listens.
Level = Literal["off", "wake word", "engaged"]
LEVELS: tuple[Level, ...] = get_args(Level)

# Who asks the desk for a level by naming it: the brain's tool, asked by voice, or a page's level button, the phone's or
# the gallery's.
Asker = Literal["tool", "page"]


def parse_level(text: str) -> Level:
    """[LAW:parse-dont-validate] the level `text` names; raises Rejected naming the levels where it names none."""
    for level in LEVELS:
        if text == level:
            return level
    raise Rejected(f"hands listens at {', '.join(LEVELS)}, not {text!r}")


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


@dataclass(frozen=True)
class Pending:
    """The desk's driver is yet to step `level`, the one the run starts at: nothing listens yet."""

    level: Level


@dataclass(frozen=True)
class At:
    """The desk listens at `level`, as its driver stepped it last."""

    level: Level


class Levels:
    """The one owner of the level the desk listens at. The run starts it at a level, off or the one a restart carried; the
    desk's driver steps that first, and tells it the level the desk is at after every step; the tools and the pages read
    it, and ask for another level, which the driver steps in order with the taps and answers with the level it found."""

    # [LAW:no-shared-mutable-globals] `stepped` is the driver's alone, which holds the concrete owner; the tools and the
    # pages hold it as Listening, which reads and asks. All of them run on the event loop.
    def __init__(self, start: Level) -> None:
        self._desk: Pending | At = Pending(start)
        self.chosen: asyncio.Queue[Choice] = asyncio.Queue()

    async def choose(self, level: Level, by: Asker) -> Level:
        """Ask the desk for `level`, and get back the level it was at as the driver stepped the choice."""
        was: asyncio.Future[Level] = asyncio.get_running_loop().create_future()
        self.chosen.put_nowait(Choice(level, by, was))
        return await was

    def stepped(self, level: Level) -> None:
        """The driver's report of the level the desk is at, as it stepped it."""
        self._desk = At(level)

    @property
    def level(self) -> Level:
        """The level the desk listens at now: off until its driver has stepped the one the run starts at."""
        match self._desk:
            case Pending():
                return "off"
            case At(level=level):
                return level

    @property
    def carried(self) -> Level:
        """The level a restart starts at, and the one the desk's driver steps first: the level the desk listens at, or the
        one the run starts at where the driver is yet to step it."""
        return self._desk.level

    @property
    def reached(self) -> bool:
        """Whether the desk's driver has stepped the level the run starts at, so `carried` is a level the desk listened at."""
        return isinstance(self._desk, At)
