"""The levels the desk listens at between the user's turns, named apart from the desk (`hands.voice.desk`) so the CLI, which
carries the level across a restart, names it with nothing of the desk's audio loaded."""

from typing import Literal

# The levels the desk listens at between turns; it is deaf under "off" alone.
Listens = Literal["wake word", "engaged"]
# One flat Literal, so a tool argument of it has a schema: "off" and every level of Listens.
Level = Literal["off", "wake word", "engaged"]
LEVELS: tuple[Level, ...] = ("off", "wake word", "engaged")
