"""The contract every Pseudoterminals keeps, run against the fake (test_pseudoterminals_contract.py) and
SpawnedPseudoterminals, on real terminals (test_pseudoterminals_contract_spawned.py): a program shows what it drew and
ends as it exited, run in the directory and the environment it was given, on a terminal's TERM; one that runs on is
stopped when asked; a program that is not there, or cannot be run, is said so on the terminal, which ends as a shell
does; and a terminal in a directory that is not there never starts."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import pytest

from hands.sessions.child import Exited, NeverStarted, Stop, Stopped
from hands.sessions.pseudoterminal import STOP_SECONDS, TERM, ClaudeCode, Pseudoterminals

# The variable a program is handed, and what it is set to.
VARIABLE, VALUE = "HANDS_CONTRACT_TERMINAL", "the terminal's own"


class World(Protocol):
    """Pseudoterminals, and programs for them to run, each named by the argv[0] a start of it is asked with."""

    @property
    def pseudoterminals(self) -> Pseudoterminals: ...

    @property
    def folder(self) -> Path:
        """A directory a program can be run in, named by its real path."""
        ...

    @property
    def environment(self) -> Mapping[str, str]:
        """The environment a start is handed, on whose PATH the world's programs are found."""
        ...

    def program(self, name: str, shown: str, code: int) -> str:
        """A program that shows `shown` and exits `code`."""
        ...

    def echoing(self, name: str) -> str:
        """A program that shows the directory it runs in, VARIABLE and TERM as it is handed them, a line each, and
        exits 0."""
        ...

    def hung(self, name: str) -> str:
        """A program that never ends of itself."""
        ...

    def unfound(self) -> str:
        """A program that is not there."""
        ...

    def unrunnable(self, name: str) -> str:
        """A program that is there and that the OS cannot run: a file no one may execute."""
        ...


async def started(world: World, program: str, environment: Mapping[str, str] | None = None) -> ClaudeCode:
    match await world.pseudoterminals.start([program], world.folder, world.environment if environment is None else environment):
        case NeverStarted(error=error):
            pytest.fail(f"{program} never started: {error}")
        case session:
            return session


class PseudoterminalsContract:
    """Subclassed with a `world` fixture of the World under test."""

    @pytest.mark.parametrize("code", [0, 3])
    async def test_a_program_shows_what_it_drew_and_ends_as_it_exited(self, world: World, code: int) -> None:
        session = await started(world, world.program("draws", "first\n\nlast\n", code))
        assert await session.exit == Exited(code)
        assert session.shown() == "first\nlast"

    async def test_a_program_runs_in_its_directory_and_environment_on_a_terminal(self, world: World) -> None:
        session = await started(world, world.echoing("echoes"), {**world.environment, VARIABLE: VALUE})
        assert await session.exit == Exited(0)
        assert session.shown() == f"{world.folder}\n{VALUE}\n{TERM}"

    async def test_a_program_that_runs_on_is_stopped_when_asked(self, world: World) -> None:
        session = await started(world, world.hung("runs"))
        assert await session.stop() == Stopped(Stop(STOP_SECONDS))
        assert await session.exit == Stopped(Stop(STOP_SECONDS))

    async def test_a_program_that_is_not_there_is_said_so_and_its_terminal_ends_127(self, world: World) -> None:
        program = world.unfound()
        session = await started(world, program)
        assert await session.exit == Exited(127)
        assert program in session.shown()

    async def test_a_program_that_cannot_be_run_is_said_so_and_its_terminal_ends_126(self, world: World) -> None:
        program = world.unrunnable("unrunnable")
        session = await started(world, program)
        assert await session.exit == Exited(126)
        assert program in session.shown()

    async def test_a_terminal_in_a_directory_that_is_not_there_never_starts(self, world: World) -> None:
        match await world.pseudoterminals.start([world.program("draws", "", 0)], world.folder / "gone", world.environment):
            case NeverStarted(error=FileNotFoundError()):
                pass
            case other:
                pytest.fail(f"started as {other}")
