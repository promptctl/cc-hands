"""The contract every Children keeps, run against the fake (test_children_contract.py) and SpawnedChildren, starting real
programs (test_children_contract_spawned.py): a run ends as its program did, handed what it was given, and held to its
limit, with what it wrote by then; a program that is not there, or cannot be run, never starts; and what no OS can be
handed is refused before anything runs."""

import asyncio
import errno
from pathlib import Path
from typing import Protocol

import pytest

from hands.sessions import wide
from hands.sessions.child import Captured, Children, Exited, Kill, Killed, Limit, NeverStarted, Ran, Traced

# The variable a program is handed, and what it is set to.
VARIABLE, VALUE = "HANDS_CONTRACT_CHILD", "the child's own"
# How long a run that would never end is let run: long enough for any program to start, short enough to wait out.
LIMIT = 0.5


class World(Protocol):
    """Children, and programs for them to run, each named by the argv[0] a run of it is asked with."""

    @property
    def children(self) -> Children: ...

    @property
    def folder(self) -> Path:
        """A directory a program can be run in, named by its real path."""
        ...

    def program(self, name: str, code: int, out: bytes, err: bytes) -> str:
        """A program that writes `out` and `err` and exits `code`."""
        ...

    def echoing(self, name: str) -> str:
        """A program that writes the directory it runs in, VARIABLE and TRACEPARENT as it is handed them (a line each,
        empty where it is handed none), then its stdin, and exits 0."""
        ...

    def hung(self, name: str, out: bytes, err: bytes) -> str:
        """A program that writes `out` and `err` and then never ends of itself."""
        ...

    def unfound(self) -> str:
        """A program that is not there."""
        ...

    def unrunnable(self, name: str) -> str:
        """A program that is there and that the OS cannot run: a file no one may execute."""
        ...

    async def past(self, seconds: float) -> None:
        """`seconds` gone by on the children's clock, while a run waits on it."""
        ...


class ChildrenContract:
    """Subclassed with a `world` fixture of the World under test."""

    @pytest.mark.parametrize("code", [0, 3])
    async def test_a_run_ends_as_its_program_exited_with_what_it_wrote(self, world: World, code: int) -> None:
        program = world.program("speaks", code, b"out\n", b"err\n")
        expected = Ran(Exited(code), Captured(b"out\n"), Captured(b"err\n"))
        assert await world.children.run(program, limit=60) == expected
        assert world.children.run_blocking(program, limit=60) == expected

    async def test_a_program_is_handed_its_directory_its_environment_and_what_is_given_it(self, world: World) -> None:
        program = world.echoing("echoes")
        match await world.children.run(program, limit=60, cwd=world.folder, env={VARIABLE: VALUE}, given=b"typed\n"):
            case Ran(ending=Exited(code=0), out=out):
                assert out.data == f"{world.folder}\n{VALUE}\n\ntyped\n".encode()
            case other:
                pytest.fail(f"ran as {other}")

    async def test_a_traced_run_is_handed_the_span_it_runs_as(self, world: World) -> None:
        program, began = world.echoing("traced"), wide.begun(world.children.clock)
        match await world.children.run(program, limit=60, cwd=world.folder, env=Traced({VARIABLE: VALUE}, began)):
            case Ran(ending=Exited(code=0), out=out):
                assert out.data == f"{world.folder}\n{VALUE}\n{began.span.traceparent}\n".encode()
            case other:
                pytest.fail(f"ran as {other}")

    async def test_a_program_that_is_not_there_never_starts_and_says_so_as_the_os_does(self, world: World) -> None:
        program = world.unfound()
        for outcome in (await world.children.run(program, limit=60), world.children.run_blocking(program, limit=60)):
            match outcome:
                case NeverStarted(error=FileNotFoundError(errno=errno.ENOENT, filename=named)):
                    assert named == program
                case other:
                    pytest.fail(f"ran as {other}")

    async def test_a_program_that_cannot_be_run_never_starts_and_says_so_as_the_os_does(self, world: World) -> None:
        program = world.unrunnable("unrunnable")
        for outcome in (await world.children.run(program, limit=60), world.children.run_blocking(program, limit=60)):
            match outcome:
                case NeverStarted(error=PermissionError(errno=errno.EACCES, filename=named)):
                    assert named == program
                case other:
                    pytest.fail(f"ran as {other}")

    async def test_a_run_past_its_limit_is_killed_for_it_with_what_it_wrote(self, world: World) -> None:
        running = asyncio.ensure_future(world.children.run(world.hung("hangs", b"out\n", b"err\n"), limit=LIMIT))
        await world.past(LIMIT)
        assert await running == Ran(Killed(Kill(Limit(LIMIT))), Captured(b"out\n"), Captured(b"err\n"))

    def test_a_blocking_run_past_its_limit_is_killed_for_it_with_what_it_wrote(self, world: World) -> None:
        program = world.hung("hangs", b"out\n", b"err\n")
        assert world.children.run_blocking(program, limit=LIMIT) == Ran(Killed(Kill(Limit(LIMIT))), Captured(b"out\n"), Captured(b"err\n"))

    @pytest.mark.parametrize(("argv", "limit"), [(("echo", "a\0b"), 60.0), (("echo",), 0.0), ((), 60.0)], ids=["a NUL", "no time", "no program"])
    async def test_what_no_os_can_be_handed_is_refused_before_anything_runs(self, world: World, argv: tuple[str, ...], limit: float) -> None:
        with pytest.raises(ValueError):
            await world.children.run(*argv, limit=limit)
        with pytest.raises(ValueError):
            world.children.run_blocking(*argv, limit=limit)

