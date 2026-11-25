"""The children contract (childrencontract.py) held to the fake (fakechildren.py): in `make check`."""

import asyncio
from pathlib import Path

import pytest

from childrencontract import VARIABLE, ChildrenContract
from fakechildren import Asked, FakeChildren, exited
from hands.sessions.child import Outcome
from hands.virtualclock import VirtualClock


class Fake:
    """Programs as values on a clock the test moves, and a directory no program is run in."""

    def __init__(self, folder: Path) -> None:
        self.clock = VirtualClock()
        self.fake = FakeChildren(self.clock)
        self._folder = folder

    @property
    def children(self) -> FakeChildren:
        return self.fake

    @property
    def folder(self) -> Path:
        return self._folder

    def program(self, name: str, code: int, out: bytes, err: bytes) -> str:
        self.fake.programs[name] = exited(code, out, err)
        return name

    def echoing(self, name: str) -> str:
        async def echoes(asked: Asked) -> Outcome:
            env = asked.env or {}
            told = f"{asked.cwd}\n{env.get(VARIABLE, '')}\n{env.get('TRACEPARENT', '')}\n".encode()
            return exited(0, told + (asked.given or b""))

        self.fake.programs[name] = echoes
        return name

    def waiting(self, name: str) -> str:
        async def waits(asked: Asked) -> Outcome:
            return await asyncio.get_running_loop().create_future()

        self.fake.programs[name] = waits
        return name

    def unfound(self) -> str:
        return "absent"

    async def past(self, seconds: float) -> None:
        await self.clock.sleeping(self.clock.now() + seconds)
        self.clock.advance(seconds)


@pytest.fixture
def world(tmp_path: Path) -> Fake:
    return Fake(tmp_path)


class TestTheFake(ChildrenContract):
    pass


def test_a_program_answered_in_a_coroutine_cannot_be_run_blocking(world: Fake) -> None:
    with pytest.raises(AssertionError, match="blocking"):
        world.children.run_blocking(world.echoing("echoes"), limit=60)


async def test_every_run_is_kept_as_its_program_is_handed_it(world: Fake) -> None:
    await world.children.run(world.program("speaks", 0, b"", b""), "an argument", limit=60, cwd=world.folder, env={VARIABLE: "x"}, given=b"typed")
    world.children.run_blocking("absent", limit=5)
    assert world.fake.asked == [Asked(("speaks", "an argument"), world.folder, {VARIABLE: "x"}, b"typed", 60), Asked(("absent",), None, None, None, 5)]
