"""The pseudoterminals contract (pseudoterminalcontract.py), kept by FakePseudoterminals."""

import errno
from collections.abc import Mapping
from pathlib import Path

import pytest

from fakechildren import NOT_THERE, Refused
from fakepseudoterminals import Asked, FakePseudoterminals, Shows
from pseudoterminalcontract import VARIABLE, PseudoterminalsContract


class FakeWorld:
    def __init__(self, folder: Path) -> None:
        self.fake = FakePseudoterminals()
        self.folder = folder

    @property
    def pseudoterminals(self) -> FakePseudoterminals:
        return self.fake

    @property
    def environment(self) -> Mapping[str, str]:
        return {}

    def program(self, name: str, shown: str, code: int) -> str:
        self.fake.programs[name] = Shows(lambda _asked: shown, code)
        return name

    def echoing(self, name: str) -> str:
        def echoed(asked: Asked) -> str:
            return f"{asked.cwd}\n{asked.environment.get(VARIABLE, '')}\n{asked.environment['TERM']}\n"

        self.fake.programs[name] = Shows(echoed, 0)
        return name

    def hung(self, name: str) -> str:
        self.fake.programs[name] = Shows(lambda _asked: "")
        return name

    def unfound(self) -> str:
        self.fake.programs["absent"] = NOT_THERE
        return "absent"

    def unrunnable(self, name: str) -> str:
        self.fake.programs[name] = Refused(errno.EACCES)
        return name


@pytest.fixture
def world(tmp_path: Path) -> FakeWorld:
    return FakeWorld(tmp_path.resolve())


class TestFakePseudoterminals(PseudoterminalsContract):
    pass


async def test_a_start_of_a_program_the_test_stands_in_for_none_of_is_the_tests_mistake(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no program 'claude' on a terminal"):
        await FakePseudoterminals().start(["claude"], tmp_path, {})


async def test_each_start_is_kept_as_its_program_is_handed_it(tmp_path: Path) -> None:
    fake = FakePseudoterminals({"claude": NOT_THERE, "sh": Shows(lambda _asked: "")})
    await fake.start(["claude", "--print"], tmp_path, {"HOME": "/h"})
    await fake.start(["sh"], tmp_path, {})
    assert fake.asked == [
        Asked(("claude", "--print"), tmp_path, {"HOME": "/h", "TERM": "xterm-256color"}),
        Asked(("sh",), tmp_path, {"TERM": "xterm-256color"}),
    ]
