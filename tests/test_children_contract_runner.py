"""The children contract (childrencontract.py) held to the Runner, on programs this test writes and the OS runs:
outside, since it starts processes."""

from pathlib import Path

import pytest

from childrencontract import VARIABLE, ChildrenContract
from hands.clock import SYSTEM
from hands.sessions import wide
from hands.sessions.child import Runner

pytestmark = pytest.mark.outside


class Real:
    """Shell scripts in a directory of the test's own, run by a Runner on the machine's clock. Each names every program
    it runs by its path, since a run is handed no PATH but the one its caller gives it."""

    def __init__(self, tmp_path: Path) -> None:
        self.events: list[wide.WideEvent] = []
        self.runner = Runner(SYSTEM, self.events.append)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self._folder = (tmp_path / "folder").resolve()
        self._folder.mkdir()

    @property
    def children(self) -> Runner:
        return self.runner

    @property
    def folder(self) -> Path:
        return self._folder

    def program(self, name: str, code: int, out: bytes, err: bytes) -> str:
        (self.bin / f"{name}.out").write_bytes(out)
        (self.bin / f"{name}.err").write_bytes(err)
        return self._script(name, f"/bin/cat '{self.bin}/{name}.out'\n/bin/cat '{self.bin}/{name}.err' >&2\nexit {code}\n")

    def echoing(self, name: str) -> str:
        return self._script(name, f"/bin/pwd -P\nprintf '%s\\n' \"${VARIABLE}\" \"$TRACEPARENT\"\nexec /bin/cat\n")

    def waiting(self, name: str) -> str:
        return self._script(name, "exec /bin/sleep 60\n")

    def unfound(self) -> str:
        return str(self.bin / "absent")

    async def past(self, seconds: float) -> None:
        # The machine's time passes of itself.
        pass

    def _script(self, name: str, body: str) -> str:
        script = self.bin / name
        script.write_text(f"#!/bin/sh\n{body}")
        script.chmod(0o755)
        return str(script)


@pytest.fixture
def world(tmp_path: Path) -> Real:
    return Real(tmp_path)


class TestTheRunner(ChildrenContract):
    pass
