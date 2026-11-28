"""The children contract (childrencontract.py) held to SpawnedChildren, on programs this test writes and the OS runs:
outside, since it starts processes."""

import asyncio
import queue
from pathlib import Path

import pytest

from childrencontract import VARIABLE, ChildrenContract
from hands.clock import SystemClock
from hands.sessions import wide
from hands.sessions.child import SpawnedChildren

pytestmark = pytest.mark.outside

# How often a wait held back by AfterWriting looks again at what has been written.
LOOK = 0.01


class AfterWriting(SystemClock):
    """[LAW:no-ambient-temporal-coupling] the machine's clock, on which a wait begins to run down only once every hung
    program has written what it writes, as the file each touches once it has says: the kill at the limit the contract
    asserts comes after the writing, however long the machine takes to start the program. A run that ends first ends
    the wait, as it ends any."""

    def __init__(self) -> None:
        self.written: list[Path] = []

    def sleep(self, seconds: float) -> asyncio.Future[None]:
        return asyncio.ensure_future(self._after_writing(seconds))

    async def _after_writing(self, seconds: float) -> None:
        while not all(marker.exists() for marker in self.written):
            await asyncio.sleep(LOOK)
        await super().sleep(seconds)

    def take[T](self, source: queue.Queue[T] | queue.SimpleQueue[T], seconds: float) -> T:
        while not all(marker.exists() for marker in self.written):
            try:
                return source.get(timeout=LOOK)
            except queue.Empty:
                pass
        return super().take(source, seconds)


class Real:
    """Shell scripts in a directory of the test's own, run by a SpawnedChildren on the machine's clock, held back until
    each hung one has written. Each names every program it runs by its path, since a run is handed no PATH but the one
    its caller gives it."""

    def __init__(self, tmp_path: Path) -> None:
        self.events: list[wide.WideEvent] = []
        self.clock = AfterWriting()
        self.runner = SpawnedChildren(self.clock, self.events.append)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self._folder = (tmp_path / "folder").resolve()
        self._folder.mkdir()

    @property
    def children(self) -> SpawnedChildren:
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

    def hung(self, name: str, out: bytes, err: bytes) -> str:
        (self.bin / f"{name}.out").write_bytes(out)
        (self.bin / f"{name}.err").write_bytes(err)
        written = self.bin / f"{name}.written"
        self.clock.written.append(written)
        return self._script(name, f"/bin/cat '{self.bin}/{name}.out'\n/bin/cat '{self.bin}/{name}.err' >&2\n/usr/bin/touch '{written}'\nexec /bin/sleep 60\n")

    def unfound(self) -> str:
        return str(self.bin / "absent")

    def unrunnable(self, name: str) -> str:
        script = self.bin / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o644)
        return str(script)

    async def past(self, seconds: float) -> None:
        # The machine's time passes of itself, from once every hung program has written.
        pass

    def _script(self, name: str, body: str) -> str:
        script = self.bin / name
        script.write_text(f"#!/bin/sh\n{body}")
        script.chmod(0o755)
        return str(script)


@pytest.fixture
def world(tmp_path: Path) -> Real:
    return Real(tmp_path)


class TestSpawnedChildren(ChildrenContract):
    pass
