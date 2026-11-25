"""The processes contract (processescontract.py) held to the fake table: in `make check`."""

from itertools import count

import pytest

from fakeprocesses import DEV_TTY, FakeProcesses, Running, processes
from hands.sessions.processes import ControllingTerminal
from hands.virtualclock import VirtualClock
from processescontract import Child, ProcessesContract, Start

OURS = 4000


class Fake:
    """The fake table, in which a child is a row put there, started on a terminal of its own, and gone once a signal or the
    test ends it."""

    def __init__(self) -> None:
        self.clock = VirtualClock()
        self.table: FakeProcesses = processes(Running(OURS, start=self.clock.wall().timestamp()))
        self._pids = count(OURS + 1)
        self._ttys = count(0x10000000)

    @property
    def processes(self) -> FakeProcesses:
        return self.table

    @property
    def ours(self) -> int:
        return OURS

    def wall(self) -> float:
        return self.clock.wall().timestamp()

    def started(self, start: Start) -> Child:
        pid, tty = next(self._pids), None if start.at == "none" else next(self._ttys)
        # Its stdin and stdout as the kernel's child has them: a pipe is no device.
        stdin, stdout = {"none": (None, None), "interactive": (tty, tty), "piped": (tty, None), "reopened": (DEV_TTY, tty), "behind": (tty, tty)}[start.at]
        self.clock.advance(0.001)
        self.table.add(
            Running(
                pid, parent=OURS, terminal=None if tty is None else ControllingTerminal(tty, foreground=start.at != "behind"), start=self.wall(),
                cwd=start.directory, started_in=start.cwd, path=start.program, arguments=start.arguments, environment=start.environment,
                stdin=stdin, stdout=stdout, caught=frozenset() if start.catches is None else frozenset({start.catches}),
            )
        )
        self.clock.advance(0.001)
        return Child(pid, tty)

    def reaped(self, child: Child) -> None:
        assert child.pid not in self.table.running, f"pid {child.pid} has not exited"

    def caught(self, child: Child) -> None:
        assert child.pid in self.table.running, f"pid {child.pid} has exited"

    def exited(self, child: Child) -> None:
        # Through Processes, a process that has exited is gone, reaped or not.
        self.table.end(child.pid)

    def stopped(self, child: Child) -> None:
        self.table.end(child.pid)


@pytest.fixture
def world() -> Fake:
    return Fake()


class TestTheFake(ProcessesContract):
    pass
