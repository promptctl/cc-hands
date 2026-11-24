"""`Processes` as values a test sets: a table of processes, each with what the kernel would say of it, and the signals sent
to them. tests/processescontract.py holds it to what `hands.sessions.kernel.KERNEL` says of real processes."""

import errno
import signal
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from hands.sessions.processes import Process, StartedAs, TerminalProcess, Undescribed

# The user a fake's processes run as, and hands with them, unless a test names another.
USER = 501
# launchd: root's, the first process, whose parent the kernel keeps out of the table.
LAUNCHD = 1


@dataclass(frozen=True)
class Running:
    """A process in a fake table, as the kernel would describe it."""

    pid: int
    parent: int = LAUNCHD
    uid: int = USER
    # The device of its controlling terminal; None for one with none.
    tty: int | None = None
    # Whether its group is the one in front of its terminal.
    in_front: bool = False
    # When it started, in wall-clock seconds.
    start: float = 0.0
    cwd: Path = Path("/")
    # The path it was exec'd by, as execve was given it.
    path: Path = Path("/bin/sh")
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict[str, str])
    # Whether its stdin and stdout are its terminal.
    interactive: bool = False
    # The call the kernel refuses with EIO about it, as one caught mid-exec is refused; None for one it describes.
    refused: str | None = None
    # The signals it catches and lives on; every other one sent ends it, as each signal's default action does.
    caught: frozenset[signal.Signals] = frozenset()


@dataclass
class FakeProcesses:
    """[LAW:one-source-of-truth] one table of `Running` values, which every question is answered from."""

    running: dict[int, Running]
    user: int = USER
    # Every signal sent through it, to a pid that had a process or not, in order.
    signalled: list[tuple[int, signal.Signals]] = field(default_factory=list[tuple[int, signal.Signals]])

    def add(self, *running: Running) -> None:
        self.running.update((each.pid, each) for each in running)

    def end(self, pid: int) -> None:
        """The process exits and is reaped: the kernel has nothing more of it."""
        del self.running[pid]

    def table(self) -> dict[int, Process]:
        return {each.pid: Process(each.pid, each.parent, each.uid, each.tty, each.tty if each.in_front else None) for each in self.running.values()}

    def starts(self, pids: Collection[int]) -> dict[int, float]:
        return {pid: self.running[pid].start for pid in pids if pid in self.running}

    def started_as(self, process: Process) -> StartedAs | Undescribed | None:
        match self._described(process):
            case Running(path=path, arguments=arguments, environment=environment):
                return StartedAs(path, arguments, environment)
            case other:
                return other

    def directory_of(self, process: Process) -> Path | Undescribed | None:
        match self._described(process):
            case Running(cwd=cwd):
                return cwd
            case other:
                return other

    def interactive(self, process: TerminalProcess) -> bool | Undescribed:
        match self._described(Process(process.pid, process.parent, self.user, process.tty, None)):
            case Running(interactive=interactive):
                return interactive
            case None:
                return False
            case Undescribed() as refused:
                return refused

    def signal(self, pid: int, signum: signal.Signals) -> bool:
        self.signalled.append((pid, signum))
        sent = self.running.get(pid)
        if sent is None:
            return False
        if signum not in sent.caught:
            self.end(pid)
        return True

    def _described(self, process: Process) -> Running | Undescribed | None:
        found = self.running.get(process.pid)
        if found is None or found.refused is None:
            return found
        return Undescribed(found.pid, found.parent, found.refused, errno.EIO)


def processes(*running: Running, user: int = USER) -> FakeProcesses:
    """A fake table holding launchd and `running`."""
    return FakeProcesses({LAUNCHD: Running(LAUNCHD, parent=0, uid=0)} | {each.pid: each for each in running}, user)
