"""`Processes` as values a test sets: a table of processes, each with the facts the kernel answers from, and the signals sent
to them. tests/processescontract.py holds it to what `hands.sessions.kernel.KERNEL` says of real processes."""

import errno
import os
import signal
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from hands.sessions.processes import ControllingTerminal, Process, StartedAs, TerminalProcess, Undescribed
from processescontract import KILL, LAUNCHD, PIDPATH, PROCARGS2, VNODEPATHINFO

# The user a fake's processes run as, and hands with them, unless a test names another.
USER = 501
# The device of /dev/tty, the name each process has for its own controlling terminal: major 2, minor 0.
DEV_TTY = 2 << 24
# The call that says when a process started, as the OSError it raises names it.
PROC_PID = "kern.proc.pid"


def fd_info(fd: int) -> str:
    """The call that says which file a process has open as `fd`."""
    return f"proc_pidfdinfo of fd {fd}"


@dataclass(frozen=True)
class Running:
    """A process in a fake table, as the facts the kernel answers from."""

    pid: int
    parent: int = LAUNCHD
    uid: int = USER
    # The device of its controlling terminal; None for one with none.
    tty: int | None = None
    # Whether its group is its terminal's foreground process group.
    foreground: bool = False
    # When it started, in wall-clock seconds: long before anything a test writes, unless it says otherwise.
    start: float = 0.0
    cwd: Path = Path("/")
    # The path it was exec'd by, as execve was given it.
    path: Path = Path("/bin/sh")
    # The file it runs, as the kernel names it, links resolved; None where the kernel names none, as for one deleted.
    program: Path | None = None
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict[str, str])
    # The device of the file it has open as its stdin and as its stdout, as `st_rdev`; None for a pipe, or none open.
    stdin: int | None = None
    stdout: int | None = None
    # Each call the kernel refuses about it, with the errno it refuses with: as kern.procargs2 refuses one caught
    # mid-exec, kill another user's process, and kern.proc.pid any other process in a sandbox that holds a run to itself.
    refused: Mapping[str, int] = field(default_factory=dict[str, int])
    # The signals it catches and lives on; any other ends it.
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
        return {each.pid: Process(each.pid, each.parent, each.uid, None if each.tty is None else ControllingTerminal(each.tty, each.foreground)) for each in self.running.values()}

    def starts(self, pids: Collection[int]) -> dict[int, float]:
        for pid in pids:
            if (refusal := self._refusal(pid, PROC_PID)) is not None:
                raise OSError(refusal, f"sysctl could not say when pid {pid} started: {os.strerror(refusal)}")
        return {pid: self.running[pid].start for pid in pids if pid in self.running}

    def started_as(self, process: Process) -> StartedAs | Undescribed | None:
        match self._described(process.pid, PROCARGS2):
            case Running(path=path, arguments=arguments, environment=environment):
                return StartedAs(path, arguments, environment)
            case other:
                return other

    def program_of(self, process: Process) -> Path | Undescribed | None:
        match self._described(process.pid, PIDPATH):
            case Running(program=Path() as program):
                return program
            case Running(pid=pid, parent=parent):
                return Undescribed(pid, parent, PIDPATH, errno.ENOENT)
            case other:
                return other

    def directory_of(self, process: Process) -> Path | Undescribed | None:
        match self._described(process.pid, VNODEPATHINFO):
            case Running(cwd=cwd):
                return cwd
            case other:
                return other

    def interactive(self, process: TerminalProcess) -> bool | Undescribed | None:
        # As the kernel reads them: fd 0's file, then fd 1's, each its terminal under its device or /dev/tty's.
        match (self._described(process.pid, fd_info(0)), self._described(process.pid, fd_info(1))):
            case (Running(stdin=stdin, stdout=stdout), Running()):
                return {stdin, stdout} <= {process.tty, DEV_TTY}
            case (Undescribed() as refused, _) | (_, Undescribed() as refused):
                return refused
            case _:
                return None

    def signal(self, pid: int, signum: signal.Signals) -> bool:
        if (refusal := self._refusal(pid, KILL)) is not None:
            # As os.kill raises it, naming nothing but the errno.
            raise OSError(refusal, os.strerror(refusal))
        self.signalled.append((pid, signum))
        sent = self.running.get(pid)
        if sent is None:
            return False
        if signum not in sent.caught:
            self.end(pid)
        return True

    def _refusal(self, pid: int, call: str) -> int | None:
        found = self.running.get(pid)
        return None if found is None else found.refused.get(call)

    def _described(self, pid: int, call: str) -> Running | Undescribed | None:
        found = self.running.get(pid)
        if found is None or call not in found.refused:
            return found
        return Undescribed(found.pid, found.parent, call, found.refused[call])


def processes(*running: Running) -> FakeProcesses:
    """A fake table holding launchd, root's, which the kernel will not say what it runs or where, nor signal, and
    `running`."""
    launchd = Running(LAUNCHD, parent=0, uid=0, refused={PROCARGS2: errno.EINVAL, VNODEPATHINFO: errno.EPERM, KILL: errno.EPERM})
    return FakeProcesses({LAUNCHD: launchd} | {each.pid: each for each in running})
