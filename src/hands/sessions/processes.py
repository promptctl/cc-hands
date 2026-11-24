"""The processes on the Mac as hands asks after them: the process table, when each started, what one was started as and
where it runs, whether it reads and writes its terminal, and a signal sent to one; and, from those, which of the user's
processes run at a terminal and whether a pid still names the process it named before.

[LAW:effects-at-boundaries] `Processes` is the one way hands reaches another process. `hands.sessions.kernel.KERNEL` asks
the kernel, the composition root hands it on, and tests/fakeprocesses.py holds a table as values; tests/processescontract.py
holds the two to one contract. Children hands starts and waits on are `hands.sessions.child`'s.

A pid is a number the OS hands out again. A running process under a remembered pid is the remembered process only if it
was already running when it was remembered; one that started later took the number of one that is dead. The session
sweep and the daemon's heartbeat both ask that, and both answer it here.
"""

import signal
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hands.sessions.payload import Rejected

# The largest pid macOS hands out.
PID_MAX = 99999

# A process starts before it can write down that it is running, so its start is never later than that record. The
# slack covers a wall clock stepped back between the two, not pid reuse, which on macOS does not come round in a second.
START_SLACK_SECONDS = 1.0


@dataclass(frozen=True)
class Process:
    """A process of any user's: its parent, its effective user, and the device number of the terminal it is controlled
    by, which is `st_rdev` of that terminal's /dev path; None for a process with no controlling terminal, as an app or a
    daemon has none."""

    pid: int
    parent: int
    uid: int
    tty: int | None
    # Its controlling terminal again when its process group is the one in front of that terminal, the group a key typed
    # there goes to; None for a process with no terminal, or one behind another group at it, as a stopped job is.
    front: int | None


@dataclass(frozen=True)
class StartedAs:
    """What a process was started as: the path it was exec'd by, as execve was given it, its arguments after the
    program, and its environment."""

    path: Path
    arguments: tuple[str, ...]
    environment: Mapping[str, str]


@dataclass(frozen=True)
class Undescribed:
    """A process of this user's the kernel would not describe while it runs: one caught mid-exec, or with its memory
    torn down on the way out, is answered EIO by kern.procargs2."""

    pid: int
    parent: int
    # The question the kernel refused, and its errno.
    call: str
    errno: int


@dataclass(frozen=True)
class TerminalProcess:
    """A process of this user's with a controlling terminal."""

    pid: int
    parent: int
    # The program it was started as, with links resolved as they are now: a file deleted since, as an updater prunes
    # old versions, is still named.
    executable: Path
    cwd: Path
    environment: Mapping[str, str]
    # Its arguments after the program, as it was started with them.
    arguments: tuple[str, ...]
    # The device of its controlling terminal, as Process.tty.
    tty: int


@dataclass(frozen=True)
class TerminalProcesses:
    """This user's processes at a terminal, and those of them the kernel would not say what they run."""

    found: list[TerminalProcess]
    unread: list[Undescribed]


class Processes(Protocol):
    """The Mac's processes, as they stand when asked. A question about a process that has exited since it was listed is
    answered None, and one the kernel refused about a process that runs is answered with the refusal."""

    @property
    def user(self) -> int:
        """The effective uid hands runs as: whose processes are the user's own."""
        ...

    def table(self) -> dict[int, Process]:
        """Every process on the Mac but the kernel's, by pid: root's too, as the /usr/bin/login a terminal app starts each
        tab through, so a line of parents runs unbroken from a session up to its app."""
        ...

    def starts(self, pids: Collection[int]) -> dict[int, float]:
        """When each pid's process started, in wall-clock seconds; a pid no process has is absent."""
        ...

    def started_as(self, process: Process) -> StartedAs | Undescribed | None:
        """What a process of this user's was started as."""
        ...

    def directory_of(self, process: Process) -> Path | Undescribed | None:
        """The directory a process of this user's runs in."""
        ...

    def interactive(self, process: TerminalProcess) -> bool | Undescribed:
        """Whether its stdin and stdout are both its terminal, as `[ -t 0 ] && [ -t 1 ]` finds them at a terminal: its
        controlling terminal's device, or /dev/tty, which is that terminal under the name each process has for its own.
        False for one that has exited since it was listed, which reads nothing."""
        ...

    def signal(self, pid: int, signum: signal.Signals) -> bool:
        """Send `signum` to the process `pid` names: whether it was sent, False where no process has the pid. Raises
        OSError where the kernel refused it."""
        ...


def parse_pid(number: int) -> int:
    """A number read from a file, as a pid some process could have."""
    # [LAW:parse-dont-validate] refused here, so nothing ever asks the kernel about a number no process can have.
    if not 0 < number <= PID_MAX:
        raise Rejected(f"pid {number} is not a process id")
    return number


def still_running(pid: int, seen_at: float, starts: Mapping[int, float]) -> bool:
    """Whether the process running under pid now is the one that was running under it at seen_at (wall-clock seconds)."""
    start = starts.get(pid)
    # [LAW:single-enforcer] the one test for a reused pid: a process that started after seen_at took the number of
    # the one that was seen, which is dead.
    return start is not None and start <= seen_at + START_SLACK_SECONDS


def terminal_processes(processes: Processes) -> TerminalProcesses:
    """Every process of this user's that has a controlling terminal: interactive programs, never daemons or apps."""
    own = processes.user
    read = [_at_terminal(processes, process, process.tty) for process in processes.table().values() if process.uid == own and process.tty is not None]
    # [LAW:no-silent-failure] one process the kernel will not describe is named, and hides none of the others.
    return TerminalProcesses([each for each in read if isinstance(each, TerminalProcess)], [each for each in read if isinstance(each, Undescribed)])


def environment_of(processes: Processes, process: Process) -> Mapping[str, str] | Undescribed | None:
    """The environment a process of this user's was started with."""
    match processes.started_as(process):
        case StartedAs(environment=environment):
            return environment
        case other:
            return other


def _at_terminal(processes: Processes, process: Process, tty: int) -> TerminalProcess | Undescribed | None:
    """The process at a terminal, with where it runs and what it was started as."""
    match (processes.directory_of(process), processes.started_as(process)):
        case (Path() as cwd, StartedAs(path=path, arguments=arguments, environment=environment)):
            # A path exec'd relative to the directory the process was started in, which a session leaves for a worktree
            # and the kernel keeps no record of. [LAW:one-source-of-truth] The PWD its shell started it with names that
            # directory, in the one record the path itself is read from; a program started with none, or one that is no
            # absolute path, is taken to be where it started.
            return TerminalProcess(process.pid, process.parent, (cwd / environment.get("PWD", "") / path).resolve(), cwd, environment, arguments, tty)
        case (Undescribed() as refused, _) | (_, Undescribed() as refused):
            return refused
        case _:
            # It exited between the two questions.
            return None


def ancestors(pid: int, processes: Mapping[int, Process]) -> Iterator[Process]:
    """pid's line of ancestors in `processes`, pid itself first."""
    process = processes.get(pid)
    while process is not None:
        yield process
        process = processes.get(process.parent)


def ancestor_terminals(pid: int, processes: Mapping[int, Process]) -> Iterator[int]:
    """Every terminal on pid's line of ancestors in `processes`, its own first."""
    return (process.tty for process in ancestors(pid, processes) if process.tty is not None)


def front_terminal(pid: int, processes: Mapping[int, Process]) -> Iterator[int]:
    """pid's controlling terminal while pid is in front of it, so a key typed there reaches it; nothing otherwise."""
    process = processes.get(pid)
    if process is not None and process.front is not None:
        yield process.front
