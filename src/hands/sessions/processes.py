"""The processes on the Mac as hands asks after them: the process table, when each started, what one was started as and
where it runs, whether it reads and writes its terminal, and a signal sent to one; and, from those, which of the user's
processes run at a terminal and whether a pid still names the process it named before.

[LAW:effects-at-boundaries] `Processes` is the way the daemon and its sessions reach another process.
`hands.sessions.kernel.KERNEL` asks the kernel, the composition root hands it on, and tests/fakeprocesses.py holds a table as
values; tests/processescontract.py holds the two to one contract. Children hands starts and waits on are
`hands.sessions.child`'s. The brain still reads and signals its own pane's processes itself, each read marked as the
exception it is, until promptctl-testing-c3u.fhu puts the brain on its fakes.

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
class ControllingTerminal:
    """The terminal a process is controlled by: its device number, which is `st_rdev` of that terminal's /dev path, and
    whether the process's group is that terminal's foreground process group, the one a key typed there goes to; a job
    stopped there is in the background."""

    device: int
    foreground: bool


@dataclass(frozen=True)
class Process:
    """A process of any user's: its parent, its effective user, and its controlling terminal; None for a process with
    none, as an app or a daemon has none."""

    pid: int
    parent: int
    uid: int
    terminal: ControllingTerminal | None


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
    # The program it runs: by the path it was exec'd by where that is absolute, so a file deleted since, as an updater
    # prunes old versions, is still named, and a link is named as the link; by the file the kernel says it runs where it
    # was exec'd by a relative one.
    executable: Path
    cwd: Path
    environment: Mapping[str, str]
    # Its arguments after the program, as it was started with them.
    arguments: tuple[str, ...]
    # The device of its controlling terminal, as ControllingTerminal.device.
    tty: int


@dataclass(frozen=True)
class TerminalProcesses:
    """This user's processes at a terminal, and those of them the kernel would not say what they run."""

    found: list[TerminalProcess]
    unread: list[Undescribed]


class Processes(Protocol):
    """The Mac's processes, as they stand when asked. A process that has exited is gone, whether or not its parent has
    reaped it yet: absent from the table and the starts, sent no signal, and a question about it, since it was listed,
    answered None. A question the kernel refused about a process that runs is answered with the refusal; the table, the
    starts and a signal, which the kernel answers for every process, raise OSError where it refused."""

    @property
    def user(self) -> int:
        """The effective uid hands runs as: whose processes are the user's own."""
        ...

    def table(self) -> dict[int, Process]:
        """Every process on the Mac but the kernel's, by pid: root's too, as the /usr/bin/login a terminal app starts each
        tab through, so a line of parents runs unbroken from a session up to its app."""
        ...

    def starts(self, pids: Collection[int]) -> dict[int, float]:
        """When each pid's process started, in wall-clock seconds; a pid no process has is absent. Raises OSError where
        the kernel refused one."""
        ...

    def started_as(self, process: Process) -> StartedAs | Undescribed | None:
        """What a process of this user's was started as."""
        ...

    def program_of(self, process: Process) -> Path | Undescribed | None:
        """The file a process of this user's runs, as the kernel names it, its links resolved; refused for one deleted since
        it started, which the kernel names none of."""
        ...

    def directory_of(self, process: Process) -> Path | Undescribed | None:
        """The directory a process of this user's runs in."""
        ...

    def interactive(self, process: TerminalProcess) -> bool | Undescribed | None:
        """Whether its stdin and stdout are both its terminal, as `[ -t 0 ] && [ -t 1 ]` finds them at a terminal: its
        controlling terminal's device, or /dev/tty, which is that terminal under the name each process has for its own."""
        ...

    def signal(self, pid: int, signum: signal.Signals) -> bool:
        """Send `signum` to the process `pid` names: whether it was sent, False where no process has the pid. Raises
        Rejected for a number no process could have, as 0 or a negative one, which would name a group; and OSError where
        the kernel refused it."""
        ...


@dataclass(frozen=True)
class TableRefused:
    """The kernel refused the process table, and what it said."""

    reason: str


async def table_of(processes: Processes) -> dict[int, Process] | TableRefused:
    """[LAW:single-enforcer] the process table, read off the event loop, or the refusal the kernel answered with, which
    Processes declares and which is no fault of hands'."""
    # Imported here rather than with the module, which the hook shim imports: as hands.clock has it.
    import asyncio

    try:
        return await asyncio.to_thread(processes.table)
    except OSError as refused:
        return TableRefused(f"the processes could not be read: {refused}")


def parse_pid(number: int) -> int:
    """A number read from a file, as a pid some process could have."""
    # [LAW:parse-dont-validate] refused here, so no pid a file names, the only pids hands signals, is 0 or negative: a
    # signal to either goes to a whole group of processes.
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
    read = [_at_terminal(processes, process, process.terminal.device) for process in processes.table().values() if process.uid == own and process.terminal is not None]
    # [LAW:no-silent-failure] one process the kernel will not describe is named, and hides none of the others.
    return TerminalProcesses([each for each in read if isinstance(each, TerminalProcess)], [each for each in read if isinstance(each, Undescribed)])


def _at_terminal(processes: Processes, process: Process, tty: int) -> TerminalProcess | Undescribed | None:
    """The process at a terminal, with where it runs and what it was started as."""
    match (processes.directory_of(process), processes.started_as(process)):
        case (Path() as cwd, StartedAs(path=path, arguments=arguments, environment=environment)):
            # A path exec'd relative to the directory the process was started in, which a session leaves for a worktree
            # and the kernel keeps no record of, names no file once it has moved; nor does its PWD, which its parent may
            # never have kept to where it ran. [LAW:one-source-of-truth] The kernel's own name for the file it runs does.
            match path if path.is_absolute() else processes.program_of(process):
                case Path() as executable:
                    return TerminalProcess(process.pid, process.parent, executable, cwd, environment, arguments, tty)
                case unnamed:
                    return unnamed
        case (Undescribed() as refused, _) | (_, Undescribed() as refused):
            return refused
        case _:
            # It exited between the two questions.
            return None


def ancestors(pid: int, table: Mapping[int, Process]) -> Iterator[Process]:
    """pid's line of ancestors in the process table, pid itself first."""
    process = table.get(pid)
    while process is not None:
        yield process
        process = table.get(process.parent)


def ancestor_terminals(pid: int, table: Mapping[int, Process]) -> Iterator[int]:
    """Every terminal on pid's line of ancestors in the process table, its own first."""
    return (process.terminal.device for process in ancestors(pid, table) if process.terminal is not None)


def foreground_terminal(pid: int, table: Mapping[int, Process]) -> Iterator[int]:
    """pid's controlling terminal while pid is in its foreground process group, so a key typed there reaches it; nothing
    otherwise."""
    match table.get(pid):
        case Process(terminal=ControllingTerminal(device=device, foreground=True)):
            yield device
        case _:
            pass
