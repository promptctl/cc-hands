"""What hands relies on of the Mac's processes, as one contract held to two parties: the fake table (tests/fakeprocesses.py),
in `make check` (test_processes_contract.py), and the kernel itself through `hands.sessions.kernel.KERNEL`, on children this
test starts (test_processes_contract_kernel.py, outside). The cases are written once, here, and each of those modules runs
all of them against its own `world`, so each asserts the same of both: that is what holds the fake to the kernel. A claim the
kernel no longer keeps fails the kernel's run, and the fake is changed to match."""

import signal
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import pytest

from hands.sessions.processes import Processes, StartedAs, TerminalProcess, ancestors, terminal_processes

# How a child is started on a terminal: none, with no controlling terminal at all; or controlled by a terminal of the test's
# own, reading and writing it (`interactive`), reading it and writing a pipe (`piped`), or writing it and reading it by
# the name each process has for its own, /dev/tty, as `xargs -o` and `< /dev/tty` give a run its stdin (`reopened`).
At = Literal["none", "interactive", "piped", "reopened"]
# How far a start may be read before the wall clock the test read just before it: the kernel keeps microseconds.
START_SLACK = 0.01


@dataclass(frozen=True)
class Start:
    """A child for the world to start, as a test of the contract asks for it."""

    program: Path
    arguments: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] = field(default_factory=dict[str, str])
    at: At = "none"


@dataclass(frozen=True)
class Child:
    """A child the world started: its pid, and the device of the terminal it was started on, if any."""

    pid: int
    tty: int | None


class World(Protocol):
    """The processes the contract asks after, and the children it starts among them."""

    @property
    def processes(self) -> Processes: ...

    @property
    def ours(self) -> int:
        """The pid every child the world starts is started by."""
        ...

    def wall(self) -> float:
        """The wall clock, in seconds."""
        ...

    def started(self, start: Start) -> Child:
        """A child running `start`'s program, once the kernel says it runs it."""
        ...

    def reaped(self, child: Child) -> None:
        """Once `child` has exited by itself or a signal, reaped, so the kernel has nothing more of it."""
        ...

    def stopped(self, child: Child) -> None:
        """`child` ended and reaped."""
        ...


# The first process, root's, which every other descends from.
LAUNCHD = 1
# What a child runs: Python waiting on a signal, not /bin/sleep, since the kernel says nothing of the environment of a
# program of the system's own, whoever started it.
PYTHON = Path(sys.executable)
WAITING = ("-c", "import signal; signal.pause()")


class ProcessesContract:
    """The claims, one method each. A holder subclasses it and gives a `world` fixture, so every claim added here is held to
    both parties without either holder naming it."""

    def test_a_child_is_in_the_table_under_its_parent_as_the_users_own_with_no_terminal_unless_started_on_one(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, WAITING, tmp_path))
        try:
            process = world.processes.table()[child.pid]
            assert (process.parent, process.uid, process.tty, process.front) == (world.ours, world.processes.user, None, None)
        finally:
            world.stopped(child)

    def test_a_child_started_on_a_terminal_is_controlled_by_it_and_in_front_of_it(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, WAITING, tmp_path, at="interactive"))
        try:
            process = world.processes.table()[child.pid]
            assert child.tty is not None
            assert (process.tty, process.front) == (child.tty, child.tty)
        finally:
            world.stopped(child)

    def test_a_childs_start_is_when_it_was_started_and_a_gone_pid_has_none(self, world: World, tmp_path: Path) -> None:
        before = world.wall()
        child = world.started(Start(PYTHON, WAITING, tmp_path))
        after = world.wall()
        try:
            # launchd's is root's, and its start is said all the same.
            starts = world.processes.starts({child.pid, world.ours, LAUNCHD})
            assert before - START_SLACK <= starts[child.pid] <= after
            assert starts[LAUNCHD] <= starts[world.ours] <= starts[child.pid]
        finally:
            world.stopped(child)
        assert world.processes.starts({child.pid}) == {}
        assert world.processes.starts(set()) == {}

    def test_what_a_child_was_started_as_and_where_it_runs_are_said(self, world: World, tmp_path: Path) -> None:
        folder = tmp_path.resolve()
        child = world.started(Start(PYTHON, WAITING, folder, {"HANDS_CONTRACT": "yes"}))
        try:
            process = world.processes.table()[child.pid]
            match world.processes.started_as(process):
                case StartedAs(path=path, arguments=arguments, environment=environment):
                    assert (path, arguments, environment.get("HANDS_CONTRACT")) == (PYTHON, WAITING, "yes"), (path, arguments, environment)
                case other:
                    pytest.fail(f"started as {other}")
            assert world.processes.directory_of(process) == folder
        finally:
            world.stopped(child)

    def test_a_child_gone_since_it_was_listed_is_described_as_gone(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, WAITING, tmp_path, at="interactive"))
        process = world.processes.table()[child.pid]
        [listed] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
        world.stopped(child)
        assert child.pid not in world.processes.table()
        assert (world.processes.started_as(process), world.processes.directory_of(process), world.processes.interactive(listed)) == (None, None, False)

    @pytest.mark.parametrize(("at", "reads"), [("interactive", True), ("piped", False), ("reopened", True)])
    def test_a_child_reads_and_writes_its_terminal_only_where_its_stdin_and_stdout_are_both_it(self, world: World, tmp_path: Path, at: At, reads: bool) -> None:
        child = world.started(Start(PYTHON, WAITING, tmp_path, at=at))
        try:
            [process] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
            assert isinstance(process, TerminalProcess) and process.executable == PYTHON.resolve()
            assert world.processes.interactive(process) is reads
        finally:
            world.stopped(child)

    def test_a_signal_is_sent_to_a_running_child_and_ends_it_and_none_is_sent_to_a_gone_pid(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, WAITING, tmp_path))
        assert world.processes.signal(child.pid, signal.SIGTERM)
        world.reaped(child)
        assert child.pid not in world.processes.table()
        assert not world.processes.signal(child.pid, signal.SIGTERM)

    def test_a_childs_line_of_parents_ends_at_launchd_which_is_roots_and_whose_parent_is_not_in_the_table(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, WAITING, tmp_path))
        try:
            table = world.processes.table()
            line = list(ancestors(child.pid, table))
        finally:
            world.stopped(child)
        assert [line[0].pid, line[1].pid] == [child.pid, world.ours]
        # A terminal app starts each tab's shell through /usr/bin/login, which runs as root: a table of this user's processes
        # alone breaks the line between a session and its app there.
        assert (line[-1].pid, line[-1].uid) == (LAUNCHD, 0)
        assert line[-1].parent not in table
