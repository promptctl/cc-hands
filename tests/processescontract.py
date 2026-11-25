"""What hands relies on of the Mac's processes, as one contract held to two parties: the fake table (tests/fakeprocesses.py),
in `make check` (test_processes_contract.py), and the kernel itself through `hands.sessions.kernel.KERNEL`, on children this
test starts (test_processes_contract_kernel.py, outside). The cases are written once, here, and each of those modules runs
all of them against its own `world`, so each asserts the same of both: that is what holds the fake to the kernel. A claim the
kernel no longer keeps fails the kernel's run, and the fake is changed to match."""

import os
import signal
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import pytest

from hands.sessions.payload import Rejected
from hands.sessions.processes import ControllingTerminal, Processes, StartedAs, TerminalProcess, Undescribed, ancestors, terminal_processes

# How a child is started on a terminal: none, with no controlling terminal at all; or controlled by a terminal of the test's
# own, reading and writing it (`interactive`), reading it and writing a pipe (`piped`), or writing it and reading it by
# the name each process has for its own, /dev/tty, as `xargs -o` and `< /dev/tty` give a run its stdin (`reopened`); or
# reading and writing it with another of its session's process groups in its foreground, as a job stopped there has
# (`behind`).
At = Literal["none", "interactive", "piped", "reopened", "behind"]
# How far a start may be read before the wall clock the test read just before it: the kernel keeps microseconds.
START_SLACK = 0.01


@dataclass(frozen=True)
class Start:
    """A child for the world to start, as a test of the contract asks for it: `program`, a Python, started in `cwd`, which
    moves to `runs_in`, as a session moves into a worktree, and then waits on a signal."""

    program: Path
    cwd: Path
    environment: Mapping[str, str] = field(default_factory=dict[str, str])
    at: At = "none"
    runs_in: Path | None = None
    # A signal it catches, and lives on: it moves into `caught` under where it runs once it has.
    catches: signal.Signals | None = None

    @property
    def directory(self) -> Path:
        """Where it runs once it has moved."""
        return self.cwd if self.runs_in is None else self.runs_in

    @property
    def arguments(self) -> tuple[str, ...]:
        # Python waiting on a signal, not /bin/sleep, since the kernel says nothing of the environment of a program of
        # the system's own, whoever started it.
        if self.catches is not None:
            return ("-c", _CATCHES, str(self.directory), self.catches.name)
        return ("-c", _BEHIND if self.at == "behind" else _WAITS, str(self.directory))


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
        """A child running `start`'s program, once the kernel says it runs it where it moves to."""
        ...

    def reaped(self, child: Child) -> None:
        """Once `child` has exited by itself or a signal, reaped, so the kernel has nothing more of it."""
        ...

    def caught(self, child: Child) -> None:
        """Once `child` has caught the signal it catches."""
        ...

    def exited(self, child: Child) -> None:
        """`child` ended by a signal, and not yet reaped: its parent has not waited on it."""
        ...

    def stopped(self, child: Child) -> None:
        """`child` ended and reaped."""
        ...


# The first process, root's, which every other descends from.
LAUNCHD = 1
# The kernel's calls a question about one process is answered by, as an Undescribed names the one it refused.
PROCARGS2 = "kern.procargs2"
VNODEPATHINFO = "proc_pidinfo flavor 9"
PIDPATH = "proc_pidpath"
# The call a signal is sent by.
KILL = "kill"
# What a child runs: it moves where it is told, then waits. Behind its terminal, it first starts a process of its own in
# a group of its own, which waits too, and puts that group in the terminal's foreground: it moves only once it is
# behind, so a world that waits for it to move has it behind.
_WAITS = "import os, signal, sys; os.chdir(sys.argv[1]); signal.pause()"
_BEHIND = "\n".join([
    "import os, signal, sys",
    "front = os.fork()",
    "if front == 0:",
    "    os.setpgid(0, 0)",
    "    signal.pause()",
    "os.setpgid(front, front)",
    "os.tcsetpgrp(0, front)",
    "os.chdir(sys.argv[1])",
    "signal.pause()",
])  # fmt: skip
_CATCHES = "\n".join([
    "import os, signal, sys",
    "caught = os.path.join(sys.argv[1], 'caught')",
    "def moved(*_):",
    "    os.makedirs(caught, exist_ok=True)",
    "    os.chdir(caught)",
    "signal.signal(signal.Signals[sys.argv[2]], moved)",
    "os.chdir(sys.argv[1])",
    "while True:",
    "    signal.pause()",
])  # fmt: skip
# What a child runs.
PYTHON = Path(sys.executable)


class ProcessesContract:
    """The claims, one method each. A holder subclasses it and gives a `world` fixture, so every claim added here is held to
    both parties without either holder naming it."""

    def test_a_child_is_in_the_table_under_its_parent_as_the_users_own_with_no_terminal_unless_started_on_one(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, tmp_path))
        try:
            process = world.processes.table()[child.pid]
            assert (process.parent, process.uid, process.terminal) == (world.ours, world.processes.user, None)
        finally:
            world.stopped(child)

    def test_a_child_started_on_a_terminal_is_controlled_by_it_and_in_its_foreground(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, tmp_path, at="interactive"))
        try:
            process = world.processes.table()[child.pid]
            assert child.tty is not None
            assert process.terminal == ControllingTerminal(child.tty, foreground=True)
        finally:
            world.stopped(child)

    def test_a_child_whose_terminal_has_another_group_in_its_foreground_is_in_its_background(self, world: World, tmp_path: Path) -> None:
        (tmp_path / "behind").mkdir()
        child = world.started(Start(PYTHON, tmp_path, at="behind", runs_in=tmp_path / "behind"))
        try:
            process = world.processes.table()[child.pid]
            assert child.tty is not None
            assert process.terminal == ControllingTerminal(child.tty, foreground=False)
        finally:
            world.stopped(child)

    def test_a_childs_start_is_when_it_was_started_and_a_gone_pid_has_none(self, world: World, tmp_path: Path) -> None:
        before = world.wall()
        child = world.started(Start(PYTHON, tmp_path))
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
        child = world.started(Start(PYTHON, folder, {"HANDS_CONTRACT": "yes"}))
        try:
            process = world.processes.table()[child.pid]
            match world.processes.started_as(process):
                case StartedAs(path=path, arguments=arguments, environment=environment):
                    assert (path, arguments, environment.get("HANDS_CONTRACT")) == (PYTHON, Start(PYTHON, folder).arguments, "yes"), (path, arguments, environment)
                case other:
                    pytest.fail(f"started as {other}")
            assert world.processes.directory_of(process) == folder
            assert world.processes.program_of(process) == Path(os.path.realpath(PYTHON))
        finally:
            world.stopped(child)

    def test_a_child_gone_since_it_was_listed_is_described_as_gone(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, tmp_path, at="interactive"))
        process = world.processes.table()[child.pid]
        [listed] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
        world.stopped(child)
        assert child.pid not in world.processes.table()
        said = (world.processes.started_as(process), world.processes.program_of(process), world.processes.directory_of(process), world.processes.interactive(listed))
        assert said == (None, None, None, None)

    def test_a_child_that_has_exited_and_is_not_yet_reaped_is_gone_from_every_question(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, tmp_path, at="interactive"))
        process = world.processes.table()[child.pid]
        [listed] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
        world.exited(child)
        try:
            listing = (child.pid in world.processes.table(), world.processes.starts({child.pid}))
            questions = (world.processes.started_as(process), world.processes.program_of(process), world.processes.directory_of(process), world.processes.interactive(listed))
            assert (listing, questions, world.processes.signal(child.pid, signal.SIGTERM)) == ((False, {}), (None, None, None, None), False)
        finally:
            world.reaped(child)

    def test_a_program_started_by_a_relative_path_is_named_by_the_file_it_runs_after_it_moves(self, world: World, tmp_path: Path) -> None:
        folder = tmp_path.resolve()
        (folder / "bin").mkdir()
        (folder / "bin" / "python").symlink_to(PYTHON)
        worktree = folder / "worktree"
        worktree.mkdir()
        # As the shim starts the real `claude` under a relative PATH entry, and as a session then moves into a worktree:
        # the kernel keeps the path as it was given, and no record of the directory it was given in. Its PWD, which a
        # parent need not keep to where it runs, names the worktree, where no bin/python is.
        child = world.started(Start(Path("bin/python"), folder, {"PWD": str(worktree)}, at="interactive", runs_in=worktree))
        try:
            [process] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
            assert (process.executable, process.cwd) == (Path(os.path.realpath(PYTHON)), worktree)
        finally:
            world.stopped(child)

    def test_a_program_deleted_since_it_started_is_still_named(self, world: World, tmp_path: Path) -> None:
        # As an updater prunes the version a running session started on.
        version = tmp_path.resolve() / "9.9.8"
        version.symlink_to(PYTHON)
        child = world.started(Start(version, tmp_path, at="interactive"))
        try:
            version.unlink()
            [process] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
            assert process.executable == version
        finally:
            world.stopped(child)

    def test_what_the_kernel_will_not_say_of_a_running_process_is_answered_with_the_call_it_refused(self, world: World) -> None:
        # Root's launchd, which the kernel will not describe to this user: what it was started as, and where it runs.
        launchd = world.processes.table()[LAUNCHD]
        said = (world.processes.started_as(launchd), world.processes.directory_of(launchd))
        assert [(each.pid, each.parent, each.call) if isinstance(each, Undescribed) else each for each in said] == [(LAUNCHD, 0, PROCARGS2), (LAUNCHD, 0, VNODEPATHINFO)]

    @pytest.mark.parametrize(("at", "reads"), [("interactive", True), ("piped", False), ("reopened", True)])
    def test_a_child_reads_and_writes_its_terminal_only_where_its_stdin_and_stdout_are_both_it(self, world: World, tmp_path: Path, at: At, reads: bool) -> None:
        child = world.started(Start(PYTHON, tmp_path, at=at))
        try:
            [process] = [each for each in terminal_processes(world.processes).found if each.pid == child.pid]
            assert isinstance(process, TerminalProcess) and process.executable == PYTHON
            assert world.processes.interactive(process) is reads
        finally:
            world.stopped(child)

    def test_a_signal_is_sent_to_a_running_child_and_ends_it_and_none_is_sent_to_a_gone_pid(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, tmp_path))
        assert world.processes.signal(child.pid, signal.SIGTERM)
        world.reaped(child)
        assert child.pid not in world.processes.table()
        assert not world.processes.signal(child.pid, signal.SIGTERM)

    def test_a_signal_a_child_catches_is_sent_and_leaves_it_running(self, world: World, tmp_path: Path) -> None:
        # As a claude wedged in its handler for SIGTERM lives on through it. It moves once its handler is in, so a world
        # that waits for it to move signals it only then.
        (tmp_path / "catching").mkdir()
        child = world.started(Start(PYTHON, tmp_path, catches=signal.SIGTERM, runs_in=tmp_path / "catching"))
        try:
            assert world.processes.signal(child.pid, signal.SIGTERM)
            world.caught(child)
            assert isinstance(world.processes.started_as(world.processes.table()[child.pid]), StartedAs)
        finally:
            world.stopped(child)

    @pytest.mark.parametrize("pid", [0, -1])
    def test_a_signal_is_never_sent_to_a_number_that_names_a_group(self, world: World, pid: int) -> None:
        # SIGCONT, which would leave each process of the group running were it sent.
        with pytest.raises(Rejected, match="is not a process id"):
            world.processes.signal(pid, signal.SIGCONT)

    def test_a_signal_the_kernel_will_not_send_raises_the_errno_it_refused_with(self, world: World) -> None:
        # Root's launchd, which this user may not signal; SIGCONT, which would leave it running were it sent.
        with pytest.raises(PermissionError, match=r"^\[Errno 1\] Operation not permitted$"):
            world.processes.signal(LAUNCHD, signal.SIGCONT)

    def test_a_childs_line_of_parents_ends_at_launchd_which_is_roots_and_whose_parent_is_not_in_the_table(self, world: World, tmp_path: Path) -> None:
        child = world.started(Start(PYTHON, tmp_path))
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
