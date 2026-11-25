"""The processes contract (processescontract.py) held to the kernel, through `hands.sessions.kernel.KERNEL`, on children
this test starts: outside, since it starts processes and reads the machine's process table."""

import fcntl
import os
import subprocess
import termios
from collections.abc import Iterator

import pytest

from hands.clock import SYSTEM
from hands.sessions.kernel import KERNEL, KernelProcesses
from hands.sessions.processes import Process, StartedAs
from processescontract import Child, ProcessesContract, Start

pytestmark = pytest.mark.outside

# How long a child has to exec its program once started.
EXEC_SECONDS = 10.0


class Kernel:
    """Children of this test's own process, each on a pseudo-terminal of the test's own where the contract asks for one."""

    def __init__(self) -> None:
        self.children: dict[int, tuple[subprocess.Popen[bytes], tuple[int, ...]]] = {}

    @property
    def processes(self) -> KernelProcesses:
        return KERNEL

    @property
    def ours(self) -> int:
        return os.getpid()

    def wall(self) -> float:
        return SYSTEM.wall().timestamp()

    def started(self, start: Start) -> Child:
        argv, env = [str(start.program), *start.arguments], dict(start.environment)
        match start.at:
            case "none":
                # A session of its own, so it holds no controlling terminal, whatever pytest's is.
                popen = subprocess.Popen(argv, cwd=start.cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, start_new_session=True)
                opened, tty = (), None
            case "interactive" | "piped" | "reopened" as at:
                controller, terminal = os.openpty()

                def control() -> None:
                    os.setsid()
                    fcntl.ioctl(terminal, termios.TIOCSCTTY, 0)
                    if at == "reopened":
                        os.dup2(os.open("/dev/tty", os.O_RDWR), 0)

                stdin = subprocess.DEVNULL if at == "reopened" else terminal
                out = subprocess.PIPE if at == "piped" else terminal
                popen = subprocess.Popen(argv, cwd=start.cwd, env=env, stdin=stdin, stdout=out, stderr=terminal, preexec_fn=control)
                opened, tty = (controller, terminal), os.fstat(terminal).st_rdev
        self.children[popen.pid] = (popen, opened)
        self._exec(popen.pid, start)
        return Child(popen.pid, tty)

    def _exec(self, pid: int, start: Start) -> None:
        """Until the child runs `start`'s program with its arguments, not the forked Python it began as, which may be the
        same program, in the directory it moves to: polled, since nothing of the child's says when it has exec'd."""
        deadline = SYSTEM.now() + EXEC_SECONDS
        while not _runs(process := KERNEL.table()[pid], start):
            assert SYSTEM.now() < deadline, f"pid {pid} never ran {start.program} in {start.directory}: {KERNEL.started_as(process)} in {KERNEL.directory_of(process)}"
            SYSTEM.sleep_blocking(0.01)

    def reaped(self, child: Child) -> None:
        popen, opened = self.children.pop(child.pid)
        popen.wait(timeout=EXEC_SECONDS)
        _closed(popen, opened)

    def stopped(self, child: Child) -> None:
        popen, opened = self.children.pop(child.pid)
        popen.kill()
        popen.wait()
        _closed(popen, opened)

    def close(self) -> None:
        for pid in list(self.children):
            self.stopped(Child(pid, None))


def _runs(process: Process, start: Start) -> bool:
    match KERNEL.started_as(process):
        case StartedAs(path=start.program, arguments=start.arguments):
            return KERNEL.directory_of(process) == start.directory
        case _:
            return False


def _closed(popen: subprocess.Popen[bytes], opened: tuple[int, ...]) -> None:
    if popen.stdout is not None:
        popen.stdout.close()
    for fd in opened:
        os.close(fd)


@pytest.fixture
def world() -> Iterator[Kernel]:
    kernel = Kernel()
    try:
        yield kernel
    finally:
        kernel.close()


class TestTheKernel(ProcessesContract):
    pass
