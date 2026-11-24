"""What `KernelProcesses` makes of a call the kernel fails, which no child can be made to draw on demand: outside, since it
reads the machine's processes. What it says of a process it describes is the contract's (processescontract.py)."""

import ctypes
import errno
import os
import subprocess

import pytest

from hands.sessions import kernel
from hands.sessions.kernel import KERNEL
from hands.sessions.processes import StartedAs, Undescribed

pytestmark = pytest.mark.outside


def test_a_running_process_the_kernel_will_not_describe_is_named_with_the_call_and_its_errno(monkeypatch: pytest.MonkeyPatch) -> None:
    # As kern.procargs2 answers a process caught mid-exec, or on its way out: refused, of a process that runs.
    def refusing(pid: int) -> StartedAs:
        ctypes.set_errno(errno.EIO)
        kernel._raise_unless_exited(pid, "kern.procargs2")  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(kernel, "_started_as", refusing)
    own = KERNEL.table()[os.getpid()]
    assert KERNEL.started_as(own) == Undescribed(own.pid, own.parent, "kern.procargs2", errno.EIO)


def test_a_process_that_has_ended_is_no_process_the_kernel_would_not_describe() -> None:
    process = subprocess.Popen(["/usr/bin/true"])
    process.wait()
    ctypes.set_errno(errno.EIO)
    with pytest.raises(kernel._Exited):  # pyright: ignore[reportPrivateUsage]
        kernel._raise_unless_exited(process.pid, "kern.procargs2")  # pyright: ignore[reportPrivateUsage]
