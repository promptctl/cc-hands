"""Darwin's process libraries, bound once: libproc and libc, each keeping the errno of the call that set it."""

import ctypes
import ctypes.util
import os
from typing import NoReturn

libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def raise_errno(what: str) -> NoReturn:
    """[LAW:no-silent-failure] the call that just failed, raised with the errno it set."""
    failure = ctypes.get_errno()
    raise OSError(failure, f"{what} could not be read: {os.strerror(failure)}")


def child_pids(pid: int) -> tuple[int, ...]:
    """The pids of the processes `pid` started that are still its children, as the kernel lists them now: none for a
    pid that is gone. Up to 64, more than any process hands asks after starts."""
    found = (ctypes.c_int * 64)()
    count = libproc.proc_listchildpids(pid, found, ctypes.sizeof(found))
    if count < 0:
        raise_errno(f"the children of {pid}")
    return tuple(found[:count])
