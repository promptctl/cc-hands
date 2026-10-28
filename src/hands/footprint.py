"""The process's physical footprint: the memory macOS charges it, as Activity Monitor's Memory column shows it.

Not the resident set size `ps -o rss` gives: memory that only the GPU has written, as MLX's arrays are, is never resident
in that sense, and a Whisper load that grows the footprint by 2.4 GB grows the resident set by 1.6 GB.
"""

import ctypes
import os
from collections.abc import Callable

from hands.sessions.wide import Fact

# `proc_pid_rusage`'s flavor for `rusage_info_v0`, as <sys/resource.h> numbers it.
_RUSAGE_INFO_V0 = 0
_MEGABYTE = 2**20


class _RusageInfoV0(ctypes.Structure):
    """<sys/resource.h>'s `rusage_info_v0`, whose `ri_phys_footprint` is the physical footprint."""

    _fields_ = (
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    )


_libc = ctypes.CDLL(None, use_errno=True)
_libc.proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV0))
_libc.proc_pid_rusage.restype = ctypes.c_int


def footprint() -> int:
    """The process's physical footprint, in bytes. [LAW:no-silent-failure] a call the kernel refuses raises with its errno."""
    info = _RusageInfoV0()
    if _libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V0, ctypes.byref(info)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"proc_pid_rusage: {os.strerror(error)}")
    return info.ri_phys_footprint


type Measured = int | OSError


def measured(read: Callable[[], int]) -> Measured:
    """The footprint, or why it could not be read: a measurement that fails is a fact for an event, never the failure of
    the work it measures."""
    try:
        return read()
    except OSError as error:
        return error


def size(reading: Measured) -> dict[str, Fact]:
    """[LAW:single-enforcer] what an event says of one reading: the footprint, or `footprint_error`, why it was not read."""
    match reading:
        case int():
            return {"footprint_mb": megabytes(reading)}
        case OSError():
            return {"footprint_error": str(reading)}


def megabytes(size: int) -> float:
    return round(size / _MEGABYTE, 1)
