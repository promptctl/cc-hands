"""`Processes` read from the kernel through libproc and sysctl, the sources ps and lsof read: the one adapter hands
reaches another process through."""

import ctypes
import errno
import os
import signal
import struct
from collections.abc import Callable, Collection
from pathlib import Path
from typing import NoReturn

from hands.darwin import libc, libproc, raise_errno
from hands.sessions.processes import Process, StartedAs, TerminalProcess, Undescribed


class KernelProcesses:
    """[LAW:effects-at-boundaries] the Mac's processes as its kernel says they stand."""

    @property
    def user(self) -> int:
        return os.geteuid()

    def table(self) -> dict[int, Process]:
        mib = (ctypes.c_int * 3)(*_KERN_PROC_ALL)
        size = ctypes.c_size_t(0)
        if libc.sysctl(mib, len(mib), None, ctypes.byref(size), None, 0) != 0:
            raise_errno("kern.proc.all")
        # With room to spare for processes started between the two calls.
        size.value += 256 * _KINFO_PROC_SIZE
        table = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, len(mib), table, ctypes.byref(size), None, 0) != 0:
            raise_errno("kern.proc.all")
        if size.value % _KINFO_PROC_SIZE:
            # A record of another size is a kernel whose struct is not the one laid out below; no errno says so.
            raise OSError(f"kern.proc.all gave {size.value} bytes, not a whole number of this kernel's {_KINFO_PROC_SIZE}-byte kinfo_proc")
        processes = (_process(*fields) for fields in _KINFO_PROC.iter_unpack(table.raw[: size.value]))
        # Not kernel_task, pid 0, which is its own parent: without it every line of parents ends, at launchd.
        return {process.pid: process for process in processes if process.pid != 0}

    def starts(self, pids: Collection[int]) -> dict[int, float]:
        # One sysctl a pid, about ten microseconds, so it is asked on the event loop and five times a second from the menu
        # bar alike.
        return {pid: start for pid in pids if (start := _process_start(pid)) is not None}

    def started_as(self, process: Process) -> StartedAs | Undescribed | None:
        return _described(process.pid, process.parent, lambda: _started_as(process.pid))

    def directory_of(self, process: Process) -> Path | Undescribed | None:
        return _described(process.pid, process.parent, lambda: _cwd(process.pid))

    def interactive(self, process: TerminalProcess) -> bool | Undescribed:
        terminal = {process.tty, os.stat("/dev/tty").st_rdev}
        read = _described(process.pid, process.parent, lambda: all(_device(process.pid, fd) in terminal for fd in (0, 1)))
        return False if read is None else read

    def signal(self, pid: int, signum: signal.Signals) -> bool:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            return False
        return True


# [LAW:single-enforcer] the one KernelProcesses: the composition roots hand it on, and nothing else builds one.
KERNEL = KernelProcesses()


def _process(flag: int, pid: int, uid: int, parent: int, group: int, tty: int, foreground: int) -> Process:
    controlling = tty if flag & _P_CONTROLT else None
    return Process(pid, parent, uid, controlling, controlling if group == foreground else None)


# sysctl kern.proc.all: one struct kinfo_proc of 648 bytes per process; its extern_proc's p_flag and p_pid, then its
# eproc's effective uid (e_ucred.cr_uid), parent's pid, process group, controlling terminal's device, and the process
# group in front of that terminal (e_tpgid), at these offsets.
_KERN_PROC_ALL = (1, 14, 0)  # CTL_KERN, KERN_PROC, KERN_PROC_ALL
_KINFO_PROC = struct.Struct("=32xi4xi376xI136xii4xii68x")
_KINFO_PROC_SIZE = _KINFO_PROC.size
_P_CONTROLT = 0x2
# kern.proc.pid.<pid>: the same `struct kinfo_proc`, the record ps itself reads. It opens with the process's start, a
# `struct timeval`: 8 bytes of seconds, then 4 of microseconds.
_KERN_PROC_PID = (1, 14, 1)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
_KINFO_PROC_ROOM = 1024  # more than the 648 bytes the record takes, so a larger one in a later macOS still fits
# libproc: struct proc_bsdinfo (136 bytes), asked only whether a process still exists, struct proc_vnodepathinfo
# (two vnode_info_path of 1176 bytes, the cwd's first, its path after a 152-byte vnode_info), and struct
# vnode_fdinfowithpath (a 24-byte proc_fileinfo, then a vnode_info_path), whose vnode's vinfo_stat has st_rdev at 116, a dev_t as e_tdev is.
_PROC_PIDTBSDINFO, _BSDINFO_SIZE = 3, 136
_PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE, _CWD_PATH_AT, _MAXPATHLEN = 9, 2352, 152, 1024
_PROC_PIDFDVNODEPATHINFO, _FDVNODEPATHINFO_SIZE = 2, 1200
_FD_RDEV = struct.Struct("=140xi1056x")
# kern.procargs2.<pid>: argc, the path the process was exec'd by as execve was given it, padding, its arguments, and
# its environment, each string ending in a NUL.
_KERN_PROCARGS2 = (1, 49)  # CTL_KERN, KERN_PROCARGS2


def _process_start(pid: int) -> float | None:
    mib = (ctypes.c_int * 4)(*_KERN_PROC_PID, pid)
    record = ctypes.create_string_buffer(_KINFO_PROC_ROOM)
    size = ctypes.c_size_t(_KINFO_PROC_ROOM)
    if libc.sysctl(mib, len(mib), record, ctypes.byref(size), None, 0) != 0:
        failure = ctypes.get_errno()
        # [LAW:no-silent-failure] the kernel refusing a question it always answers is raised, never taken for "gone".
        raise OSError(failure, f"sysctl could not say when pid {pid} started: {os.strerror(failure)}")
    if size.value == 0:
        return None  # no process has the pid; the kernel answers with an empty record
    return ctypes.c_int64.from_buffer(record, 0).value + ctypes.c_int32.from_buffer(record, 8).value / 1_000_000


def _described[T](pid: int, parent: int, read: Callable[[], T]) -> T | Undescribed | None:
    """What `read` reads of pid from the kernel; None if pid has exited since it was listed, and the refusal if the
    kernel would not say.

    [LAW:single-enforcer] the one place a question about a process that ran when it was listed is told apart from one
    the kernel refused.
    """
    try:
        return read()
    except _Exited:
        return None
    except _Refused as refused:
        return Undescribed(pid, parent, refused.call, refused.errno)


def _cwd(pid: int) -> Path:
    return Path(os.fsdecode(_string(_pidinfo(pid, _PROC_PIDVNODEPATHINFO, _VNODEPATHINFO_SIZE).raw[_CWD_PATH_AT : _CWD_PATH_AT + _MAXPATHLEN])))


def _device(pid: int, fd: int) -> int | None:
    """The device a file pid has open as fd is, as `st_rdev`; None for a pipe, a socket, or an fd it has not open."""
    record = ctypes.create_string_buffer(_FDVNODEPATHINFO_SIZE)
    filled = libproc.proc_pidfdinfo(pid, fd, _PROC_PIDFDVNODEPATHINFO, record, _FDVNODEPATHINFO_SIZE)
    if filled <= 0:
        failure = ctypes.get_errno()
        # EBADF is an fd that is no file, or no fd at all: nothing a terminal is.
        if failure == errno.EBADF:
            return None
        _raise_unless_exited(pid, f"proc_pidfdinfo of fd {fd}")
    if filled != _FDVNODEPATHINFO_SIZE:
        raise OSError(f"proc_pidfdinfo gave {filled} bytes for fd {fd} of pid {pid}, where this kernel's struct was taken to be {_FDVNODEPATHINFO_SIZE}")
    # A regular file's st_rdev is 0, never a terminal's device.
    (device,) = _FD_RDEV.unpack(record.raw)
    return device


def _started_as(pid: int) -> StartedAs:
    mib = (ctypes.c_int * 3)(*_KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, len(mib), None, ctypes.byref(size), None, 0) != 0:
        _raise_unless_exited(pid, "kern.procargs2")
    record = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, len(mib), record, ctypes.byref(size), None, 0) != 0:
        _raise_unless_exited(pid, "kern.procargs2")
    raw = record.raw[: size.value]
    argc = ctypes.c_int.from_buffer_copy(raw, 0).value
    path, rest = raw[ctypes.sizeof(ctypes.c_int) :].split(b"\0", 1)
    strings = rest.lstrip(b"\0").split(b"\0")
    # The environment runs from after the arguments to the first empty string.
    after = strings[argc:]
    entries = after[: after.index(b"")] if b"" in after else after
    arguments = tuple(os.fsdecode(argument) for argument in strings[1:argc])
    return StartedAs(Path(os.fsdecode(path)), arguments, {name: value for name, _, value in (os.fsdecode(entry).partition("=") for entry in entries)})


def _string(raw: bytes) -> bytes:
    return raw.split(b"\0", 1)[0]


class _Exited(Exception):
    pass


class _Refused(Exception):
    def __init__(self, call: str, errno: int) -> None:
        super().__init__(call, errno)
        self.call = call
        self.errno = errno


def _pidinfo(pid: int, flavor: int, size: int) -> ctypes.Array[ctypes.c_char]:
    record = ctypes.create_string_buffer(size)
    filled = libproc.proc_pidinfo(pid, flavor, 0, record, size)
    if filled <= 0:
        _raise_unless_exited(pid, f"proc_pidinfo flavor {flavor}")
    if filled != size:
        # A record of another size is a kernel whose struct is not the one laid out above; no errno says so.
        raise OSError(f"proc_pidinfo flavor {flavor} gave {filled} bytes for pid {pid}, where this kernel's struct was taken to be {size}")
    return record


def _raise_unless_exited(pid: int, call: str) -> NoReturn:
    """Raised after `call` failed for pid: _Exited when the process is gone, or a zombie, and _Refused when it runs."""
    failure = ctypes.get_errno()
    # [LAW:single-enforcer] the one test for an exit, whichever call failed: libproc has nothing on a process that
    # exited, or is a zombie waiting on its parent.
    probe = ctypes.create_string_buffer(_BSDINFO_SIZE)
    if libproc.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, probe, _BSDINFO_SIZE) <= 0 and ctypes.get_errno() == errno.ESRCH:
        raise _Exited
    # [LAW:no-silent-failure] anything else is the kernel refusing a question about one of this user's own processes.
    raise _Refused(call, failure)
