"""Each speech model's load, as a unit of work of its own that says how big the process was as it ended.

What the models cost in memory is read off these events: each says the process's resident set size as its load ended,
and how much the process grew across it, measured as `ps -o rss` measures (promptctl-voice-i15t.99s.u9w). Both are facts
about the process, not the model: a load that ran alone grew it by the model's own cost, and one that overlapped another
load, which the events' own times show, grew it by both. The last load's event is the total with every model loaded.
"""

import ctypes
from collections.abc import Callable, Generator
from contextlib import contextmanager

from hands.sessions.wide import Fact, WideEvent, annotate, unit

# Mach's `task_info` flavor for `mach_task_basic_info`, as <mach/task_info.h> numbers it.
_MACH_TASK_BASIC_INFO = 20
_MEGABYTE = 2**20


class _TimeValue(ctypes.Structure):
    _fields_ = (("seconds", ctypes.c_int), ("microseconds", ctypes.c_int))


class _MachTaskBasicInfo(ctypes.Structure):
    """<mach/task_info.h>'s `mach_task_basic_info`, whose `resident_size` is the resident set size `ps` reports."""

    _fields_ = (
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _TimeValue),
        ("system_time", _TimeValue),
        ("policy", ctypes.c_int),
        ("suspend_count", ctypes.c_int),
    )


_libc = ctypes.CDLL(None)
_libc.task_info.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(_MachTaskBasicInfo), ctypes.POINTER(ctypes.c_uint32))
_libc.task_info.restype = ctypes.c_int
_TASK = ctypes.c_uint32.in_dll(_libc, "mach_task_self_").value


def resident() -> int:
    """The process's resident set size, in bytes. [LAW:no-silent-failure] a call Mach refuses raises with its code."""
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    returned = _libc.task_info(_TASK, _MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count))
    if returned != 0:
        raise OSError(f"task_info returned kern_return_t {returned}")
    return info.resident_size


type Measured = int | OSError


def measured(read: Callable[[], int]) -> Measured:
    """The process's resident set size, or why it could not be read: a measurement that fails is a fact for an event,
    never the failure of the work it measures."""
    try:
        return read()
    except OSError as error:
        return error


def size(reading: Measured) -> dict[str, Fact]:
    """[LAW:single-enforcer] what an event says of one reading: the resident size, or `resident_error`, why it was not read."""
    match reading:
        case int():
            return {"resident_mb": _megabytes(reading)}
        case OSError():
            return {"resident_error": str(reading)}


def memory(before: Measured, after: Measured) -> dict[str, Fact]:
    """What a load's event says of memory: its size as it ended, and how much the process grew since `before`."""
    match before, after:
        case int(), int():
            return {**size(after), "process_grew_mb": _megabytes(after - before)}
        case OSError(), _:
            return size(before)
        case int(), OSError():
            return size(after)


def _megabytes(size: int) -> float:
    return round(size / _MEGABYTE, 1)


@contextmanager
def loading(event: str, emit: Callable[[WideEvent], None], read: Callable[[], int] = resident) -> Generator[None]:
    """Run a model's load as the unit of work `event`, which says how big the process was as it ended and how much it
    grew across it, however the load ends. [LAW:nothing-unseen] a measurement that fails is on the event as
    `resident_error`, and the load goes on as it would have."""
    with unit(event, emit):
        before = measured(read)
        try:
            yield
        finally:
            annotate(**memory(before, measured(read)))
