"""[LAW:single-enforcer] the one owner of the hold that keeps a test run off the network: the profile, the first macOS
whose sandbox applies it faithfully, and applying it. hands' run takes it from here (conftest) once its fetches are
done; wake-training's suite loads this file by path; fritter's Go tests read the profile and SANDBOXED_MACOS from here.
Standard library and pytest only, so any of the repository's Python suites can load it."""

import ctypes
import os
from pathlib import Path

import pytest

# The sandbox profile a test run is held to.
OFFLINE = Path(__file__).with_name("offline.sb")
# The first macOS whose sandbox holds a run to OFFLINE without refusing loopback: on macOS 14, CI's macos-14 refused
# some ten tests' binds and connects on 127.0.0.1 with EPERM on every run, under this profile and under Bazel's.
SANDBOXED_MACOS = 15


def sandboxable(release: str) -> None:
    """[LAW:no-silent-failure] a macOS `release` whose sandbox cannot hold a run offline stops it before any test,
    saying why, never a run whose loopback connections fail at random."""
    if int(release.split(".")[0]) < SANDBOXED_MACOS:
        raise pytest.UsageError(f"the run holds itself offline with macOS's sandbox, which refuses loopback at random before macOS {SANDBOXED_MACOS}: this is macOS {release}")


def hold() -> None:
    """The calling process in the OFFLINE sandbox, which every process it starts inherits and none can leave, so a
    connection a test, or any child of one, makes past loopback is refused by the OS before a packet leaves. It is
    macOS's sandbox_init, the call sandbox-exec makes, on the run's own process, as Bazel's macOS sandbox holds a test
    with the network denied. [LAW:no-silent-failure] a profile the OS does not apply stops the run before any test, with
    the OS's reason, and whether the run was in a sandbox already, sandbox-exec's or Claude Code's sandboxed Bash among
    them, inside which macOS applies no other."""
    sandbox = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
    sandbox.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
    sandbox.sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    error = ctypes.c_char_p()
    if sandbox.sandbox_init(OFFLINE.read_bytes(), 0, ctypes.byref(error)):
        held = bool(sandbox.sandbox_check(os.getpid(), None, 0))
        raise pytest.UsageError(f"the run cannot hold itself to {OFFLINE}: {error.value!r}; in a sandbox already: {held}")
