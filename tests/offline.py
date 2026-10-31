"""[LAW:single-enforcer] the one owner of the hold that keeps a test run off the network: the profile, the first macOS
whose sandbox applies it faithfully, and applying it. hands' run takes it from here (conftest) once its fetches are
done; wake-training's suite loads this file by path; fritter's Go tests read the profile and SANDBOXED_MACOS from here.
Standard library and pytest only, so any of the repository's Python suites can load it."""

import ctypes
import errno
import getpass
import os
import socket
from pathlib import Path

import pytest

# The sandbox profile a test run is held to.
OFFLINE = Path(__file__).with_name("offline.sb")
# The first macOS whose sandbox holds a run to OFFLINE without refusing loopback: on macOS 14, CI's macos-14 refused
# some ten tests' binds and connects on 127.0.0.1 with EPERM on every run, under this profile and under Bazel's.
SANDBOXED_MACOS = 15
# Where the pinned Claude Code (2.1.308 when this was read from it) reads a machine's policy on macOS: the folder of its
# settings, their drop-ins and its MCP config, and its MDM profiles, the machine's and the account's. OFFLINE has each
# read as absent.
POLICY = (
    Path("/Library/Application Support/ClaudeCode"),
    Path("/Library/Managed Preferences/com.anthropic.claudecode.plist"),
    Path("/Library/Managed Preferences") / getpass.getuser() / "com.anthropic.claudecode.plist",
)
# TEST-NET-1 (RFC 5737), routed nowhere: outside the sandbox a connection to it times out, inside it the OS refuses it
# at once.
UNROUTED = ("192.0.2.1", 443)


def sandboxable(release: str) -> None:
    """[LAW:no-silent-failure] a macOS `release` whose sandbox cannot hold a run offline stops it before any test,
    saying why, never a run whose loopback connections fail at random."""
    if int(release.split(".")[0]) < SANDBOXED_MACOS:
        raise pytest.UsageError(f"the run holds itself offline with macOS's sandbox, which refuses loopback at random before macOS {SANDBOXED_MACOS}: this is macOS {release}")


def hold() -> str:
    """The calling process in the OFFLINE sandbox, which every process it starts inherits and none can leave, so a
    connection a test, or any child of one, makes past loopback is refused by the OS before a packet leaves, and the
    machine's Claude Code policy is nowhere it looks; and how it is held. It is macOS's sandbox_init, the call sandbox-exec makes, on the run's own process, as Bazel's macOS sandbox
    holds a test with the network denied.

    [LAW:single-enforcer] what is proven is what the profile promises, a connection past loopback refused, loopback
    open and no policy found, not that this call applied it: a run started inside a held one, as wake-training's test of its own conftest
    starts one, is held by its parent's sandbox, and macOS 15 refuses a second sandbox_init in a process held already.
    [LAW:no-silent-failure] a run that is not so held stops before any test, with the OS's reason, whether the run was in
    a sandbox already, sandbox-exec's or Claude Code's sandboxed Bash among them, and which half of the promise failed."""
    sandbox = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
    sandbox.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
    sandbox.sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    error = ctypes.c_char_p()
    sandbox.sandbox_init(OFFLINE.read_bytes(), 0, ctypes.byref(error))
    refused, looped, found = _refused(), _looped(), _found()
    if not (refused and looped):
        held = bool(sandbox.sandbox_check(os.getpid(), None, 0))
        raise pytest.UsageError(
            f"the run cannot hold itself to {OFFLINE}: {error.value!r}; in a sandbox already: {held}; past loopback refused: {refused}; loopback open: {looped}; Claude Code policy found: {found}"
        )
    return f"{OFFLINE}" if error.value is None else f"{OFFLINE}, by the sandbox the run started in: sandbox_init said {error.value.decode()!r}"


def _found() -> list[str]:
    """Each place this process finds the machine's Claude Code policy. A sandbox that refuses the read, as one the run
    started in may, raises: the policy is there, unread here but not in every Claude Code that asks."""
    return [str(path) for path in POLICY if path.exists()]


def _refused() -> bool:
    """Whether the OS refuses this process a connection past loopback."""
    try:
        socket.create_connection(UNROUTED, timeout=1).close()
    except OSError as failed:
        return failed.errno == errno.EPERM
    return False


def _looped() -> bool:
    """Whether this process can listen on loopback and connect to what listens there."""
    try:
        with socket.create_server(("127.0.0.1", 0)) as server:
            socket.create_connection(server.getsockname(), timeout=1).close()
    except OSError:
        return False
    return True
