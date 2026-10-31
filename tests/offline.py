"""[LAW:single-enforcer] the one owner of the hold that keeps a test run off the network and the machine's Claude Code
policy out of it: the profile, the first macOS whose sandbox applies it faithfully, and applying it. hands' run takes it
from here (conftest) once its fetches are done; wake-training's suite loads this file by path; fritter's Go tests read
the profile and SANDBOXED_MACOS from here. Standard library and pytest only, so any of the repository's Python suites
can load it."""

import ctypes
import errno
import os
import socket
from pathlib import Path

import pytest

# The sandbox profile a test run is held to.
OFFLINE = Path(__file__).with_name("offline.sb")
# The first macOS whose sandbox holds a run to OFFLINE without refusing loopback: on macOS 14, CI's macos-14 refused
# some ten tests' binds and connects on 127.0.0.1 with EPERM on every run, under this profile and under Bazel's.
SANDBOXED_MACOS = 15
# Where Claude Code reads a machine's policy on macOS (docs/claude-code-behavior.md, machine policy): the folder of its
# settings, their drop-ins and its MCP config, and its MDM plists, the machine's and each account's. OFFLINE has each read
# as absent.
POLICY_FOLDER = Path("/Library/Application Support/ClaudeCode")
MANAGED_PREFERENCES = Path("/Library/Managed Preferences")
POLICY_PLIST = "com.anthropic.claudecode.plist"
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
    machine's Claude Code policy is nowhere it looks; and how it is held. It is macOS's sandbox_init, the call sandbox-exec
    makes, on the run's own process, as Bazel's macOS sandbox holds a test with the network denied.

    [LAW:single-enforcer] what is proven is what the profile promises, a connection past loopback refused, loopback
    open and no policy found, not that this call applied it: a run started inside a held one, as wake-training's test of
    its own conftest starts one, is held by its parent's sandbox, and macOS 15 refuses a second sandbox_init in a process
    held already.
    [LAW:no-silent-failure] a run that is not so held stops before any test, with the OS's reason, whether the run was in
    a sandbox already, sandbox-exec's or Claude Code's sandboxed Bash among them, and which part of the promise failed."""
    sandbox = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
    sandbox.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
    sandbox.sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    error = ctypes.c_char_p()
    sandbox.sandbox_init(OFFLINE.read_bytes(), 0, ctypes.byref(error))
    refused, looped, found = _refused(), _looped(), policy_found()
    if not (refused and looped and not found):
        held = bool(sandbox.sandbox_check(os.getpid(), None, 0))
        raise pytest.UsageError(
            f"the run cannot hold itself to {OFFLINE}: {error.value!r}; in a sandbox already: {held}; past loopback refused: {refused}; loopback open: {looped}; Claude Code policy found: {found}"
        )
    return f"{OFFLINE}" if error.value is None else f"{OFFLINE}, by the sandbox the run started in: sandbox_init said {error.value.decode()!r}"


def policy_found() -> list[str]:
    """Each place the machine's Claude Code policy lives that does not answer this process "no such file", every account's
    plist among them, as the profile hides every account's. A refused read, as a sandbox the run started in may answer, is
    found: the policy is unread here but not in every Claude Code that asks. os.walk lists a plist the profile hides, so
    the look is what keeps it out of the answer, and each folder it cannot list gets the same look, so a refused one is
    found and a missing MANAGED_PREFERENCES is not. These are os.walk and open because they answer alike on every Python
    the project allows, where 3.13's glob and 3.14's exists() hide a refused read."""
    unlisted: list[OSError] = []
    listed = [Path(folder, POLICY_PLIST) for folder, _, files in os.walk(MANAGED_PREFERENCES, onerror=unlisted.append) if POLICY_PLIST in files]
    places = [POLICY_FOLDER, *listed, *(Path(failed.filename) for failed in unlisted)]
    return sorted(str(path) for path in places if not _absent(path))


def _absent(path: Path) -> bool:
    """Whether a read of `path` answers "no such file", as on a machine that has no policy. A folder that is there
    answers that it is a folder."""
    try:
        path.open().close()
    except OSError as failed:
        return failed.errno == errno.ENOENT
    return False


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
