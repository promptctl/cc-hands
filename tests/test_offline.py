"""The run is offline once its fetches are done (conftest, offline.py): nothing a test does, nor anything a test starts, connects
past loopback, and loopback is as open as it was."""

import errno
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import offline
from offline import UNROUTED, sandboxable


def test_a_connection_a_test_makes_past_loopback_is_refused_by_the_os() -> None:
    with pytest.raises(PermissionError) as refused:
        socket.create_connection(UNROUTED, timeout=5)
    assert refused.value.errno == errno.EPERM


def test_a_connection_a_test_s_child_makes_past_loopback_is_refused_by_the_os() -> None:
    tried = subprocess.run(["nc", "-zv", "-w", "5", *map(str, UNROUTED)], capture_output=True, text=True, timeout=30)
    assert tried.returncode != 0
    assert "Operation not permitted" in tried.stderr, tried.stderr


def test_a_name_off_this_machine_is_not_resolved() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.com", 443)


def test_loopback_is_reachable_by_address_and_by_localhost() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        for host in ("127.0.0.1", "localhost"):
            socket.create_connection((host, port), timeout=5).close()


@pytest.mark.parametrize("release", ["15.0", "15.7.9", "26.5.2"])
def test_a_macos_whose_sandbox_holds_loopback_starts_the_run(release: str) -> None:
    sandboxable(release)


def test_a_macos_whose_sandbox_refuses_loopback_at_random_stops_the_run_saying_so() -> None:
    with pytest.raises(pytest.UsageError, match=r"refuses loopback at random before macOS 15: this is macOS 14\.8\.9$"):
        sandboxable("14.8.9")


def test_a_run_started_inside_the_held_run_is_held_and_says_how() -> None:
    held = subprocess.run([sys.executable, "-c", "import offline; print(offline.hold())"], cwd=Path(offline.__file__).parent, capture_output=True, text=True, timeout=30)
    assert held.returncode == 0, held.stderr
    assert held.stdout.startswith(str(offline.OFFLINE)), held.stdout
