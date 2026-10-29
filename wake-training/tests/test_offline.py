"""The suite is held offline from its start (conftest): nothing a test does connects past loopback. 192.0.2.1 is
TEST-NET-1 (RFC 5737), routed nowhere: outside the sandbox a connection to it times out, inside it the OS refuses it at
once."""

import errno
import socket

import pytest


def test_a_connection_a_test_makes_past_loopback_is_refused_by_the_os() -> None:
    with pytest.raises(PermissionError) as refused:
        socket.create_connection(("192.0.2.1", 443), timeout=5)
    assert refused.value.errno == errno.EPERM


def test_a_name_off_this_machine_is_not_resolved() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.com", 443)
