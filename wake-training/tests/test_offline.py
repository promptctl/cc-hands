"""The suite is held offline from its start (conftest): nothing a test does connects past loopback."""

import errno
import socket

import pytest
from conftest import offline


def test_a_connection_a_test_makes_past_loopback_is_refused_by_the_os() -> None:
    with pytest.raises(PermissionError) as refused:
        socket.create_connection(offline.UNROUTED, timeout=5)
    assert refused.value.errno == errno.EPERM


def test_a_name_off_this_machine_is_not_resolved() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.com", 443)
