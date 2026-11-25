"""brioche, the brain's HTTPS proxy, as hands runs it: the way to Anthropic's API it prints for the brain, what it says
and why it stopped, in hands' audit log, and its refusal to start, said. What brioche does with each request is its own
tests' (fritter/brioche), and the brain through it is tests/test_pinned_brain.py's."""

import os
import signal
from pathlib import Path

import pytest

from conftest import children_on
from hands.brain.brioche import BriocheFailed, proxied
from hands.virtualclock import VirtualClock
from hands.sessions.audit import BriocheSaid, Entry

pytestmark = pytest.mark.outside

# hands' proxy, as brioche is told of it: on loopback, and never reached by these tests.
PROXY = "http://127.0.0.1:1"


async def test_brioche_gives_the_brain_its_way_to_the_api_and_says_why_it_stops_once_hands_stops_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    trust = tmp_path / "trusted.pem"
    async with proxied(children_on(VirtualClock()), PROXY, trust, {"PATH": os.environ["PATH"]}, recorded.append) as brioche:
        assert brioche.environment["NODE_EXTRA_CA_CERTS"] == str(trust)
        assert brioche.environment["HTTPS_PROXY"].startswith("http://127.0.0.1:")
        assert trust.read_bytes().count(b"BEGIN CERTIFICATE") == 1
    assert recorded[-1] == BriocheSaid("brioche: stopping: told to by terminated")


async def test_a_brioche_that_cannot_start_is_refused_saying_why(tmp_path: Path) -> None:
    # A variable brioche cannot read as the brain would: relative to a directory it does not know.
    inherited = {"PATH": os.environ["PATH"], "NODE_EXTRA_CA_CERTS": "corp.pem"}
    with pytest.raises(BriocheFailed, match=r'did not start: exited 1: brioche: NODE_EXTRA_CA_CERTS names "corp\.pem", relative'):
        async with proxied(children_on(VirtualClock()), PROXY, tmp_path / "trusted.pem", inherited, lambda _entry: None):
            pass


async def test_a_brioche_that_ends_while_hands_runs_says_how_and_the_last_it_said(tmp_path: Path) -> None:
    async with proxied(children_on(VirtualClock()), PROXY, tmp_path / "trusted.pem", {"PATH": os.environ["PATH"]}, lambda _entry: None) as brioche:
        os.kill(brioche._running.pid, signal.SIGTERM)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match=r"brioche, the brain's HTTPS proxy, ended while hands was running: exited 0: brioche: stopping: told to by terminated$"):
            await brioche.outlived()
