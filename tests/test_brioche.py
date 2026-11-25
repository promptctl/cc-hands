"""brioche, the brain's HTTPS proxy, as hands runs it: the way to Anthropic's API it prints for the brain, what it says
and why it stopped, in hands' audit log, and its refusal to start, said. What brioche does with each request is its own
tests' (fritter/brioche), and the brain through it is tests/test_pinned_brain.py's."""

import asyncio
import json
import os
import signal
from pathlib import Path

import pytest

from conftest import children_on
from hands.brain.brioche import _Heard, _report, _through, proxied  # pyright: ignore[reportPrivateUsage]
from hands.brain.configdir import Unstartable
from hands.virtualclock import VirtualClock
from hands.sessions.audit import BriocheSaid, Entry

# hands' proxy, as brioche is told of it: on loopback, and never reached by these tests.
PROXY = "http://127.0.0.1:1"
# What brioche prints for a client, as fritter/intercept's Environment gives it.
PRINTED = b"NODE_EXTRA_CA_CERTS=/h/trusted.pem\nHTTPS_PROXY=http://127.0.0.1:9\nhttps_proxy=http://127.0.0.1:9\nHTTP_PROXY=http://127.0.0.1:9\nhttp_proxy=http://127.0.0.1:9\nNO_PROXY=\nno_proxy=\n"


def test_the_brains_way_to_the_api_is_each_variable_brioche_printed_for_a_client() -> None:
    assert dict(_through(PRINTED, Path("/h/trusted.pem")).variables) == {
        "NODE_EXTRA_CA_CERTS": "/h/trusted.pem", **dict.fromkeys(("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"), "http://127.0.0.1:9"), "NO_PROXY": "", "no_proxy": "",
    }


@pytest.mark.parametrize("printed", [
    pytest.param(PRINTED.replace(b"HTTP_PROXY=http://127.0.0.1:9\n", b""), id="one left out, as a print that failed part way leaves it"),
    pytest.param(PRINTED + b"HTTPS_PROXY=http://127.0.0.1:10\n", id="one printed twice"),
    pytest.param(PRINTED + b"ANTHROPIC_BASE_URL=http://elsewhere\n", id="one no interceptor sets"),
    pytest.param(PRINTED.replace(b"/h/trusted.pem", b"/h/other.pem"), id="a trust file hands did not name"),
    pytest.param(PRINTED[:-1], id="its last line not ended"),
    pytest.param(PRINTED + b"\xff\n", id="no text"),
])
def test_a_way_to_the_api_brioche_did_not_print_whole_refuses_the_start(printed: bytes) -> None:
    with pytest.raises(Unstartable, match="^brioche, the brain's HTTPS proxy, printed "):
        _through(printed, Path("/h/trusted.pem"))


def test_a_trust_file_whose_path_holds_a_line_break_other_than_brioches_own_is_one_variable() -> None:
    # Python's str.splitlines breaks on \x0c and U+2028 too; brioche ends its lines with \n alone.
    trust = Path("/h/a\x0cb c/trusted.pem")
    printed = PRINTED.replace(b"/h/trusted.pem", str(trust).encode())
    assert _through(printed, trust).variables["NODE_EXTRA_CA_CERTS"] == str(trust)


def test_each_line_brioche_says_is_the_report_it_is_judged_as_brioche_judged_it() -> None:
    line = json.dumps({"kind": "handshake_refused", "level": "error", "said": "the client refused brioche's answer for api.anthropic.com:443: tls: bad certificate", "target": "api.anthropic.com:443"})
    assert _report(line.encode()) == BriocheSaid("handshake_refused", "error", "the client refused brioche's answer for api.anthropic.com:443: tls: bad certificate", {"target": "api.anthropic.com:443"})


@pytest.mark.parametrize("line", [b"panic: runtime error: index out of range", b'{"kind": "opened", "level": "loud", "said": "x"}', b'{"kind": "answered", "level": "info", "said": "x", "status": true}', b"\xff"])
def test_a_line_that_is_no_report_is_an_error_said_as_it_was_written(line: bytes) -> None:
    assert _report(line) == BriocheSaid("unframed", "error", line.decode(errors="replace"), {})


async def test_every_report_is_heard_however_long_and_the_last_is_kept() -> None:
    recorded: list[Entry] = []
    reads, writes = os.pipe()
    with os.fdopen(reads, "rb", 0) as pipe:
        heard = _Heard(pipe, recorded.append)
        # Longer than a StreamReader takes as one line.
        long = json.dumps({"kind": "refused", "level": "warning", "said": "x" * 200_000}).encode() + b"\n"
        last = json.dumps({"kind": "stopping", "level": "info", "said": "stopping: its input ended"}).encode()
        def write() -> None:
            with os.fdopen(writes, "wb") as writing:
                writing.write(long + last)

        # Off the loop, as brioche writes from its own process: more than a pipe holds, read as it is written.
        await asyncio.gather(asyncio.to_thread(write), heard.drained)
    assert [(said.kind, len(said.said)) for said in recorded if isinstance(said, BriocheSaid)] == [("refused", 200_000), ("stopping", 25)]
    assert heard.last == recorded[-1]


@pytest.mark.outside
async def test_brioche_gives_the_brain_its_way_to_the_api_and_says_why_it_stops_once_hands_stops_it(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    trust = tmp_path / "trusted.pem"
    async with proxied(children_on(VirtualClock()), PROXY, trust, {"PATH": os.environ["PATH"]}, recorded.append) as brioche:
        assert brioche.route.variables["NODE_EXTRA_CA_CERTS"] == str(trust)
        assert brioche.route.variables["HTTPS_PROXY"].startswith("http://127.0.0.1:")
        assert trust.read_bytes().count(b"BEGIN CERTIFICATE") == 1
    assert [(said.kind, said.level) for said in recorded if isinstance(said, BriocheSaid)] == [("stopping", "info")]


@pytest.mark.outside
async def test_brioche_runs_in_hands_environment_as_it_is_passed_on_never_in_a_session_tap_hands_was_started_in(tmp_path: Path) -> None:
    # A tap's own authority, as the session hands was started in trusts it.
    tap = tmp_path / "tap.pem"
    async with proxied(children_on(VirtualClock()), PROXY, tap, {"PATH": os.environ["PATH"]}, lambda _entry: None):
        pass
    tapped = {"PATH": os.environ["PATH"], "FRITTER_TAP": "http://127.0.0.1:40000", "HTTPS_PROXY": "http://127.0.0.1:40000", "NODE_EXTRA_CA_CERTS": str(tap)}
    trust = tmp_path / "trusted.pem"
    async with proxied(children_on(VirtualClock()), PROXY, trust, tapped, lambda _entry: None):
        # Only brioche's own authority: the session's tap, and what it trusted, are none of the brain's.
        assert trust.read_bytes().count(b"BEGIN CERTIFICATE") == 1


@pytest.mark.outside
async def test_a_brioche_that_cannot_start_is_refused_saying_why(tmp_path: Path) -> None:
    # A variable brioche cannot read as the brain would: relative to a directory it does not know.
    inherited = {"PATH": os.environ["PATH"], "NODE_EXTRA_CA_CERTS": "corp.pem"}
    with pytest.raises(Unstartable, match=r'did not start: exited 1: it said \'NODE_EXTRA_CA_CERTS names "corp\.pem", relative'):
        async with proxied(children_on(VirtualClock()), PROXY, tmp_path / "trusted.pem", inherited, lambda _entry: None):
            pass


@pytest.mark.outside
async def test_a_brioche_that_ends_while_hands_runs_says_how_and_the_last_it_said(tmp_path: Path) -> None:
    async with proxied(children_on(VirtualClock()), PROXY, tmp_path / "trusted.pem", {"PATH": os.environ["PATH"]}, lambda _entry: None) as brioche:
        os.kill(brioche.running.pid, signal.SIGTERM)
        with pytest.raises(RuntimeError, match=r"brioche, the brain's HTTPS proxy, ended while hands was running: exited 0: it said 'stopping: told to by terminated'$"):
            await brioche.outlived()
