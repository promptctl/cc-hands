"""Giving back what a session's tap replaced: the one rule, read the same in Python and in sh."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from hands.sessions.untap import KEPT, REPLACED, TRUST, tap, tapped, untap_script, untapped

TAP = "http://127.0.0.1:40000"
# A session's environment as fritter left it, its proxy and trust both replaced, and what they were before kept.
SESSION = {
    "HOME": "/home/a",
    "FRITTER_SOCKET": "/tmp/fritter-1/session.sock",
    "FRITTER_TAP": TAP,
    "HTTPS_PROXY": TAP,
    "https_proxy": TAP,
    "HTTP_PROXY": TAP,
    "http_proxy": TAP,
    "NO_PROXY": "",
    "no_proxy": "",
    "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem",
    "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128",
    "FRITTER_OUTER_NO_PROXY": ".anthropic.com",
    "FRITTER_OUTER_NODE_EXTRA_CA_CERTS": "",
}

# hands' own environment, with an outer proxy, an empty NO_PROXY, and no certificates of its own named.
HANDS = {"HOME": "/home/a", "HTTPS_PROXY": "http://corp:3128", "no_proxy": "", "PATH": "/usr/bin:/bin"}
# The brain's, given hands' proxy in the tap's own mark.
BRAIN = HANDS | tap(HANDS, "http://127.0.0.1:50000", "/tmp/hands/trusted.pem", "127.0.0.1,localhost")

CASES = {
    "tapped": SESSION,
    "given hands' proxy": BRAIN,
    "tapped with nothing replaced that was set": {name: value for name, value in SESSION.items() if not name.startswith("FRITTER_OUTER_")},
    "a proxy set again since": {**SESSION, "HTTPS_PROXY": "http://other:8080"},
    "not a session's": {"HOME": "/home/a", "HTTPS_PROXY": "http://corp:3128"},
    # The proxy unset inside a tapped session: what the tap kept of an earlier one is not given back to it.
    "a tap's leavings": {"HOME": "/home/a", "FRITTER_TAP": TAP, "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128", "https_proxy": TAP},
}


def test_a_tapped_session_s_environment_is_given_back_what_was_there_before() -> None:
    assert untapped(SESSION) == {
        "HOME": "/home/a",
        # Not the tap's: the session's own, which a claude started from it runs under a fritter of its own.
        "FRITTER_SOCKET": "/tmp/fritter-1/session.sock",
        "HTTPS_PROXY": "http://corp:3128",
        # A NO_PROXY that covered the API's host is the session's own again, as the tap had none.
        "NO_PROXY": ".anthropic.com",
        # Set and empty before is set and empty again, not unset.
        "NODE_EXTRA_CA_CERTS": "",
    }
    assert tapped(SESSION) and not tapped(untapped(SESSION))


@pytest.mark.outside
@pytest.mark.parametrize("environ", CASES.values(), ids=CASES.keys())
def test_sh_gives_back_what_python_does(environ: dict[str, str]) -> None:
    printed = subprocess.run(["/bin/sh", "-c", untap_script() + "env -0"], env=environ, capture_output=True, text=True, check=True).stdout
    after = dict(entry.split("=", 1) for entry in printed.split("\0") if entry)
    watched = {*environ, *REPLACED, *KEPT}
    assert {name: value for name, value in after.items() if name in watched} == untapped(environ)


def test_what_a_client_given_hands_proxy_runs_is_given_hands_own_environment_back() -> None:
    assert tapped(BRAIN)
    assert {name: BRAIN[name] for name in REPLACED} == {
        **dict.fromkeys(("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"), "http://127.0.0.1:50000"),
        **dict.fromkeys(("NO_PROXY", "no_proxy"), "127.0.0.1,localhost"),
        TRUST: "/tmp/hands/trusted.pem",
    }
    assert untapped(BRAIN) == HANDS


def test_a_tap_s_leavings_are_gone_whether_or_not_its_session_is_still_tapped() -> None:
    assert untapped(CASES["a tap's leavings"]) == {"HOME": "/home/a", "https_proxy": TAP}


@pytest.mark.outside
def test_what_is_given_back_is_every_variable_a_real_fritter_replaces(fritter: Path) -> None:
    # [LAW:one-source-of-truth] REPLACED restates what fritter sets in a client's environment (fritter/intercept), which
    # Python cannot import: this is where the two are held to one another.
    # Short on purpose: macOS caps a unix socket path near 104 bytes, and fritter's lives in this directory.
    with tempfile.TemporaryDirectory(prefix="untap-", dir="/tmp") as short:
        machine = {"PATH": "/usr/bin:/bin", "HOME": short}
        printed = Path(short) / "env"
        tap = [str(fritter), "--socket-dir", short, "--tap", "https://api.example.com", "--tap-ca", TRUST, "--tap-to", f"{short}/wire.sock", "--"]

        def seen(wrapped: list[str]) -> set[str]:
            """The names sh -c sees run under `wrapped`, the shell's own among them."""
            controller, terminal = os.openpty()
            try:
                subprocess.run([*wrapped, "sh", "-c", f"env -0 > {printed}"], env=machine, stdin=terminal, stdout=terminal, stderr=terminal, check=True)
            finally:
                os.close(terminal)
                os.close(controller)
            return {entry.split("=", 1)[0] for entry in printed.read_text().split("\0") if entry}

        assert {name for name in seen(tap) - seen([]) if not name.startswith("FRITTER_")} == set(REPLACED)
