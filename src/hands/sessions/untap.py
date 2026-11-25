"""What fritter's tap replaces in a session's environment, and how what the session runs is given it back; and the same
mark set by hands for its brain, whose HTTPS proxy is hands' own proxy.

A tapped session reaches its API with fritter as its proxy, trusting the authority fritter answers the API's host with
(fritter/intercept). Both end with the session, so what is given back here is not left with them: a claude run inside it
taps itself or reaches the API directly, the session's shell commands run as they would outside it, and hands' brain is
the same brain wherever hands was started. Each is given back what the session's own environment held before fritter.
What Claude Code starts on its own - its MCP servers, its hooks - inherits the tap as it is.
"""

import shlex
from collections.abc import Mapping

# The variable Claude Code reads the certificates it trusts besides its own from.
TRUST = "NODE_EXTRA_CA_CERTS"

# [LAW:one-source-of-truth] what fritter sets in a tapped session, as fritter/intercept's Environment says: its address in
# each proxy variable, each NO_PROXY empty, and in TRUST a file holding its authority. Go's list, restated where Python
# cannot import it, and held to the real fritter by tests/test_untap.py. It keeps each one's earlier value in
# FRITTER_OUTER_<name>.
PROXIES = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
EXEMPT = ("NO_PROXY", "no_proxy")
REPLACED = (*PROXIES, *EXEMPT, TRUST)

# The base URL a Claude Code reaches in the API's place, a gateway's.
BASE_URL = "ANTHROPIC_BASE_URL"

# [LAW:one-source-of-truth] what decides where a Claude Code's requests to the API go: its proxy and the certificates it
# trusts, which a tap replaces, and the base URL it would reach in the API's place.
ROUTING = (*REPLACED, BASE_URL)


def _outer(name: str) -> str:
    return f"FRITTER_OUTER_{name}"


def tapped(environ: Mapping[str, str]) -> bool:
    """Whether environ is a tapped session's, its proxy still the tap's: one set again since is what it was set to."""
    return bool(address := environ.get("FRITTER_TAP")) and environ.get("HTTPS_PROXY") == address


# What fritter keeps of a tap beside what it replaced: never left behind, tapped or not, so a tap's earlier values are
# never given back to a session they were not taken from.
KEPT = ("FRITTER_TAP", *map(_outer, REPLACED))


def tap(environ: Mapping[str, str], proxy: str, trust: str, exempt: str) -> dict[str, str]:
    """What a client started from `environ` is given to reach the world through the proxy at `proxy`, trusting the file
    `trust`, every host `exempt` names reached directly: in the mark fritter's tap leaves, so `untapped` and
    `untap_script` give what the client runs what `environ` held before."""
    given = {**dict.fromkeys(PROXIES, proxy), **dict.fromkeys(EXEMPT, exempt), TRUST: trust}
    return {"FRITTER_TAP": proxy, **given, **{_outer(name): environ[name] for name in REPLACED if name in environ}}


def untapped(environ: Mapping[str, str]) -> dict[str, str]:
    """environ with what its session's tap replaced given back, and the tap's own variables gone."""
    kept = {name: value for name, value in environ.items() if name not in KEPT}
    if not tapped(environ):
        return kept
    return {name: value for name, value in kept.items() if name not in REPLACED} | {name: environ[_outer(name)] for name in REPLACED if _outer(name) in environ}


def untap_script() -> str:
    """untapped, as sh run in the environment to give back."""
    given = "".join(
        f'  if [ -n "${{{_outer(name)}+x}}" ]; then export {name}="${_outer(name)}"; else unset {name}; fi\n' for name in REPLACED
    )
    return f'if [ -n "${{FRITTER_TAP-}}" ] && [ "${{HTTPS_PROXY-}}" = "$FRITTER_TAP" ]; then\n{given}fi\nunset {shlex.join(KEPT)}\n'
