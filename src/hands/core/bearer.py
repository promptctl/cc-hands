"""A bearer token (RFC 6750) as hands' servers take one from their clients: the Authorization header that carries it,
and the check that a request's header is that one.

[LAW:one-way-deps] here, below the phone's pages (`hands.voice.phoneaddress`) and the brain's MCP server
(`hands.brain.mcp`), so both check a token the one way.
"""

import hmac


def bearer(token: str) -> str:
    """The Authorization header that carries `token`."""
    return f"Bearer {token}"


def bears(authorization: str, token: str) -> bool:
    """Whether `authorization` is the header that carries `token`.

    [LAW:single-enforcer] the one check of a bearer token, compared in constant time; a header aiohttp decoded with
    surrogateescape is compared as the bytes it arrived as, so no header a client sends makes the check raise.
    """
    return hmac.compare_digest(authorization.encode("utf-8", "surrogateescape"), bearer(token).encode())
