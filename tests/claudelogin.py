"""What Claude Code's `claude auth status --json` prints of the login it holds: [LAW:one-source-of-truth] as fake_claude
prints it, and as a test standing in for that claude gives it (conftest.LOGGED_IN)."""

import json

# What it says of whoever holds a login, by the login's method.
HOLDERS: dict[str, dict[str, str]] = {"claude.ai": {"email": "brain@example.com"}, "api_key": {"email": "brain@example.com", "apiKeySource": "/login managed key"}, "oauth_token": {}}


def auth_status(logged_in: bool, method: str = "claude.ai", provider: str = "firstParty") -> str:
    """The line it prints for a login made by `method` through `provider`."""
    return json.dumps({"loggedIn": logged_in, "authMethod": method, "apiProvider": provider, **HOLDERS[method]})
