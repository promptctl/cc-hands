"""What Claude Code's StopFailure hook says failed a turn, as a value: parsed by the brain's process, a fact of the
model's that the system channel says (`hands.sessions.model_facts`). [LAW:one-way-deps] here in core, which loads no
Pipecat, so the commands that load the brain's process answer without it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StopFailed:
    """What Claude Code's StopFailure hook says failed a turn: the kind of error, and the error text it showed. Its
    contract always sends the kind and may not show text; either missing is None, since the hook is never refused over
    what only narrates its turn. Where the wire named no cause, a turn that failed in Claude Code before the API
    answered it, as an expired login does, this is the only account of why."""

    kind: str | None
    message: str | None

    @property
    def login_expired(self) -> bool:
        """Whether the brain's login has expired: [LAW:single-enforcer] the one reading of Claude Code's words for it. The
        pinned Claude Code says it two ways, both starting so: kind `invalid_request` for a token that expired with no
        refresh, and `authentication_failed` for a refresh token that is no longer valid, so the kind is not read."""
        return self.message is not None and self.message.startswith("Login expired")

    def __str__(self) -> str:
        match (self.kind, self.message):
            case (str() as kind, str() as message):
                return f"{kind}: {message}"
            case (str() as kind, None):
                return kind
            case (None, str() as message):
                return message
            case (None, None):
                return "no error named"
