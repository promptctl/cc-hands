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


# The expired login said once, in hands' words, and hands' fix for it: what the voice, the heartbeat and `hands check`
# all say of it. Claude Code's own text names its /login, which is not how hands' brain logs in.
LOGIN_EXPIRY = "the brain's login has expired, so every turn fails"
LOGIN_RENEWAL = "`hands login`, then `hands restart`"
# [LAW:one-source-of-truth] the two said as one, where the voice and the heartbeat say them; `hands check` gives each
# its own place in a finding.
LOGIN_EXPIRED_SAID = f"{LOGIN_EXPIRY}: {LOGIN_RENEWAL}"


@dataclass(frozen=True)
class LoginExpired:
    """A turn the brain's Claude Code failed because its login expired, as its StopFailure said (`expired`)."""

    stop: StopFailed

    def __str__(self) -> str:
        return str(self.stop)


def expired(stop: StopFailed) -> StopFailed | LoginExpired:
    """What `stop` says failed the turn: an expired login wherever its text starts "Login expired", as both of the
    wordings Claude Code 2.1.308 ends a turn with when its login expired do (docs/claude-code-behavior.md, Brain), and
    otherwise the StopFailure as it is. The shared start is those two texts', not a grouping of Claude Code's own.

    [LAW:parse-dont-validate] the one place the text is read: whatever reads a turn's end matches on the type."""
    match stop:
        case StopFailed(message=str() as message) if message.startswith("Login expired"):
            return LoginExpired(stop)
        case _:
            return stop
