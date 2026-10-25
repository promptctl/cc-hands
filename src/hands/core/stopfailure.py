"""What Claude Code's StopFailure hook says failed a turn, as a value: parsed by the brain's process, a fact of the
model's that the system channel says (`hands.sessions.model_facts`). [LAW:one-way-deps] here in core, which loads no
Pipecat, so the commands that load the brain's process answer without it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StopFailed:
    """What Claude Code's StopFailure hook says failed a turn: the kind of error, always sent, and the error text it
    showed, when it showed one. Where the wire named no cause, a turn that failed in Claude Code before the API answered
    it, as an expired login does, this is the only account of why."""

    kind: str
    message: str | None

    def __str__(self) -> str:
        match self.message:
            case str() as message:
                return f"{self.kind}: {message}"
            case None:
                return self.kind
