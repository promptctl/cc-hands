"""What the brain is doing now, as the phone's page shows it, and what came of a Stop pressed there.

The brain's stage (`hands.brain.stage`) says it, from the turn it has in flight and the calls it heard that turn's replies
make; the phone's page (`hands.voice.brainpage`) shows it. [LAW:one-way-deps] here, below both, so neither imports the other.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

from hands.core.wire import Unreadable

# Wall-clock seconds since the Unix epoch: what a phone counts a call's time from. Read by the stage, never here.
Epoch = NewType("Epoch", float)


@dataclass(frozen=True)
class Running:
    """A call the brain's turn made that is running now: its tool as the model called it, its input, and when it began."""

    tool: str
    input: Mapping[str, object] | Unreadable
    since: Epoch


@dataclass(frozen=True)
class Idle:
    """No turn of the brain's is in flight."""


@dataclass(frozen=True)
class Working:
    """A turn of the brain's is in flight: which one, so a Stop names the turn it was pressed on, since when, and the calls
    it is running now, none while the model is replying."""

    turn: str
    since: Epoch
    calls: tuple[Running, ...]


BrainWork = Idle | Working


@dataclass(frozen=True)
class Stopping:
    """The turn pressed on is being stopped, as Escape at the brain's keyboard stops it."""


@dataclass(frozen=True)
class NotWorking:
    """There was no turn of the brain's to stop: none in flight, or the brain had not yet taken it."""


@dataclass(frozen=True)
class AlreadyEnded:
    """The turn pressed on had ended, and the brain may be on another: that one is not stopped."""


StopOutcome = Stopping | NotWorking | AlreadyEnded
