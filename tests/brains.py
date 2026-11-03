"""What the brain's tests read a turn, a permission, or a failure by, whether the brain is a launched one or built bare."""

from collections.abc import Sequence

import pytest
from loguru import logger

from conftest import events
from hands.brain.process import Asked, Brain
from hands.core.session import Held
from hands.sessions.audit import Entry
from hands.sessions.wide import Fact


def unasked(asked: Asked) -> None:
    """A turn's teller of permissions, for a turn that asks none."""
    pytest.fail(f"a turn that asks nothing held {asked.permission}")


def permissions(recorded: Sequence[Entry]) -> list[tuple[Fact, Fact, Fact]]:
    """The permissions the brain's setup asked about, as the log says each was settled."""
    return [(event.facts["prompt"], event.facts["tool"], event.facts["decision"]) for event in events(recorded, "brain.permission")]


def failures() -> tuple[list[str], int]:
    """What is said on the terminal as an error, as it is said, and the sink that hears it."""
    said: list[str] = []
    return said, logger.add(lambda message: said.append(message.record["message"]), level="ERROR")


def dialogs(brain: Brain) -> tuple[Held, ...]:
    """The dialogs the brain's model holds."""
    held = brain._modelled()  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    return held.dialogs
