"""What the brain's tests read a turn, a permission, or a failure by, whether the brain is a launched one or built bare."""

from collections.abc import Iterator, Sequence

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


@pytest.fixture
def said() -> Iterator[list[str]]:
    """[LAW:no-silent-failure] every error hands says while a test that uses it runs, what the brain's own work raised
    among them (`_broke`): one the test does not take (`taken`) fails it, so work that broke where the test did not look
    is never a pass. A module of the brain's tests uses it for each of its tests (`pytestmark`)."""
    said: list[str] = []
    sink = logger.add(lambda message: said.append(message.record["message"]), level="ERROR", filter="hands")
    try:
        yield said
        assert said == [], f"said and never asserted: {said}"
    finally:
        logger.remove(sink)


def taken(said: list[str]) -> list[str]:
    """What `said` holds, taken out of it, for the test to assert."""
    lines = said[:]
    said.clear()
    return lines


def dialogs(brain: Brain) -> tuple[Held, ...]:
    """The dialogs the brain's model holds."""
    held = brain._modelled()  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    return held.dialogs
