"""A page's read that waits for what the page shows to change: answered at once when the page has seen nothing or
something else, once a change comes while it waits, and as it is at the end of its wait."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from hands.sessions.wide import WideEvent, begun, unit
from hands.virtualclock import VirtualClock
from hands.voice.longpoll import POLL_SECONDS, WAIT_SECONDS, version_of, waited


class Shown:
    """What a page shows, as a test sets it, and the look a read takes of it."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.failure: Exception | None = None

    async def look(self) -> tuple[str, str]:
        if self.failure is not None:
            raise self.failure
        return version_of(self.text), self.text


async def read(clock: VirtualClock, look: Callable[[], Awaitable[tuple[str, str]]], seen: str | None) -> tuple[tuple[str, str] | BaseException, WideEvent]:
    """What the read answers, or what it raised, and the event of the unit it ran in."""
    recorded: list[WideEvent] = []
    answered: tuple[str, str] | BaseException
    try:
        with unit("page.read", recorded.append, begun(clock)):
            answered = await waited(clock, look, seen)
    except Exception as error:
        answered = error
    [event] = recorded
    return answered, event


async def test_a_page_that_has_seen_nothing_is_answered_at_once() -> None:
    clock, shown = VirtualClock(), Shown("charles")
    answered, event = await read(clock, shown.look, None)
    assert answered == (version_of("charles"), "charles")
    assert (event.facts["looks"], event.facts["waited_ms"], event.facts["changed"]) == (1, 0.0, True)


async def test_a_page_that_has_seen_something_else_is_answered_at_once() -> None:
    clock, shown = VirtualClock(), Shown("charles")
    answered, event = await read(clock, shown.look, version_of("a run before"))
    assert answered == (version_of("charles"), "charles")
    assert (event.facts["looks"], event.facts["changed"]) == (1, True)


async def test_a_page_that_has_seen_what_is_shown_is_answered_once_it_changes() -> None:
    clock, shown = VirtualClock(), Shown("charles")
    reading = asyncio.create_task(read(clock, shown.look, version_of("charles")))
    await clock.sleeping(POLL_SECONDS)
    assert not reading.done()
    shown.text = "bill_boerst"
    clock.advance(POLL_SECONDS)
    answered, event = await reading
    assert answered == (version_of("bill_boerst"), "bill_boerst")
    assert (event.facts["looks"], event.facts["waited_ms"], event.facts["changed"]) == (2, POLL_SECONDS * 1000, True)


async def test_a_page_that_has_seen_what_is_shown_is_answered_with_it_unchanged_at_the_end_of_the_wait() -> None:
    clock, shown = VirtualClock(), Shown("charles")
    reading = asyncio.create_task(read(clock, shown.look, version_of("charles")))
    await clock.sleeping(POLL_SECONDS)
    clock.advance(WAIT_SECONDS)
    answered, event = await reading
    assert answered == (version_of("charles"), "charles")
    assert (event.facts["waited_ms"], event.facts["changed"]) == (WAIT_SECONDS * 1000, False)


async def test_a_look_that_fails_partway_through_the_wait_says_how_far_the_wait_had_come() -> None:
    clock, shown = VirtualClock(), Shown("charles")
    reading = asyncio.create_task(read(clock, shown.look, version_of("charles")))
    await clock.sleeping(POLL_SECONDS)
    shown.failure = OSError("the voice file is unreadable")
    clock.advance(POLL_SECONDS)
    answered, event = await reading
    assert isinstance(answered, OSError)
    assert (event.outcome, event.facts["looks"], event.facts["waited_ms"]) == ("failed", 2, POLL_SECONDS * 1000)
    assert "changed" not in event.facts


@pytest.mark.parametrize("shown", [{"level": "engaged", "voices": ["charles"]}, {"voices": ["charles"], "level": "engaged"}])
def test_a_version_is_of_what_is_shown_whatever_order_it_was_built_in(shown: dict[str, object]) -> None:
    assert version_of(shown) == version_of({"level": "engaged", "voices": ["charles"]})
    assert version_of(shown) != version_of({"level": "off", "voices": ["charles"]})
