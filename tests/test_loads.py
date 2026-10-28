"""A model's load says how big the process was as it ended and how much it grew across it, and a reading that fails is
on its event without failing the load."""

from collections.abc import Callable, Iterator

import pytest

from hands.sessions.wide import WideEvent
from hands.voice import loads

MEGABYTE = 2**20


def readings(*sizes: int | OSError) -> Callable[[], int]:
    """A read of the footprint that gives `sizes` in turn, raising where one is an error."""
    given: Iterator[int | OSError] = iter(sizes)

    def read() -> int:
        match next(given):
            case OSError() as error:
                raise error
            case int() as size:
                return size

    return read


def test_a_load_says_the_size_it_ended_at_and_how_much_the_process_grew_across_it() -> None:
    events: list[WideEvent] = []
    with loads.loading("model.loaded", events.append, readings(100 * MEGABYTE, 1124 * MEGABYTE)):
        pass
    assert [(event.event, event.outcome, event.facts) for event in events] == [("model.loaded", "ok", {"footprint_mb": 1124.0, "process_grew_mb": 1024.0})]


def test_a_load_that_fails_says_its_memory_all_the_same() -> None:
    events: list[WideEvent] = []
    with pytest.raises(FileNotFoundError), loads.loading("model.loaded", events.append, readings(100 * MEGABYTE, 150 * MEGABYTE)):
        raise FileNotFoundError("no weights")
    assert [(event.outcome, event.error, event.facts) for event in events] == [
        ("failed", "FileNotFoundError: no weights", {"footprint_mb": 150.0, "process_grew_mb": 50.0})
    ]


REFUSED = "proc_pid_rusage: Operation not permitted"


@pytest.mark.parametrize(
    ("sizes", "facts"),
    [
        ((OSError(REFUSED), 150 * MEGABYTE), {"footprint_mb": 150.0, "footprint_before_error": REFUSED}),
        ((100 * MEGABYTE, OSError(REFUSED)), {"footprint_error": REFUSED}),
        ((OSError(REFUSED), OSError(REFUSED)), {"footprint_error": REFUSED, "footprint_before_error": REFUSED}),
    ],
)
def test_a_reading_that_fails_is_on_the_event_beside_what_was_read_and_the_load_goes_on(sizes: tuple[int | OSError, int | OSError], facts: dict[str, float | str]) -> None:
    events: list[WideEvent] = []
    ran: list[str] = []
    with loads.loading("model.loaded", events.append, readings(*sizes)):
        ran.append("load")
    assert ran == ["load"]
    assert [(event.outcome, event.facts) for event in events] == [("ok", facts)]
