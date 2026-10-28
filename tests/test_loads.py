"""A model's load says how big the process was as it ended and how much it grew across it, and a reading that fails is
on its event without failing the load."""

from collections.abc import Callable, Iterator

import numpy as np
import pytest

from hands.sessions.wide import WideEvent
from hands.voice import loads

MEGABYTE = 2**20


def readings(*sizes: int | OSError) -> Callable[[], int]:
    """A read of the resident size that gives `sizes` in turn, raising where one is an error."""
    given: Iterator[int | OSError] = iter(sizes)

    def read() -> int:
        match next(given):
            case OSError() as error:
                raise error
            case int() as size:
                return size

    return read


def test_the_resident_size_is_the_memory_the_process_holds() -> None:
    before = loads.resident()
    held = np.ones(200 * MEGABYTE, np.uint8)
    grew = (loads.resident() - before) / MEGABYTE
    assert held.sum() == 200 * MEGABYTE
    assert 190 < grew < 260


def test_a_load_says_the_size_it_ended_at_and_how_much_the_process_grew_across_it() -> None:
    events: list[WideEvent] = []
    with loads.loading("model.loaded", events.append, readings(100 * MEGABYTE, 1124 * MEGABYTE)):
        pass
    assert [(event.event, event.outcome, event.facts) for event in events] == [("model.loaded", "ok", {"resident_mb": 1124.0, "process_grew_mb": 1024.0})]


def test_a_load_that_fails_says_its_memory_all_the_same() -> None:
    events: list[WideEvent] = []
    with pytest.raises(FileNotFoundError), loads.loading("model.loaded", events.append, readings(100 * MEGABYTE, 150 * MEGABYTE)):
        raise FileNotFoundError("no weights")
    assert [(event.outcome, event.error, event.facts) for event in events] == [
        ("failed", "FileNotFoundError: no weights", {"resident_mb": 150.0, "process_grew_mb": 50.0})
    ]


@pytest.mark.parametrize("sizes", [(OSError("task_info returned kern_return_t 5"), 150 * MEGABYTE), (100 * MEGABYTE, OSError("task_info returned kern_return_t 5"))])
def test_a_reading_that_fails_is_on_the_event_and_the_load_goes_on(sizes: tuple[int | OSError, int | OSError]) -> None:
    events: list[WideEvent] = []
    ran: list[str] = []
    with loads.loading("model.loaded", events.append, readings(*sizes)):
        ran.append("load")
    assert ran == ["load"]
    assert [(event.outcome, event.facts) for event in events] == [("ok", {"resident_error": "task_info returned kern_return_t 5"})]


def test_one_reading_says_its_size_or_why_it_was_not_read() -> None:
    assert loads.size(loads.measured(readings(512 * MEGABYTE))) == {"resident_mb": 512.0}
    assert loads.size(loads.measured(readings(OSError("refused")))) == {"resident_error": "refused"}
