"""Each speech model's load, as a unit of work of its own that says how big the process was as it ended.

What the models cost in memory is read off these events: each says the process's physical footprint as its load ended,
and how much the process grew across it (hands.footprint). Both are facts about the process, not the model: a load that
ran alone grew it by the model's own cost, and one that overlapped another load, which the events' own times show, grew
it by both. The last load's event is the process with every model it then holds loaded.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager

from hands import footprint
from hands.footprint import Measured
from hands.sessions.wide import Fact, WideEvent, annotate, unit


def memory(before: Measured, after: Measured) -> dict[str, Fact]:
    """What a load's event says of memory: its footprint as it ended, and how much the process grew since `before`."""
    match before, after:
        case int(), int():
            return {**footprint.size(after), "process_grew_mb": footprint.megabytes(after - before)}
        case OSError(), _:
            return {**footprint.size(after), "footprint_before_error": str(before)}
        case int(), OSError():
            return footprint.size(after)


@contextmanager
def loading(event: str, emit: Callable[[WideEvent], None], read: Callable[[], int] = footprint.footprint) -> Generator[None]:
    """Run a model's load as the unit of work `event`, which says how big the process was as it ended and how much it
    grew across it, however the load ends. [LAW:nothing-unseen] a measurement that fails is on the event as
    `footprint_error` or `footprint_before_error`, and the load goes on as it would have."""
    with unit(event, emit):
        before = footprint.measured(read)
        try:
            yield
        finally:
            annotate(**memory(before, footprint.measured(read)))
