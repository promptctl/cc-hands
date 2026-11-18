"""Metrics: views over the wide events the audit log records, pushed to VictoriaMetrics in the Prometheus text format.

    [telemetry]
    metrics = "https://victoriametrics.example"

[LAW:one-source-of-truth] no metric is kept on its own: each is an aggregate over the events as they are recorded, so the
log and the metrics never say two things of one run. Every event counts once in `hands_events_total` and adds its
duration to `hands_event_seconds_total`, by its name and how it ended; each count it declared adds to
`hands_event_counts_total`, a count of zero written as zero; and the last `memory.sampled` event's sizes are
`hands_memory_bytes`, by what each measures. Every label is an event's name, an outcome, a count's name or a size's, all
fixed by the code and none by what a run saw, so the series stay as few as the code's events.

[LAW:nothing-unseen] each push is an Exported line of the metrics signal: how long it took and, where VictoriaMetrics did
not take it, why. A run of them is a degradation, as for the collector's signals (hands.sessions.otlp.Exports). The push
runs on a thread of its own, so a store that is slow or gone costs a unit of work nothing.

The metrics are pushed rather than scraped, so nothing need reach into this Mac, and pushed as each memory sample is
recorded, once a minute (hands.voice.memory).
"""

import queue
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from urllib.request import Request

from hands.clock import Clock
from hands.sessions.audit import Entry, Exported, Record
from hands.sessions.otlp import SERVICE, STOPPED, TIMEOUT_SECONDS, post, traced, why
from hands.sessions.wide import Outcome, WideEvent

# The wide event whose sizes are hands_memory_bytes: the process's memory, sampled as the run goes (hands.voice.memory).
MEMORY = "memory.sampled"
# VictoriaMetrics' import of the Prometheus text exposition format, appended to the address [telemetry] metrics names.
IMPORT = "/api/v1/import/prometheus"
# A size's fact is named for what it measures, in megabytes, as every event in hands writes a size (hands.footprint).
_MB = "_mb"
_MEGABYTE = 2**20


class Metrics:
    """`record`, folding each wide event it records into the metrics, which `exposition` writes out."""

    def __init__(self, record: Record) -> None:
        self._record = record
        # [LAW:no-shared-mutable-globals] written by `record` alone, from whatever thread records, under the lock;
        # `exposition` reads them under it.
        self._folding = threading.Lock()
        self._events: dict[tuple[str, Outcome], tuple[int, float]] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._memory: Mapping[str, float] = {}

    def record(self, entry: Entry) -> None:
        self._record(entry)
        # [LAW:single-enforcer] a line is a unit of work's event where the trace says it is: the proxy's exchanges too.
        event = traced(entry)
        if event is None:
            return
        with self._folding:
            ran, seconds = self._events.get((event.event, event.outcome), (0, 0.0))
            self._events[(event.event, event.outcome)] = (ran + 1, seconds + event.duration_ms / 1000)
            for name, n in event.counts.items():
                self._counts[(event.event, name)] = self._counts.get((event.event, name), 0) + n
            if event.event == MEMORY:
                self._memory = {name.removesuffix(_MB): float(size) for name, size in event.facts.items() if name.endswith(_MB) and isinstance(size, int | float)}

    def exposition(self) -> str:
        """Every metric as the Prometheus text exposition format writes it."""
        with self._folding:
            events, counts, memory = dict(self._events), dict(self._counts), self._memory
        lines = [
            "# TYPE hands_events_total counter",
            *(_sample("hands_events_total", {"event": name, "outcome": outcome}, ran) for (name, outcome), (ran, _) in events.items()),
            "# TYPE hands_event_seconds_total counter",
            *(_sample("hands_event_seconds_total", {"event": name, "outcome": outcome}, seconds) for (name, outcome), (_, seconds) in events.items()),
            "# TYPE hands_event_counts_total counter",
            *(_sample("hands_event_counts_total", {"event": name, "count": counted}, n) for (name, counted), n in counts.items()),
            "# TYPE hands_memory_bytes gauge",
            *(_sample("hands_memory_bytes", {"kind": kind}, round(size * _MEGABYTE)) for kind, size in memory.items()),
        ]
        return "\n".join(lines) + "\n"


def _sample(metric: str, labels: Mapping[str, str], value: float) -> str:
    written = ",".join(f'{name}="{_escaped(value)}"' for name, value in {"service": SERVICE, **labels}.items())
    return f"{metric}{{{written}}} {value}"


def _escaped(value: str) -> str:
    # The exposition format's escapes inside a label value: backslash, double quote, and line feed.
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@contextmanager
def pushing(address: str | None, record: Record, clock: Clock, send: Callable[[Request, float], bytes] = post, timeout: float = TIMEOUT_SECONDS) -> Generator[Record]:
    """`record`, and where `address` names a VictoriaMetrics, each line recorded through it also folded into the
    metrics, which are pushed there as each memory sample is recorded, and once more as this ends, from a thread of its
    own. Each push is recorded as an Exported line.

    [LAW:no-ambient-temporal-coupling] the sampler (hands.voice.memory) owns the period, so the metrics have no clock of
    their own; and the pusher alone writes the metrics out, as it takes each push, so no push carries older metrics than
    the one before it."""
    if address is None:
        yield record
        return
    metrics = Metrics(record)
    # Each push asked for: True for the last, as this ends.
    last: queue.SimpleQueue[bool] = queue.SimpleQueue()

    def push() -> None:
        began = clock.now()
        try:
            send(Request(f"{address}{IMPORT}", data=metrics.exposition().encode(), headers={"Content-Type": "text/plain"}, method="POST"), timeout)
            error = None
        except Exception as failed:
            # Unreachable or an HTTP error status: the push is lost to the store alike, and said alike.
            error = why(failed)
        record(Exported(address, "metrics", (), (clock.now() - began) * 1000, error, None))

    def run() -> None:
        while not last.get():
            push()
        push()

    def pushed(entry: Entry) -> None:
        metrics.record(entry)
        if isinstance(entry, WideEvent) and entry.event == MEMORY:
            last.put(False)

    pusher = threading.Thread(target=run, name="metrics push", daemon=True)
    pusher.start()
    try:
        yield pushed
    finally:
        last.put(True)
        # [LAW:no-ambient-temporal-coupling] a store that is gone holds up a stop by one push's timeout and a second, as it
        # does the collector's exporter (hands.sessions.otlp.Exporter.close); a push still waiting then, and so the last
        # push, is said here, as the process may end before it.
        pusher.join(timeout + 1)
        if pusher.is_alive():
            record(Exported(address, "metrics", (), 0.0, f"{STOPPED}: a push was still waiting on the store", None))
