"""The metrics: views over the wide events as they are recorded, pushed to VictoriaMetrics as each memory sample is, and
once more as the run ends; each push said in the log, with why where the store did not take it."""

from urllib.error import URLError
from urllib.request import Request

from hands.sessions.audit import Entry, Exported
from hands.sessions.metrics import IMPORT, MEMORY, Metrics, pushing
from hands.sessions.otlp import FAILING_BATCHES, Exports
from hands.sessions.wide import WideEvent, annotate, begun, count, fail, unit
from hands.virtualclock import VirtualClock

STORE = "https://victoriametrics.example"


def test_each_event_counts_once_by_name_and_outcome_its_seconds_and_counts_add_up_and_a_zero_count_is_written() -> None:
    clock, kept = VirtualClock(), list[Entry]()
    metrics = Metrics(kept.append)
    for _ in range(2):
        with unit("delta.read", metrics.record, begun(clock), ("commits", "files")):
            count(commits=3)
            clock.advance(0.25)
    with unit("delta.read", metrics.record, begun(clock), ("commits", "files")):
        fail("the forge refused")
    # Every line reaches the log as it was recorded, folded or not.
    assert [entry.event for entry in kept if isinstance(entry, WideEvent)] == ["delta.read"] * 3
    assert metrics.exposition() == (
        "# TYPE hands_events_total counter\n"
        'hands_events_total{service="hands",event="delta.read",outcome="ok"} 2\n'
        'hands_events_total{service="hands",event="delta.read",outcome="failed"} 1\n'
        "# TYPE hands_event_seconds_total counter\n"
        'hands_event_seconds_total{service="hands",event="delta.read",outcome="ok"} 0.5\n'
        'hands_event_seconds_total{service="hands",event="delta.read",outcome="failed"} 0.0\n'
        "# TYPE hands_event_counts_total counter\n"
        'hands_event_counts_total{service="hands",event="delta.read",count="commits"} 6\n'
        'hands_event_counts_total{service="hands",event="delta.read",count="files"} 0\n'
        "# TYPE hands_memory_bytes gauge\n"
    )


def test_the_last_memory_sample_is_the_memory_in_bytes_by_what_each_size_measures() -> None:
    clock = VirtualClock()
    metrics = Metrics(lambda _: None)
    with unit(MEMORY, metrics.record, begun(clock)):
        annotate(footprint_mb=9000.0, mlx_cache_mb=4000.0)
    with unit(MEMORY, metrics.record, begun(clock)):
        # A footprint the kernel refused is no size: the sample says why, and the metric is left out until one is read.
        annotate(footprint_error="[Errno 1] Operation not permitted", mlx_active_mb=2587.5, mlx_cache_mb=1024.0)
    assert metrics.exposition().split("# TYPE hands_memory_bytes gauge\n")[1] == (
        f'hands_memory_bytes{{service="hands",kind="mlx_active"}} {round(2587.5 * 2**20)}\nhands_memory_bytes{{service="hands",kind="mlx_cache"}} {2**30}\n'
    )


def test_a_label_value_is_escaped_as_the_exposition_format_spells_it() -> None:
    metrics = Metrics(lambda _: None)
    with unit('a "quoted"\\name\n', metrics.record, begun(VirtualClock())):
        pass
    assert 'event="a \\"quoted\\"\\\\name\\n"' in metrics.exposition()


class Store:
    """A stand-in for VictoriaMetrics' import: it keeps each request, and raises `refusing` where one is given."""

    def __init__(self, refusing: Exception | None = None) -> None:
        self.requests: list[Request] = []
        self.refusing = refusing

    def send(self, request: Request, timeout: float) -> bytes:
        self.requests.append(request)
        if self.refusing is not None:
            raise self.refusing
        return b""


def test_the_metrics_are_pushed_as_each_memory_sample_is_recorded_and_once_more_as_the_run_ends() -> None:
    clock, kept, store = VirtualClock(), list[Entry](), Store()
    with pushing(STORE, Metrics(kept.append), kept.append, clock, store.send) as record:
        with unit("hearing", record, begun(clock)):
            pass
        with unit(MEMORY, record, begun(clock)):
            annotate(footprint_mb=6354.8)
    sent = [(request.full_url, request.get_method(), request.data) for request in store.requests]
    assert [(url, method) for url, method, _ in sent] == [(f"{STORE}{IMPORT}", "POST")] * 2
    # The push the sample set off carries that sample.
    assert isinstance(first := sent[0][2], bytes)
    assert 'hands_events_total{service="hands",event="hearing",outcome="ok"} 1' in first.decode()
    assert f'hands_memory_bytes{{service="hands",kind="footprint"}} {round(6354.8 * 2**20)}' in first.decode()
    # The last push carries the sample's own event, recorded as it ended.
    assert isinstance(last := sent[1][2], bytes)
    assert f'hands_events_total{{service="hands",event="{MEMORY}",outcome="ok"}} 1' in last.decode()
    pushes = [entry for entry in kept if isinstance(entry, Exported)]
    assert [(push.collector, push.signal, push.spans, push.error) for push in pushes] == [(STORE, "metrics", (), None)] * 2


def test_a_push_the_store_did_not_take_is_said_with_why_and_a_run_of_them_is_a_degradation() -> None:
    clock, kept = VirtualClock(), list[Entry]()
    exports = Exports(kept.append, clock)
    with pushing(STORE, Metrics(exports.record), exports.record, clock, Store(URLError("connection refused")).send) as record:
        for _ in range(FAILING_BATCHES - 1):
            with unit(MEMORY, record, begun(clock)):
                pass
    pushes = [entry for entry in kept if isinstance(entry, Exported)]
    assert [push.error for push in pushes] == ["URLError: <urlopen error connection refused>"] * FAILING_BATCHES
    [degraded] = exports.degraded()
    assert degraded.brief == "can't export metrics"


def test_with_no_store_the_metrics_are_folded_and_pushed_nowhere() -> None:
    clock, kept = VirtualClock(), list[Entry]()
    with pushing(None, Metrics(kept.append), kept.append, clock, Store(AssertionError("nothing is pushed")).send) as record:
        with unit(MEMORY, record, begun(clock)):
            pass
    assert [type(entry).__name__ for entry in kept] == ["WideEvent"]
