"""conftest's rule for a quarantined test, run under pytester: a failure in any phase is an expected failure with its ticket as
the reason, its traceback kept, and the run stays green; a failure without the marker still fails the run; a marker that
names no ticket stops the run before any test."""

from types import SimpleNamespace

import pytest

import conftest

SUITE = """
import pytest

@pytest.fixture
def model():
    raise RuntimeError("setup: the model did not load")

@pytest.fixture
def daemon():
    yield
    raise RuntimeError("teardown: the daemon did not stop")

@pytest.mark.quarantined("t-1")
def test_setup_fails(model):
    pass

@pytest.mark.quarantined("t-1")
def test_call_fails():
    raise AssertionError("call: the stream said 3, not 4")

@pytest.mark.quarantined("t-1")
def test_teardown_fails(daemon):
    pass

@pytest.mark.quarantined("t-1")
def test_passes():
    pass
"""

PLAIN_FAILURE = """
def test_fails_without_the_marker():
    assert False
"""

NO_TICKET = """
import pytest

@pytest.mark.quarantined
def test_flaky():
    pass
"""

# The rule and the collection check, as plugins of the run, not the whole conftest beside them.
HOOKS = SimpleNamespace(
    pytest_runtest_makereport=conftest.pytest_runtest_makereport,
    pytest_collection_modifyitems=conftest.pytest_collection_modifyitems,
)


class Reports:
    """Each phase's report as the run logs it, after the rule has rewritten it."""

    def __init__(self) -> None:
        self.reports: list[pytest.TestReport] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.reports.append(report)


def test_a_quarantined_failure_in_any_phase_is_an_expected_failure_with_its_ticket(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SUITE)
    reports = Reports()
    pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS, reports])
    assert [report for report in reports.reports if report.failed] == []
    skipped = {report.when: report for report in reports.reports if report.skipped}
    assert set(skipped) == {"setup", "call", "teardown"}
    assert all(report.wasxfail == "quarantined by t-1" for report in skipped.values())


def test_the_original_traceback_survives_the_rewrite(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SUITE)
    reports = Reports()
    pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS, reports])
    texts = {report.when: str(report.longrepr) for report in reports.reports if report.skipped}
    assert "the model did not load" in texts["setup"]
    assert "the stream said 3, not 4" in texts["call"]
    assert "the daemon did not stop" in texts["teardown"]


def test_a_quarantined_run_exits_green(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SUITE)
    assert pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS]).ret == 0


def test_a_failure_without_the_marker_fails_the_run(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(PLAIN_FAILURE)
    assert pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS]).ret != 0


def test_a_marker_that_names_no_ticket_stops_the_run(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(NO_TICKET)
    ran = pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS])
    assert ran.ret != 0
    ran.stderr.fnmatch_lines(["*is quarantined without its ticket*"])
