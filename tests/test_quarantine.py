"""conftest's rule for a quarantined test, run under pytester on a day the test sets, so no clock is read: a failure
in any phase is an expected failure with its ticket as the reason, its traceback kept, and the run stays green; a failure
without the marker still fails the run; a quarantine whose date is past, or a marker that is not a ticket and a date, stops
the run before any test."""

from datetime import date
from types import SimpleNamespace

import pytest

import conftest


@pytest.fixture(autouse=True)
def the_tenth_of_january(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "utc_today", lambda: date(2026, 1, 10))


FAILURES = """
import pytest

@pytest.fixture
def model():
    raise RuntimeError("setup: the model did not load")

@pytest.fixture
def daemon():
    yield
    raise RuntimeError("teardown: the daemon did not stop")

@pytest.mark.quarantined("t-1", until="2026-01-12")
def test_setup_fails(model):
    pass

@pytest.mark.quarantined("t-1", until="2026-01-12")
def test_call_fails():
    raise AssertionError("call: the stream said 3, not 4")

@pytest.mark.quarantined("t-1", until="2026-01-12")
def test_teardown_fails(daemon):
    pass

@pytest.mark.quarantined("t-1", until="2026-01-12")
def test_passes():
    pass
"""

# The rule and the collection check, as plugins of the run, not the whole conftest beside them.
HOOKS = SimpleNamespace(pytest_collection_modifyitems=conftest.pytest_collection_modifyitems)


class Reports:
    """Each phase's report as the run logs it, after the rule has rewritten it."""

    def __init__(self) -> None:
        self.reports: list[pytest.TestReport] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.reports.append(report)


def test_a_quarantined_failure_in_any_phase_is_an_expected_failure_with_its_ticket_and_date(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(FAILURES)
    reports = Reports()
    pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS, reports])
    assert [report for report in reports.reports if report.failed] == []
    skipped = {report.when: report for report in reports.reports if report.skipped}
    assert set(skipped) == {"setup", "call", "teardown"}
    assert all(report.wasxfail == "quarantined by t-1 until 2026-01-12" for report in skipped.values())


def test_the_original_traceback_survives_the_rewrite(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(FAILURES)
    reports = Reports()
    pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS, reports])
    texts = {report.when: str(report.longrepr) for report in reports.reports if report.skipped}
    assert "the model did not load" in texts["setup"]
    assert "the stream said 3, not 4" in texts["call"]
    assert "the daemon did not stop" in texts["teardown"]


def test_a_quarantined_run_exits_green(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(FAILURES)
    assert pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS]).ret == 0


def test_a_failure_without_the_marker_fails_the_run(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_fails_without_the_marker():\n    assert False\n")
    assert pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS]).ret != 0


def test_a_quarantine_whose_week_is_past_stops_the_run_before_any_test_naming_it(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        def test_innocent():
            pass

        @pytest.mark.quarantined("t-2", until="2026-01-09")
        def test_flaky():
            pass
        """
    )
    reports = Reports()
    ran = pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS, reports])
    assert ran.ret != 0 and reports.reports == []
    ran.stderr.fnmatch_lines(["*test_flaky was quarantined by t-2 until 2026-01-09, and that week is past*"])


@pytest.mark.parametrize(
    "marker",
    [
        "@pytest.mark.quarantined",
        '@pytest.mark.quarantined("t-3")',
        '@pytest.mark.quarantined(until="2026-01-12")',
        '@pytest.mark.quarantined("", until="2026-01-12")',
        '@pytest.mark.quarantined("t-3", "t-4", until="2026-01-12")',
        '@pytest.mark.quarantined("t-3", until="2026-01-12", reason="slow")',
        '@pytest.mark.quarantined("t-3", until=20260101)',
    ],
)
def test_a_marker_that_is_not_a_ticket_and_a_date_stops_the_run(pytester: pytest.Pytester, marker: str) -> None:
    pytester.makepyfile(f"import pytest\n\n{marker}\ndef test_flaky():\n    pass\n")
    ran = pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS])
    assert ran.ret != 0
    ran.stderr.fnmatch_lines(["*is quarantined without its ticket and its date*"])


def test_an_until_that_is_not_a_date_stops_the_run(pytester: pytest.Pytester) -> None:
    pytester.makepyfile('import pytest\n\n@pytest.mark.quarantined("t-3", until="next week")\ndef test_flaky():\n    pass\n')
    ran = pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS])
    assert ran.ret != 0
    ran.stderr.fnmatch_lines(['*is quarantined until "next week", which is not a date*'])


def test_a_marker_on_a_module_or_a_class_quarantines_each_test_under_it(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        pytestmark = pytest.mark.quarantined("t-4", until="2026-01-12")

        class TestSome:
            def test_one(self):
                assert False

        @pytest.mark.parametrize("n", [1, 2])
        def test_each_case(n):
            assert False
        """
    )
    reports = Reports()
    ran = pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS, reports])
    assert ran.ret == 0 and [report for report in reports.reports if report.failed] == []
    assert len([report for report in reports.reports if report.skipped]) == 3


@pytest.mark.parametrize(("until", "stops"), [("2026-01-09", True), ("2026-01-10", False), ("2026-01-17", False), ("2026-01-18", True)])
def test_a_quarantine_holds_from_today_through_a_week_ahead_and_nothing_beyond_either_end(pytester: pytest.Pytester, until: str, stops: bool) -> None:
    pytester.makepyfile(f'import pytest\n\n@pytest.mark.quarantined("t-5", until="{until}")\ndef test_flaky():\n    assert False\n')
    ran = pytester.runpytest_inprocess("-p", "no:asyncio", plugins=[HOOKS])
    assert (ran.ret != 0) is stops
