"""tests/costs.py reads what each test cost out of a run's report log: from records written here, in process, and once from
the log the real plugin writes, so a release of it that changes a record's shape fails here."""

import json
from pathlib import Path

import pytest

from costs import report

SUITE = """
def test_passes():
    pass

def test_fails():
    assert False
"""


def phase(test: str, when: str, duration: float) -> str:
    return json.dumps({"$report_type": "TestReport", "nodeid": test, "when": when, "duration": duration}) + "\n"


def spent(test: str, setup: float, call: float, teardown: float) -> str:
    return phase(test, "setup", setup) + phase(test, "call", call) + phase(test, "teardown", teardown)


STARTED = json.dumps({"$report_type": "SessionStart", "pytest_version": "9.1.1"}) + "\n"
FINISHED = json.dumps({"$report_type": "SessionFinish", "exitstatus": 0}) + "\n"


def test_a_report_gives_the_run_then_each_module_then_each_test_slowest_first() -> None:
    log = STARTED + spent("tests/test_a.py::test_quick", 0.5, 0.25, 0.25) + spent("tests/test_b.py::test_slow", 2, 4, 1) + spent("tests/test_a.py::test_slower", 0.5, 1, 0.5) + FINISHED
    assert report(log) == (
        "3 tests, 10.000 s: the run ended with exit status 0\n"
        "\n"
        "  seconds share tests  module\n"
        "    7.000   70%     1  tests/test_b.py\n"
        "    3.000   30%     2  tests/test_a.py\n"
        "\n"
        "    total     setup      call  teardown  test\n"
        "    7.000     2.000     4.000     1.000  tests/test_b.py::test_slow\n"
        "    2.000     0.500     1.000     0.500  tests/test_a.py::test_slower\n"
        "    1.000     0.500     0.250     0.250  tests/test_a.py::test_quick\n"
    )


def test_a_failed_run_says_how_it_ended() -> None:
    finished = json.dumps({"$report_type": "SessionFinish", "exitstatus": 1}) + "\n"
    assert report(STARTED + spent("tests/test_a.py::test_fails", 0.5, 0.5, 0.5) + finished).startswith("1 tests, 1.500 s: the run ended with exit status 1\n")


def test_a_log_with_no_end_says_it_stopped_and_gives_the_tests_it_holds() -> None:
    told = report(STARTED + spent("tests/test_a.py::test_finished", 1, 1, 1) + phase("tests/test_a.py::test_cancelled_in_its_body", "setup", 1))
    assert told.startswith("2 tests, 4.000 s: the log stops before the run ended\n")
    assert "    1.000     1.000     0.000     0.000  tests/test_a.py::test_cancelled_in_its_body\n" in told


def test_a_line_still_being_written_is_no_record() -> None:
    assert report(STARTED + spent("tests/test_a.py::test_finished", 1, 1, 1) + phase("tests/test_a.py::test_next", "setup", 1)[:40]).startswith("1 tests, 3.000 s: the log stops before the run ended\n")


def test_a_run_stopped_before_any_test_reports_none() -> None:
    assert report(STARTED).startswith("0 tests, 0.000 s: the log stops before the run ended\n")


def test_a_finished_line_that_is_no_record_is_loud() -> None:
    with pytest.raises(json.JSONDecodeError):
        report(STARTED + "not a record\n" + FINISHED)


def test_the_plugins_own_log_reads_as_a_report(pytester: pytest.Pytester, tmp_path: Path) -> None:
    pytester.makepyfile(test_suite=SUITE)
    log = tmp_path / "report-log.jsonl"
    pytester.runpytest_inprocess("-p", "no:asyncio", f"--report-log={log}", "--report-log-exclude-logs-on-passed-tests").assert_outcomes(passed=1, failed=1)
    told = report(log.read_text())
    assert told.startswith("2 tests, ")
    assert told.split("\n")[0].endswith(": the run ended with exit status 1")
    assert "test_suite.py::test_passes\n" in told and "test_suite.py::test_fails\n" in told
