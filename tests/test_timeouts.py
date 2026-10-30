"""tests/timeouts.py, in runs of their own: a fixture shared by many tests is set up under the suite's bound and counts
against none of them, and each test's own work stays bounded by its own. In another process, as pytest-timeout's timer
and its signal are the whole process's, and its thread method ends the process it fires in."""

import ast
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

import pytest

import timeouts

# Each unit sleeps longer than half the bound, so two of them together overrun it and either alone does not.
BOUND = ["--timeout=2"]

SHARED = """
import time
import pytest

@pytest.fixture(scope="module")
def shared():
    time.sleep(1.3)
    yield
    time.sleep(1.3)

def test_between_a_shared_setup_and_teardown(shared):
    time.sleep(1.3)
"""

ASYNC_SHARED = """
import asyncio
import pytest
import pytest_asyncio

@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def shared():
    await asyncio.sleep(1.3)

@pytest.mark.asyncio(loop_scope="module")
async def test_after_a_shared_async_setup(shared):
    await asyncio.sleep(1.3)
"""

MARKED = """
import time
import pytest

@pytest.fixture(scope="module")
def shared():
    time.sleep(1.3)

@pytest.mark.timeout(1)
def test_with_a_shorter_bound_of_its_own(shared):
    pass
"""

# A session-scoped fixture first set up after a function-scoped one, as pytest-asyncio's runner is: the test's time spent
# before it stays spent.
LATE = """
import time
import pytest

@pytest.fixture(scope="session")
def late():
    pass

@pytest.fixture
def first(request):
    time.sleep(1.3)
    request.getfixturevalue("late")

def test_overruns_with_its_own_setup(first):
    time.sleep(1.3)
"""

# The test's own bound is shorter than the suite's, so the bound that ends the setup says which one it ran under: the
# marker is written past the test's bound and short of the suite's.
SUITE_MARKER = "past the test's own bound"
HANGS = f"""
import sys
import time
import pytest

@pytest.fixture(scope="module")
def shared_never_returns():
    time.sleep(1.5)
    print({SUITE_MARKER!r}, file=sys.stderr, flush=True)
    time.sleep(60)

@pytest.mark.timeout(1)
def test_after_a_shared_setup_that_hangs(shared_never_returns):
    pass
"""

OVERRUNS = """
import time

def test_that_never_returns():
    time.sleep(60)
"""

# Past the 0.7 s a bound never rescoped would have left after the shared setup, and short of the 2 s a re-armed one has.
MARKER = "past the unrescoped bound"
REARMED = f"""
import sys
import time
import pytest

@pytest.fixture(scope="module")
def shared():
    time.sleep(1.3)

def test_overruns_its_own_bound(shared):
    time.sleep(1.0)
    print({MARKER!r}, file=sys.stderr, flush=True)
    time.sleep(60)
"""


# A failed setup's report outlasts the test's own bound: once the suite's bound has fired on a shared setup, the test's
# stays spent, as pytest-timeout's single timer would, so nothing fires while pytest builds the report. pytest-timeout
# cancels the test's timer once the report is built.
SLOW_REPORT = """
import time
import pytest

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "setup" and report.failed:
        time.sleep(1.5)
    return report
"""


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    """A run's directory holding the plugin, as it is loaded by name, and each suite above as a module of its own."""
    pytester.makeconftest(SLOW_REPORT)
    pytester.makepyfile(
        timeouts=Path(timeouts.__file__).read_text(),
        test_shared=SHARED,
        test_async_shared=ASYNC_SHARED,
        test_marked=MARKED,
        test_late=LATE,
        test_hangs=HANGS,
        test_overruns=OVERRUNS,
        test_rearmed=REARMED,
    )
    return pytester


def rescoped(suite: pytest.Pytester) -> list[timeouts.Rescoping]:
    """Each rescoping the run's report.xml records, in the order its tests ran."""
    properties = ElementTree.parse(suite.path / "report.xml").iterfind(".//property")
    return [cast(timeouts.Rescoping, ast.literal_eval(p.attrib["value"])) for p in properties if p.attrib["name"] == timeouts.RESCOPED]


def run(suite: pytest.Pytester, method: str, *args: str) -> pytest.RunResult:
    # A timer the plugin never restored hangs the run: the run's own bound ends it, failing this test.
    return suite.runpytest_subprocess("-p", "timeouts", *BOUND, f"--timeout-method={method}", *args, timeout=60)


def test_under_a_signal_a_shared_setup_counts_against_no_test_and_each_test_is_bounded_by_its_own_work(suite: pytest.Pytester) -> None:
    # -vv: the summary's lines whole, not cut to the terminal's width.
    ran = run(
        suite, "signal", "-vv", "--junitxml=report.xml", "test_shared.py", "test_async_shared.py", "test_marked.py", "test_late.py", "test_hangs.py", "test_overruns.py"
    )
    ran.assert_outcomes(passed=3, failed=2, errors=1)
    # The suite's bound fired on the hung setup alone, so it alone left its test's bound spent.
    assert [(r["fixture"], r["phase"]) for r in rescoped(suite) if r["fired"]] == [("shared_never_returns", "setup")]
    ran.stdout.fnmatch_lines_random(
        [
            "ERROR test_hangs.py::test_after_a_shared_setup_that_hangs - Failed: Timeout (>2.0s)*",
            # What the test had left once its own setup had run.
            "FAILED test_late.py::test_overruns_with_its_own_setup - Failed: Timeout (>0.*s)*",
            "FAILED test_overruns.py::test_that_never_returns - Failed: Timeout (>2.0s)*",
        ]
    )


def test_a_shared_setup_and_teardown_are_on_the_report_of_the_test_they_ran_in(suite: pytest.Pytester) -> None:
    run(suite, "signal", "--junitxml=report.xml", "test_shared.py").assert_outcomes(passed=1)
    # pytest-asyncio's session-scoped fixtures are rescoped too.
    recorded = [r for r in rescoped(suite) if r["fixture"] == "shared"]
    assert [(r["phase"], r["bound"], r["fired"]) for r in recorded] == [("setup", 2.0, False), ("teardown", 2.0, False)]
    # Each took its 1.3 s outside the test's bound, which at teardown had what setup left it less the call's own 1.3 s.
    assert all(r["seconds"] >= 1.3 for r in recorded)
    setup, teardown = recorded
    assert setup["left"] <= 2.0 and teardown["left"] <= setup["left"] - 1.3


def test_under_the_thread_method_a_shared_setup_and_teardown_count_against_no_test(suite: pytest.Pytester) -> None:
    run(suite, "thread", "test_shared.py").assert_outcomes(passed=1)


def test_under_the_thread_method_a_shared_setup_that_hangs_ends_the_run_in_its_own_bound(suite: pytest.Pytester) -> None:
    ran = run(suite, "thread", "test_hangs.py")
    assert ran.ret == 1
    ran.stdout.fnmatch_lines_random(["*+ Timeout +*", "*in shared_never_returns*", SUITE_MARKER])


def test_under_the_thread_method_a_test_is_bounded_by_its_own_work_after_a_shared_setup(suite: pytest.Pytester) -> None:
    ran = run(suite, "thread", "test_rearmed.py")
    assert ran.ret == 1
    # The timer writes what the test had written by the time it fired, the marker among it.
    ran.stdout.fnmatch_lines_random(["*+ Timeout +*", "*in test_overruns_its_own_bound*", MARKER])
