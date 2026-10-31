"""tests/timeouts.py, in runs of their own: a fixture shared by many tests is set up under the suite's bound and counts
against none of them, and each test's own work stays bounded by its own. In another process, as pytest-timeout's timer
and its signal are the whole process's, and its thread method ends the process it fires in."""

import ast
from pathlib import Path
from xml.etree import ElementTree

import pytest

import timeouts

# Each unit sleeps longer than half the bound, so two of them together overrun it and either alone does not.
BOUND = ["--timeout=2"]
# A bound past any setup, for a run whose firing the test drives itself.
UNFIRED = ["--timeout=30"]

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

# Each phase's report takes longer to build than its test has left, as on a loaded machine: a shared setup that raises
# after the test's own setup spent 1.3 s of its 2, a test that fails after 1.3 s of its 2, and a test that passes though
# pytest's logging of its start alone takes longer than its whole bound.
REPORTED = """
import time
import pytest

@pytest.fixture(scope="session")
def shared_raises():
    raise RuntimeError("the shared setup failed")

@pytest.fixture
def first(request):
    time.sleep(1.3)
    request.getfixturevalue("shared_raises")

def test_after_a_shared_setup_that_raises(first):
    pass

def test_that_fails():
    time.sleep(1.3)
    assert False

@pytest.mark.timeout(0.5)
def test_that_passes():
    pass
"""

# Skipped as its setup runs, after a fixture whose teardown never returns and with a shared one torn down after it.
SKIPS = """
import time
import pytest

@pytest.fixture(scope="module")
def shared():
    yield

@pytest.fixture
def never_torn_down(shared):
    yield
    time.sleep(60)

@pytest.fixture
def skips(never_torn_down):
    pytest.skip("skipped as its setup runs")

def test_that_skips(skips):
    pass
"""

# pytest's own logging of a test's start and its reports of each phase, each slow, as on a loaded machine.
# The bound is driven by its signal, with the main thread in the loop's select where a stuck async test sits, so what is
# reported depends on no clock.
AWAITS = """
import asyncio
import signal
import sys
import threading
import time

import pytest

def fire_in_select():
    main = threading.main_thread().ident
    while sys._current_frames()[main].f_code.co_name != "select":
        time.sleep(0.001)
    signal.pthread_kill(main, signal.SIGALRM)

async def deepest():
    threading.Thread(target=fire_in_select, daemon=True).start()
    await asyncio.Event().wait()

async def outermost():
    await deepest()

@pytest.mark.asyncio
async def test_that_awaits():
    await outermost()
"""

BLOCKS = """
import signal

def test_that_blocks():
    signal.raise_signal(signal.SIGALRM)
"""

# A run ended before the session reports its failures, as a job cancelled at its limit is.
ENDED_BEFORE_ITS_REPORT = """
import os

def pytest_runtest_logreport(report):
    if report.failed:
        os._exit(99)
"""

SLOW_REPORTING = """
import time
import pytest

def pytest_runtest_logstart(nodeid, location):
    time.sleep(0.8)

@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    time.sleep(0.8)
    return report
"""


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    """A run's directory holding the plugin, as it is loaded by name, and each suite above as a module of its own."""
    pytester.makepyfile(
        timeouts=Path(timeouts.__file__).read_text(),
        test_shared=SHARED,
        test_async_shared=ASYNC_SHARED,
        test_marked=MARKED,
        test_late=LATE,
        test_hangs=HANGS,
        test_overruns=OVERRUNS,
        test_rearmed=REARMED,
        test_reported=REPORTED,
        test_skips=SKIPS,
        test_awaits=AWAITS,
        test_blocks=BLOCKS,
    )
    return pytester


def run(suite: pytest.Pytester, method: str, *args: str) -> pytest.RunResult:
    # A timer the plugin never restored hangs the run: the run's own bound ends it, failing this test.
    return suite.runpytest_subprocess("-p", "timeouts", *BOUND, f"--timeout-method={method}", *args, timeout=60)


def test_under_a_signal_a_shared_setup_counts_against_no_test_and_each_test_is_bounded_by_its_own_work(suite: pytest.Pytester) -> None:
    # -vv: the summary's lines whole, not cut to the terminal's width.
    ran = run(suite, "signal", "-vv", "test_shared.py", "test_async_shared.py", "test_marked.py", "test_late.py", "test_hangs.py", "test_overruns.py")
    ran.assert_outcomes(passed=3, failed=2, errors=1)
    ran.stdout.fnmatch_lines_random(
        [
            "ERROR test_hangs.py::test_after_a_shared_setup_that_hangs - Failed: Timeout (>2.0s)*",
            # What the test had left once its own setup had run.
            "FAILED test_late.py::test_overruns_with_its_own_setup - Failed: Timeout (>0.*s)*",
            "FAILED test_overruns.py::test_that_never_returns - Failed: Timeout (>1.*s)*",
        ]
    )


def test_under_a_signal_no_timer_runs_while_a_phase_is_reported(suite: pytest.Pytester) -> None:
    # pytest-timeout's one timer spans the whole protocol, reporting and all, so a timer that fires while pytest logs or
    # reports fires inside pytest's own code, and the run ends in INTERNALERROR with outcomes uncounted.
    suite.makeconftest(SLOW_REPORTING)
    ran = run(suite, "signal", "test_hangs.py", "test_reported.py")
    ran.assert_outcomes(passed=1, failed=1, errors=2)
    assert "INTERNALERROR" not in ran.stdout.str()


def test_under_a_signal_a_test_skipped_in_its_setup_is_bounded_through_its_teardown(suite: pytest.Pytester) -> None:
    # The bound fires once: the shared teardown after it re-arms nothing, so the error is the one timeout, not a group.
    ran = run(suite, "signal", "-rsE", "test_skips.py")
    ran.assert_outcomes(skipped=1, errors=1)
    ran.stdout.fnmatch_lines_random(["ERROR test_skips.py::test_that_skips - Failed: Timeout (>1.*s)*"])


def outermost_first(said: str) -> bool:
    return said.index("in test_that_awaits") < said.index("in outermost") < said.index("in deepest")


def test_under_a_signal_a_timeouts_failure_says_where_each_task_waited(suite: pytest.Pytester) -> None:
    ran = run(suite, "signal", *UNFIRED, "test_awaits.py")
    ran.assert_outcomes(failed=1)
    said = ran.stdout.str()
    assert outermost_first(said[said.index("FAILURES") :])


def test_under_a_signal_a_timeout_says_where_each_task_waited_in_a_run_ended_before_its_report(suite: pytest.Pytester) -> None:
    suite.makeconftest(ENDED_BEFORE_ITS_REPORT)
    said = run(suite, "signal", *UNFIRED, "test_awaits.py").stdout.str()
    assert "FAILURES" not in said
    assert outermost_first(said[said.index("test_awaits.py::test_that_awaits timed out") :])


def test_under_a_signal_a_timeout_in_a_sync_test_says_where_its_thread_was_and_that_no_loop_runs(suite: pytest.Pytester) -> None:
    suite.makeconftest(ENDED_BEFORE_ITS_REPORT)
    said = run(suite, "signal", *UNFIRED, "test_blocks.py").stdout.str()
    assert "FAILURES" not in said
    where = said[said.index("test_blocks.py::test_that_blocks timed out") :]
    assert "in test_that_blocks" in where and "no event loop is running on this thread" in where


def test_a_shared_setup_and_teardown_are_on_the_report_of_the_test_they_ran_in(suite: pytest.Pytester) -> None:
    run(suite, "signal", "--junitxml=report.xml", "test_shared.py").assert_outcomes(passed=1)
    properties = ElementTree.parse(suite.path / "report.xml").iterfind(".//property")
    rescoped = [ast.literal_eval(p.attrib["value"]) for p in properties if p.attrib["name"] == timeouts.RESCOPED]
    # pytest-asyncio's session-scoped fixtures are rescoped too.
    recorded = [r for r in rescoped if r["fixture"] == "shared"]
    assert [(r["phase"], r["bound"]) for r in recorded] == [("setup", 2.0), ("teardown", 2.0)]
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
