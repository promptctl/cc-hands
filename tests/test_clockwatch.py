"""The runtime record of what test modules reach of the machine's time, and its verdict against UNMOVED."""

from pathlib import Path

import pytest

from clockwatch import verdict

SAMPLE = """
import datetime, threading, time, uuid
from time import monotonic
from hands.clock import SYSTEM


def tick():
    time.time()


def test_reaches_by_every_route():
    time.sleep(0)
    monotonic()
    uuid.uuid4()
    threading.Event().wait(0.001)
    threading.Event().wait(0)
    datetime.datetime.now()
    done = threading.Event()
    done.set()
    done.wait()
    thread = threading.Thread(target=tick)
    thread.start()
    thread.join(1)


def test_a_thread_hands_started_carries_the_test_that_started_it():
    thread = threading.Thread(target=SYSTEM.now)
    thread.start()
    thread.join()
"""


def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *args: str, unmoved: str = "frozenset()", **modules: str) -> pytest.RunResult:
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent))
    pytester.makepyfile(**{f"tests/{name}": source for name, source in modules.items()})
    pytester.makepyfile(**{"tests/conftest": f"import clockwatch\n\ndef pytest_configure(config):\n    clockwatch.register(config, {unmoved})\n"})
    return pytester.runpytest_subprocess("tests", "-p", "no:randomly", *args)


def reached(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, **modules: str) -> list[str]:
    result = run(pytester, monkeypatch, **modules)
    result.assert_outcomes(passed=sum(source.count("def test_") for source in modules.values()))
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    return result.outlines


def test_a_run_names_each_module_by_what_it_reached(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = reached(pytester, monkeypatch, test_sample=SAMPLE)
    assert "test_sample.py: ['SystemClock.now', 'datetime.datetime.now()', 'threading.Event.wait', 'threading.Thread.join', 'time.monotonic', 'time.sleep', 'time.time', 'uuid.uuid4']" in lines


def test_a_thread_that_outlives_its_test_is_charged_to_the_test_that_started_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = reached(
        pytester,
        monkeypatch,
        test_one="import queue, threading\nfrom hands.clock import SYSTEM\nLATER = queue.Queue()\n\ndef test_starts():\n    threading.Thread(target=SYSTEM.take, args=(LATER, 5), daemon=True).start()\n",
        test_two="from test_one import LATER\n\ndef test_ends():\n    LATER.put(1)\n",
    )
    assert "test_one.py: ['SystemClock.take']" in lines
    assert not any(line.startswith("test_two.py") for line in lines)


def test_a_module_unmoved_does_not_hold_fails_any_run_and_one_never_reached_only_the_whole_suite() -> None:
    record = {"test_a.py": {"time.sleep"}, "test_b.py": {"time.monotonic"}}
    unmoved = frozenset({"test_b.py", "test_c.py"})
    assert verdict(record, unmoved, whole=False) == ({"test_a.py": {"time.sleep"}}, set())
    assert verdict(record, unmoved, whole=True) == ({"test_a.py": {"time.sleep"}}, {"test_c.py"})


def test_an_entry_never_reached_fails_a_run_of_the_whole_suite_and_no_part_of_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    unmoved = "frozenset({'test_quiet.py'})"
    quiet = "def test_waits_on_nothing():\n    pass\n\ndef test_neither():\n    pass\n"
    whole = run(pytester, monkeypatch, unmoved=unmoved, test_quiet=quiet)
    assert whole.ret == pytest.ExitCode.TESTS_FAILED
    assert "test_quiet.py" in whole.outlines
    assert run(pytester, monkeypatch, "-k", "nothing", unmoved=unmoved, test_quiet=quiet).ret == pytest.ExitCode.OK
