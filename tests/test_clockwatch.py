"""The runtime record of what test modules reach of the machine's time, and its verdict against the modules held to move
and held for good."""

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
    time.sleep(0.001)
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

# What needs the outside in a run's own suite, given to the watch as conftest gives it `needs_the_outside`.
OUTSIDE = "lambda item: item.get_closest_marker('outside') is not None"


def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *args: str, to_move: str = "frozenset()", permanent: str = "{}", hooks: str = "", at: str = "tests", **modules: str) -> pytest.RunResult:
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent))
    pytester.makepyfile(**{f"tests/{name}": source for name, source in modules.items()})
    pytester.makepyfile(**{"tests/conftest": f"import clockwatch\n\ndef pytest_configure(config):\n    clockwatch.register(config, {to_move}, {permanent}, {OUTSIDE})\n{hooks}"})
    return pytester.runpytest_subprocess(at, "-p", "no:randomly", *args)


def reached(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, **modules: str) -> list[str]:
    result = run(pytester, monkeypatch, **modules)
    result.assert_outcomes(passed=sum(source.count("def test_") for source in modules.values()))
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    return result.outlines


@pytest.mark.outside
def test_a_run_names_each_module_by_what_it_reached(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = reached(pytester, monkeypatch, test_sample=SAMPLE)
    assert "test_sample.py: ['SystemClock.now', 'datetime.datetime.now()', 'threading.Event.wait', 'threading.Thread.join', 'time.monotonic', 'time.sleep', 'time.time', 'uuid.uuid4']" in lines


@pytest.mark.outside
def test_a_thread_that_outlives_its_test_is_charged_to_the_test_that_started_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = reached(
        pytester,
        monkeypatch,
        test_one="import queue, threading\nfrom hands.clock import SYSTEM\nLATER = queue.Queue()\n\ndef test_starts():\n    threading.Thread(target=SYSTEM.take, args=(LATER, 5), daemon=True).start()\n",
        test_two="from test_one import LATER\n\ndef test_ends():\n    LATER.put(1)\n",
    )
    assert "test_one.py: ['SystemClock.take']" in lines
    assert not any(line.startswith("test_two.py") for line in lines)


def test_a_module_not_held_that_reaches_fails_any_run_and_a_held_one_that_never_does_only_the_whole_suite() -> None:
    record = {"test_a.py": {"time.sleep"}, "test_b.py": {"time.monotonic"}}
    held = frozenset({"test_b.py", "test_c.py"})
    assert verdict(record, held, whole=False) == ({"test_a.py": {"time.sleep"}}, set())
    assert verdict(record, held, whole=True) == ({"test_a.py": {"time.sleep"}}, {"test_c.py"})


@pytest.mark.outside
def test_a_module_held_both_to_move_and_for_good_stops_the_run_before_any_test(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    result = run(pytester, monkeypatch, to_move="frozenset({'test_quiet.py'})", permanent="{'test_quiet.py': 'a party'}", test_quiet="def test_waits_on_nothing():\n    pass\n")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "held both to move and for good (tests/clockwatch.py): ['test_quiet.py']" in result.stderr.str()


@pytest.mark.outside
def test_a_test_of_a_module_held_for_good_that_the_unit_layer_would_run_fails_the_run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each reaches the machine's time, as a module held for good does, so the run is whole and none of them is stale.
    reaching = "import time\n\nimport pytest\n\n"
    marked = f"{reaching}@pytest.mark.outside\ndef test_outside():\n    time.monotonic()\n"
    held = {"test_module_marked": f"{reaching}pytestmark = pytest.mark.outside\n\ndef test_outside():\n    time.monotonic()\n", "test_each_marked": marked}
    assert run(pytester, monkeypatch, permanent=repr({f"{name}.py": "a party" for name in held}), **held).ret == pytest.ExitCode.OK
    modules = {**held, "test_mixed": f"{marked}\n\ndef test_unit():\n    time.monotonic()\n"}
    result = run(pytester, monkeypatch, permanent=repr({f"{name}.py": "a party" for name in modules}), **modules)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert result.outlines[result.outlines.index("test_mixed.py, held for good: a party") + 1] == "tests/test_mixed.py::test_unit"
    assert not {f"tests/{name}.py::test_outside" for name in modules} & set(result.outlines)


@pytest.mark.outside
def test_a_run_through_a_symlink_to_the_checkout_is_watched_as_one_through_its_path(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    link = pytester.path.parent / f"{pytester.path.name}-link"
    link.symlink_to(pytester.path)
    result = run(pytester, monkeypatch, "--rootdir", str(link), at=str(link / "tests"), test_waits="import time\n\ndef test_waits():\n    time.monotonic()\n")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert any(line.startswith("test_waits.py: ") for line in result.outlines)


@pytest.mark.outside
def test_an_entry_never_reached_fails_a_run_of_the_whole_suite_and_no_part_of_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    to_move = "frozenset({'test_quiet.py'})"
    quiet = "def test_waits_on_nothing():\n    pass\n\ndef test_neither():\n    pass\n"
    whole = run(pytester, monkeypatch, to_move=to_move, test_quiet=quiet)
    assert whole.ret == pytest.ExitCode.TESTS_FAILED
    assert "test_quiet.py" in whole.outlines
    assert run(pytester, monkeypatch, "-k", "nothing", to_move=to_move, test_quiet=quiet).ret == pytest.ExitCode.OK


@pytest.mark.outside
def test_a_test_is_charged_for_what_its_helper_reaches(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = reached(pytester, monkeypatch, helper="import time\n\ndef wait():\n    time.strftime('%H')\n", test_user="from helper import wait\n\ndef test_waits():\n    wait()\n")
    assert "test_user.py: ['time.strftime']" in lines
    assert not any(line.startswith("helper.py") for line in lines)


@pytest.mark.outside
def test_a_thread_starts_with_none_of_its_starters_context(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    source = "import contextvars, threading\nOPEN = contextvars.ContextVar('open', default=None)\n\ndef test_fresh():\n    OPEN.set(1)\n    seen = []\n    thread = threading.Thread(target=lambda: seen.append(OPEN.get()))\n    thread.start()\n    thread.join()\n    assert seen == [None]\n"
    run(pytester, monkeypatch, test_fresh=source).assert_outcomes(passed=1)


@pytest.mark.outside
def test_a_run_that_runs_no_test_or_ignores_a_module_is_not_the_whole_suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    to_move = "frozenset({'test_quiet.py'})"
    quiet = "def test_waits_on_nothing():\n    pass\n"
    assert run(pytester, monkeypatch, "--co", to_move=to_move, test_quiet=quiet, test_other=quiet).ret == pytest.ExitCode.OK
    assert run(pytester, monkeypatch, "--ignore", "tests/test_other.py", to_move=to_move, test_quiet=quiet, test_other=quiet).ret == pytest.ExitCode.OK


@pytest.mark.outside
def test_an_xfail_does_not_keep_the_whole_suite_from_proving_an_entry_stale(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    quiet = "import pytest\n\n@pytest.mark.xfail(strict=True)\ndef test_fails():\n    assert False\n"
    result = run(pytester, monkeypatch, to_move="frozenset({'test_quiet.py'})", test_quiet=quiet)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "test_quiet.py" in result.outlines


@pytest.mark.outside
def test_a_run_that_says_it_deselected_nothing_is_still_the_whole_suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = "\ndef pytest_collection_modifyitems(config, items):\n    config.hook.pytest_deselected(items=[])\n"
    result = run(pytester, monkeypatch, hooks=hooks, to_move="frozenset({'test_quiet.py'})", test_quiet="def test_waits_on_nothing():\n    pass\n")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "test_quiet.py" in result.outlines


@pytest.mark.outside
def test_code_a_test_writes_for_another_interpreter_is_charged_to_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = reached(pytester, monkeypatch, test_child='import subprocess, sys\n\ndef test_spawns():\n    subprocess.run([sys.executable, "-c", "import time"], check=True)\n')
    assert any(line.startswith("test_child.py: ") and "'time'" in line for line in lines)


@pytest.mark.outside
def test_a_hook_is_charged_to_its_own_module_and_not_the_test_it_runs_for(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = "\nimport time\n\ndef pytest_runtest_setup(item):\n    time.monotonic()\n"
    result = run(pytester, monkeypatch, hooks=hooks, test_quiet="def test_nothing():\n    pass\n")
    assert "conftest.py: ['time.monotonic']" in result.outlines
    assert not any(line.startswith("test_quiet.py") for line in result.outlines)


@pytest.mark.outside
def test_a_task_a_test_spawns_is_charged_to_the_test_and_a_wait_of_none_to_no_one(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = "import time\n\nasync def poll():\n    time.strftime('%H')\n"
    user = "import asyncio, threading, time\nfrom helper import poll\n\ndef test_spawns():\n    async def main():\n        await asyncio.create_task(poll())\n    asyncio.run(main())\n\ndef test_yields():\n    time.sleep(0)\n    threading.Timer(0, lambda: None).cancel()\n"
    lines = reached(pytester, monkeypatch, helper=helper, test_user=user)
    # The loop asyncio.run drives reads the machine's monotonic clock too.
    assert "test_user.py: ['time.monotonic', 'time.strftime']" in lines
    assert not any(line.startswith("helper.py") for line in lines)


@pytest.mark.outside
def test_a_decorated_test_is_charged_and_a_keyword_wait_of_none_is_no_reach(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = "import functools, time\n\ndef later():\n    time.strftime('%H')\n\ndef wrapped(test):\n    @functools.wraps(test)\n    def run():\n        test()\n    return run\n"
    user = "import threading\nfrom helper import later, wrapped\n\n@wrapped\ndef test_decorated():\n    later()\n\ndef test_polls():\n    threading.Timer(interval=0, function=lambda: None).cancel()\n"
    lines = reached(pytester, monkeypatch, helper=helper, test_user=user)
    assert "test_user.py: ['time.strftime']" in lines
    assert not any(line.startswith("helper.py") for line in lines)


@pytest.mark.outside
def test_a_semaphore_polled_is_no_reach_whatever_clock_it_reads_and_one_waited_on_is_one(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # A thread pool polls one so to ask whether a worker is idle, as `asyncio.to_thread` has it do.
    polls = "import threading\n\ndef test_polls():\n    threading.Semaphore(0).acquire(timeout=0)\n"
    waits = "import threading\n\ndef test_waits():\n    threading.Semaphore(0).acquire(True, 0.001)\n"
    lines = reached(pytester, monkeypatch, test_polls=polls, test_waits=waits)
    assert "test_waits.py: ['.acquire(seconds)', 'threading.Semaphore.acquire']" in lines
    assert not any(line.startswith("test_polls.py") for line in lines)


@pytest.mark.outside
def test_a_shared_fixture_torn_down_after_a_test_is_charged_to_the_harness_and_not_the_test(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # This repo's timeouts.py, run from the run's own tests/ and registered as its conftest does.
    hooks = "\nimport timeouts\n\ndef pytest_configure(config):\n    config.pluginmanager.register(timeouts, 'timeouts')\n    clockwatch.register(config, frozenset({'timeouts.py'}), {}, " + OUTSIDE + ")\n"
    shared = "import pytest\n\n@pytest.fixture(scope='module')\ndef shared():\n    yield\n\ndef test_uses(shared):\n    pass\n"
    timeouts = (Path(__file__).parent / "timeouts.py").read_text()
    result = run(pytester, monkeypatch, "--timeout", "60", hooks=hooks, timeouts=timeouts, test_quiet=shared)
    result.assert_outcomes(passed=1)
    assert not any(line.startswith("test_quiet.py") for line in result.outlines)
    assert result.ret == pytest.ExitCode.OK
