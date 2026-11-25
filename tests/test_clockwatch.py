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


def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *args: str, unmoved: str = "frozenset()", hooks: str = "", **modules: str) -> pytest.RunResult:
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent))
    pytester.makepyfile(**{f"tests/{name}": source for name, source in modules.items()})
    pytester.makepyfile(**{"tests/conftest": f"import clockwatch\n\ndef pytest_configure(config):\n    clockwatch.register(config, {unmoved})\n{hooks}"})
    return pytester.runpytest_subprocess("tests", "-p", "no:randomly", *args)


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


def test_a_module_unmoved_does_not_hold_fails_any_run_and_one_never_reached_only_the_whole_suite() -> None:
    record = {"test_a.py": {"time.sleep"}, "test_b.py": {"time.monotonic"}}
    unmoved = frozenset({"test_b.py", "test_c.py"})
    assert verdict(record, unmoved, whole=False) == ({"test_a.py": {"time.sleep"}}, set())
    assert verdict(record, unmoved, whole=True) == ({"test_a.py": {"time.sleep"}}, {"test_c.py"})


@pytest.mark.outside
def test_an_entry_never_reached_fails_a_run_of_the_whole_suite_and_no_part_of_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    unmoved = "frozenset({'test_quiet.py'})"
    quiet = "def test_waits_on_nothing():\n    pass\n\ndef test_neither():\n    pass\n"
    whole = run(pytester, monkeypatch, unmoved=unmoved, test_quiet=quiet)
    assert whole.ret == pytest.ExitCode.TESTS_FAILED
    assert "test_quiet.py" in whole.outlines
    assert run(pytester, monkeypatch, "-k", "nothing", unmoved=unmoved, test_quiet=quiet).ret == pytest.ExitCode.OK


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
    unmoved = "frozenset({'test_quiet.py'})"
    quiet = "def test_waits_on_nothing():\n    pass\n"
    assert run(pytester, monkeypatch, "--co", unmoved=unmoved, test_quiet=quiet, test_other=quiet).ret == pytest.ExitCode.OK
    assert run(pytester, monkeypatch, "--ignore", "tests/test_other.py", unmoved=unmoved, test_quiet=quiet, test_other=quiet).ret == pytest.ExitCode.OK


@pytest.mark.outside
def test_an_xfail_does_not_keep_the_whole_suite_from_proving_an_entry_stale(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    quiet = "import pytest\n\n@pytest.mark.xfail(strict=True)\ndef test_fails():\n    assert False\n"
    result = run(pytester, monkeypatch, unmoved="frozenset({'test_quiet.py'})", test_quiet=quiet)
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
    # The loop asyncio.run drives reads the machine's clock as it runs, and bounds its executor's shutdown: the loop's own.
    assert "test_user.py: ['time.strftime']" in lines
    assert not any(line.startswith("helper.py") for line in lines)


@pytest.mark.outside
def test_a_test_is_charged_for_the_waits_it_asks_the_loop_for_and_its_reads_of_the_loops_clock(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    def ran(body: str) -> str:
        return f"import asyncio\n\ndef test_runs():\n    async def main():\n{body}\n    asyncio.run(main())\n"

    lines = reached(
        pytester, monkeypatch,
        test_sleeps=ran("        await asyncio.sleep(0.001)"),
        test_bounds=ran("        async with asyncio.timeout(1):\n            pass"),
        test_reads=ran("        asyncio.get_running_loop().time()"),
        test_named=ran("        asyncio.get_running_loop().call_later(delay=1, callback=lambda: None).cancel()"),
        # asyncio's own coroutines handed the loop straight: the waits are the test's all the same.
        test_handed="import asyncio\n\ndef test_runs():\n    asyncio.run(asyncio.sleep(0.001))\n",
        test_bounded="import asyncio\n\ndef test_runs():\n    asyncio.run(asyncio.wait_for(asyncio.sleep(0), 1))\n",
        # The loop's own clock read by the test, for a timer already due.
        test_late=ran("        loop = asyncio.get_running_loop()\n        loop.call_at(loop.time() - 1, lambda: None)"),
        # A job on the loop's executor, whose shutdown asyncio.run bounds by a timer of its own; and a log record.
        test_polls=ran("        loop = asyncio.get_running_loop()\n        loop.call_later(0, lambda: None)\n        loop.call_soon(lambda: None)\n        await asyncio.sleep(0)\n        await asyncio.to_thread(int)\n        import logging\n        logging.getLogger('x').warning('w')"),
    )  # fmt: skip
    assert "test_sleeps.py: ['asyncio.BaseEventLoop.call_later']" in lines
    assert "test_bounds.py: ['asyncio.BaseEventLoop.call_at']" in lines
    assert "test_reads.py: ['time.monotonic']" in lines
    assert "test_named.py: ['asyncio.BaseEventLoop.call_later']" in lines
    assert "test_handed.py: ['asyncio.BaseEventLoop.call_later']" in lines
    assert "test_bounded.py: ['asyncio.BaseEventLoop.call_at']" in lines
    assert "test_late.py: ['time.monotonic']" in lines
    assert not any(line.startswith("test_polls.py") for line in lines)


@pytest.mark.outside
def test_a_wait_handed_to_a_thread_or_an_executor_job_is_charged_to_the_test_that_handed_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # A library outside tests/ that starts a thread for itself, as tqdm does its monitor: the thread's waits are no test's.
    pytester.makepyfile(library="import threading\n\nclass Monitor(threading.Thread):\n    def run(self):\n        threading.Event().wait(0.001)\n\ndef start():\n    monitor = Monitor()\n    monitor.start()\n    monitor.join()\n")
    lines = reached(
        pytester, monkeypatch,
        test_monitored="import library\n\ndef test_runs():\n    library.start()\n",
        # No frame of tests/ is on the stack that waits: the wait is the callable the test handed over.
        test_offloads="import asyncio, threading\n\ndef test_runs():\n    asyncio.run(asyncio.to_thread(threading.Event().wait, 0.001))\n",
        test_slept="import asyncio, time\n\ndef test_runs():\n    asyncio.run(asyncio.to_thread(time.sleep, 0.001))\n",
        test_threaded="import threading, time\n\ndef test_runs():\n    thread = threading.Thread(target=time.sleep, args=(0.001,))\n    thread.start()\n    thread.join()\n",
    )  # fmt: skip
    assert "test_offloads.py: ['threading.Event.wait']" in lines
    assert "test_slept.py: ['time.sleep']" in lines
    assert "test_threaded.py: ['time.sleep']" in lines
    assert not any(line.startswith("test_monitored.py") for line in lines)


@pytest.mark.outside
def test_a_task_a_wider_fixtures_setup_starts_is_charged_to_the_fixture_and_not_the_test_it_was_set_up_for(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # A decorator from outside tests/ is the fixture's outermost layer: its own module is the innermost layer's.
    swap = "import functools\n\ndef wrapped(fixture):\n    @functools.wraps(fixture)\n    async def run():\n        async for value in fixture():\n            yield value\n    return run\n"
    hooks = (
        "\nimport asyncio, time\nimport pytest_asyncio\nfrom swap import wrapped\n\n"
        "@pytest_asyncio.fixture(scope='session', loop_scope='session')\n@wrapped\nasync def started():\n"
        "    asked, stamped = asyncio.Event(), asyncio.Event()\n\n"
        "    async def stamp():\n        await asked.wait()\n        time.strftime('%H')\n        stamped.set()\n\n"
        "    task = asyncio.create_task(stamp())\n    yield asked, stamped\n    await task\n"
    )
    user = "import pytest\n\n@pytest.mark.asyncio(loop_scope='session')\nasync def test_uses(started):\n    asked, stamped = started\n    asked.set()\n    await stamped.wait()\n"
    pytester.makepyfile(swap=swap)
    result = run(pytester, monkeypatch, hooks=hooks, test_user=user)
    result.assert_outcomes(passed=1)
    # The task reaches the clock while the test runs, and is the fixture's work all the same.
    assert "conftest.py: ['time.strftime']" in result.outlines
    assert not any(line.startswith("test_user.py") for line in result.outlines)


@pytest.mark.outside
def test_a_decorated_test_is_charged_and_a_keyword_wait_of_none_is_no_reach(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = "import functools, time\n\ndef later():\n    time.strftime('%H')\n\ndef wrapped(test):\n    @functools.wraps(test)\n    def run():\n        test()\n    return run\n"
    user = "import threading\nfrom helper import later, wrapped\n\n@wrapped\ndef test_decorated():\n    later()\n\ndef test_polls():\n    threading.Timer(interval=0, function=lambda: None).cancel()\n"
    lines = reached(pytester, monkeypatch, helper=helper, test_user=user)
    assert "test_user.py: ['time.strftime']" in lines
    assert not any(line.startswith("helper.py") for line in lines)


@pytest.mark.outside
def test_a_shared_fixture_torn_down_after_a_test_is_charged_to_the_harness_and_not_the_test(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # This repo's timeouts.py, run from the run's own tests/ and registered as its conftest does.
    hooks = "\nimport timeouts\n\ndef pytest_configure(config):\n    config.pluginmanager.register(timeouts, 'timeouts')\n    clockwatch.register(config, frozenset({'timeouts.py'}))\n"
    shared = "import pytest\n\n@pytest.fixture(scope='module')\ndef shared():\n    yield\n\ndef test_uses(shared):\n    pass\n"
    timeouts = (Path(__file__).parent / "timeouts.py").read_text()
    result = run(pytester, monkeypatch, "--timeout", "60", hooks=hooks, timeouts=timeouts, test_quiet=shared)
    result.assert_outcomes(passed=1)
    assert not any(line.startswith("test_quiet.py") for line in result.outlines)
    assert result.ret == pytest.ExitCode.OK
