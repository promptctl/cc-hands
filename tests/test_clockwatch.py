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
    # The loop asyncio.run drives reads the machine's clock as it runs: the loop's own.
    assert "test_user.py: ['time.strftime']" in lines
    assert not any(line.startswith("helper.py") for line in lines)


def runs(body: str) -> dict[str, str]:
    """A test whose body is `body`'s lines, as a sync test under `asyncio.run` and as an async test under pytest-asyncio."""
    lines = [line for line in body.split("\n") if line]
    return {
        "sync": "import asyncio, logging\n\ndef test_runs():\n    async def main():\n" + "".join(f"        {line}\n" for line in lines) + "    asyncio.run(main())\n",
        "async": "import asyncio, logging, pytest\n\n@pytest.mark.asyncio\nasync def test_runs():\n" + "".join(f"    {line}\n" for line in lines),
    }


@pytest.mark.outside
def test_a_test_is_charged_for_the_waits_it_and_its_tasks_ask_the_loop_for_however_the_loop_is_run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    charged: dict[str, tuple[str, list[str]]] = {
        "sleeps": ("await asyncio.sleep(0.001)", ["asyncio.BaseEventLoop.call_later"]),
        "bounds": ("async with asyncio.timeout(1):\n    pass", ["asyncio.BaseEventLoop.call_at"]),
        "reads": ("asyncio.get_running_loop().time()", ["time.monotonic"]),
        "spawns": ("await asyncio.create_task(asyncio.sleep(0.001))", ["asyncio.BaseEventLoop.call_later"]),
        "gathers": ("await asyncio.gather(asyncio.sleep(0.001))", ["asyncio.BaseEventLoop.call_later"]),
        "groups": ("async with asyncio.TaskGroup() as group:\n    group.create_task(asyncio.wait_for(asyncio.sleep(0), 1))", ["asyncio.BaseEventLoop.call_at"]),
        # The test's own read of the loop's clock, for a timer already due.
        "late": ("loop = asyncio.get_running_loop()\nloop.call_at(loop.time() - 1, lambda: None)", ["time.monotonic"]),
        # Polls, a log record, and a job on the loop's executor, whose shutdown asyncio.run bounds by a timer of its own.
        "polls": ("loop = asyncio.get_running_loop()\nloop.call_later(0, lambda: None)\nloop.call_soon(lambda: None)\nawait asyncio.sleep(0)\nlogging.getLogger('x').warning('w')\nawait asyncio.to_thread(int)", []),
    }  # fmt: skip
    modules = {f"test_{way}_{name}": source for name, (body, _) in charged.items() for way, source in runs(body).items()}
    result = run(pytester, monkeypatch, **modules)
    result.assert_outcomes(passed=len(modules))
    for name, (_, facilities) in charged.items():
        for way in ("sync", "async"):
            expected = [f"test_{way}_{name}.py: {facilities}"] if facilities else []
            assert [line for line in result.outlines if line.startswith(f"test_{way}_{name}.py:")] == expected, (way, name)


# A library outside tests/: a thread and a task it starts for itself, and a wait it is handed.
LIBRARY = """
import asyncio, threading, time
from hands.clock import SYSTEM

class Monitor(threading.Thread):
    def run(self):
        threading.Event().wait(0.001)

def watch():
    monitor = Monitor()
    monitor.start()
    monitor.join()

def offload():
    return asyncio.ensure_future(asyncio.to_thread(time.sleep, 0.001))

def snooze():
    return asyncio.ensure_future(asyncio.sleep(0.001))

def nap(*_):
    time.sleep(0.001)

async def dozes():
    time.sleep(0.001)

def clocked():
    thread = threading.Thread(target=SYSTEM.now)
    thread.start()
    thread.join()
"""

# A pool and a SerialThread another module makes, shared by the tests that hand them work.
SHARED = """
import concurrent.futures
import library, pytest
from hands.threads import SerialThread

@pytest.fixture(scope='session')
def pool():
    with concurrent.futures.ThreadPoolExecutor(1, initializer=library.nap) as pool:
        yield pool

@pytest.fixture(scope='session')
def serial():
    return SerialThread('serial')
"""


@pytest.mark.outside
def test_a_wait_handed_to_a_thread_a_job_or_a_serial_thread_is_charged_to_the_test_that_handed_it(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    pytester.makepyfile(library=LIBRARY)
    handed: dict[str, tuple[str, list[str]]] = {
        "test_offloads": ("import asyncio, threading\n\ndef test_runs():\n    asyncio.run(asyncio.to_thread(threading.Event().wait, 0.001))\n", ["threading.Event.wait"]),
        "test_slept": ("import asyncio, time\n\ndef test_runs():\n    asyncio.run(asyncio.to_thread(time.sleep, 0.001))\n", ["time.sleep"]),
        "test_threaded": ("import threading, time\n\ndef test_runs():\n    thread = threading.Thread(target=time.sleep, args=(0.001,))\n    thread.start()\n    thread.join()\n", ["time.sleep"]),
        "test_submitted": ("import concurrent.futures, time\n\ndef test_runs():\n    with concurrent.futures.ThreadPoolExecutor(1) as pool:\n        pool.submit(time.sleep, 0.001).result()\n", ["time.sleep"]),
        "test_off_loop": ("import asyncio, functools, time\nfrom hands.threads import off_loop\n\ndef test_runs():\n    asyncio.run(off_loop(functools.partial(time.sleep, 0.001), 'slow'))\n", ["time.sleep"]),
        # Callbacks and a coroutine handed to the loop, by the test and by a thread it started.
        "test_scheduled": ("import asyncio, library\n\ndef test_runs():\n    async def main():\n        asyncio.get_running_loop().call_soon(library.nap)\n        await asyncio.sleep(0)\n    asyncio.run(main())\n", ["time.sleep"]),
        "test_told": ("import asyncio, threading, library\n\ndef test_runs():\n    async def main():\n        thread = threading.Thread(target=asyncio.get_running_loop().call_soon_threadsafe, args=(library.nap,))\n        thread.start()\n        thread.join()\n        await asyncio.sleep(0)\n    asyncio.run(main())\n", ["time.sleep"]),
        "test_threadsafe": ("import asyncio, library\n\ndef test_runs():\n    async def main():\n        future = await asyncio.to_thread(asyncio.run_coroutine_threadsafe, library.dozes(), asyncio.get_running_loop())\n        await asyncio.wrap_future(future)\n    asyncio.run(main())\n", ["time.sleep"]),
        "test_gathered": ("import asyncio, pytest, time\n\n@pytest.mark.asyncio\nasync def test_runs():\n    await asyncio.gather(asyncio.to_thread(time.sleep, 0.001))\n", ["time.sleep"]),
        # Work handed to a SerialThread and a pool conftest.py made, which charges conftest.py the pool's initializer.
        "test_given": ("import asyncio, functools, time\n\ndef test_runs(serial):\n    asyncio.run(serial.run(functools.partial(time.sleep, 0.001)))\n", ["time.sleep"]),
        "test_a_fills": ("def test_runs(pool):\n    pool.submit(int).result()\n", []),
        # A done callback run on the worker the first module started.
        "test_b_calls": ("import threading, library\n\ndef test_runs(pool):\n    gate = threading.Event()\n    future = pool.submit(gate.wait)\n    future.add_done_callback(library.nap)\n    gate.set()\n    future.result()\n", ["time.sleep"]),
        # A thread and a task the library starts for itself.
        "test_monitored": ("import library\n\ndef test_runs():\n    library.watch()\n", []),
        "test_offloaded": ("import asyncio, library\n\ndef test_runs():\n    async def main():\n        await library.offload()\n    asyncio.run(main())\n", []),
        # A thread the library starts for the test's sake reads the clock for hands, which is the test's read.
        "test_clocked": ("import library\n\ndef test_runs():\n    library.clocked()\n", ["SystemClock.now"]),
        "test_snoozed": ("import asyncio, library\n\ndef test_runs():\n    async def main():\n        await library.snooze()\n    asyncio.run(main())\n", []),
    }  # fmt: skip
    result = run(pytester, monkeypatch, hooks=SHARED, **{name: source for name, (source, _) in handed.items()})
    result.assert_outcomes(passed=len(handed))
    assert "conftest.py: ['time.sleep']" in result.outlines
    for name, (_, facilities) in handed.items():
        expected = [f"{name}.py: {facilities}"] if facilities else []
        assert [line for line in result.outlines if line.startswith(f"{name}.py:")] == expected, name


@pytest.mark.outside
def test_a_task_a_wider_fixtures_setup_starts_is_charged_to_the_fixture_and_not_the_test_it_was_set_up_for(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # A decorator from outside tests/ is the fixture's outermost layer: its own module is its innermost layer's.
    pytester.makepyfile(swap="import functools\n\ndef wrapped(fixture):\n    @functools.wraps(fixture)\n    async def run():\n        async for value in fixture():\n            yield value\n    return run\n")
    hooks = (
        "\nimport asyncio\nimport helper, pytest_asyncio\nfrom swap import wrapped\n\n"
        "@pytest_asyncio.fixture(scope='session', loop_scope='session')\n@wrapped\nasync def started():\n"
        "    asked, stamped = asyncio.Event(), asyncio.Event()\n\n"
        "    async def stamp():\n        await asked.wait()\n        helper.stamp()\n        stamped.set()\n\n"
        "    task = asyncio.create_task(stamp())\n    yield asked, stamped\n    await task\n"
    )
    user = "import pytest\n\n@pytest.mark.asyncio(loop_scope='session')\nasync def test_uses(started):\n    asked, stamped = started\n    asked.set()\n    await stamped.wait()\n"
    result = run(pytester, monkeypatch, hooks=hooks, helper="import time\n\ndef stamp():\n    time.strftime('%H')\n", test_user=user)
    result.assert_outcomes(passed=1)
    # The task reaches the clock, through a helper, while the test runs, and is the fixture's work all the same.
    assert "conftest.py: ['time.strftime']" in result.outlines
    assert not any(line.startswith(("test_user.py", "helper.py")) for line in result.outlines)


# A wider fixture's setup, which starts a thread through a helper of tests/ or the library, and its teardown: undecorated,
# and wrapped by a decorator from outside tests/, so the decorator is its outermost layer and the fixture's own its
# innermost. Each with what it reaches.
WIDER = {
    "decorated": ("@pytest.fixture(scope='module')\n@wrapped\ndef started():\n    asked = threading.Event()\n    thread = helper.start(asked)\n    yield asked\n    thread.join()\n", "time.strftime"),
    "plain": ("@pytest.fixture(scope='module')\ndef started():\n    asked = threading.Event()\n    thread = helper.start(asked)\n    yield asked\n    thread.join()\n", "time.strftime"),
    "torn_down": ("@pytest.fixture(scope='session')\n@wrapped\ndef started():\n    yield threading.Event()\n    helper.stamp()\n", "time.strftime"),
    "library_thread": ("@pytest.fixture(scope='module')\ndef started():\n    library.clocked()\n    yield threading.Event()\n", "SystemClock.now"),
}  # fmt: skip


@pytest.mark.outside
@pytest.mark.parametrize(("fixture", "facility"), WIDER.values(), ids=WIDER.keys())
def test_what_a_wider_fixture_does_through_a_helper_is_charged_to_the_fixture_and_not_the_helper_or_the_test(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, fixture: str, facility: str) -> None:
    pytester.makepyfile(library=LIBRARY)
    pytester.makepyfile(swap="import functools\n\ndef wrapped(fixture):\n    @functools.wraps(fixture)\n    def run():\n        yield from fixture()\n    return run\n")
    helper = "import threading, time\n\ndef stamp():\n    time.strftime('%H')\n\ndef start(asked):\n    thread = threading.Thread(target=lambda: (asked.wait(), stamp()))\n    thread.start()\n    return thread\n"
    hooks = f"\nimport threading\nimport helper, library, pytest\nfrom swap import wrapped\n\n{fixture}"
    user = "def test_uses(started):\n    started.set()\n"
    result = run(pytester, monkeypatch, hooks=hooks, helper=helper, test_user=user)
    result.assert_outcomes(passed=1)
    assert f"conftest.py: ['{facility}']" in result.outlines
    assert not any(line.startswith(("test_user.py", "helper.py")) for line in result.outlines)


@pytest.mark.outside
def test_a_decorated_test_is_charged_and_a_keyword_wait_of_none_is_no_reach(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = "import functools, time\n\ndef later():\n    time.strftime('%H')\n\ndef wrapped(test):\n    @functools.wraps(test)\n    def run():\n        test()\n    return run\n"
    user = "import threading\nfrom helper import later, wrapped\n\n@wrapped\ndef test_decorated():\n    later()\n\ndef test_polls():\n    threading.Timer(interval=0, function=lambda: None).cancel()\n"
    # A decorator from outside tests/ is the test's own layer: what it reaches, with no frame of tests/ under it, is the test's.
    pytester.makepyfile(retry="import functools, threading, time\n\ndef retried(test):\n    @functools.wraps(test)\n    def run():\n        time.strftime('%H')\n        waited = threading.Thread(target=time.sleep, args=(0.001,))\n        waited.start()\n        waited.join()\n        test()\n    return run\n")
    retried = "from retry import retried\n\n@retried\ndef test_retried():\n    pass\n"
    lines = reached(pytester, monkeypatch, helper=helper, test_user=user, test_retried=retried)
    assert "test_user.py: ['time.strftime']" in lines
    assert "test_retried.py: ['time.sleep', 'time.strftime']" in lines
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
