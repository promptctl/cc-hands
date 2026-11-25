"""Which test modules reach the machine's time, ids, and randomness as they run, held to UNMOVED.

[LAW:single-enforcer] This is the one check on tests/: a reach is caught by what a test's run calls, not by how its
source spells the call. What no wrapper sees is taken from the code scan in tests/test_clock_boundary.py, whose own
tests hold src and the composition roots, which run under no harness: datetime's reads of the wall clock inside its C
type, a C lock's timed acquire, and code run in another process, written out by a test or a script under fixtures/. A
reference to a facility taken before the watch installs, as a default argument or a class attribute, escapes both.

A reach is charged by the unit of work it runs in: a call the harness makes of tests/ code, a hook, a task, a callback
handed to the event loop, a thread, an executor job, a piece of work given to a `SerialThread`, a done callback on a
concurrent future, or a finalizer. Each unit keeps a record made when it was made: the module its work is charged to,
and whether test code, or the harness calling it, handed it that work. A test's call, and a fixture's setup, are the
item's for a test and a fixture of its scope, a wider fixture's own module's for one wider; a finalizer runs as the call
that registered it, so a fixture's teardown as its setup; a hook is its own module's. A task's record is kept by the
task and the rest on a stack per thread, so no context a caller made is touched. A walk out from a reach stops at its
unit's edge, the loop's dispatch, the frame that runs a unit, or the harness's own, and never crosses it. With a frame
of a module under tests/ inside the unit, the reach is charged to the unit's module, or to that frame's with none, so a
test is charged for what its helpers wait on. With none, it is charged to the unit's module if the reach is
`SystemClock`'s or the unit was handed its work, as `asyncio.to_thread(time.sleep, 1)` is; a thread or a task a library
starts for itself, as tqdm's monitor, is no one's. Nothing reads which test runs now, so a thread that outlives its test
is charged to that test and not to the next. A done callback on an asyncio future, whose C type takes no wrapper, runs
as the unit that resolves the future, not the one that added it.

The event loop's clock and timers are charged as the unit that asks the loop for them: a test's `loop.time()`, its
`asyncio.sleep` or `asyncio.timeout`, but not the loop's own reads of its clock as it runs, nor the timers of
`asyncio.run`'s close, asyncio's own shutdown. A log record's timestamp is no reach.
"""

import asyncio
import asyncio.events
import asyncio.runners
import concurrent.futures
import functools
import logging
import os
import random
import secrets
import sys
import threading
import time
import uuid
import weakref
from collections import defaultdict
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import CodeType, FrameType
from typing import Any, NamedTuple, cast

import _pytest.fixtures
import _pytest.python
import pluggy
import pytest
from _pytest.nodes import Node

import hands.threads
from hands.clock import SystemClock
from hands.threads import SerialThread
from test_clock_boundary import offenders, reached, written

# The clock's own tests, which prove the adapter against the machine it reads, and the two checks', whose samples
# reach it: their reaches are the adapter's, not a test's.
CLOCK_TESTS = ("test_clock.py", "test_clock_boundary.py", "test_clockwatch.py")

# Test modules whose verdict still waits on the machine's time, struck off as each moves onto a VirtualClock it
# advances. test_pinned_brain.py, test_session_contract_pinned.py and test_terminal_contract_pinned.py are not among
# those to move: they run a real Claude Code, and nothing tells them when it has done what they wait on; nor is
# test_processes_contract_kernel.py, whose children's exec and exit the kernel's table alone says, polled on the machine's
# time; nor are test_pages.py and test_phonewakepage.py, whose verdict waits on a real Chrome drawing the pages on the machine's time, as the routes behind them run on it. Which facilities a module reaches is not recorded: a library reaches some only on a race, as
# subprocess's communicate sleeps only while its child still runs, and a module is moved whole or not at all.
UNMOVED = frozenset({
    "conftest.py",
    "fixtures/standin_daemon.py",
    "fixtures/watched_run.py",
    "test_answer_permission.py",
    "test_asides.py",
    "test_atdesk.py",
    "test_attention.py",
    "test_audit.py",
    "test_backlogpage.py",
    "test_brain.py",
    "test_brain_turns.py",
    "test_brainpage.py",
    "test_child.py",
    "test_clipowner.py",
    "test_command_event.py",
    "test_conversation.py",
    "test_conversationpage.py",
    "test_cues.py",
    "test_daemon_config.py",
    "test_delta.py",
    "test_devices.py",
    "test_display.py",
    "test_echoscore.py",
    "test_engaged_models.py",
    "test_filing.py",
    "test_focus.py",
    "test_front.py",
    "test_gallerypage.py",
    "test_hermetic.py",
    "test_input_tools.py",
    "test_library.py",
    "test_liveness.py",
    "test_microphone.py",
    "test_narrator.py",
    "test_npm_registry.py",
    "test_occurrences.py",
    "test_offline.py",
    "test_otlp.py",
    "test_own_model.py",
    "test_pages.py",
    "test_patience.py",
    "test_phone.py",
    "test_phone_listening.py",
    "test_phonepage.py",
    "test_phonewake.py",
    "test_phonewakepage.py",
    "test_pinned.py",
    "test_pinned_brain.py",
    "test_pinnedfile_served.py",
    "test_playback.py",
    "test_processes_contract_kernel.py",
    "test_progress.py",
    "test_proxy.py",
    "test_readiness.py",
    "test_recall.py",
    "test_render.py",
    "test_restart.py",
    "test_sentences.py",
    "test_session_contract_pinned.py",
    "test_shim.py",
    "test_shortcut.py",
    "test_smoke.py",
    "test_spoken.py",
    "test_spotify.py",
    "test_stage.py",
    "test_start.py",
    "test_startsession.py",
    "test_status.py",
    "test_streaming.py",
    "test_subagents.py",
    "test_summary_store.py",
    "test_system.py",
    "test_tail.py",
    "test_talking_over.py",
    "test_tap.py",
    "test_terminal_contract_pinned.py",
    "test_timeouts.py",
    "test_tool_runs.py",
    "test_transcript.py",
    "test_transcription_weights.py",
    "test_typing.py",
    "test_update.py",
    "test_usage.py",
    "test_utterance.py",
    "test_voice_turns.py",
    "test_voiceprints.py",
    "test_voices.py",
    "test_wake.py",
    "test_watchdog.py",
    "test_whisper.py",
    "test_whisper_recorded.py",
    "test_whisper_takes.py",
    "test_wide.py",
    "test_wrapper.py",
    "timeouts.py",
})  # fmt: skip

# Facilities the code scan finds in tests/, as it names them.
SCANNED = frozenset({"datetime.datetime.now()", "datetime.datetime.utcnow()", "datetime.datetime.today()", "datetime.date.today()", ".acquire(timeout=)", ".acquire(seconds)", ".get(timeout=)"})

# The machine's time, ids, and randomness, each read by the module function a test or a library calls.
FACILITIES: dict[str, tuple[object, tuple[str, ...]]] = {
    "time": (time, ("monotonic", "monotonic_ns", "time", "time_ns", "sleep", "perf_counter", "perf_counter_ns", "process_time", "process_time_ns", "thread_time", "thread_time_ns", "clock_gettime", "clock_gettime_ns", "localtime", "gmtime", "ctime", "asctime", "strftime")),
    "os": (os, ("urandom",)),
    "uuid": (uuid, ("uuid1", "uuid4")),
    "secrets": (secrets, ("token_bytes", "token_hex", "token_urlsafe", "choice", "randbelow", "randbits")),
    "random": (random, ("random", "uniform", "triangular", "randint", "randrange", "choice", "choices", "shuffle", "sample", "getrandbits", "randbytes", "gauss", "normalvariate", "lognormvariate", "expovariate", "vonmisesvariate", "gammavariate", "betavariate", "paretovariate", "weibullvariate", "binomialvariate")),
}  # fmt: skip

# Functions that read the machine's time only when not handed one: how many arguments they take before it.
GIVEN = {"time.localtime": 0, "time.gmtime": 0, "time.ctime": 0, "time.asctime": 0, "time.strftime": 1}

# Waits handed their seconds at a place or by a keyword, which wait on none when handed none: a sleep of none is a yield.
POLLED = {"time.sleep": (0, "secs"), "threading.Timer": (1, "interval"), "SystemClock.sleep": (1, "seconds"), "SystemClock.sleep_blocking": (1, "seconds"), "SystemClock.take": (2, "seconds")}

# Waits that block on a C lock's timeout and read no Python clock, each with the place its timeout is passed. Every `acquire(timeout=)` is the scan's: Lock and RLock are C types no attribute reaches.
TIMED: dict[str, tuple[type, str, int, str]] = {
    "threading.Event.wait": (threading.Event, "wait", 0, "timeout"),
    "threading.Condition.wait": (threading.Condition, "wait", 0, "timeout"),
    "threading.Thread.join": (threading.Thread, "join", 0, "timeout"),
    # ThreadPoolExecutor.submit polls its idle semaphore with a timeout of none.
    "threading.Semaphore.acquire": (threading.Semaphore, "acquire", 1, "timeout"),
}  # fmt: skip

# The event loop's clock and timers, each with whether a call reaches: the loop's clock read by asyncio as it runs is its
# own, and a timer due now or before, or after no delay, is a poll.
LOOP: dict[str, tuple[str, Callable[[asyncio.BaseEventLoop, tuple[Any, ...], dict[str, Any]], bool]]] = {
    "time.monotonic": ("time", lambda _loop, _args, _kwargs: not asking().f_code.co_filename.startswith(ASYNCIO)),
    "asyncio.BaseEventLoop.call_later": ("call_later", lambda _loop, args, kwargs: argument(args, kwargs, 0, "delay") > 0),
    "asyncio.BaseEventLoop.call_at": ("call_at", lambda loop, args, kwargs: argument(args, kwargs, 0, "when") > loop.time()),
}  # fmt: skip

ASYNCIO = f"{Path(asyncio.__file__).parent}{os.sep}"
# This module and the code that makes units of work on a caller's behalf: a unit's maker is the first frame outside them.
RELAYS = (__file__, threading.__file__, f"{Path(concurrent.futures.__file__).parent}{os.sep}", ASYNCIO, hands.threads.__file__)
# The loop's dispatch: every task's step and every callback the loop runs is called from it.
DISPATCH = asyncio.events.Handle._run.__code__  # pyright: ignore[reportPrivateUsage]
# asyncio's own work, done for no caller: the tasks `asyncio.run` makes as it closes, to cancel what is left, close async
# generators, and shut its executor down within a timeout.
OWN = frozenset({asyncio.runners.Runner.close.__code__})
# pluggy, whose call of a hook is the hook's edge.
PLUGGY = f"{Path(pluggy.__file__).parent}{os.sep}"
# pytest's frames that call tests/ code: a test, a fixture's setup, a yield fixture's teardown. Each is the edge of the
# call pytest pushed around it.
CALLS = frozenset({
    _pytest.python.pytest_pyfunc_call.__code__,
    _pytest.fixtures.call_fixture_func.__code__,  # pyright: ignore[reportUnknownMemberType]
    _pytest.fixtures._teardown_yield_fixture.__code__,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
})  # fmt: skip
# asyncio's own task, whose instances the watch's replaces.
TASK = asyncio.Task

class Unit(NamedTuple):
    """What a unit of work records when it is made: the module its work is charged to, and whether test code handed it
    that work."""

    origin: str | None
    handed: bool


NOBODY = Unit(None, False)


class Running(threading.local):
    """The units running on a thread now, innermost last, each pushed by the frame that runs it."""

    def __init__(self) -> None:
        self.units: list[Unit] = []


class Watch:
    """The record of one run: each module under `tests` with the facilities its run reached."""

    def __init__(self, tests: Path, patch: pytest.MonkeyPatch, unmoved: frozenset[str]) -> None:
        self.tests = tests
        self.unmoved = unmoved
        self._patch = patch
        self._originals: dict[int, Callable[..., object]] = {}
        self._within = threading.local()
        self._running = Running()
        self._tasks: weakref.WeakKeyDictionary[asyncio.Task[Any], Unit] = weakref.WeakKeyDictionary()
        # A pool's record, which its own threads run as: an executor starts them as it is handed its first jobs.
        self._pools: weakref.WeakKeyDictionary[ThreadPoolExecutor, Unit] = weakref.WeakKeyDictionary()
        self.record: defaultdict[str, set[str]] = defaultdict(set)
        self._modules: dict[str, str | None] = {}
        self._ran: set[str] = set()
        self._skipped = False
        self._deselected = False

    def _module(self, filename: str) -> str | None:
        """The module `filename` is, relative to tests/; "" for a file outside it, and this one, that a test may have
        called into; None for the harness's, pytest's, pluggy's and pytest-asyncio's, which run every test."""
        if filename not in self._modules:
            path = Path(filename)
            self._modules[filename] = (
                path.relative_to(self.tests).as_posix() if path.is_relative_to(self.tests) and path != Path(__file__)
                else None if {"_pytest", "pluggy", "pytest_asyncio"} & set(path.parts) else ""
            )  # fmt: skip
        return self._modules[filename]

    def _edge(self, frame: FrameType, entered: FrameType | None) -> Unit | None:
        """The record of the unit whose edge `frame` is, `entered` the frame it called, None for a frame inside a unit.
        The loop's dispatch is the edge of the task it steps: every other callback is a unit of its own, run inside it.
        The frame that runs a unit is the edge of the unit it pushed. pluggy's call of a hook is the edge of the hook, its
        own module's whichever item runs; the rest of the harness's frames, of the call of a test or a fixture the
        harness has pushed on its thread, or of nobody between them."""
        code = frame.f_code
        if code is DISPATCH:
            # The task the handle this frame runs steps, read off the handle: an eager task, whose first step runs inline
            # in its maker's frames, is the current task there, and the dispatch below it is its maker's. Every task is
            # recorded as it is made.
            task = stepping(frame.f_locals["self"]._callback)
            return NOBODY if task is None else self._tasks[task]
        if code is RUNS:
            return self._running.units[-1]
        if code.co_filename.startswith(PLUGGY):
            return NOBODY if entered is None else Unit(self._module(entered.f_code.co_filename) or None, False)
        if self._module(code.co_filename) is None:
            if code in CALLS:
                return self._current()
            # The harness's own code is no one's work, in the module of the unit it runs in.
            return Unit(self._enclosing(frame.f_back, frame).origin, False)
        return None

    def _current(self) -> Unit:
        """The record of the unit this thread runs now, as pushed: nobody's on a thread that runs none."""
        return self._running.units[-1] if self._running.units else NOBODY

    def _enclosing(self, frame: FrameType | None, entered: FrameType | None = None) -> Unit:
        """The record of the unit `frame`, which called `entered`, runs in: its first edge out, or nobody's at a thread's
        bottom."""
        while frame is not None:
            if (unit := self._edge(frame, entered)) is not None:
                return unit
            entered, frame = frame, frame.f_back
        return NOBODY

    def _charge(self, frame: FrameType | None, carried: bool) -> str | None:
        """The module a reach made at `frame` is charged to, from the frames out to its unit's edge and the unit's
        record. With a frame of tests/ code inside the unit, the unit's module, or, as no item runs while a module's
        code runs as it is collected, the innermost such frame's. With none, the unit's module if the reach is `carried`
        or the unit was handed its work, else no one."""
        innermost, entered, unit = "", None, NOBODY
        while frame is not None:
            if (edge := self._edge(frame, entered)) is not None:
                unit = edge
                break
            innermost = innermost or self._module(frame.f_code.co_filename) or ""
            entered, frame = frame, frame.f_back
        if innermost:
            return unit.origin or innermost
        return unit.origin if carried or unit.handed else None

    def _made(self) -> Unit:
        """The record of a unit of work made now, from its maker, the first frame out outside RELAYS and the harness.
        Test code hands it the module it would be charged to. asyncio's own close, src and a library hand it nothing, and
        give it the module of the unit they run in. An edge met first, as when asyncio makes a task from a callback, a
        pool starts its threads, or pytest-asyncio makes a test's task in the call pytest makes of it, gives it that
        edge's unit."""
        frame: FrameType | None = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]
        entered = None
        while frame is not None and frame.f_code not in (DISPATCH, RUNS, *CALLS, *OWN) and self._relays(frame.f_code.co_filename):
            entered, frame = frame, frame.f_back
        if frame is None or frame.f_code in (DISPATCH, RUNS, *CALLS) or frame.f_code.co_filename.startswith(PLUGGY):
            return self._enclosing(frame, entered)
        if self._module(frame.f_code.co_filename):
            return Unit(self._charge(frame, carried=True), True)
        return Unit(self._enclosing(frame).origin, False)

    def _relays(self, filename: str) -> bool:
        """Whether code of `filename` makes units on its caller's behalf, as the harness does for the call it runs."""
        return filename.startswith(RELAYS) or self._module(filename) is None and not filename.startswith(PLUGGY)

    def _runs(self, unit: Unit, run: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        """`run` called as the unit `unit` records: this frame is the unit's edge."""
        units = self._running.units
        units.append(unit)
        try:
            return run(*args, **kwargs)
        finally:
            units.pop()

    def handing(self, work: Callable[..., object]) -> Callable[..., object]:
        """`work`, handed off to run later, wherever it runs, as a unit of its own made by the code handing it."""
        handed = functools.partial(self._runs, self._made(), work)
        # What finds the source of the work, as asyncio's log of a callback that raised, finds the work's, not this watch's.
        handed.__wrapped__ = work  # pyright: ignore[reportAttributeAccessIssue]
        return handed

    def _through(self, facility: str, charged: Callable[[], str | None], function: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        """`function` called, as a reach of `facility` by the module `charged` names unless it is made inside another
        facility's call: a uuid4 reads os.urandom and an Event's wait a Condition's, and the reach is the call the test
        made, not its parts. A timer armed after the call returns, as `asyncio.timeout`'s is on entry, is a reach of its
        own. Only the outer call walks the stack."""
        outer = not getattr(self._within, "facility", False)
        try:
            # Set first: what the watch reads to work out the charge, as the loop's time for a timer, is no reach.
            self._within.facility = True
            if outer and (module := charged()) is not None and module not in CLOCK_TESTS:
                self.record[module].add(facility)
            return function(*args, **kwargs)
        finally:
            self._within.facility = not outer

    def _charged(self, carried: bool) -> str | None:
        return self._charge(sys._getframe(1), carried)  # pyright: ignore[reportPrivateUsage]

    def wrap(self, facility: str, function: Callable[..., object], carried: bool) -> Callable[..., object]:
        """`function`, recording each call as a reach of `facility`, by a test frame or a unit handed its work, or,
        `carried`, by the module of the unit it runs in."""

        given, polled = GIVEN.get(facility), POLLED.get(facility)

        @functools.wraps(function)
        def reached(*args: object, **kwargs: object) -> object:
            reads = (given is None or len(args) <= given or args[given] is None) and (polled is None or argument(args, kwargs, *polled) != 0)
            return self._through(facility, functools.partial(self._charged, carried) if reads else lambda: None, function, *args, **kwargs)

        return reached

    def timed(self, facility: str, method: Callable[..., object], at: int, keyword: str) -> Callable[..., object]:
        """`method`, recording each call given a timeout as a reach of `facility`."""

        @functools.wraps(method)
        def waited(owner: object, *args: object, **kwargs: object) -> object:
            timeout = argument(args, kwargs, at, keyword)
            # A wait of no seconds is a poll, as a sleep of none is a yield: it waits on no time.
            waits = isinstance(timeout, int | float) and timeout > 0
            return self._through(facility, functools.partial(self._charged, False) if waits else lambda: None, method, owner, *args, **kwargs)

        return waited

    def looped(self, facility: str, method: Callable[..., object], reaches: Callable[[asyncio.BaseEventLoop, tuple[Any, ...], dict[str, Any]], bool]) -> Callable[..., object]:
        """`method` of the event loop, recording each call that `reaches` as a reach of `facility`."""

        @functools.wraps(method)
        def asked(loop: asyncio.BaseEventLoop, *args: object, **kwargs: object) -> object:
            return self._through(facility, lambda: self._charged(False) if reaches(loop, args, kwargs) else None, method, loop, *args, **kwargs)

        return asked

    def install(self) -> None:
        patch, originals = self._patch, self._originals
        for name, (module, functions) in FACILITIES.items():
            for function in functions:
                original = getattr(module, function)
                originals[id(original)] = self.wrap(f"{name}.{function}", original, carried=False)
                patch.setattr(module, function, originals[id(original)])
        for method in ("now", "wall", "sleep", "call_at", "timeout", "token", "uuid", "sleep_blocking", "take"):
            patch.setattr(SystemClock, method, self.wrap(f"SystemClock.{method}", getattr(SystemClock, method), carried=True))
        for facility, (owner, method, at, keyword) in TIMED.items():
            patch.setattr(owner, method, self.timed(facility, getattr(owner, method), at, keyword))
        for facility, (method, reaches) in LOOP.items():
            patch.setattr(asyncio.BaseEventLoop, method, self.looped(facility, getattr(asyncio.BaseEventLoop, method), reaches))
        timer = threading.Timer.__init__
        patch.setattr(threading.Timer, "__init__", self.wrap("threading.Timer", timer, carried=False))
        stamp = logging.LogRecord.__init__

        @functools.wraps(stamp)
        def stamped(record: logging.LogRecord, *args: Any, **kwargs: Any) -> None:
            # A record stamps the time it was made: no wait, and no test's reach.
            self._through("logging.LogRecord", lambda: None, stamp, record, *args, **kwargs)

        patch.setattr(logging.LogRecord, "__init__", stamped)
        handle = asyncio.events.Handle.__init__

        def scheduled(scheduled: asyncio.events.Handle, callback: Callable[..., object], *args: Any) -> None:
            # A task's step or wakeup runs as the task; every other callback is a unit of its own, made by the code that
            # hands it to the loop.
            handle(scheduled, callback if stepping(callback) else self.handing(callback), *args)

        patch.setattr(asyncio.events.Handle, "__init__", scheduled)
        watch = self

        class Task(TASK):  # pyright: ignore[reportGeneralTypeIssues, reportUntypedBaseClass]
            """asyncio's task, recorded as it is made: before its first step, which an eager task takes inline."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                watch._tasks[self] = watch._made()  # pyright: ignore[reportPrivateUsage]
                super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]

        # Every task is made by these names, looked up as it is made: loop.create_task's, aiohttp's own.
        for module in (asyncio, asyncio.tasks):
            patch.setattr(module, "Task", Task)
            patch.setattr(module, "eager_task_factory", asyncio.create_eager_task_factory(Task))
        start = threading.Thread.start

        def started(thread: threading.Thread) -> None:
            # The instance's run, which a subclass's override is.
            thread.run = self.handing(thread.run)  # pyright: ignore[reportAttributeAccessIssue]
            start(thread)

        patch.setattr(threading.Thread, "start", started)

        def registered(add: Callable[[Any, Callable[[], object]], None]) -> Callable[[Any, Callable[[], object]], None]:
            def finalizing(owner: Any, finalizer: Callable[[], object]) -> None:
                # A finalizer runs as the call that registers it: a fixture's teardown as its setup, whoever tears it down.
                add(owner, self.handing(finalizer))

            return finalizing

        patch.setattr(pytest.FixtureDef, "addfinalizer", registered(pytest.FixtureDef[object].addfinalizer))
        patch.setattr(Node, "addfinalizer", registered(Node.addfinalizer))
        init, adjust = ThreadPoolExecutor.__init__, ThreadPoolExecutor._adjust_thread_count  # pyright: ignore[reportPrivateUsage]

        def made(pool: ThreadPoolExecutor, *args: Any, **kwargs: Any) -> None:
            self._pools[pool] = self._made()
            init(pool, *args, **kwargs)

        def adjusted(pool: ThreadPoolExecutor) -> None:
            # The threads a pool starts for itself are its maker's: each job they run is a unit of its own.
            self._runs(self._pools.get(pool, NOBODY), adjust, pool)

        patch.setattr(ThreadPoolExecutor, "__init__", made)
        patch.setattr(ThreadPoolExecutor, "_adjust_thread_count", adjusted)
        submit, give = ThreadPoolExecutor.submit, SerialThread.give
        add: Callable[[Future[object], Callable[[Future[object]], object]], None] = Future.add_done_callback  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        def submitted(pool: ThreadPoolExecutor, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future[object]:
            return submit(pool, self.handing(fn), *args, **kwargs)

        def given(serial: SerialThread, work: Callable[[], None]) -> None:
            give(serial, self.handing(work))  # pyright: ignore[reportArgumentType]

        def added(future: Future[object], fn: Callable[[Future[object]], object]) -> None:
            add(future, self.handing(fn))

        patch.setattr(ThreadPoolExecutor, "submit", submitted)
        patch.setattr(SerialThread, "give", given)
        patch.setattr(Future, "add_done_callback", added)

    def rebind(self) -> None:
        """Every module attribute bound to a facility by `from time import monotonic` and its like, bound to its wrapper."""
        for module in [module for module in list(sys.modules.values()) if hasattr(module, "__dict__")]:
            for name, value in list(vars(module).items()):
                if (wrapper := self._originals.get(id(value))) is not None and value is not wrapper:
                    self._patch.setattr(module, name, wrapper)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.rebind()

    @pytest.hookimpl(wrapper=True)
    def pytest_fixture_setup(self, fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest) -> Generator[None, object, object]:
        """A fixture's setup, and the teardown it registers, are a call of tests/ code: the item's, for a fixture of the
        test's scope; the fixture's own module's, for one wider, whichever test it is set up or torn down for; no one's,
        for one defined outside tests/, a plugin's."""
        # The fixture's own function, innermost of the layers a decorator or a plugin wraps it in, as pytest-asyncio's.
        *_, own = layers(fixturedef.func)
        if not (module := self._module(own.co_filename)):
            return (yield from self._running_as(NOBODY))
        node = cast(Node, request.node)  # pyright: ignore[reportUnknownMemberType]
        return (yield from self._running_as(Unit(self._item(node) if fixturedef.scope == "function" else module, True)))

    @pytest.hookimpl(wrapper=True, trylast=True)
    def pytest_pyfunc_call(self, pyfuncitem: pytest.Function) -> Generator[None, object, object]:
        """The test's call, as pytest or pytest-asyncio makes it, is the item's, its decorators' code too: pytest's frame
        that calls it is the edge of the unit pushed here."""
        return (yield from self._running_as(Unit(self._item(pyfuncitem), True)))

    def pytest_runtest_protocol(self, item: pytest.Item) -> None:
        if (module := self._item(item)) is not None:
            self._ran.add(module)

    def _item(self, node: Node) -> str | None:
        """The module under tests/ `node` was collected from, None for one outside it, as pytester's runs are."""
        return node.path.relative_to(self.tests).as_posix() if node.path.is_relative_to(self.tests) else None

    def _running_as(self, unit: Unit) -> Generator[None, object, object]:
        """A hook wrapper's yield, with the harness's thread running `unit` until the hooks it wraps return."""
        units = self._running.units
        units.append(unit)
        try:
            return (yield)
        finally:
            units.pop()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # An xfail ran, and its report says skipped.
        self._skipped |= report.skipped and not hasattr(report, "wasxfail")

    def pytest_deselected(self) -> None:
        self._deselected = True

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        # The whole suite ran when every test module did, each test in it neither skipped nor deselected.
        modules = {path.relative_to(self.tests).as_posix() for path in self.tests.rglob("test_*.py")}
        whole = session.exitstatus == 0 and not self._skipped and not self._deselected and modules <= self._ran
        for module, found in scanned(self.tests).items():
            self.record[module] |= found
        self.new, self.stale = verdict({module: found for module, found in dict(self.record).items() if found}, self.unmoved, whole)
        # Only a run that would pass fails for the watch: an interrupted or empty one keeps its own status.
        if (self.new or self.stale) and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if self.new:
            terminalreporter.section("reached the machine's time, not in UNMOVED (tests/clockwatch.py)", red=True)
            for module, facilities in sorted(self.new.items()):
                terminalreporter.line(f"{module}: {sorted(facilities)}")
        if self.stale:
            terminalreporter.section("in UNMOVED, never reached the machine's time", red=True)
            terminalreporter.line(", ".join(sorted(self.stale)))


def verdict(record: dict[str, set[str]], unmoved: frozenset[str], whole: bool) -> tuple[dict[str, set[str]], set[str]]:
    """Each module a run saw reach the machine's time that `unmoved` does not hold, with what it reached, and, from a run
    of the whole suite, each it holds that never did. A part of the suite sees only its part's reaches, so only the whole
    can prove an entry stale."""
    return {module: found for module, found in record.items() if module not in unmoved}, set(unmoved - record.keys()) if whole else set()


# The frame that runs a unit of work: the unit's edge.
RUNS = Watch._runs.__code__  # pyright: ignore[reportPrivateUsage]


def stepping(callback: object) -> asyncio.Task[Any] | None:
    """The task `callback` is bound to, its step or its wakeup or a method of its own, none of which reaches the clock;
    None for any other callback the loop runs."""
    task: object = getattr(callback, "__self__", None)
    return cast(asyncio.Task[Any], task) if isinstance(task, TASK) else None


def argument(args: tuple[Any, ...], kwargs: dict[str, Any], at: int, keyword: str) -> Any:
    """A call's argument at place `at` or by `keyword`, None if given neither way."""
    return args[at] if len(args) > at else kwargs.get(keyword)


def asking() -> FrameType:
    """The frame that called into the watch now: the innermost outside this module."""
    depth = 1
    while (frame := sys._getframe(depth)).f_code.co_filename == __file__:  # pyright: ignore[reportPrivateUsage]
        depth += 1
    return frame


def layers(function: Callable[..., object]) -> Iterator[CodeType]:
    """The code of `function` and of each function it wraps: pytest calls a decorated test's outermost layer."""
    layer: object = function
    while layer is not None:
        if isinstance(code := getattr(layer, "__code__", None), CodeType):
            yield code
        layer = getattr(layer, "__wrapped__", None)


@functools.cache
def scanned(tests: Path) -> dict[str, set[str]]:
    """What each module under `tests` reaches that no wrapper sees: the scan's facilities in its own code, and any in
    code run in another process, written out by it or, a script under fixtures/, itself."""
    exempt = (*CLOCK_TESTS, Path(__file__).name)
    found = {module: facilities & SCANNED for module, facilities in offenders(tests, exempt).items()}
    for path in tests.rglob("*.py"):
        if (module := path.relative_to(tests).as_posix()) not in exempt:
            source = path.read_text()
            found.setdefault(module, set()).update(written(source) | (reached(source) if module.startswith("fixtures/") else set()))
    return {module: facilities for module, facilities in found.items() if facilities}


def register(config: pytest.Config, unmoved: frozenset[str]) -> None:
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    watch = Watch(config.rootpath.resolve() / "tests", patch, unmoved)
    watch.install()
    config.pluginmanager.register(watch, "clockwatch")
