"""Which test modules reach the machine's time, ids, and randomness as they run, held to UNMOVED.

[LAW:single-enforcer] This is the one check on tests/: a reach is caught by what a test's run calls, not by how its
source spells the call, so no spelling escapes it. Two reaches no wrapper sees, datetime's reads of the wall clock inside
its C type and a C lock's timed acquire, are taken from the code scan in tests/test_clock_boundary.py, whose own tests
hold src and the composition roots, which run under no harness.

A reach is credited to the innermost frame of a module under tests/. A read the machine's clock makes for hands code,
through `SystemClock`, with no such frame on the stack, is credited to the origin the running code carries: the module
whose item was running when the task, the thread, or the executor job that makes it was started. Nothing reads which
test runs now, so a thread that outlives its test is charged to that test and not to the next.
"""

import contextvars
import functools
import os
import random
import secrets
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from hands.clock import SystemClock
from test_clock_boundary import offenders

# The clock's own tests, which prove the adapter against the machine it reads, and the two checks', whose samples
# reach it: their reaches are the adapter's, not a test's.
CLOCK_TESTS = ("test_clock.py", "test_clock_boundary.py", "test_clockwatch.py")

# Test modules whose verdict still waits on the machine's time, struck off as each moves onto a VirtualClock it
# advances. Which facilities a module reaches is not recorded: a library reaches some only on a race, as
# subprocess's communicate sleeps only while its child still runs, and a module is moved whole or not at all.
UNMOVED = frozenset({
    "conftest.py",
    "fetches.py",
    "fixtures/standin_daemon.py",
    "npm_registry.py",
    "test_answer_permission.py",
    "test_asides.py",
    "test_atdesk.py",
    "test_attention.py",
    "test_audit.py",
    "test_backlogpage.py",
    "test_brain.py",
    "test_brain_e2e.py",
    "test_brain_turns.py",
    "test_brainpage.py",
    "test_catch_up.py",
    "test_child.py",
    "test_closesession.py",
    "test_command_event.py",
    "test_conversation.py",
    "test_conversationpage.py",
    "test_daemon_config.py",
    "test_delta.py",
    "test_devices.py",
    "test_display.py",
    "test_engaged.py",
    "test_filing.py",
    "test_focus.py",
    "test_front.py",
    "test_gallerypage.py",
    "test_hold.py",
    "test_holdowner.py",
    "test_input_tools.py",
    "test_library.py",
    "test_liveness.py",
    "test_marketplace.py",
    "test_microphone.py",
    "test_naming.py",
    "test_narrator.py",
    "test_npm_registry.py",
    "test_occurrences.py",
    "test_offline.py",
    "test_otlp.py",
    "test_own_model.py",
    "test_patience.py",
    "test_phone.py",
    "test_phonepage.py",
    "test_pinned.py",
    "test_playback.py",
    "test_progress.py",
    "test_proxy.py",
    "test_read_session.py",
    "test_readiness.py",
    "test_recall.py",
    "test_recorded_turns.py",
    "test_recorder.py",
    "test_refocus.py",
    "test_restart.py",
    "test_sentences.py",
    "test_shim.py",
    "test_smoke.py",
    "test_spoken.py",
    "test_spotify.py",
    "test_stage.py",
    "test_start.py",
    "test_startsession.py",
    "test_status.py",
    "test_steps.py",
    "test_streaming.py",
    "test_subagents.py",
    "test_summary_store.py",
    "test_system.py",
    "test_tail.py",
    "test_talking_over.py",
    "test_tap.py",
    "test_tool_runs.py",
    "test_typing.py",
    "test_update.py",
    "test_usage.py",
    "test_utterance.py",
    "test_vocabulary.py",
    "test_voice_turns.py",
    "test_voiceprints.py",
    "test_voices.py",
    "test_wake.py",
    "test_watchdog.py",
    "test_whisper.py",
    "test_wide.py",
    "test_wrapper.py",
    "timeouts.py",
})  # fmt: skip

# Facilities the code scan finds in tests/, as it names them.
SCANNED = frozenset({"datetime.datetime.now()", "datetime.datetime.utcnow()", "datetime.datetime.today()", "datetime.date.today()", ".acquire(timeout=)", ".acquire(seconds)"})

# The machine's time, ids, and randomness, each read by the module function a test or a library calls.
FACILITIES: dict[str, tuple[object, tuple[str, ...]]] = {
    "time": (time, ("monotonic", "monotonic_ns", "time", "time_ns", "sleep", "perf_counter", "perf_counter_ns")),
    "os": (os, ("urandom",)),
    "uuid": (uuid, ("uuid1", "uuid4")),
    "secrets": (secrets, ("token_bytes", "token_hex", "token_urlsafe", "choice", "randbelow", "randbits")),
    "random": (random, ("random", "uniform", "triangular", "randint", "randrange", "choice", "choices", "shuffle", "sample", "getrandbits", "randbytes", "gauss", "normalvariate", "lognormvariate", "expovariate", "vonmisesvariate", "gammavariate", "betavariate", "paretovariate", "weibullvariate", "binomialvariate")),
}  # fmt: skip

# Waits that block on a C lock's timeout and read no Python clock, each with the place its timeout is passed. Every `acquire(timeout=)` is the scan's: Lock and RLock are C types no attribute reaches.
TIMED: dict[str, tuple[type, str, int, str]] = {
    "threading.Event.wait": (threading.Event, "wait", 0, "timeout"),
    "threading.Condition.wait": (threading.Condition, "wait", 0, "timeout"),
    "threading.Thread.join": (threading.Thread, "join", 0, "timeout"),
}  # fmt: skip

ORIGIN: contextvars.ContextVar[str] = contextvars.ContextVar("clockwatch.origin")


class Watch:
    """The record of one run: each module under `tests` with the facilities its run reached."""

    def __init__(self, tests: Path, patch: pytest.MonkeyPatch, unmoved: frozenset[str]) -> None:
        self.tests = tests
        self.unmoved = unmoved
        self._patch = patch
        self._originals: dict[int, Callable[..., object]] = {}
        self._within = threading.local()
        self.record: defaultdict[str, set[str]] = defaultdict(set)
        self._modules: dict[str, str | None] = {}
        self._skipped = False
        self._deselected = False

    def _module(self, filename: str) -> str | None:
        """The module `filename` is, relative to tests/; "" for a file outside it, and this one, that a test may have
        called into; None for pytest's and pluggy's, below which no test frame lies, since pytest calls every test."""
        if filename not in self._modules:
            path = Path(filename)
            self._modules[filename] = (
                path.relative_to(self.tests).as_posix() if path.is_relative_to(self.tests) and path != Path(__file__)
                else None if {"_pytest", "pluggy"} & set(path.parts) else ""
            )  # fmt: skip
        return self._modules[filename]

    def _innermost(self) -> str | None:
        frame = sys._getframe(2)  # pyright: ignore[reportPrivateUsage]
        while frame is not None and (module := self._module(frame.f_code.co_filename)) is not None:
            if module:
                return module
            frame = frame.f_back
        return None

    def _reach(self, facility: str, module: str | None) -> None:
        if module is not None and module not in CLOCK_TESTS:
            self.record[module].add(facility)

    def _through(self, facility: str, module: str | None, function: Callable[..., object], *args: object, **kwargs: object) -> object:
        """`function` called, as a reach of `facility` by `module` unless it is made inside another facility: a uuid4
        reads os.urandom and an Event's wait a Condition's, and the reach is the call the test made, not its parts."""
        outer = not getattr(self._within, "facility", False)
        if outer:
            self._reach(facility, module)
        self._within.facility = True
        try:
            return function(*args, **kwargs)
        finally:
            self._within.facility = not outer

    def wrap(self, facility: str, function: Callable[..., object], carried: bool) -> Callable[..., object]:
        """`function`, recording each call as a reach of `facility`: by a test frame alone, or, `carried`, by the
        origin the running code carries when no test frame is on the stack."""

        @functools.wraps(function)
        def reached(*args: object, **kwargs: object) -> object:
            module = self._innermost()
            return self._through(facility, module if module is not None or not carried else ORIGIN.get(None), function, *args, **kwargs)

        return reached

    def timed(self, facility: str, method: Callable[..., object], at: int, keyword: str) -> Callable[..., object]:
        """`method`, recording each call given a timeout as a reach of `facility`."""

        @functools.wraps(method)
        def waited(owner: object, *args: object, **kwargs: object) -> object:
            timeout = args[at] if len(args) > at else kwargs.get(keyword)
            # A wait of no seconds is a poll, as a sleep of none is a yield: it waits on no time.
            module = self._innermost() if isinstance(timeout, int | float) and timeout > 0 else None
            return self._through(facility, module, method, owner, *args, **kwargs)

        return waited

    def carrying(self, run: Callable[..., object]) -> Callable[..., object]:
        """`run`, made to carry the origin of the code that starts it, wherever it runs."""
        context = contextvars.copy_context()
        if (origin := self._innermost() or ORIGIN.get(None)) is not None:
            context.run(ORIGIN.set, origin)
        return functools.partial(context.run, run)

    def install(self) -> None:
        patch, originals = self._patch, self._originals
        for name, (module, functions) in FACILITIES.items():
            for function in functions:
                original = getattr(module, function)
                originals[id(original)] = self.wrap(f"{name}.{function}", original, carried=False)
                patch.setattr(module, function, originals[id(original)])
        for method in ("now", "wall", "pause", "call_at", "timeout", "token", "uuid", "rest", "take"):
            patch.setattr(SystemClock, method, self.wrap(f"SystemClock.{method}", getattr(SystemClock, method), carried=True))
        for facility, (owner, method, at, keyword) in TIMED.items():
            patch.setattr(owner, method, self.timed(facility, getattr(owner, method), at, keyword))
        timer = threading.Timer.__init__
        patch.setattr(threading.Timer, "__init__", self.wrap("threading.Timer", timer, carried=False))
        start = threading.Thread.start

        def started(thread: threading.Thread) -> None:
            # The instance's run, which a subclass's override is, made to carry the origin of the code starting it.
            thread.run = self.carrying(thread.run)  # pyright: ignore[reportAttributeAccessIssue]
            start(thread)

        patch.setattr(threading.Thread, "start", started)
        submit = ThreadPoolExecutor.submit

        def submitted(pool: ThreadPoolExecutor, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future[object]:
            return submit(pool, self.carrying(fn), *args, **kwargs)

        patch.setattr(ThreadPoolExecutor, "submit", submitted)

    def rebind(self) -> None:
        """Every module attribute bound to a facility by `from time import monotonic` and its like, bound to its wrapper."""
        for module in list(sys.modules.values()):
            for name, value in list(vars(module).items()):
                if (wrapper := self._originals.get(id(value))) is not None and value is not wrapper:
                    self._patch.setattr(module, name, wrapper)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.rebind()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> Generator[None, object, object]:
        token = ORIGIN.set(item.path.relative_to(self.tests).as_posix() if item.path.is_relative_to(self.tests) else "")
        try:
            return (yield)
        finally:
            ORIGIN.reset(token)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._skipped |= report.skipped

    def pytest_deselected(self) -> None:
        self._deselected = True

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        whole = session.exitstatus == 0 and not self._skipped and not self._deselected and [(session.config.invocation_params.dir / arg).resolve() for arg in session.config.args] == [self.tests]
        for module, found in offenders(self.tests, (*CLOCK_TESTS, Path(__file__).name)).items():
            self.record[module] |= found & SCANNED
        self.new, self.stale = verdict({module: found for module, found in self.record.items() if found}, self.unmoved, whole)
        if self.new or self.stale:
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


def register(config: pytest.Config, unmoved: frozenset[str]) -> None:
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    watch = Watch(config.rootpath.resolve() / "tests", patch, unmoved)
    watch.install()
    config.pluginmanager.register(watch, "clockwatch")
