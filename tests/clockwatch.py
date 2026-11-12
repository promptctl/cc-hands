"""Which test modules reach the machine's time, ids, and randomness as they run, held to UNMOVED.

[LAW:single-enforcer] This is the one check on tests/: a reach is caught by what a test's run calls, not by how its
source spells the call. What no wrapper sees is taken from the code scan in tests/test_clock_boundary.py, whose own
tests hold src and the composition roots, which run under no harness: datetime's reads of the wall clock inside its C
type, a C lock's timed acquire, and code run in another process, written out by a test or a script under fixtures/. A
reference to a facility taken before the watch installs, as a default argument or a class attribute, escapes both.

A reach made with a frame of a module under tests/ on the stack is charged to the origin the running code carries: the
module whose item was running when it, or the task, the thread, or the executor job that makes it, was started, so a
test is charged for what its helpers wait on; the harness's own code, a hook or a finalizer, is charged to its module. A read the machine's clock makes for hands code, through `SystemClock`,
is charged to that origin with no such frame. Nothing reads which test runs now, so a thread that outlives its test is
charged to that test and not to the next.
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
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import CodeType

import pytest

from hands.clock import SystemClock
from test_clock_boundary import offenders, reached, written

# The clock's own tests, which prove the adapter against the machine it reads, and the two checks', whose samples
# reach it: their reaches are the adapter's, not a test's.
CLOCK_TESTS = ("test_clock.py", "test_clock_boundary.py", "test_clockwatch.py")

# Test modules whose verdict still waits on the machine's time, struck off as each moves onto a VirtualClock it
# advances. Which facilities a module reaches is not recorded: a library reaches some only on a race, as
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
    "test_brain_e2e.py",
    "test_brain_turns.py",
    "test_brainpage.py",
    "test_catch_up.py",
    "test_child.py",
    "test_closesession.py",
    "test_command_event.py",
    "test_command_imports.py",
    "test_conversation.py",
    "test_cues.py",
    "test_conversationpage.py",
    "test_daemon_config.py",
    "test_delta.py",
    "test_devices.py",
    "test_display.py",
    "test_engaged_models.py",
    "test_filing.py",
    "test_focus.py",
    "test_front.py",
    "test_gallerypage.py",
    "test_hermetic.py",
    "test_hold.py",
    "test_holdowner.py",
    "test_input_tools.py",
    "test_library.py",
    "test_liveness.py",
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
    "test_timeouts.py",
    "test_tap.py",
    "test_tool_runs.py",
    "test_transcript.py",
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
POLLED = {"time.sleep": (0, "secs"), "threading.Timer": (1, "interval"), "SystemClock.pause": (1, "seconds"), "SystemClock.rest": (1, "seconds"), "SystemClock.take": (2, "seconds")}

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
        self._ran: set[str] = set()
        # The code of every test and fixture an item has run: what pytest enters tests/ through for a test.
        self._tests: set[CodeType] = set()
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

    def _entered(self) -> tuple[str, bool] | None:
        """The innermost frame's module under tests/, and whether the code it runs in is the harness's: entered from
        pytest's own frame directly through anything but a test or a fixture of the item's own, as a hook or a finalizer
        is. A task or a thread, entered from the event loop or the thread's start, is the code that started it. None
        with no frame under tests/."""
        frame, innermost, harness = sys._getframe(), "", False  # pyright: ignore[reportPrivateUsage]
        while frame is not None and (module := self._module(frame.f_code.co_filename)) is not None:
            if module:
                innermost = innermost or module
                caller = frame.f_back
                harness = caller is not None and self._module(caller.f_code.co_filename) is None and frame.f_code not in self._tests
            frame = frame.f_back
        return (innermost, harness) if innermost else None

    def _charged(self, carried: bool) -> str | None:
        """The module a reach is charged to: with a frame under tests/ on the stack, the origin, unless the harness
        makes it, or no item runs, as a module's code runs as it is collected: then the innermost such frame's module.
        With none, the origin if `carried`, else none."""
        if (entered := self._entered()) is None:
            return ORIGIN.get(None) if carried else None
        innermost, harness = entered
        return innermost if harness else ORIGIN.get(innermost)

    def _through(self, facility: str, charged: Callable[[], str | None], function: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        """`function` called, as a reach of `facility` by the module `charged` names unless it is made inside another
        facility: a uuid4 reads os.urandom and an Event's wait a Condition's, and the reach is the call the test made,
        not its parts. Only the outer call walks the stack."""
        outer = not getattr(self._within, "facility", False)
        if outer and (module := charged()) is not None and module not in CLOCK_TESTS:
            self.record[module].add(facility)
        self._within.facility = True
        try:
            return function(*args, **kwargs)
        finally:
            self._within.facility = not outer

    def wrap(self, facility: str, function: Callable[..., object], carried: bool) -> Callable[..., object]:
        """`function`, recording each call as a reach of `facility`: by a test frame alone, or, `carried`, by the
        origin the running code carries when no test frame is on the stack."""

        given, polled = GIVEN.get(facility), POLLED.get(facility)

        @functools.wraps(function)
        def reached(*args: object, **kwargs: object) -> object:
            reads = (given is None or len(args) <= given or args[given] is None) and (polled is None or (args[polled[0]] if len(args) > polled[0] else kwargs.get(polled[1])) != 0)
            return self._through(facility, functools.partial(self._charged, carried) if reads else lambda: None, function, *args, **kwargs)

        return reached

    def timed(self, facility: str, method: Callable[..., object], at: int, keyword: str) -> Callable[..., object]:
        """`method`, recording each call given a timeout as a reach of `facility`."""

        @functools.wraps(method)
        def waited(owner: object, *args: object, **kwargs: object) -> object:
            timeout = args[at] if len(args) > at else kwargs.get(keyword)
            # A wait of no seconds is a poll, as a sleep of none is a yield: it waits on no time.
            waits = isinstance(timeout, int | float) and timeout > 0
            return self._through(facility, functools.partial(self._charged, False) if waits else lambda: None, method, owner, *args, **kwargs)

        return waited

    def carrying(self, run: Callable[..., object]) -> Callable[..., object]:
        """`run`, made to carry the origin of the code that starts it, wherever it runs, and nothing else of its context:
        a thread starts in an empty one, as it does unwatched."""
        context = contextvars.Context()
        if (origin := self._charged(carried=True)) is not None:
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
        for module in [module for module in list(sys.modules.values()) if hasattr(module, "__dict__")]:
            for name, value in list(vars(module).items()):
                if (wrapper := self._originals.get(id(value))) is not None and value is not wrapper:
                    self._patch.setattr(module, name, wrapper)

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.rebind()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> Generator[None, object, object]:
        if not item.path.is_relative_to(self.tests):
            return (yield)
        module = item.path.relative_to(self.tests).as_posix()
        self._ran.add(module)
        if isinstance(item, pytest.Function):
            functions: list[Callable[..., object]] = [item.function]  # pyright: ignore[reportUnknownMemberType]
            # A fixture of a wider scope is set up and torn down for whichever test comes first or last: the harness's.
            functions += (definition.func for definitions in item._fixtureinfo.name2fixturedefs.values() for definition in definitions if definition.scope == "function")  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
            self._tests.update(code for function in functions for code in layers(function))
        token = ORIGIN.set(module)
        try:
            return (yield)
        finally:
            ORIGIN.reset(token)

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
