"""pytest-timeout's bound, scoped to the work it bounds: a fixture wider than one function is set up inside whichever test
first asks for it and torn down inside the last, yet it is the work of every test that shares it, so its setup and its
teardown run under the suite's bound and the test's own bound resumes with what it had left; and the test's bound runs
only inside its phases, never while pytest reports one. A plugin of its own, so a run in another process can load it by name with
`-p timeouts`; conftest registers it for the suite."""

import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import cast

import pytest
from pytest_timeout import Settings, get_env_settings  # pyright: ignore[reportUnknownVariableType]  (untyped in pytest-timeout)


@dataclass(frozen=True)
class Armed:
    """The timer pytest-timeout armed for `item`, which fires at `deadline` on the monotonic clock."""

    item: pytest.Item
    settings: Settings
    deadline: float


# The suite's own settings, resolved once as the run is configured: `--timeout`, PYTEST_TIMEOUT and the ini file, never a
# test's marker, and never the environment as a test has left it.
SUITE = pytest.StashKey[Settings]()


@dataclass(frozen=True)
class Paused:
    """`item`'s bound stopped between two of its phases with `left` of it still to run."""

    item: pytest.Item
    settings: Settings
    left: float


# [LAW:no-shared-mutable-globals] [LAW:one-source-of-truth] the test's bound right now: armed, paused between its phases,
# or none. Written only by the two hooks that arm and cancel a timer, and by `pause` and `resume`.
BOUND = pytest.StashKey[Armed | Paused | None]()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    config.stash[SUITE] = get_env_settings(config)
    config.stash[BOUND] = None


@pytest.hookimpl(wrapper=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: Settings) -> Generator[None, object, object]:
    """[LAW:single-enforcer] pytest-timeout arms every timer, the ones this plugin asks for among them; this only reads
    off what it armed, once it has: the deadline never comes before the timer's own, so a bound with nothing left by it
    has expired."""
    armed = yield
    item.config.stash[BOUND] = Armed(item, settings, time.monotonic() + settings.timeout)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  (untyped in pytest-timeout)
    return armed


@pytest.hookimpl(wrapper=True)
def pytest_timeout_cancel_timer(item: pytest.Item) -> Generator[None, object, object]:
    """pytest-timeout ending the test's bound ends it paused or armed: after a failure, in pytest_exception_interact, the
    phases left run unbounded, as pytest-timeout's own would."""
    item.config.stash[BOUND] = None
    return (yield)


@contextmanager
def shared(config: pytest.Config, phase: str, fixture: str) -> Generator[None]:
    """Work every test sharing `fixture` owns, under the suite's bound: the armed timer stopped meanwhile and re-armed
    with the time it had left, nested work restoring innermost first. What it did lands on the test's report, and so in
    junitxml."""
    # The timer armed on entry: a test's, or an outer shared setup's.
    armed = config.stash[BOUND]
    # With no timer armed, as under `timeout_func_only` or between a test's phases, nothing bounds the work to rescope.
    if not isinstance(armed, Armed):
        yield
        return
    started = time.monotonic()
    paused = pause(config, armed)
    suite = config.stash[SUITE]
    bound = cast(float | None, suite.timeout)  # pyright: ignore[reportUnknownMemberType]  (untyped in pytest-timeout)
    # A suite with no bound gives its shared fixtures none.
    if bound:
        config.hook.pytest_timeout_set_timer(item=armed.item, settings=suite)
    try:
        yield
    finally:
        config.hook.pytest_timeout_cancel_timer(item=armed.item)
        # [LAW:nothing-unseen] the bound the work ran under, how long it took, and what the test had left: a timeout
        # reported after it names that remainder, not the test's configured bound. Recorded before the re-arm, which a
        # bound with next to nothing left fires at once.
        armed.item.user_properties.append(
            (RESCOPED, {"phase": phase, "fixture": fixture, "bound": bound, "seconds": round(time.monotonic() - started, 3), "left": round(paused.left, 3)})
        )
        resume(config, paused)


def pause(config: pytest.Config, armed: Armed) -> Paused:
    """[LAW:single-enforcer] `armed`'s timer cancelled, keeping what it had left for `resume`."""
    left = armed.deadline - time.monotonic()
    config.hook.pytest_timeout_cancel_timer(item=armed.item)
    return Paused(armed.item, armed.settings, left)


def resume(config: pytest.Config, paused: Paused) -> None:
    """[LAW:single-enforcer] `paused`'s bound re-armed with what it had left. A bound with nothing left has fired, and
    pytest-timeout's timer fires once: a timer re-armed with nothing left would fire at once, inside pytest's own machinery
    rather than in any work it bounds, and report a second timeout for the one the test already has."""
    if paused.left > 0:
        config.hook.pytest_timeout_set_timer(item=paused.item, settings=paused.settings._replace(timeout=paused.left))
    else:
        config.stash[BOUND] = None


# The name a rescoping is recorded under on a test's report.
RESCOPED = "timeouts.shared"


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest) -> Generator[None, object, object]:
    """A fixture wider than its test is set up and torn down as shared work, whatever order the test's fixtures are set up
    in (pytest-asyncio's runner, a session-scoped fixture, comes after the function-scoped ones) and whichever test it is
    torn down in: the last to use it."""
    # A function-scoped fixture is its test's own work.
    if fixturedef.scope == "function":
        return (yield)
    config, fixture = request.config, fixturedef.argname
    # Finalizers run last-added first, so these two bracket the teardown the fixture's own setup registers. The second
    # is added before the test's timer is re-armed, which a bound with next to nothing left fires at once.
    teardown = ExitStack()
    fixturedef.addfinalizer(teardown.close)
    with shared(config, "setup", fixture):
        value = yield
        fixturedef.addfinalizer(lambda: teardown.enter_context(shared(config, "teardown", fixture)))
    return value


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    """pytest-timeout arms the test's bound as its protocol starts, before pytest logs the test's start: inside
    pytest-timeout's wrapper, it waits paused for the setup phase."""
    armed = item.config.stash[BOUND]
    if isinstance(armed, Armed):
        item.config.stash[BOUND] = pause(item.config, armed)
    return (yield)


def phase(item: pytest.Item) -> Generator[None, None, None]:
    """[LAW:single-enforcer] the test's bound runs only inside its phases. pytest-timeout arms one timer for the whole
    runtest protocol, so pytest's reporting of each phase runs under it, and a timer that fires there ends the run in
    INTERNALERROR. So the bound pauses as each phase ends, still inside pytest's call of it, where a timer that fires is
    that phase's error, and resumes with what it had left as the next phase starts: the reporting between counts against
    no bound."""
    config = item.config
    paused = config.stash[BOUND]
    if isinstance(paused, Paused):
        resume(config, paused)
    try:
        return (yield)
    finally:
        armed = config.stash[BOUND]
        # Under `timeout_func_only` pytest-timeout's own call hook has already ended the bound.
        if isinstance(armed, Armed):
            config.stash[BOUND] = pause(config, armed)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:
    return (yield from phase(item))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    return (yield from phase(item))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, None, None]:
    return (yield from phase(item))
