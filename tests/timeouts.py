"""pytest-timeout's bound, scoped to the work it bounds: a fixture wider than one function is set up inside whichever test
first asks for it and torn down inside the last, yet it is the work of every test that shares it, so its setup and its
teardown run under the suite's bound and the test's own bound resumes with what it had left. A plugin of its own, so a
run in another process can load it by name with `-p timeouts`; conftest registers it for the suite."""

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
# [LAW:no-shared-mutable-globals] the timer armed right now, written only by the two hooks that arm and cancel it.
ARMED = pytest.StashKey[Armed | None]()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    config.stash[SUITE] = get_env_settings(config)
    config.stash[ARMED] = None


@pytest.hookimpl(wrapper=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: Settings) -> Generator[None, object, object]:
    """[LAW:single-enforcer] pytest-timeout arms every timer, the ones this plugin asks for among them; this only reads
    off what it armed."""
    item.config.stash[ARMED] = Armed(item, settings, time.monotonic() + settings.timeout)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  (untyped in pytest-timeout)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_timeout_cancel_timer(item: pytest.Item) -> Generator[None, object, object]:
    item.config.stash[ARMED] = None
    return (yield)


@contextmanager
def shared(config: pytest.Config, phase: str, fixture: str) -> Generator[None]:
    """Work every test sharing `fixture` owns, under the suite's bound: the armed timer stopped meanwhile and re-armed
    with the time it had left, nested work restoring innermost first. What it did lands on the test's report, and so in
    junitxml."""
    # The timer armed on entry: a test's, or an outer shared setup's.
    armed = config.stash[ARMED]
    # With no timer armed, as under `timeout_func_only`, nothing bounds the work to rescope.
    if armed is None:
        yield
        return
    started = time.monotonic()
    left = armed.deadline - started
    suite = config.stash[SUITE]
    bound = cast(float | None, suite.timeout)  # pyright: ignore[reportUnknownMemberType]  (untyped in pytest-timeout)
    config.hook.pytest_timeout_cancel_timer(item=armed.item)
    # A suite with no bound gives its shared fixtures none.
    if bound:
        config.hook.pytest_timeout_set_timer(item=armed.item, settings=suite)
    try:
        yield
    finally:
        config.hook.pytest_timeout_cancel_timer(item=armed.item)
        # [LAW:nothing-unseen] the bound the work ran under, how long it took, and what the test had left: a timeout
        # reported after it names that remainder, not the test's configured bound. Recorded before the re-arm, which a
        # spent bound fires at once.
        armed.item.user_properties.append(
            (RESCOPED, {"phase": phase, "fixture": fixture, "bound": bound, "seconds": round(time.monotonic() - started, 3), "left": round(left, 3)})
        )
        config.hook.pytest_timeout_set_timer(item=armed.item, settings=armed.settings._replace(timeout=max(left, SPENT)))


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
    # is added before the test's timer is re-armed, which a spent bound fires at once.
    teardown = ExitStack()
    fixturedef.addfinalizer(teardown.close)
    with shared(config, "setup", fixture):
        value = yield
        fixturedef.addfinalizer(lambda: teardown.enter_context(shared(config, "teardown", fixture)))
    return value


# The shortest timer either method arms: a signal's interval timer takes no zero, which disarms it.
SPENT = 1e-6
