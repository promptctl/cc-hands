"""pytest-timeout's bound, scoped to the work it bounds: a fixture wider than one function is set up inside whichever test
first asks for it, yet it is the work of every test that shares it, so its setup runs under the suite's bound and the
test's own bound resumes with what it had left. A plugin of its own, so a run in another process can load it by name
with `-p timeouts`; conftest registers it for the suite."""

import time
from collections.abc import Generator
from dataclasses import dataclass

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


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest) -> Generator[None, object, object]:
    """A shared fixture's setup under the suite's bound, the test's timer stopped meanwhile and re-armed with the time it
    had left, whatever order the test's fixtures are set up in: pytest-asyncio's runner, a session-scoped fixture, comes
    after the function-scoped ones. Nested setups restore innermost first."""
    config = request.config
    # The timer armed on entry: the test's, or an outer shared setup's.
    armed = config.stash[ARMED]
    # A function-scoped fixture is its test's own work; with no timer armed, as under `timeout_func_only`, nothing bounds
    # setup to scope.
    if fixturedef.scope == "function" or armed is None:
        return (yield)
    left = armed.deadline - time.monotonic()
    suite = config.stash[SUITE]
    config.hook.pytest_timeout_cancel_timer(item=armed.item)
    # A suite with no bound gives its shared fixtures none.
    if suite.timeout:  # pyright: ignore[reportUnknownMemberType]  (untyped in pytest-timeout)
        config.hook.pytest_timeout_set_timer(item=armed.item, settings=suite)
    try:
        return (yield)
    finally:
        config.hook.pytest_timeout_cancel_timer(item=armed.item)
        # A bound already spent fires at once.
        config.hook.pytest_timeout_set_timer(item=armed.item, settings=armed.settings._replace(timeout=max(left, SPENT)))


# The shortest timer either method arms: a signal's interval timer takes no zero, which disarms it.
SPENT = 1e-6
