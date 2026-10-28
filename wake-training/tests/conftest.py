"""The environment every test runs in, and what it is made from.

[LAW:single-enforcer] conftest is the one owner of the environment the suite runs in: a test that needs a name sets it
on top, and builds no environment of its own to keep the developer's out. It imports nothing of wake_training, so it
is loaded, and the run's environment made, before any test module imports what reads it."""

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pytest

PASSED = ("TMPDIR",)


def hermetic(machine: Mapping[str, str], home: Path) -> dict[str, str]:
    """The environment the run and each test run in, from the `machine`'s: its TMPDIR, which tempfile and pytest's own
    temporary directories are made under, the system's tools, which are all the suite's children need (curl and tar;
    the trainer is run as sys.executable), one UTF-8 locale, and a `home` of its own. Every other name the machine has,
    a credential, a proxy, a Claude Code session's or a HOME with a ~/.curlrc in it, is absent."""
    return {name: machine[name] for name in PASSED if name in machine} | {
        "PATH": "/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "HOME": str(home),
    }


def entered(monkeypatch: pytest.MonkeyPatch, names: Mapping[str, str]) -> None:
    """`names`, and nothing else but pytest's own, as the process's environment until `monkeypatch` is undone."""
    for name in [name for name in os.environ if not name.startswith("PYTEST_")]:
        monkeypatch.delenv(name)
    for name, value in names.items():
        monkeypatch.setenv(name, value)


_run = pytest.StashKey[tuple[pytest.MonkeyPatch, Path, dict[str, str]]]()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """The run's environment from before the first test module is imported, since importing onnxruntime, which the
    suite's modules do, writes under HOME. [LAW:no-ambient-temporal-coupling] made here, not in a fixture, so no import
    runs before it; and last of the run's configuring, so pytest's own reporter, which is the developer's and not the
    tests', has read their terminal's names (PY_COLORS, NO_COLOR, TERM) before they go."""
    machine = dict(os.environ)
    home = Path(tempfile.mkdtemp(prefix="wake-training-home-", dir="/tmp"))
    monkeypatch = pytest.MonkeyPatch()
    entered(monkeypatch, hermetic(machine, home))
    config.stash[_run] = (monkeypatch, home, machine)


def pytest_unconfigure(config: pytest.Config) -> None:
    monkeypatch, home, _ = config.stash[_run]
    monkeypatch.undo()
    shutil.rmtree(home)


@pytest.fixture(scope="session")
def machine(pytestconfig: pytest.Config) -> dict[str, str]:
    """The machine's environment as the run began."""
    return pytestconfig.stash[_run][2]


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, machine: dict[str, str]) -> dict[str, str]:
    """Each test runs in the hermetic environment afresh, whatever a test before it left, with a HOME of its own that
    holds nothing, as the run's own does, so a default under ~ lands there and a curl finds no ~/.curlrc."""
    names = hermetic(machine, tmp_path_factory.mktemp("home"))
    entered(monkeypatch, names)
    return names
