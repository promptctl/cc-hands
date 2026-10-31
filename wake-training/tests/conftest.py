"""The environment every test runs in, and what it is made from.

[LAW:single-enforcer] conftest is the one owner of the environment the suite runs in: a test that needs a name sets it
on top, and builds no environment of its own to keep the developer's out. It imports nothing of wake_training, so it
is loaded, and the run's environment made and held offline, before any test module imports what reads it."""

import importlib.util
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest


def _loaded(path: Path) -> ModuleType:
    """The module at `path`, loaded by its path: the repository's tests/ is not on this suite's import path, and its
    conftest would shadow this one if it were."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"{path} is no module Python can load"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# [LAW:one-source-of-truth] the hold hands' run is kept to, off the network and away from the machine's Claude Code
# policy, its profile and the macOS it needs.
offline = _loaded(Path(__file__).parents[2] / "tests" / "offline.py")

# The names a test takes from the machine as they are: macOS's per-user temporary directory, and CI's word that the run
# is in CI, which pytest reads as each report is made, to show a failure whole.
PASSED = ("TMPDIR", "CI")


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


_machine = pytest.StashKey[dict[str, str]]()
# How the run is held, as the hold says.
_held = pytest.StashKey[str]()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """The run's environment from before the first test module is imported, since importing onnxruntime, which the
    suite's modules do, writes under HOME. [LAW:no-ambient-temporal-coupling] made here, not in a fixture, so no import
    runs before it; and last of the run's configuring, so pytest's own reporter, which is the developer's and not the
    tests', has read their terminal's names (PY_COLORS, NO_COLOR, TERM) before they go."""
    machine = dict(os.environ)
    home = Path(tempfile.mkdtemp(prefix="wake-training-home-", dir="/tmp"))
    monkeypatch = pytest.MonkeyPatch()

    def left() -> None:
        monkeypatch.undo()
        shutil.rmtree(home)

    config.add_cleanup(left)
    entered(monkeypatch, hermetic(machine, home))
    config.stash[_machine] = machine
    # The suite fetches nothing while it runs, so it is held from before its first test to its last, and so is every
    # process a test starts.
    offline.sandboxable(platform.mac_ver()[0])
    config.stash[_held] = offline.hold()


def pytest_report_header(config: pytest.Config) -> str:
    return f"offline: nothing past loopback and no Claude Code policy ({config.stash[_held]})"


@pytest.fixture(scope="session")
def machine(pytestconfig: pytest.Config) -> dict[str, str]:
    """The machine's environment as the run began."""
    return pytestconfig.stash[_machine]


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, machine: dict[str, str]) -> dict[str, str]:
    """Each test runs in the hermetic environment afresh, whatever a test before it left, with a HOME of its own that
    holds nothing, as the run's own does, so a default under ~ lands there and a curl finds no ~/.curlrc."""
    names = hermetic(machine, tmp_path_factory.mktemp("home"))
    entered(monkeypatch, names)
    return names
