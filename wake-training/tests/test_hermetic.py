"""Every test runs in the hermetic environment conftest makes, and so does the import of every test module: nothing of
the developer's but their temporary directory, and a HOME of its own."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import hermetic

# As pytest imports this module, collecting the suite, with whatever the modules before it imported.
IMPORTED = dict(os.environ)


def test_the_hermetic_environment_keeps_nothing_else_of_the_machine_s() -> None:
    machine = {
        "TMPDIR": "/var/folders/t/",
        "PATH": "/opt/homebrew/bin:/usr/bin",
        "LANG": "fr_FR.ISO8859-1",
        "LC_ALL": "fr_FR.ISO8859-1",
        "CI": "true",
        "HOME": "/Users/developer",
        "HF_TOKEN": "hf_developer",
        "ANTHROPIC_API_KEY": "sk-ant-developer",
        "CLAUDE_CODE_OAUTH_TOKEN": "developer's",
        "CLAUDE_CONFIG_DIR": "/Users/developer/.claude",
        "CLAUDECODE": "1",
        "HTTPS_PROXY": "http://127.0.0.1:8080",
        "CURL_HOME": "/Users/developer",
        "VIRTUAL_ENV": "/Users/developer/code/hands/wake-training/.venv",
    }

    assert hermetic(machine, Path("/tmp/home")) == {
        "TMPDIR": "/var/folders/t/",
        "PATH": "/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "HOME": "/tmp/home",
    }


def test_a_test_runs_in_the_hermetic_environment(environment: dict[str, str]) -> None:
    assert {name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")} == environment
    assert Path.home() == Path(environment["HOME"])


def test_a_test_module_is_imported_in_the_hermetic_environment() -> None:
    home = Path(IMPORTED.get("HOME", ""))
    assert home.parent == Path("/tmp") and home.name.startswith("wake-training-home-")
    assert "ANTHROPIC_API_KEY" not in IMPORTED


def test_a_developer_s_credential_never_reaches_a_module_s_import(environment: dict[str, str]) -> None:
    """The suite run anew by a developer who holds a credential, so the test above has one to find. A library imported
    with the suite may add names of its own, with values a developer may also hold, so a planted name, not the
    machine's every name, is what says nothing of theirs got through."""
    ran = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{__file__}::{test_a_test_module_is_imported_in_the_hermetic_environment.__name__}"],
        env=environment | {"ANTHROPIC_API_KEY": "sk-ant-developer"},
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr


@pytest.mark.parametrize("test", ["first", "second"])
def test_what_a_test_leaves_in_its_home_no_other_test_finds(test: str) -> None:
    assert list(Path.home().iterdir()) == []
    (Path.home() / test).write_text(test)
