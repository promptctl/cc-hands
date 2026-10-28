"""Every test runs in the hermetic environment conftest makes: nothing of the developer's but their tools, and a HOME and
a tmux socket directory of its own."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from conftest import executable, hermetic, without_claude
from hands.voice.engaged import SMART_TURN


def test_a_path_without_claude_finds_the_tools_beside_it(tmp_path: Path) -> None:
    lit = executable(tmp_path / "local" / "lit", "#!/bin/sh\n")
    executable(tmp_path / "local" / "claude", "#!/bin/sh\n")
    other = executable(tmp_path / "other" / "tool", "#!/bin/sh\n").parent
    (links := tmp_path / "links").mkdir()

    path = without_claude(f"{lit.parent}{os.pathsep}{other}", links)

    assert shutil.which("claude", path=path) is None
    assert Path(found := shutil.which("lit", path=path) or "").resolve() == lit.resolve(), found
    # A directory with no Claude Code in it is the same directory.
    assert path.split(os.pathsep)[1] == str(other)


@pytest.mark.parametrize("named", ["claude", "Claude"])
def test_a_path_without_claude_holds_no_claude_however_the_disk_spells_it(tmp_path: Path, named: str) -> None:
    executable(tmp_path / "local" / named, "#!/bin/sh\n")
    (links := tmp_path / "links").mkdir()

    assert shutil.which("claude", path=without_claude(str(tmp_path / "local"), links)) is None


def test_a_path_without_claude_holds_only_absolute_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each of these is the directory the test is in, wherever that is, and here it holds a claude.
    executable(tmp_path / "bin" / "claude", "#!/bin/sh\n")
    monkeypatch.chdir(tmp_path)
    (links := tmp_path / "links").mkdir()

    assert without_claude(os.pathsep.join(["", ".", "bin", "/usr/bin"]), links) == "/usr/bin"


def test_the_hermetic_environment_keeps_nothing_else_of_the_machine_s() -> None:
    machine = {
        "PATH": "/opt/homebrew/bin:/usr/bin",
        "TMPDIR": "/var/folders/t/",
        "CI": "true",
        "LANG": "fr_FR.ISO8859-1",
        "LC_ALL": "fr_FR.ISO8859-1",
        "HF_HUB_OFFLINE": "0",
        "HF_ENDPOINT": "https://hub.example.com",
        "HF_TOKEN": "hf_developer",
        "GIT_AUTHOR_NAME": "Developer",
        "HOME": "/Users/developer",
        "ANTHROPIC_API_KEY": "sk-ant-developer",
        "CLAUDE_CODE_OAUTH_TOKEN": "developer's",
        "CLAUDE_CONFIG_DIR": "/Users/developer/.claude",
        "CLAUDE_CODE_SESSION_ID": "the developer's session",
        "CLAUDECODE": "1",
        "HANDS_HOME": "/Users/developer/.hands",
        "FRITTER_SOCKET": "/tmp/fritter-1/session.sock",
        "HTTPS_PROXY": "http://127.0.0.1:8080",
        "GIT_DIR": "/Users/developer/code/hands/.git",
        "TMUX": "/private/tmp/tmux-501/default,1,0",
    }

    tested = hermetic(machine, "/links/0:/usr/bin", "/Users/developer/.cache/huggingface/hub", Path("/tmp/home"), Path("/tmp/sockets"))

    assert tested == {
        "PATH": "/links/0:/usr/bin",
        "TMPDIR": "/var/folders/t/",
        "CI": "true",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_CACHE": "/Users/developer/.cache/huggingface/hub",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "LANG": "en_US.UTF-8",
        "HOME": "/tmp/home",
        "TMUX_TMPDIR": "/tmp/sockets",
    }


def test_a_test_runs_in_the_hermetic_environment(environment: dict[str, str], machine: dict[str, str]) -> None:
    assert {name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")} == environment
    assert Path.home() == Path(environment["HOME"])
    assert shutil.which("claude") is None
    # Nor does the hub in this process read the developer's token, from under the HOME it found as it was imported.
    assert not Path(huggingface_hub.constants.HF_TOKEN_PATH).is_relative_to(machine["HOME"])
    assert huggingface_hub.get_token() is None


@pytest.fixture(scope="module")
def wider() -> dict[str, str]:
    """The environment a fixture wider than a test is set up in."""
    return {name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")}


def test_a_fixture_wider_than_a_test_runs_in_the_hermetic_environment_too(wider: dict[str, str], environment: dict[str, str], machine: dict[str, str]) -> None:
    # Of the machine's names, what a test sees, each test's own HOME and tmux sockets apart. A library may add names of
    # its own as it is imported, torch's inductor its cache directory among them, which are no developer's.
    def machine_s(names: dict[str, str]) -> dict[str, str]:
        return {name: value for name, value in names.items() if name in machine and name not in ("HOME", "TMUX_TMPDIR")}

    assert machine_s(wider) == machine_s(environment)


@pytest.mark.parametrize("test", ["first", "second"])
def test_what_a_test_leaves_in_its_home_or_its_tmux_directory_no_other_test_finds(test: str) -> None:
    for own in (Path.home(), Path(os.environ["TMUX_TMPDIR"])):
        assert list(own.iterdir()) == []
        (own / test).write_text(test)


def test_a_test_reaches_no_hub_for_a_weight() -> None:
    # The run fetched every weight before the first test, and left the hub in this process offline once it had.
    with pytest.raises(OfflineModeIsEnabled):
        huggingface_hub.HfApi().model_info(SMART_TURN)


def test_a_child_reaches_no_hub_for_a_weight_either() -> None:
    asked = subprocess.run([sys.executable, "-c", f"import huggingface_hub; huggingface_hub.HfApi().model_info({SMART_TURN!r})"], capture_output=True, text=True)

    assert asked.returncode != 0
    assert "OfflineModeIsEnabled" in asked.stderr, asked.stderr
