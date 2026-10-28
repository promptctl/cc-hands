"""Every test runs in the hermetic environment conftest makes: nothing of the developer's but their tools, and a HOME and
a tmux socket directory of its own."""

import os
import shutil
from pathlib import Path

import pytest

from conftest import executable, hermetic, without_claude


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
        "HF_HUB_CACHE": "/Users/developer/.cache/huggingface/hub",
        "GIT_CONFIG_NOSYSTEM": "1",
        "LANG": "en_US.UTF-8",
        "HOME": "/tmp/home",
        "TMUX_TMPDIR": "/tmp/sockets",
    }


def test_a_test_runs_in_the_hermetic_environment(environment: dict[str, str]) -> None:
    assert {name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")} == environment
    assert Path.home() == Path(environment["HOME"])
    assert shutil.which("claude") is None


@pytest.mark.parametrize("test", ["first", "second"])
def test_what_a_test_leaves_in_its_home_or_its_tmux_directory_no_other_test_finds(test: str) -> None:
    for own in (Path.home(), Path(os.environ["TMUX_TMPDIR"])):
        assert list(own.iterdir()) == []
        (own / test).write_text(test)
