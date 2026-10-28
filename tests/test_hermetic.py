"""Every test runs in the hermetic environment conftest makes: nothing of the developer's but their tools, and a HOME of
its own."""

import os
import shutil
from pathlib import Path

import pytest

from conftest import hermetic, without_claude


def program(at: Path) -> Path:
    at.parent.mkdir(parents=True, exist_ok=True)
    at.write_text("#!/bin/sh\n")
    at.chmod(0o755)
    return at


def test_a_path_without_claude_finds_the_tools_beside_it(tmp_path: Path) -> None:
    lit = program(tmp_path / "local" / "lit")
    program(tmp_path / "local" / "claude")
    other = program(tmp_path / "other" / "tool").parent
    (links := tmp_path / "links").mkdir()

    path = without_claude(f"{lit.parent}{os.pathsep}{other}", links)

    assert shutil.which("claude", path=path) is None
    assert Path(found := shutil.which("lit", path=path) or "").resolve() == lit.resolve(), found
    # A directory with no Claude Code in it is the same directory.
    assert path.split(os.pathsep)[1] == str(other)


def test_the_hermetic_environment_keeps_nothing_else_of_the_machine_s() -> None:
    machine = {
        "PATH": "/opt/homebrew/bin:/usr/bin",
        "TMPDIR": "/var/folders/t/",
        "CI": "true",
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

    tested = hermetic(machine, "/links/0:/usr/bin", Path("/tmp/sockets"), "/Users/developer/.cache/huggingface/hub")

    assert tested == {
        "PATH": "/links/0:/usr/bin",
        "TMPDIR": "/var/folders/t/",
        "CI": "true",
        "HF_HUB_CACHE": "/Users/developer/.cache/huggingface/hub",
        "GIT_CONFIG_NOSYSTEM": "1",
        "TMUX_TMPDIR": "/tmp/sockets",
    }


def test_a_test_runs_in_the_hermetic_environment_with_a_home_of_its_own(hermetic_environment: dict[str, str]) -> None:
    assert {name for name in os.environ if not name.startswith("PYTEST_")} == hermetic_environment.keys() | {"HOME"}
    assert {name: os.environ[name] for name in hermetic_environment} == hermetic_environment
    assert Path.home() == Path(os.environ["HOME"])
    assert shutil.which("claude") is None
    assert list(Path(os.environ["TMUX_TMPDIR"]).iterdir()) == []


@pytest.mark.parametrize("test", ["first", "second"])
def test_what_a_test_leaves_under_its_home_no_other_test_finds(test: str) -> None:
    home = Path.home()
    assert list(home.iterdir()) == []
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text(test)
