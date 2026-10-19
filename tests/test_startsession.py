"""The `start_session` tool: `claude` started in a tmux window of the tmux session named for its folder, as from a terminal
outside any session, and the session that joined hands named."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from hands.core.session import Membership, SessionId
from hands.sessions.audit import AuditLog, segment
from hands.sessions.home import Home
from hands.sessions.membership import parse_membership
from hands.sessions.startsession import SESSION_GIVEN, as_from_a_terminal, descends
from hands.sessions.terminals import Process
from hands.voice.tool import Result
from hands.voice.tools import start_session_tool


def home_in(tmp_path: Path) -> Home:
    home = Home(tmp_path / "home")
    home.memberships.mkdir(parents=True)
    home.bin.mkdir(parents=True)
    return home


def members(home: Home) -> list[Membership]:
    """The home's sessions as the registry holds them once each has written its membership."""
    return [parse_membership(SessionId(path.stem), path.read_bytes()) for path in sorted(home.memberships.glob("*.json"))]


def start(home: Home, folder: Path, model: str = "") -> dict[str, Any]:
    """The tool called as the model calls it, in the environment hands was started in."""
    start_session = start_session_tool(home, AuditLog(home.audit, clock=datetime.now).record, os.environ, lambda: members(home))

    async def called() -> Result:
        return await start_session.body(folder=str(folder), model=model)

    return dict(asyncio.run(called()))


def started(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line.get("event") == "session.start"]


def test_a_process_descends_from_itself_and_what_it_started_never_from_a_sibling() -> None:
    table = {pid: Process(pid, parent, 501, None, None) for pid, parent in {10: 1, 11: 10, 12: 11, 20: 1, 1: 0}.items()}
    assert [descends(pid, 10, table) for pid in (10, 11, 12, 20, 1, 99)] == [True, True, True, False, False, False]


def test_a_folder_that_is_not_there_is_a_symlink_loop_or_is_not_a_whole_path_is_said(tmp_path: Path) -> None:
    home = home_in(tmp_path)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    for folder in (tmp_path / "nowhere", loop):
        assert f"there is no folder {folder}" in start(home, folder)["error"]
    # Never the daemon's own working directory.
    for folder in (Path("billing"), Path("")):
        assert f"{folder} is no absolute folder, nor one from ~" in start(home, folder)["error"]
    assert [event["outcome"] for event in started(home)] == ["failed"] * 4


def test_a_terminal_outside_any_session_has_nothing_of_one_and_keeps_the_users_own_setup() -> None:
    inside = {
        **{name: "given" for name in SESSION_GIVEN},
        # fritter's tap of the session hands was started in, and the proxy it replaced.
        "FRITTER_TAP": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "FRITTER_OUTER_HTTPS_PROXY": "http://proxy.lan:3128",
    }
    users = {"PATH": "/Users/me/.hands/bin:/usr/bin", "CLAUDE_CONFIG_DIR": "/Users/me/.claude.work", "ANTHROPIC_API_KEY": "sk-mine", "HANDS_HOME": "/elsewhere"}
    assert as_from_a_terminal({**inside, **users}) == {**users, "HTTPS_PROXY": "http://proxy.lan:3128"}
