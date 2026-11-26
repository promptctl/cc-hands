"""Which terminal sessions Hands discovers from process information."""

from pathlib import Path

import pytest

from fakeprocesses import Running, processes
from hands.sessions.processes import ControllingTerminal, terminal_processes


def test_a_relative_executable_still_identifies_a_session_after_it_changes_directory(tmp_path: Path) -> None:
    install = tmp_path / "install"
    worktree = tmp_path / "worktree"
    invocation = Path("bin") / "claude"
    table = processes(Running(42, terminal=ControllingTerminal(7, foreground=True), path=invocation, started_in=install, cwd=worktree))

    [session] = terminal_processes(table).found

    assert (session.executable, session.cwd) == (install / invocation, worktree)


def test_a_process_that_exits_during_discovery_is_omitted_without_hiding_its_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = ControllingTerminal(7, foreground=True)
    table = processes(Running(42, terminal=terminal), Running(43, terminal=terminal))
    snapshot = table.table()
    table.end(42)
    monkeypatch.setattr(table, "table", snapshot.copy)

    result = terminal_processes(table)

    assert [session.pid for session in result.found] == [43]
    assert result.unread == []
