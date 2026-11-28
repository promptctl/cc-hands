"""Terminal process information used to diagnose a missing session registration."""

from pathlib import Path

import pytest

from fakeprocesses import Running, processes
from hands.sessions.processes import ControllingTerminal, terminal_processes


def test_a_relative_invocation_uses_the_executable_reported_by_the_process_host(tmp_path: Path) -> None:
    executable = tmp_path / "versions" / "1.2.3"
    worktree = tmp_path / "worktree"
    invocation = Path("bin") / "claude"
    table = processes(Running(42, terminal=ControllingTerminal(7, foreground=True), path=executable, invoked_as=invocation, cwd=worktree))

    [process] = terminal_processes(table).found

    assert (process.executable, process.cwd) == (executable, worktree)


def test_a_process_that_exits_during_discovery_is_omitted_without_hiding_its_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = ControllingTerminal(7, foreground=True)
    table = processes(Running(42, terminal=terminal), Running(43, terminal=terminal))
    snapshot = table.table()
    table.end(42)
    monkeypatch.setattr(table, "table", snapshot.copy)

    result = terminal_processes(table)

    assert [process.pid for process in result.found] == [43]
    assert result.unread == []
