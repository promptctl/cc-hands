"""Each `hands` command but `run` and `tmux-status` is one hands.command event: the command, its arguments as parsed, its exit code, and
how long it took, with the unit of work it ran in its trace."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
from loguru import logger

from hands.daemon import cli
from hands.daemon.cli import main
from hands.sessions import heartbeat, marketplace
from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.voice import phoneaddress


def events(home: Home) -> list[dict[str, Any]]:
    return [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line["type"] == "WideEvent"]


def test_a_command_that_exits_zero_and_one_that_exits_nonzero_each_leave_one_event(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    # No heartbeat: hands has never run here, which `hands status` says by exiting 1.
    assert main(["--home", str(home.root), "status"]) == 1
    said = capsys.readouterr().out.rstrip()
    assert main(["--home", str(home.root), "plugin"]) == 0
    capsys.readouterr()

    status, render, plugin = events(home)
    # The failure says why, in the words the command said it.
    assert (status["event"], status["outcome"], status["error"]) == ("hands.command", "failed", f"exited 1: {said}")
    # [LAW:nothing-unseen] what the exit code was decided by, beside it.
    assert status["facts"] == {"command": "status", "home": str(home.root), "arguments.home": str(home.root), "verdict": {"type": "NeverRan", "path": str(home.status)}, "exit_code": 1}
    assert (plugin["event"], plugin["outcome"], plugin["error"]) == ("hands.command", "ok", None)
    assert plugin["facts"] == {"command": "plugin", "home": str(home.root), "arguments.home": str(home.root), "exit_code": 0}
    assert plugin["duration_ms"] >= render["duration_ms"] > 0
    # The render is a part of the command's run, in its trace; each command is a trace of its own.
    assert render["event"] == "plugin.render"
    assert (render["trace_id"], render["parent_id"]) == (plugin["trace_id"], plugin["span_id"])
    assert plugin["parent_id"] is None and status["trace_id"] != plugin["trace_id"]


def test_a_status_of_a_daemon_that_is_up_carries_its_heartbeat(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    now = datetime.now(UTC)
    heartbeat.Heart(home.status, os.getpid(), now, heartbeat.HEARTBEAT, kick_watchdog=lambda: None).beat("running", None, 1, listening=True, degraded=())
    assert main(["--home", str(home.root), "status"]) == 0
    capsys.readouterr()
    [command] = events(home)
    assert (command["outcome"], command["facts"]["exit_code"]) == ("ok", 0)
    verdict = command["facts"]["verdict"]
    assert (verdict["type"], verdict["status"]["pid"], verdict["status"]["live_sessions"]) == ("Up", os.getpid(), 1)
    # A duration is written in milliseconds, as the event's own duration_ms is.
    assert verdict["status"]["heartbeat"] == heartbeat.HEARTBEAT.total_seconds() * 1000


def test_a_tmux_status_line_refreshing_writes_nothing_to_the_audit_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    for _ in range(3):
        assert main(["--home", str(home.root), "tmux-status"]) == 0
    capsys.readouterr()
    assert not segment(home.audit, 0).exists()


def test_arguments_are_on_the_event_as_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    assert main(["--home", str(home.root), "recall", "token", "helper", "-n", "3"]) == 0
    capsys.readouterr()
    [command] = [event for event in events(home) if event["event"] == "hands.command"]
    assert command["facts"] == {"command": "recall", "home": str(home.root), "arguments.home": str(home.root), "arguments.words": ["token", "helper"], "arguments.most": 3, "exit_code": 0}


def test_a_phone_with_no_address_a_phone_reaches_fails_saying_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def untailed() -> phoneaddress.Untailed:
        return phoneaddress.Untailed("a stand-in for Tailscale")

    monkeypatch.setattr(phoneaddress, "tailnet_name", untailed)
    monkeypatch.setattr(phoneaddress, "lan_addresses", list[str])
    home = Home(tmp_path / "home")
    assert main(["--home", str(home.root), "phone"]) == 1
    assert capsys.readouterr().err.endswith("hands: this machine has no address a phone can reach.\n")
    [command] = events(home)
    assert (command["event"], command["outcome"], command["error"]) == ("hands.command", "failed", "exited 1: this machine has no address a phone can reach.")


def test_a_command_that_raises_is_a_failed_event_with_no_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path / "home")

    def full_disk(home: Home, interpreter: str) -> marketplace.Rendered:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(marketplace, "render", full_disk)
    with pytest.raises(OSError, match="No space left"):
        main(["--home", str(home.root), "plugin"])
    [_, command] = events(home)
    assert (command["event"], command["outcome"], command["error"]) == ("hands.command", "failed", "OSError: [Errno 28] No space left on device")
    assert command["facts"] == {"command": "plugin", "home": str(home.root), "arguments.home": str(home.root)}


def test_a_check_says_each_finding_it_exited_on(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    code = main(["--home", str(home.root), "check", "-v"])
    # A finding's indented lines continue the one above them: its mark and step, then what is said of it in full.
    printed = [block.split("\n") for block in re.split(r"\n(?! )", capsys.readouterr().out.rstrip("\n"))]
    [command] = events(home)
    assert command["facts"]["exit_code"] == code != 0
    # One finding a printed block, each on the event as the block says it: its step, the fix of a missing one, and the rest in full.
    findings = command["facts"]["findings"]
    assert [finding["step"] for finding in findings] == [head.split(None, 1)[1].split("  → ")[0].rstrip() for head, *_ in printed]
    assert [finding["fix"] for finding in findings if finding["type"] == "Missing"] == [head.split("  → ")[1] for head, *_ in printed if head.startswith("✗")]
    assert [finding["detail"] for finding in findings] == ["\n".join(line.removeprefix("    ") for line in body) for _, *body in printed]


def test_the_home_a_command_ran_on_is_on_its_event_wherever_it_came_from(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path / "home")
    monkeypatch.setenv("HANDS_HOME", str(home.root))
    assert main(["status"]) == 1
    capsys.readouterr()
    [command] = events(home)
    assert (command["facts"]["home"], command["facts"]["arguments.home"]) == (str(home.root), None)


def test_a_restart_with_nothing_running_says_what_the_heartbeat_said(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    assert main(["--home", str(home.root), "restart"]) == 1
    capsys.readouterr()
    [command] = events(home)
    assert (command["outcome"], command["facts"]["outcome"]) == ("failed", {"type": "NotRunning", "verdict": {"type": "NeverRan", "path": str(home.status)}})


def test_a_command_on_a_home_whose_log_cannot_be_made_still_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    home.root.mkdir(mode=0o500)
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        # [LAW:nothing-unseen] a telemetry failure is telemetry: warned of, and the command answers as it would.
        assert main(["--home", str(home.root), "status"]) == 1
    finally:
        logger.remove(sink)
        home.root.chmod(0o700)
    assert capsys.readouterr().out.startswith("hands has not run")
    assert [("cannot be made" in warning, "failed at a WideEvent line" in warning) for warning in warnings] == [(True, False), (False, True)]
    assert not home.audit.exists()


def sinks() -> set[int]:
    """The ids of loguru's sinks now: it has no public way to list them."""
    handlers = cast(dict[int, object], getattr(getattr(logger, "_core"), "handlers"))
    return set(handlers)


def warned(home: Home) -> Literal[0]:
    """`hands status`, standing in as a command that warns."""
    logger.warning("warned inside the command")
    return 0


def test_a_command_logs_to_every_sink_there_was_before_it_and_leaves_them_as_they_were(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path / "home")
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    before = sinks()
    try:
        with patch("hands.daemon.cli.report", warned):
            assert main(["--home", str(home.root), "status"]) == 0
        after = sinks()
    finally:
        logger.remove(sink)
    capsys.readouterr()
    # [LAW:no-ambient-temporal-coupling] the command lets go of what it added, and of nothing it found.
    assert after == before and warnings == ["warned inside the command"]
    # The command's own audit sink kept it while it ran.
    logged = [line for line in map(json.loads, segment(home.audit, 0).read_text().splitlines()) if line["type"] == "Logged"]
    assert [(line["level"], line["message"]) for line in logged] == [("warning", "warned inside the command")]


def test_what_opening_the_audit_log_warns_of_is_shown_on_the_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def refused(self: Path, mode: int) -> None:
        raise PermissionError(f"not {self}'s owner")

    monkeypatch.setattr(Path, "chmod", refused)
    with cli.logging_to(Home(tmp_path)):
        pass
    assert "cannot be made the user's alone" in capsys.readouterr().err
