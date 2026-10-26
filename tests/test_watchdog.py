"""The run's watchdog: a run whose heartbeat stops ends itself and says where every thread was; one that beats lives."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from conftest import PATIENCE_SECS
from hands.sessions import audit, heartbeat
from hands.sessions.home import Home

WATCHED = Path(__file__).resolve().parent / "fixtures" / "watched_run.py"
# Far past the fixture's deadline, and short of the suite's own timeout.
BOUND_SECONDS = 30


def watched(home: Home, mode: str) -> subprocess.CompletedProcess[str]:
    # [LAW:no-ambient-temporal-coupling] the fixture's deadline is the suite's stall, so a loaded machine that is slow but
    # moving never reads as a hung run, and only a stall the suite would fail on anyway ends one.
    return subprocess.run([sys.executable, str(WATCHED), str(home.root), str(PATIENCE_SECS), mode], capture_output=True, text=True, timeout=BOUND_SECONDS)


def errors(home: Home) -> list[str]:
    return [line["message"] for line in map(json.loads, audit.tail(home.audit, 100)[0]) if (line["type"], line["level"]) == ("Logged", "error")]


def test_each_hung_run_ends_itself_and_the_next_says_where_its_threads_were_before_it_can_hang_in_turn(tmp_path: Path) -> None:
    # As the daemon hung: no Python thread, and no Python signal handler, runs while the regex holds the GIL.
    home = Home(tmp_path)
    ended = watched(home, "hangs")
    assert ended.returncode == 1, ended.stderr
    hung = home.hang.read_text()
    assert hung.startswith(f"Timeout ({timedelta(seconds=PATIENCE_SECS)})!")
    assert "in stuck_in_a_regex" in hung
    # The next run says so as it begins, then hangs at its door, before any beat, and is ended all the same.
    stalled = watched(home, "stalls")
    assert stalled.returncode == 1, stalled.stderr
    assert hung in stalled.stderr
    said = home.hang.read_text()
    assert "in stalled_at_the_door" in said
    # A run that is refused at its door says the second, empties the file for its own run, and leaves no timer behind.
    assert watched(home, "returns").returncode == 11
    assert [hung in error for error in errors(home)] == [True, False]
    assert [said in error for error in errors(home)] == [False, True]
    assert home.hang.read_text() == ""


def test_a_run_that_keeps_beating_is_never_ended_by_its_watchdog(tmp_path: Path) -> None:
    home = Home(tmp_path)
    ended = watched(home, "beats")
    assert (ended.returncode, home.hang.read_text(), errors(home)) == (0, "", []), ended.stderr


def test_every_beat_kicks_the_watchdog(tmp_path: Path) -> None:
    kicks: list[None] = []
    heart = heartbeat.Heart(tmp_path / "status.json", 4242, datetime.now(UTC), timedelta(seconds=2), lambda: kicks.append(None))
    heart.beat("starting", None, 0, listening=False, degraded=())
    heart.beat(heartbeat.Refusal("no key"), None, 0, listening=False, degraded=())
    assert len(kicks) == 2
