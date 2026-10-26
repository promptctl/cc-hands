"""The run's watchdog: a run whose heartbeat stops ends itself and says where every thread was; one that beats lives."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hands.daemon import watchdog
from hands.sessions import audit, heartbeat
from hands.sessions.home import Home

WATCHED = Path(__file__).resolve().parent / "fixtures" / "watched_run.py"
# Far past the fixture's half-second deadline, and short of the suite's own timeout.
BOUND_SECONDS = 30


def watched(home: Home, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(WATCHED), str(home.root), mode], capture_output=True, text=True, timeout=BOUND_SECONDS)


def test_a_run_stuck_behind_a_thread_that_holds_the_gil_ends_itself_and_the_next_start_says_where_that_thread_was(tmp_path: Path) -> None:
    # As the daemon hung: no Python thread, and no Python signal handler, runs while the regex holds the GIL.
    home = Home(tmp_path)
    ended = watched(home, "hangs")
    assert ended.returncode == 1, ended.stderr
    said = home.hang.read_text()
    assert said.startswith("Timeout (0:00:00.500000)!")
    assert "in stuck_in_a_regex" in said
    # The next start, refused at its door here, carries the tracebacks to the log, and empties the file for its own run.
    # The hung run, ended as it started, wrote no start of its own.
    assert watched(home, "returns").returncode == 11
    starts = [line["facts"] for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line["event"] == "hands.start"]
    assert [facts["hung_before"] for facts in starts] == [said]
    assert home.hang.read_text() == ""


def test_a_run_that_has_returned_leaves_no_watchdog_to_end_its_process(tmp_path: Path) -> None:
    ended = watched(Home(tmp_path), "returns")
    assert ended.returncode == 11, ended.stderr


def test_a_run_that_keeps_beating_is_never_ended_by_its_watchdog(tmp_path: Path) -> None:
    home = Home(tmp_path)
    ended = watched(home, "beats")
    assert (ended.returncode, home.hang.read_text()) == (0, ""), ended.stderr


def test_every_beat_kicks_the_watchdog(tmp_path: Path) -> None:
    kicks: list[None] = []
    heart = heartbeat.Heart(tmp_path / "status.json", 4242, datetime.now(UTC), timedelta(seconds=2), lambda: kicks.append(None))
    heart.beat("starting", None, 0, listening=False, degraded=())
    heart.beat(heartbeat.Refusal("no key"), None, 0, listening=False, degraded=())
    assert len(kicks) == 2


def test_what_the_run_before_dumped_is_read_as_its_tracebacks_as_empty_or_as_never_kept(tmp_path: Path) -> None:
    dump = tmp_path / "hang.txt"
    assert watchdog.dumped(dump) is None
    dump.write_text("")
    assert watchdog.dumped(dump) == ""
    dump.write_text("Timeout (0:01:00)!\n")
    assert watchdog.dumped(dump) == "Timeout (0:01:00)!\n"
