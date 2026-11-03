"""tests/quarantine_gate.py: a quarantine's age is the history's record of its marker. The rule is checked in process, and the
clock is checked on real repositories built here, where each marker's date is the date of the commit that adds it."""

import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import quarantine_gate as gate

TODAY = date(2026, 11, 2)
TODAY_NOON = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
THE_ADD = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)  # eight days before TODAY
WITHIN_THE_WEEK = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)  # six days before TODAY

MARKED = '''import pytest

@pytest.mark.quarantined("t-old")
def test_flaky():
    pass
'''


def test_a_quarantine_past_its_week_is_overdue() -> None:
    assert gate.overdue({"t-old": THE_ADD}, TODAY) == ["t-old: quarantined for 8 days, past its 7"]


def test_a_quarantine_within_its_week_is_not_overdue() -> None:
    assert gate.overdue({"t-new": WITHIN_THE_WEEK}, TODAY) == []


def git(root: Path, *args: str, author: datetime | None = None, committer: datetime | None = None) -> None:
    # The environment's GIT_ variables are the caller's; these commits take only the identity and dates the test sets.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env |= {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if author is not None:
        env["GIT_AUTHOR_DATE"] = author.isoformat()
    if committer is not None:
        env["GIT_COMMITTER_DATE"] = committer.isoformat()
    subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)


def commit(root: Path, message: str, author: datetime, committer: datetime | None = None, amend: bool = False) -> None:
    git(root, "add", "-A")
    args = ["-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", message]
    if amend:
        args.append("--amend")
    git(root, *args, author=author, committer=committer or author)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_flaky.py").write_text(MARKED)
    commit(tmp_path, "add the flaky test", author=THE_ADD)
    return tmp_path


def test_a_marker_eight_days_old_fails_the_gate(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate.main(repo, TODAY) == 1
    assert "t-old: quarantined for 8 days" in capsys.readouterr().out


def test_a_marker_six_days_old_passes_the_gate(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_flaky.py").write_text(MARKED)
    commit(tmp_path, "add the flaky test", author=WITHIN_THE_WEEK)
    assert gate.main(tmp_path, TODAY) == 0


def test_a_move_does_not_reset_the_clock(repo: Path) -> None:
    git(repo, "mv", "tests/test_flaky.py", "tests/test_moved.py")
    commit(repo, "move the flaky test", author=TODAY_NOON)
    assert gate.main(repo, TODAY) == 1


def test_an_edit_of_the_marker_line_does_not_reset_the_clock(repo: Path) -> None:
    (repo / "tests" / "test_flaky.py").write_text(MARKED.replace('@pytest.mark.quarantined("t-old")', '@pytest.mark.quarantined( "t-old" )'))
    commit(repo, "reformat the marker", author=TODAY_NOON)
    assert gate.main(repo, TODAY) == 1


def test_two_tests_under_one_ticket_fail_the_gate_loudly(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (repo / "tests" / "test_second.py").write_text(MARKED)
    commit(repo, "a second test under the same ticket", author=TODAY_NOON)
    assert gate.main(repo, TODAY) == 1
    assert "quarantines more than one test" in capsys.readouterr().out


def test_both_spellings_of_the_marker_are_read(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_spellings.py").write_text(
        'import pytest\nfrom pytest import mark\n\n@pytest.mark.quarantined("t-a")\ndef test_a():\n    pass\n\n'
        '@mark.quarantined("t-b")\ndef test_b():\n    pass\n'
    )
    assert gate.markers(tmp_path / "tests") == ["t-a", "t-b"]


def test_a_marker_looked_up_with_getattr_is_refused(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_getattr.py").write_text('import pytest\n\ngetattr(pytest.mark, "quarantined")("t-a")\n')
    with pytest.raises(ValueError, match="looked up with getattr"):
        gate.markers(tmp_path / "tests")


def test_a_start_after_today_fails_loudly() -> None:
    future = datetime(2026, 12, 20, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="starts 48 days after today"):
        gate.overdue({"t-future": future}, TODAY)


def test_a_marker_without_its_ticket_as_a_string_is_refused(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_bare.py").write_text("import pytest\n\n@pytest.mark.quarantined\ndef test_a():\n    pass\n")
    with pytest.raises(ValueError, match="must be called with its ticket"):
        gate.markers(tmp_path / "tests")


def test_a_recommit_with_a_new_committer_date_does_not_reset_the_clock(repo: Path) -> None:
    commit(repo, "add the flaky test", author=THE_ADD, committer=TODAY_NOON, amend=True)
    assert gate.started(repo, "t-old") == THE_ADD
    assert gate.main(repo, TODAY) == 1


def test_a_marker_not_in_the_history_fails_loudly(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("history, without the marker\n")
    commit(tmp_path, "an unrelated commit", author=THE_ADD)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_flaky.py").write_text(MARKED)
    with pytest.raises(ValueError, match="not committed here"):
        gate.started(tmp_path, "t-old")
