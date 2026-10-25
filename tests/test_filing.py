"""Warnings and errors from the audit log grouped by cause, and what `hands file-tickets` files for each group."""

import asyncio
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args

import pytest

from hands.daemon import cli
from hands.sessions import audit, backlog, filing, grouping
from hands.sessions.audit import AuditLog, Entry, Logged, Transcribed
from hands.sessions.backlog import parse_export

LIT = shutil.which("lit")
real_lit = pytest.mark.skipif(LIT is None, reason="the tickets are lit's, so it is lit that is driven")

START = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
# Where a ticket says the audit log is.
LOGS = filing.where(Path("/Users/someone/.hands/audit"), Path("/Users/someone"))


class Clock:
    """A wall clock that moves a minute each time it is read, so each line is written a minute after the one before."""

    def __init__(self, at: datetime = START) -> None:
        self.at = at

    def __call__(self) -> datetime:
        self.at += timedelta(minutes=1)
        return self.at


def broke(session: str, path: str, chars: int) -> Logged:
    """One cause, said with what differs between its sightings: a session id, a path, a count, and a quoted excerpt."""
    return Logged(
        level="error", source="hands.sessions.tap:read",
        message=f"hands could not read session {session}'s copy at {path}: not JSON at char {chars} in b'{{\"kind\":\"req",
        where="/x/tap.py:159", trace=(),
    )


def write_audit(directory: Path, clock: Clock) -> AuditLog:
    """Two groups and a repeat: the copy error three times over, a warning once, and an info line beside them."""
    log = AuditLog(directory, clock=clock)
    log.record(broke("ee909f6e-b7ac-4256-9af5-a55e54d100d3", "/Users/someone/.claude/projects/a.jsonl", 404))
    log.record(Transcribed("send it"))
    log.record(Logged(level="warning", source="hands.sessions.delta:_git", message="git add in /code/x did not answer in 0.1s", where="/x/delta.py:515", trace=()))
    log.record(broke("652a74a6-0a46-4efc-b66e-ad035c7f6a07", "/Users/someone/.claude/projects/b.jsonl", 47))
    log.record(broke("0d8d2611-26bc-4663-94ed-87b5c47ee882", "/tmp/c.jsonl", 3))
    return log


def test_sightings_of_one_cause_are_one_group_whatever_ids_paths_numbers_and_quoted_text_they_carry(tmp_path: Path) -> None:
    write_audit(tmp_path / "audit", Clock())
    read = grouping.read(audit.forwards(tmp_path / "audit"))
    assert [(group.origin, group.cause, len(group.sightings)) for group in read.groups] == [
        ("hands.sessions.tap:read", "hands could not read session <id>'s copy at <path>: not JSON at char <n> in <text>", 3),
        ("hands.sessions.delta:_git", "git add in <path> did not answer in <n>s", 1),
    ]
    copy, git = read.groups
    assert (copy.first.at, copy.last.at) == (START + timedelta(minutes=1), START + timedelta(minutes=5))
    assert copy.label == f"audit-{copy.fingerprint}" and len(copy.fingerprint) == 12 and copy.fingerprint != git.fingerprint
    # The info line is read, counted, and part of no group.
    assert (read.lines, read.unreadable, read.unparsed) == (5, 0, 0)
    assert (read.oldest, read.newest) == (START + timedelta(minutes=1), START + timedelta(minutes=5))


def test_a_line_that_is_no_json_object_is_counted_unreadable_and_one_of_a_kind_hands_cannot_read_is_still_grouped(tmp_path: Path) -> None:
    lines = [
        '{"at": "2026-10-24T12:01:00.000+00:00", "level": "error", "type": "Retired", "said": "x"}',
        '{"at": "2026-10-24T12:02:00.000+00:00", "level": "error", "type": "Exchanged", "method": "GET", "path": "/v1/x", "reply": {"type": "Bounced"}}',
        '{"at": "2026-10-24T12:03:00.000+00:00", "level": "error", "type": "WideEvent", "event": "name.judged"',
        '{"at": "2026-10-24T12:04:00.000+00:00", "level": "error", "type": "Failure", "source": "hands.x:f", "message": "broke", "where": "/x.py:1", "trace": []}',
    ]
    read = grouping.read(lines)
    assert [(group.origin, group.cause) for group in read.groups] == [
        ("Retired", "a Retired line, a kind this hands does not write"),
        ("Exchanged", "a Exchanged line this hands cannot read: a reply of a kind this hands does not write: <text>"),
        # Written before Logged was its name, read as it.
        ("hands.x:f", "broke"),
    ]
    assert (read.lines, read.unreadable, read.unparsed) == (4, 1, 2)


# The kinds of line the log never judges a warning or an error.
ALWAYS_INFO = {"Typing", "CopiesLost", "Transcribed", "HoldHeard", "Replied", "CutOff", "UserTurn", "Announced", "Cued", "Rolled"}


def test_every_kind_of_line_the_log_can_judge_a_warning_or_an_error_says_its_cause() -> None:
    kinds = {kind.__name__ for kind in get_args(Entry)}
    assert ALWAYS_INFO <= kinds
    assert kinds - ALWAYS_INFO <= set(grouping.CAUSES)


def issue(id: str, rank: str, status: str | None, description: str = "", **extra: object) -> dict[str, object]:
    made: dict[str, object] = {"id": id, "title": id, "description": description, "issue_type": "bug" if status else "epic", "topic": "warnings", "rank": rank, **extra}
    closed = {"closed_at": "2026-10-24T12:00:00Z"} if status == "closed" and "closed_at" not in extra else {}
    return made if status is None else {**made, "status": status, **closed}


def export(issues: list[dict[str, object]], labels: dict[str, str] = {}, comments: list[tuple[str, str]] = []) -> bytes:
    return json.dumps({
        "version": 5, "workspace_id": "w", "issues": issues, "relations": [],
        "comments": [{"id": f"c{at}", "issue_id": on, "body": body, "created_by": "hands", "created_at": f"2026-10-24T13:{at:02d}:00Z"} for at, (on, body) in enumerate(comments)],
        "labels": [{"issue_id": on, "name": name, "created_at": "", "created_by": ""} for on, name in labels.items()],
        "events": [], "purges": [], "plan_reviews": [],
    }).encode()


def groups_of(tmp_path: Path) -> tuple[grouping.Group, ...]:
    return grouping.read(audit.forwards(tmp_path / "audit")).groups


def test_each_group_no_ticket_has_is_a_new_ticket_and_the_same_lines_against_those_tickets_file_nothing(tmp_path: Path) -> None:
    clock = Clock()
    write_audit(tmp_path / "audit", clock)
    groups = groups_of(tmp_path)
    first = filing.filings(groups, parse_export(export([issue("e", "0", None)])))
    assert [type(each).__name__ for each in first] == ["New", "New"]
    # The tickets those make, as lit would hold them.
    filed = parse_export(export(
        [issue("e", "0", None), *(issue(f"t{at}", f"{at + 1}", "open", filing.description(each.group, LOGS)) for at, each in enumerate(first))],
        labels={f"t{at}": each.group.label for at, each in enumerate(first)},
    ))
    assert filing.filings(groups, filed) == ()
    description = filing.description(first[0].group, LOGS)
    assert "hands logged this 3 times, from 2026-10-24T12:01:00.000+00:00 to 2026-10-24T12:05:00.000+00:00" in description
    assert description.endswith(f"Last seen: 2026-10-24T12:05:00.000+00:00 ({first[0].group.label})")


def test_a_ticket_carries_none_of_the_line_and_the_command_it_gives_reads_the_lines_where_they_are(tmp_path: Path) -> None:
    clock = Clock()
    write_audit(tmp_path / "audit", clock)
    [copy, _] = groups_of(tmp_path)
    description = filing.description(copy, filing.where(tmp_path / "audit", Path("/Users/someone")))
    # lit's tickets are pushed with the repository: no id, path, or excerpt a line said leaves the machine.
    assert not any(said in description for said in ("ee909f6e", "/Users/someone", "0d8d2611", '"kind"', "/x/tap.py"))
    [command] = [line for line in description.splitlines() if line.startswith("grep ")]
    found = subprocess.run(command, shell=True, check=True, capture_output=True, text=True).stdout
    assert [json.loads(line)["message"].split(" ")[5] for line in found.splitlines()] == ["ee909f6e-b7ac-4256-9af5-a55e54d100d3's", "0d8d2611-26bc-4663-94ed-87b5c47ee882's"]


def test_a_group_seen_again_is_a_comment_on_its_open_ticket_and_a_new_ticket_once_its_ticket_is_closed(tmp_path: Path) -> None:
    clock = Clock()
    log = write_audit(tmp_path / "audit", clock)
    [copy, git] = groups_of(tmp_path)
    told = {"t1": filing.description(copy, LOGS), "t2": filing.description(git, LOGS)}
    log.record(broke("4ce425fb-670d-4936-b5c2-f9b045875d8d", "/tmp/d.jsonl", 9))
    log.record(Logged(level="warning", source="hands.sessions.delta:_git", message="git add in /code/y did not answer in 0.2s", where="/x/delta.py:515", trace=()))
    again = groups_of(tmp_path)
    held = parse_export(export(
        [issue("e", "0", None), issue("t1", "1", "in_progress", told["t1"]), issue("t2", "2", "closed", told["t2"])],
        labels={"t1": copy.label, "t2": git.label},
    ))
    [comment, new] = filing.filings(again, held)
    assert isinstance(comment, filing.Again) and comment.ticket == "t1" and len(comment.group.sightings) == 1
    assert filing.comment(comment.group, LOGS).startswith("Seen again once, from 2026-10-24T12:06:00.000+00:00 to 2026-10-24T12:06:00.000+00:00.")
    # Only what came after the closed ticket's last sighting is the new ticket's.
    assert isinstance(new, filing.New) and [sighting.at for sighting in new.group.sightings] == [START + timedelta(minutes=7)]
    # A comment's Last seen line moves the cursor as a description's does.
    commented = parse_export(export(
        [issue("e", "0", None), issue("t1", "1", "open", told["t1"]), issue("t3", "3", "open", filing.description(new.group, LOGS))],
        labels={"t1": copy.label, "t3": git.label}, comments=[("t1", filing.comment(comment.group, LOGS))],
    ))
    assert filing.filings(again, commented) == ()


@pytest.mark.parametrize("resolution", ["duplicate", "superseded"])
def test_a_group_whose_ticket_was_closed_into_another_is_a_comment_on_that_one(tmp_path: Path, resolution: str) -> None:
    clock = Clock()
    log = write_audit(tmp_path / "audit", clock)
    [copy, _] = groups_of(tmp_path)
    log.record(broke("4ce425fb-670d-4936-b5c2-f9b045875d8d", "/tmp/d.jsonl", 9))
    again = groups_of(tmp_path)
    held = parse_export(export(
        [issue("e", "0", None), issue("t1", "1", "closed", filing.description(copy, LOGS), resolution=resolution, redirect_target="t9"), issue("t9", "2", "open")],
        labels={"t1": copy.label},
    ))
    [comment] = [each for each in filing.filings(again, held) if each.group.label == copy.label]
    assert isinstance(comment, filing.Again) and comment.ticket == "t9" and len(comment.group.sightings) == 1


def test_an_archived_ticket_still_says_how_far_its_group_was_told(tmp_path: Path) -> None:
    clock = Clock()
    write_audit(tmp_path / "audit", clock)
    groups = groups_of(tmp_path)
    archived = parse_export(export(
        [issue("e", "0", None), *(issue(f"t{at}", f"{at + 1}", "closed", filing.description(group, LOGS), archived_at="2026-10-24T14:00:00Z") for at, group in enumerate(groups))],
        labels={f"t{at}": group.label for at, group in enumerate(groups)},
    ))
    assert archived.tickets.keys() == {"e"}
    assert filing.filings(groups, archived) == ()


def test_two_groups_closed_into_one_ticket_keep_their_own_cursors(tmp_path: Path) -> None:
    clock = Clock()
    log = write_audit(tmp_path / "audit", clock)
    [copy, git] = groups_of(tmp_path)
    log.record(broke("4ce425fb-670d-4936-b5c2-f9b045875d8d", "/tmp/d.jsonl", 9))
    later = next(group for group in groups_of(tmp_path) if group.label == copy.label)
    # The git group was last told after the copy group's new sighting; its line on the shared ticket is not the copy group's.
    held = parse_export(export(
        [issue("e", "0", None), issue("t1", "1", "closed", filing.description(copy, LOGS), resolution="duplicate", redirect_target="t9"),
         issue("t2", "2", "closed", "", resolution="duplicate", redirect_target="t9"), issue("t9", "3", "open")],
        labels={"t1": copy.label, "t2": git.label},
        comments=[("t9", f"Last seen: 2026-10-24T13:00:00.000+00:00 ({git.label})")],
    ))
    [comment] = filing.filings([later], held)
    assert isinstance(comment, filing.Again) and len(comment.group.sightings) == 1


def lit(project: Path, *arguments: str) -> str:
    assert LIT is not None
    return subprocess.run([LIT, *arguments], cwd=project, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, str]:
    """A lit project with one epic, and the epic's id."""
    repository = tmp_path / "project"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    lit(repository, "init")
    epic = lit(repository, "new", "--type", "epic", "--topic", "warnings", "--title", "Warnings").split()[0]
    return repository, epic


def bugs(project: Path) -> list[dict[str, object]]:
    exported = json.loads(lit(project, "export"))
    return [issue for issue in exported["issues"] if issue["issue_type"] == "bug"]


@real_lit
def test_file_tickets_files_each_group_once_however_often_it_runs_and_comments_on_one_seen_again(tmp_path: Path, project: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    repository, epic = project
    home = tmp_path / "home"
    clock = Clock()
    log = write_audit(home / "audit", clock)
    monkeypatch.chdir(repository)
    assert cli.main(["--home", str(home), "file-tickets", epic]) == 0
    filed = bugs(repository)
    assert sorted(str(issue["title"]) for issue in filed) == [
        "hands.sessions.delta:_git: git add in <path> did not answer in <n>s",
        "hands.sessions.tap:read: hands could not read session <id>'s copy at <path>: not JSON at char <n> in <text>",
    ]
    assert all(issue["status"] == "open" for issue in filed)
    assert cli.main(["--home", str(home), "file-tickets", epic]) == 0
    assert len(bugs(repository)) == 2
    log.record(broke("4ce425fb-670d-4936-b5c2-f9b045875d8d", "/tmp/d.jsonl", 9))
    assert cli.main(["--home", str(home), "file-tickets", epic]) == 0
    assert len(bugs(repository)) == 2
    exported = json.loads(lit(repository, "export"))
    assert [comment["body"].splitlines()[0] for comment in exported["comments"]] == ["Seen again once, from 2026-10-24T12:06:00.000+00:00 to 2026-10-24T12:06:00.000+00:00."]
    printed = capsys.readouterr().out
    assert "2 groups of warnings and errors: 1 filed or commented, 1 with nothing new" in printed
    # Each pass is one event, counting what it read and what came of it.
    passes = [line for line in map(json.loads, audit.forwards(home / "audit")) if line.get("event") == "tickets.file"]
    assert [line["counts"] for line in passes] == [
        {"lines": 5, "unreadable": 0, "unparsed": 0, "retold": 0, "sightings": 4, "groups": 2, "filed": 2, "commented": 0, "unchanged": 0},
        {"lines": 7, "unreadable": 0, "unparsed": 0, "retold": 0, "sightings": 4, "groups": 2, "filed": 0, "commented": 0, "unchanged": 2},
        {"lines": 10, "unreadable": 0, "unparsed": 0, "retold": 0, "sightings": 5, "groups": 2, "filed": 0, "commented": 1, "unchanged": 1},
    ]
    assert passes[0]["facts"]["oldest"] == "2026-10-24T12:01:00.000+00:00" and passes[0]["facts"]["epic"] == epic


@real_lit
def test_file_tickets_names_an_epic_the_backlog_lists_or_fails_saying_so(tmp_path: Path, project: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    repository, _ = project
    home = tmp_path / "home"
    write_audit(home / "audit", Clock())
    monkeypatch.chdir(repository)
    assert cli.main(["--home", str(home), "file-tickets", "no-such-epic"]) == 1
    assert f"hands file-tickets: the backlog in {repository} lists no epic no-such-epic" in capsys.readouterr().err
    assert bugs(repository) == []
    [event] = [line for line in map(json.loads, audit.forwards(home / "audit")) if line.get("event") == "tickets.file"]
    assert (event["outcome"], event["error"]) == ("failed", f"the backlog in {repository} lists no epic no-such-epic")
    # The command failed because the pass inside it did: one cause, and the next pass files it once.
    found = grouping.read(audit.forwards(home / "audit"))
    assert [group.origin for group in found.groups].count("tickets.file") == 1
    assert "hands.command" not in [group.origin for group in found.groups] and found.retold == 1



def unit(event: str, span: str, parent: str | None, at: int, error: str | None, exit_code: int | None = None) -> str:
    """A unit of work's line in trace t, failed where it says an error, a command's carrying the code it exited with."""
    return json.dumps({
        "at": f"2026-10-24T12:{at:02d}:00.000+00:00", "level": "info" if error is None else "error", "type": "WideEvent",
        "event": event, "trace_id": "t", "span_id": span, "parent_id": parent, "outcome": "ok" if error is None else "failed", "error": error,
        "facts": {} if exit_code is None else {"exit_code": exit_code},
    })


def test_a_failed_unit_is_a_sighting_of_its_own_unless_a_unit_inside_it_failed_first() -> None:
    nested = grouping.read([unit("inner", "c", "b", 1, "broke"), unit("middle", "b", "a", 2, "broke"), unit("hands.command", "a", None, 3, "exited 1: broke", 1)])
    assert [(group.origin, group.cause) for group in nested.groups] == [("inner", "broke")] and nested.retold == 2
    # A unit inside it that went right leaves the failure the outer unit's own.
    alone = grouping.read([unit("inner", "c", "a", 1, None), unit("hands.command", "a", None, 2, "exited 2: undone", 2)])
    assert [group.origin for group in alone.groups] == ["hands.command"] and alone.retold == 0
    # An outer unit that failed of its own, after one inside it failed of another cause, is a sighting of each.
    apart = grouping.read([unit("tool.call", "c", "a", 1, "the tool refused"), unit("brain.turn", "a", None, 2, "TimeoutError: no reply")])
    assert [group.origin for group in apart.groups] == ["tool.call", "brain.turn"] and apart.retold == 0
    # One whose error only ends on the inner one's is a cause of its own.
    longer = grouping.read([unit("lit.run", "c", "a", 1, "timed out"), unit("backlog.read", "a", None, 2, "lit export timed out")])
    assert [group.origin for group in longer.groups] == ["lit.run", "backlog.read"] and longer.retold == 0


def test_a_title_is_one_line_whatever_the_cause_said() -> None:
    group = grouping.Group("hands.x:f", "request failed:\n  <text>\tagain", (grouping.Sighting("hands.x:f", "c", START),))
    assert filing.title(group) == "hands.x:f: request failed: <text> again"


def test_a_last_seen_line_the_pass_did_not_write_is_not_read_as_one(tmp_path: Path) -> None:
    write_audit(tmp_path / "audit", Clock())
    [copy, _] = groups_of(tmp_path)
    for said in (f"Last seen: yesterday ({copy.label})", f"Last seen: 2026-10-24T13:00:00 ({copy.label})"):
        held = parse_export(export([issue("e", "0", None), issue("t1", "1", "open", said)], labels={"t1": copy.label}))
        [again] = filing.filings([copy], held)
        assert isinstance(again, filing.Again) and again.group.sightings == copy.sightings


def test_a_ticket_names_the_audit_logs_segments_under_the_users_home_by_tilde_and_elsewhere_whole_quoted() -> None:
    assert filing.where(Path("/Users/someone/.hands/audit"), Path("/Users/someone")) == f"~/.hands/audit/{audit.SEGMENT_GLOB}"
    assert filing.where(Path("/Users/someone/Library/Application Support/hands/audit"), Path("/Users/someone")) == f"~/'Library/Application Support/hands/audit'/{audit.SEGMENT_GLOB}"
    assert filing.where(Path("/opt/hands/audit"), Path("/Users/someone")) == f"/opt/hands/audit/{audit.SEGMENT_GLOB}"


def test_a_ticket_closed_answers_for_every_sighting_before_it_was_closed(tmp_path: Path) -> None:
    log = write_audit(tmp_path / "audit", Clock())
    [copy, _] = groups_of(tmp_path)
    log.record(broke("4ce425fb-670d-4936-b5c2-f9b045875d8d", "/tmp/d.jsonl", 9))
    [later, _] = groups_of(tmp_path)
    # Told up to 12:05, seen again at 12:06, and closed fixed at 12:30: nothing is new.
    fixed = issue("t1", "1", "closed", filing.description(copy, LOGS), resolution="fixed", closed_at="2026-10-24T12:30:00Z")
    assert filing.filings([later], parse_export(export([issue("e", "0", None), fixed], labels={"t1": copy.label}))) == ()
    # Seen again after it was closed, it is a new ticket of what came after.
    early = issue("t1", "1", "closed", filing.description(copy, LOGS), resolution="fixed", closed_at="2026-10-24T12:05:30Z")
    [new] = filing.filings([later], parse_export(export([issue("e", "0", None), early], labels={"t1": copy.label})))
    assert isinstance(new, filing.New) and [sighting.stamp for sighting in new.group.sightings] == ["2026-10-24T12:06:00.000+00:00"]


def test_a_lit_new_that_says_no_id_is_a_failure_and_not_a_ticket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def silent(project: Path, *arguments: str, environment: Mapping[str, str] | None) -> bytes:
        return b"\n"

    monkeypatch.setattr(backlog, "_lit", silent)
    with pytest.raises(backlog.LitFailed, match="said no ticket's id"):
        asyncio.run(backlog.file_ticket(tmp_path, "e", "warnings", "a title", "a description", "audit-0123456789ab"))
