"""The backlog page: every open ticket under its epic with hands' sentence for it, and a drop that is a `lit rank`."""

import json
import shutil
import subprocess
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.sessions.audit import Entry
from hands.sessions.backlog import Above, Below, Move, Untracked, parse_export
from hands.sessions.payload import Rejected
from hands.sessions.sentences import Sentences
from hands.sessions.wide import WideEvent
from hands.voice.backlogpage import backlog_routes, parse_asked, shown
from hands.voice.sentences import Sighted, SummaryStore, sighted
from conftest import events

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
LIT = shutil.which("lit")
real_lit = pytest.mark.skipif(LIT is None, reason="the order is lit's, so it is lit that is driven")


def issue(id: str, rank: str, title: str, status: str | None = "open", kind: str = "task", **extra: str) -> dict[str, object]:
    made: dict[str, object] = {"id": id, "title": title, "description": f"The words of {title}.", "issue_type": kind, "rank": rank, **extra}
    return made if status is None else {**made, "status": status}


def inside(parent: str, child: str) -> dict[str, object]:
    return {"src_id": child, "dst_id": parent, "type": "parent-child", "created_at": "", "created_by": ""}


def export(issues: list[dict[str, object]], relations: list[dict[str, object]]) -> bytes:
    return json.dumps({"version": 5, "workspace_id": "w", "issues": issues, "relations": relations, "comments": [], "labels": [], "events": [], "purges": [], "plan_reviews": []}).encode()


# Two epics, the second ranked first; a closed child, and a follow-up filed under a closed ticket.
ISSUES = [
    issue("e2", "01", "Second", status=None, kind="epic"),
    issue("e1", "02", "First", status=None, kind="epic"),
    issue("e1.a", "03", "A"),
    issue("e1.b", "04", "B", status="in_progress"),
    issue("e1.c", "05", "C", status="closed"),
    issue("e2.a", "06", "D"),
    issue("e2.b", "07", "Old", status="closed"),
    issue("e2.b.f", "08", "Follow up"),
]
RELATIONS = [inside("e1", "e1.a"), inside("e1", "e1.b"), inside("e1", "e1.c"), inside("e2", "e2.a"), inside("e2", "e2.b"), inside("e2.b", "e2.b.f")]


def test_the_page_is_every_open_ticket_under_its_epic_in_rank_order_with_hands_sentence_and_never_its_description() -> None:
    backlog = parse_export(export(ISSUES, RELATIONS))
    page = shown(Sighted(backlog, {"e1.a": "A, said.", "e2": "Second, said."}))
    assert page["tracked"] is True
    items = cast(list[dict[str, Any]], page["items"])
    assert [(node["id"], [child["id"] for child in node["children"]]) for node in items] == [("e2", ["e2.a"]), ("e1", ["e1.a", "e1.b"]), ("e2.b.f", [])]
    second, first, follow_up = items
    assert second["summary"] == "Second, said." and first["children"][0]["summary"] == "A, said."
    # A ticket with no sentence has none, rather than its description in its place.
    assert "summary" not in first and "summary" not in first["children"][1]
    assert "The words of" not in json.dumps(page)
    assert first["children"][1]["status"] == "in_progress" and first["kind"] == "epic"
    # The frame a drop stays inside: the follow-up is drawn among the roots, and lit ranks it inside its closed parent.
    assert (second["frame"], first["children"][0]["frame"], follow_up["frame"]) == (None, "e1", "e2.b")


def test_a_project_lit_does_not_track_is_drawn_as_none() -> None:
    assert shown(Untracked(Path("/nowhere"))) == {"tracked": False}


def test_a_move_is_only_between_tickets_lit_ranks_inside_the_same_parent() -> None:
    backlog = parse_export(export(ISSUES, RELATIONS))
    assert backlog.move("e1.b", Above("e1.a")) == Move("e1.b", Above("e1.a"))
    assert backlog.move("e1", Above("e2")) == Move("e1", Above("e2"))
    # A closed sibling is still one lit ranks among.
    assert backlog.move("e1.a", Below("e1.c")) == Move("e1.a", Below("e1.c"))
    for ticket, place, why in [
        ("e1.a", Above("e2.a"), "not inside the same parent"),
        ("e2.b.f", Below("e1"), "not inside the same parent"),
        ("e1.a", Above("e1"), "not inside the same parent"),
        ("nope", Above("e1.a"), "no ticket nope"),
        ("e1.a", Below("nope"), "no ticket nope"),
        ("e1.a", Above("e1.a"), "next to itself"),
    ]:
        with pytest.raises(Rejected, match=why):
            backlog.move(ticket, place)


def test_an_archived_parent_frames_its_children_and_a_deleted_one_frames_nothing() -> None:
    issues = [
        issue("r", "01", "Root", status=None, kind="epic"),
        issue("gone", "02", "Deleted", status=None, kind="epic", deleted_at="2026-10-18T00:00:00Z"),
        issue("gone.a", "03", "Left behind"),
        issue("shelf", "04", "Archived", status=None, kind="epic", archived_at="2026-10-18T00:00:00Z"),
        issue("shelf.a", "05", "Shelved"),
    ]
    backlog = parse_export(export(issues, [inside("gone", "gone.a"), inside("shelf", "shelf.a")]))
    # Both are roots of the tree the page draws, since lit lists neither parent.
    assert backlog.roots() == ("r", "gone.a", "shelf.a") and backlog.parent == {}
    assert backlog.inside == {"shelf.a": "shelf"}
    assert backlog.move("gone.a", Below("r")) == Move("gone.a", Below("r"))
    with pytest.raises(Rejected, match="not inside the same parent"):
        backlog.move("shelf.a", Below("r"))


def test_a_move_has_landed_only_when_the_ticket_sits_directly_beside_its_target_in_their_frame() -> None:
    backlog = parse_export(export(ISSUES, RELATIONS))
    assert backlog.landed(Move("e1.a", Above("e1.b"))) and backlog.landed(Move("e1.b", Below("e1.a")))
    assert backlog.landed(Move("e1.b", Above("e1.c"))) and backlog.landed(Move("e2", Above("e1")))
    assert not backlog.landed(Move("e1.b", Above("e1.a"))) and not backlog.landed(Move("e1.a", Above("e1.c")))
    # Beside it in rank, but not in its frame.
    assert not backlog.landed(Move("e1", Below("e2.a")))


@pytest.mark.parametrize(
    "body, why",
    [
        (b"not json", "JSON"),
        (b'{"project": "/p", "ticket": "a"}', "one of above or below"),
        (b'{"project": "/p", "ticket": "a", "above": "b", "below": "c"}', "one of above or below"),
        (b'{"project": "/p", "ticket": 3, "above": "b"}', "one of above or below"),
        (b'{"project": "p", "ticket": "a", "above": "b"}', "absolute path"),
    ],
)
def test_a_move_the_page_sends_is_refused_naming_what_is_wrong(body: bytes, why: str) -> None:
    with pytest.raises(Rejected, match=why):
        parse_asked(body)


def test_a_move_the_page_sends_is_its_project_ticket_and_place() -> None:
    assert parse_asked(b'{"project": "/p", "ticket": "a", "below": "b"}') == (Path("/p"), "a", Below("b"))


# --- the routes, against lit itself


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    store: SummaryStore
    recorded: list[Entry]

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


@pytest.fixture
async def page(tmp_path: Path) -> AsyncGenerator[Page]:
    recorded: list[Entry] = []
    store = SummaryStore(Sentences(tmp_path / "sentences.db"))
    app = web.Application()
    app.add_routes(backlog_routes(store, KEY, recorded.append))
    client = TestClient(TestServer(app))
    await client.start_server()
    yield Page(client, store, recorded)
    await client.close()


@dataclass
class Workspace:
    """A git repository with a lit workspace of its own, in a temporary directory."""

    root: Path

    def lit(self, *arguments: str) -> str:
        """The id lit printed first."""
        return subprocess.run([str(LIT), *arguments], cwd=self.root, check=True, capture_output=True, text=True).stdout.split()[0]

    def order(self, *ids: str) -> list[str]:
        """`ids` in the rank order `lit export` reports them in, read raw."""
        exported = json.loads(subprocess.run([str(LIT), "export"], cwd=self.root, check=True, capture_output=True).stdout)
        return [entry["id"] for entry in sorted(exported["issues"], key=lambda entry: entry["rank"]) if entry["id"] in ids]

    def shim(self, bin: Path, before_rank: str) -> None:
        """Put a lit first on PATH that runs `before_rank` before a `lit rank`, then runs lit itself."""
        bin.mkdir(exist_ok=True)
        (bin / "lit").write_text(f'#!/bin/sh\nif [ "$1" = rank ]; then\n{before_rank}\nfi\nexec {LIT} "$@"\n')
        (bin / "lit").chmod(0o755)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([str(LIT), "init"], cwd=root, check=True, capture_output=True)
    return Workspace(root)


def epic(workspace: Workspace, title: str, *children: str) -> tuple[str, ...]:
    made = workspace.lit("new", "--type", "epic", "--title", title, "--topic", "page")
    return (made, *(workspace.lit("new", "--parent", made, "--title", child, "--topic", "page") for child in children))


async def moved(page: Page, workspace: Workspace, ticket: str, **place: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/rank", json={"project": str(workspace.root), "ticket": ticket, **place}, headers=KEYED)
    return response.status, await response.text()


async def test_the_page_is_served_to_anyone_and_its_backlog_to_the_phones_key_alone(page: Page) -> None:
    shown_page = await page.client.get("/backlog")
    assert shown_page.status == 200 and "Drag a card up or down" in await shown_page.text()
    assert (await page.client.get("/backlog/tickets?project=/tmp")).status == 401
    assert (await page.client.post("/backlog/rank", json={})).status == 401
    assert [event.error for event in page.events("backlog.read") + page.events("backlog.ranked")] == ["read without the phone's key", "ranked without the phone's key"]


@pytest.mark.parametrize("query", ["", "?project=relative/path"])
async def test_a_read_that_names_no_absolute_project_is_refused(page: Page, query: str) -> None:
    response = await page.client.get(f"/backlog/tickets{query}", headers=KEYED)
    assert response.status == 400 and "absolute path" in await response.text()


@real_lit
async def test_a_directory_lit_has_no_workspace_in_is_drawn_as_untracked(page: Page, tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    response = await page.client.get("/backlog/tickets", params={"project": str(tmp_path)}, headers=KEYED)
    assert response.status == 200 and await response.json() == {"tracked": False}
    [event] = page.events("backlog.read")
    assert event.outcome == "ok" and event.facts["tracked"] is False


@real_lit
async def test_the_read_is_lits_backlog_with_hands_sentences_and_asks_for_the_rest(page: Page, workspace: Workspace) -> None:
    voice, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    read = await sighted(page.store, workspace.root)
    assert isinstance(read, Sighted)
    # The leaves are said first; the epic waits on them.
    page.store.keep({due.digest: f"Said of {due.id}." for due in page.store.reckon(read.backlog.thing()).due})
    response = await page.client.get("/backlog/tickets", params={"project": str(workspace.root)}, headers=KEYED)
    assert response.status == 200
    [node] = (await response.json())["items"]
    assert node["id"] == voice and "summary" not in node
    assert [(child["id"], child["summary"]) for child in node["children"]] == [(mic, f"Said of {mic}."), (parser, f"Said of {parser}.")]
    [event] = page.events("backlog.read")
    assert event.outcome == "ok" and event.facts["project"] == workspace.root and event.counts == {"tickets": 3, "summarised": 2}


@real_lit
async def test_a_drop_ranks_the_ticket_in_lit_and_the_page_is_answered_with_lits_order(page: Page, workspace: Workspace) -> None:
    _, mic, parser, docs = epic(workspace, "Voice", "Fix the mic", "Fix the parser", "Write the docs")
    assert workspace.order(mic, parser, docs) == [mic, parser, docs]
    status, body = await moved(page, workspace, docs, above=mic)
    assert status == 200
    assert workspace.order(mic, parser, docs) == [docs, mic, parser]
    [node] = json.loads(body)["items"]
    assert [child["id"] for child in node["children"]] == [docs, mic, parser]
    status, _ = await moved(page, workspace, mic, below=parser)
    assert status == 200 and workspace.order(mic, parser, docs) == [docs, parser, mic]
    first, second = page.events("backlog.ranked")
    assert first.outcome == "ok" and (first.facts["ticket"], first.facts["place"]) == (docs, Above(mic))
    assert second.outcome == "ok" and second.facts["place"] == Below(parser)


@real_lit
async def test_epics_are_dragged_among_themselves(page: Page, workspace: Workspace) -> None:
    voice, *_ = epic(workspace, "Voice", "Fix the mic")
    wire, *_ = epic(workspace, "Wire", "Tap it")
    status, _ = await moved(page, workspace, wire, above=voice)
    assert status == 200 and workspace.order(voice, wire) == [wire, voice]


@real_lit
async def test_a_move_across_epics_is_refused_and_nothing_in_lit_moves(page: Page, workspace: Workspace) -> None:
    voice, mic = epic(workspace, "Voice", "Fix the mic")
    wire, tap = epic(workspace, "Wire", "Tap it")
    before = workspace.order(voice, mic, wire, tap)
    # lit itself would carry this out by moving the Wire epic above the Voice one.
    status, body = await moved(page, workspace, tap, above=mic)
    assert status == 400 and "not inside the same parent" in body
    status, body = await moved(page, workspace, "nope", above=mic)
    assert status == 400 and "no ticket nope" in body
    assert workspace.order(voice, mic, wire, tap) == before
    assert [event.outcome for event in page.events("backlog.ranked")] == ["failed", "failed"]


@real_lit
async def test_a_deleted_epics_child_is_moved_among_the_roots(page: Page, workspace: Workspace) -> None:
    root, *_ = epic(workspace, "Root", "Keep it")
    gone, left = epic(workspace, "Gone", "Left behind")
    other, *_ = epic(workspace, "Other", "Another")
    workspace.lit("delete", gone, "--reason", "Dropped.")
    status, _ = await moved(page, workspace, left, below=root)
    assert status == 200 and workspace.order(root, left, other) == [root, left, other]


@real_lit
async def test_an_archived_epics_child_is_not_moved_among_the_roots(page: Page, workspace: Workspace) -> None:
    root, *_ = epic(workspace, "Root", "Keep it")
    shelf, shelved = epic(workspace, "Shelf", "Shelved")
    workspace.lit("archive", shelf, "--reason", "Later.")
    before = workspace.order(root, shelf, shelved)
    status, body = await moved(page, workspace, shelved, above=root)
    assert status == 400 and "not inside the same parent" in body
    assert workspace.order(root, shelf, shelved) == before


@real_lit
async def test_a_move_lit_makes_otherwise_because_the_backlog_changed_meanwhile_is_a_conflict(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voice, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    wire, *_ = epic(workspace, "Wire", "Tap it")
    # An agent puts the parser in the Wire epic between the page's read and its rank: lit ranks the Wire epic instead.
    workspace.shim(tmp_path / "bin", f"{LIT} parent set --child {parser} --parent {wire} >/dev/null")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:/bin:/usr/bin")
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 409 and f"lit did not put {parser} above {mic}" in body
    assert workspace.order(voice, wire) == [wire, voice]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error is not None and "the backlog changed while it was moved" in event.error


@real_lit
async def test_a_rank_lit_fails_is_a_failure_carrying_lits_verdict(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    workspace.shim(tmp_path / "bin", 'echo "error (code=5, reason=workspace_busy): the store is locked" >&2\nexit 5')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:/bin:/usr/bin")
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and "lit rank" in body and "workspace_busy" in body
    assert workspace.order(mic, parser) == [mic, parser]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error is not None and "workspace_busy" in event.error
