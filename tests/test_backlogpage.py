"""The backlog page: every open ticket under its epic with hands' sentence for it, a drop that is a `lit rank`, and a
level that moves a ticket into its topic's epic for it."""

import asyncio
import getpass
import json
import re
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
from hands.sessions.backlog import LEVELS, Level, Leveling, Move, NewEpic, Place, Standing, Untracked, parse_export, parse_level
from hands.sessions.payload import Rejected
from hands.sessions.sentences import Sentences
from hands.sessions.wide import WideEvent, unit
from hands.voice.backlogpage import backlog_routes, parse_asked, parse_leveling, parse_note, parse_project, shown
from hands.voice.sentences import Sighted, SummaryStore, sighted
from conftest import events

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
LIT = shutil.which("lit")
real_lit = pytest.mark.skipif(LIT is None, reason="the order is lit's, so it is lit that is driven")


def issue(id: str, rank: str, title: str, status: str | None = "open", kind: str = "task", topic: str = "page", **extra: str) -> dict[str, object]:
    made: dict[str, object] = {"id": id, "title": title, "description": f"The words of {title}.", "issue_type": kind, "topic": topic, "rank": rank, **extra}
    return made if status is None else {**made, "status": status, **({"closed_at": "2026-10-19T10:00:00Z"} if status == "closed" else {})}


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
    recorded: list[Entry] = []
    with unit("backlog.read", recorded.append, counts=("tickets", "summarised")):
        page = shown(Sighted(backlog, {"e1.a": "A, said.", "e2": "Second, said."}))
    assert page["tracked"] is True
    items = cast(list[dict[str, Any]], page["items"])
    assert [(node["id"], [child["id"] for child in node["children"]]) for node in items] == [("e2", ["e2.a"]), ("e1", ["e1.a", "e1.b"]), ("e2.b.f", [])]
    second, first, follow_up = items
    assert second["summary"] == "Second, said." and first["children"][0]["summary"] == "A, said."
    # A ticket with no sentence has none, rather than its description in its place.
    assert first["summary"] is None and first["children"][1]["summary"] is None
    assert "The words of" not in json.dumps(page)
    assert first["children"][1]["status"] == "in_progress" and first["status"] is None
    assert (first["kind"], first["children"][0]["kind"]) == ("epic", "task")
    # The frame a drop stays inside: the follow-up is drawn among the roots, and lit ranks it inside its closed parent.
    assert (second["frame"], first["children"][0]["frame"], follow_up["frame"]) == (None, "e1", "e2.b")
    # A ticket in an epic stands at no level of its topic yet; an epic, or a ticket in no epic, cannot be put at one.
    assert first["children"][0]["standing"] == {"epic": "e1", "level": None}
    assert (first["standing"], follow_up["standing"]) == (None, None)
    assert page["levels"] == ("high", "normal", "low")
    # The event counts what the page is given.
    [event] = events(recorded, "backlog.read")
    assert event.counts == {"tickets": 6, "summarised": 2} and event.facts["tracked"] is True


def test_a_project_lit_does_not_track_is_drawn_as_none() -> None:
    recorded: list[Entry] = []
    with unit("backlog.read", recorded.append, counts=("tickets", "summarised")):
        assert shown(Untracked(Path("/nowhere"))) == {"tracked": False}
    [event] = events(recorded, "backlog.read")
    assert event.facts["tracked"] is False and event.counts == {"tickets": 0, "summarised": 0}


def test_a_move_is_only_between_tickets_lit_ranks_inside_the_same_parent() -> None:
    backlog = parse_export(export(ISSUES, RELATIONS))
    assert backlog.move("e1.b", Place("above", "e1.a")) == Move("e1.b", Place("above", "e1.a"))
    assert backlog.move("e1", Place("above", "e2")) == Move("e1", Place("above", "e2"))
    # A closed sibling is still one lit ranks among.
    assert backlog.move("e1.a", Place("below", "e1.c")) == Move("e1.a", Place("below", "e1.c"))
    for ticket, place, why in [
        ("e1.a", Place("above", "e2.a"), "not inside the same parent"),
        ("e2.b.f", Place("below", "e1"), "not inside the same parent"),
        ("e1.a", Place("above", "e1"), "not inside the same parent"),
        ("nope", Place("above", "e1.a"), "no ticket nope"),
        ("e1.a", Place("below", "nope"), "no ticket nope"),
        ("e1.a", Place("above", "e1.a"), "next to itself"),
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
    assert backlog.move("gone.a", Place("below", "r")) == Move("gone.a", Place("below", "r"))
    with pytest.raises(Rejected, match="not inside the same parent"):
        backlog.move("shelf.a", Place("below", "r"))


def test_a_move_has_landed_only_when_the_ticket_sits_directly_beside_its_target_in_their_frame() -> None:
    backlog = parse_export(export(ISSUES, RELATIONS))
    assert backlog.landed(Move("e1.a", Place("above", "e1.b"))) and backlog.landed(Move("e1.b", Place("below", "e1.a")))
    assert backlog.landed(Move("e1.b", Place("above", "e1.c"))) and backlog.landed(Move("e2", Place("above", "e1")))
    assert not backlog.landed(Move("e1.b", Place("above", "e1.a"))) and not backlog.landed(Move("e1.a", Place("above", "e1.c")))
    # Beside it in rank, but not in its frame.
    assert not backlog.landed(Move("e1", Place("below", "e2.a")))


@pytest.mark.parametrize(
    "body, why",
    [
        (b"not json", "JSON"),
        (b'{"project": "/", "ticket": "a"}', "one of above or below"),
        (b'{"project": "/", "ticket": "a", "above": "b", "below": "c"}', "never both"),
        (b'{"project": "/", "ticket": 3, "above": "b"}', "one of above or below"),
        (b'{"project": "p", "ticket": "a", "above": "b"}', "absolute path"),
        (b'{"project": "/no/such/directory", "ticket": "a", "above": "b"}', "no directory"),
    ],
)
def test_a_move_the_page_sends_is_refused_naming_what_is_wrong(body: bytes, why: str) -> None:
    with pytest.raises(Rejected, match=why):
        parse_asked(body)


def test_a_move_the_page_sends_is_its_project_resolved_its_ticket_and_its_place(tmp_path: Path) -> None:
    spelled = tmp_path / "project" / ".." / "project"
    (tmp_path / "project").mkdir()
    (tmp_path / "file").touch()
    body = json.dumps({"project": str(spelled), "ticket": "a", "below": "b"}).encode()
    # One directory has one name however it is spelled, so it is summarised once.
    assert parse_asked(body) == ((tmp_path / "project").resolve(), "a", Place("below", "b"))
    with pytest.raises(Rejected, match="no directory"):
        parse_project(str(tmp_path / "file"))
    with pytest.raises(Rejected, match="is no path"):
        parse_project("/a\0b")


# A topic's epic, its epic for high, its epic for low whose every ticket is closed, and a near miss of each kind.
LEVELED = [
    issue("w", "01", "Train a new wake word", status=None, kind="epic", topic="wake-word"),
    issue("w.a", "02", "Hey hands", topic="wake-word"),
    issue("w.b", "03", "Many wake words", topic="wake-word"),
    issue("wh", "04", "wake word - high", status=None, kind="epic", topic="wake-word"),
    issue("wh.a", "05", "A wake word", topic="wake-word"),
    issue("wl", "06", "wake word - low", status=None, kind="epic", topic="wake-word"),
    issue("wl.a", "07", "Landed", status="closed", topic="wake-word"),
    issue("wn", "08", "Wake word - normal", status=None, kind="epic", topic="wake-word"),
    issue("xn", "09", "wake word - normal", status=None, kind="epic", topic="wake"),
    issue("ww", "10", "Wake words", status=None, kind="epic", topic="wake-words"),
    issue("ww.a", "11", "Train your own", topic="wake-words"),
]
LEVELED_RELATIONS = [inside("w", "w.a"), inside("w", "w.b"), inside("wh", "wh.a"), inside("wl", "wl.a"), inside("ww", "ww.a")]


def test_a_ticket_stands_at_the_level_its_epics_exact_title_names_for_its_topic() -> None:
    backlog = parse_export(export(LEVELED, LEVELED_RELATIONS))
    assert backlog.standing("w.a") == Standing("w", None) and backlog.standing("wh.a") == Standing("wh", "high")
    assert backlog.standing("w") is None


def test_a_level_is_the_open_epic_of_that_topic_titled_exactly_so_or_one_made_beside_the_tickets_epic() -> None:
    backlog = parse_export(export(LEVELED, LEVELED_RELATIONS))
    assert backlog.leveling("w.a", "high") == Leveling("w.a", "high", "wh")
    # Closed, differently cased, and of another topic are none of them wake-word's epic for the level.
    # One whose every ticket is closed is the topic's epic for its level still, and is used again.
    assert backlog.leveling("w.b", "low") == Leveling("w.b", "low", "wl")
    # Differently cased and of another topic are none of them wake-word's epic for normal; one is made between high and low.
    assert backlog.leveling("w.b", "normal") == Leveling("w.b", "normal", NewEpic("wake-word", "wake word - normal", None, Place("above", "wl")))
    # A level's epic carries its topic, so a ticket in one goes to its topic's epic for another.
    assert backlog.leveling("wh.a", "normal").epic == NewEpic("wake-word", "wake word - normal", None, Place("above", "wl"))
    assert backlog.leveling("ww.a", "high").epic == NewEpic("wake-words", "wake words - high", None, Place("above", "ww"))
    refused: list[tuple[str, Level, str]] = [("wh.a", "high", "already in wh"), ("w", "low", "is an epic"), ("nope", "low", "no ticket nope")]
    for ticket, level, why in refused:
        with pytest.raises(Rejected, match=why):
            backlog.leveling(ticket, level)


def test_a_level_epic_made_is_placed_so_its_topics_levels_stay_high_over_normal_over_low() -> None:
    issues = [
        issue("w", "01", "Train a new wake word", status=None, kind="epic", topic="wake-word"),
        issue("w.a", "02", "Hey hands", topic="wake-word"),
        issue("wl", "03", "wake word - low", status=None, kind="epic", topic="wake-word"),
        issue("wl.a", "04", "Many wake words", topic="wake-word"),
    ]
    backlog = parse_export(export(issues, [inside("w", "w.a"), inside("wl", "wl.a")]))
    # Above the first lower level, wherever the ticket's own epic is.
    assert backlog.leveling("wl.a", "normal").epic == NewEpic("wake-word", "wake word - normal", None, Place("above", "wl"))
    assert backlog.leveling("w.a", "high").epic == NewEpic("wake-word", "wake word - high", None, Place("above", "wl"))


def test_drawn_leaves_out_an_epic_with_no_open_ticket_and_the_summary_keeps_it() -> None:
    issues = [issue("e", "01", "Emptied", status=None, kind="epic"), issue("f", "02", "Full", status=None, kind="epic"), issue("f.a", "03", "A")]
    backlog = parse_export(export(issues, [inside("f", "f.a")]))
    with unit("backlog.read", lambda _entry: None, counts=("tickets", "summarised")):
        page = shown(Sighted(backlog, {}))
    assert [node["id"] for node in cast(list[dict[str, Any]], page["items"])] == ["f"]
    assert [part.id for part in backlog.thing().parts] == ["e", "f"]


def test_a_level_made_inside_an_epic_is_made_in_the_same_one() -> None:
    issues = [issue("r", "01", "Room", status=None, kind="epic"), issue("r.w", "02", "Wake", status=None, kind="epic", topic="wake"), issue("r.w.a", "03", "Hey", topic="wake")]
    backlog = parse_export(export(issues, [inside("r", "r.w"), inside("r.w", "r.w.a")]))
    assert backlog.leveling("r.w.a", "high").epic == NewEpic("wake", "wake - high", "r", Place("above", "r.w"))


@pytest.mark.parametrize(
    "body, why",
    [
        (b"[", "JSON"),
        (b'{"project": "/", "ticket": "a"}', "level: one of high, normal, low"),
        (b'{"project": "/", "ticket": "a", "level": "urgent"}', "level: one of high, normal, low"),
        (b'{"project": "/", "ticket": "a", "level": "High"}', "level: one of high, normal, low"),
        (b'{"project": "p", "ticket": "a", "level": "low"}', "absolute path"),
    ],
)
def test_a_level_the_page_sends_is_refused_naming_what_is_wrong(body: bytes, why: str) -> None:
    with pytest.raises(Rejected, match=why):
        parse_leveling(body)


def test_a_level_is_parsed_from_its_name_alone() -> None:
    assert [parse_level(level) for level in LEVELS] == list(LEVELS)
    assert (parse_level("High"), parse_level("urgent"), parse_level("")) == (None, None, None)


def test_a_level_the_page_sends_is_its_project_its_ticket_and_one_of_the_levels(tmp_path: Path) -> None:
    for level in LEVELS:
        assert parse_leveling(json.dumps({"project": str(tmp_path), "ticket": "a", "level": level}).encode()) == (tmp_path.resolve(), "a", level)


# --- the routes, against lit itself


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    store: SummaryStore
    recorded: list[Entry]

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b"{", "a note is JSON"),
        (b'{"project": "/tmp", "ticket": "a"}', "a note is {project, ticket, and text"),
        (b'{"project": "/tmp", "ticket": "a", "text": "  \\n "}', "not blank"),
    ],
)
def test_a_note_the_page_sends_is_refused_naming_what_is_wrong(body: bytes, why: str) -> None:
    with pytest.raises(Rejected, match=re.escape(why)):
        parse_note(body)


def test_a_note_the_page_sends_is_its_project_resolved_its_ticket_and_its_words_trimmed(tmp_path: Path) -> None:
    assert parse_note(json.dumps({"project": str(tmp_path / "."), "ticket": "a", "text": " split it \n"}).encode()) == (tmp_path.resolve(), "a", "split it")


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


def epic(workspace: Workspace, title: str, *children: str, topic: str = "page") -> tuple[str, ...]:
    made = workspace.lit("new", "--type", "epic", "--title", title, "--topic", topic)
    return (made, *(workspace.lit("new", "--parent", made, "--title", child, "--topic", topic) for child in children))


async def moved(page: Page, workspace: Workspace, ticket: str, **place: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/rank", json={"project": str(workspace.root), "ticket": ticket, **place}, headers=KEYED)
    return response.status, await response.text()


async def leveled(page: Page, workspace: Workspace, ticket: str, level: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/level", json={"project": str(workspace.root), "ticket": ticket, "level": level}, headers=KEYED)
    return response.status, await response.text()


def titled(items: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Each root the page is given, by title, over its children's ids."""
    return [(node["title"], [child["id"] for child in node["children"]]) for node in items]


async def test_the_page_is_served_to_anyone_and_its_backlog_to_the_phones_key_alone(page: Page) -> None:
    shown_page = await page.client.get("/backlog")
    assert shown_page.status == 200 and "drag its card up or down" in await shown_page.text()
    assert (await page.client.get("/backlog/tickets?project=/tmp")).status == 401
    assert (await page.client.post("/backlog/rank", json={})).status == 401
    assert (await page.client.post("/backlog/level", json={})).status == 401
    assert (await page.client.post("/backlog/note", json={})).status == 401
    keyless = page.events("backlog.read") + page.events("backlog.ranked") + page.events("backlog.leveled") + page.events("backlog.noted")
    assert [event.error for event in keyless] == ["asked without the phone's key"] * 4


@pytest.mark.parametrize(
    "query, why",
    [("", "absolute path"), ("?project=relative/path", "absolute path"), ("?project=/no/such/directory", "no directory"), ("?project=/a%00b", "is no path")],
)
async def test_a_read_that_names_no_directory_is_refused_rather_than_asking_lit(page: Page, query: str, why: str) -> None:
    response = await page.client.get(f"/backlog/tickets{query}", headers=KEYED)
    assert response.status == 400 and why in await response.text()


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
    assert node["id"] == voice and node["summary"] is None
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
    assert first.outcome == "ok" and (first.facts["ticket"], first.facts["place"], first.facts["ranked"]) == (docs, Place("above", mic), "yes")
    # What it showed after: the epic and its three tickets, none said yet.
    assert first.counts == {"tickets": 4, "summarised": 0}
    assert second.outcome == "ok" and second.facts["place"] == Place("below", parser)


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
    # What lit ranked instead is said in lit's own words, to the page and on the event; nothing is undone.
    assert status == 409 and f"does not have {parser} above {mic}" in body and f"ranked the epic {wire} instead" in body
    assert workspace.order(voice, wire) == [wire, voice]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == body
    assert event.facts["ranked"] == "yes" and f"ranked the epic {wire} instead" in str(event.facts["lit_said"])


@real_lit
async def test_a_rank_lit_fails_is_a_failure_carrying_lits_verdict(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    workspace.shim(tmp_path / "bin", 'echo "error (code=5, reason=workspace_busy): the store is locked" >&2\nexit 5')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:/bin:/usr/bin")
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and "lit rank" in body and "workspace_busy" in body
    assert workspace.order(mic, parser) == [mic, parser]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error is not None and "workspace_busy" in event.error and event.facts["ranked"] == "no"


@real_lit
async def test_a_rank_lit_made_whose_backlog_could_not_be_read_again_says_it_was_ranked(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    ranked = tmp_path / "ranked"
    # lit ranks, and every export after it fails.
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(
        f'#!/bin/sh\nif [ "$1" = export ] && [ -e {ranked} ]; then\necho "error (code=5, reason=workspace_busy): the store is locked" >&2\nexit 5\nfi\n'
        f'if [ "$1" = rank ]; then touch {ranked}; fi\nexec {LIT} "$@"\n'
    )
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and body.startswith(f"lit ranked {parser} as it was asked to go above {mic}, and the backlog could not be read again") and "workspace_busy" in body
    assert workspace.order(mic, parser) == [parser, mic]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == body and event.facts["ranked"] == "yes"


@real_lit
async def test_a_rank_lit_stops_without_a_reason_is_said_to_be_unknown_and_never_claimed_either_way(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    # lit writes the rank, then is killed before it can say so.
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(f'#!/bin/sh\nif [ "$1" = rank ]; then\n{LIT} "$@" >/dev/null\nkill -9 $$\nfi\nexec {LIT} "$@"\n')
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and body.startswith(f"whether lit ranked {parser} above {mic} is not known")
    assert workspace.order(mic, parser) == [parser, mic]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == body and event.facts["ranked"] == "unknown"


@real_lit
@pytest.mark.parametrize("export, why", [('echo "error (code=5, reason=workspace_busy): locked" >&2; exit 5', "workspace_busy"), ("echo not json", "JSON")])
async def test_a_read_lit_cannot_give_is_a_failure_on_the_page_and_its_event(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export: str, why: str) -> None:
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(f'#!/bin/sh\nif [ "$1" = export ]; then\n{export}\nfi\nexec {LIT} "$@"\n')
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    response = await page.client.get("/backlog/tickets", params={"project": str(workspace.root)}, headers=KEYED)
    body = await response.text()
    assert response.status == 502 and body.startswith(f"the backlog in {workspace.root} could not be read") and why in body
    [event] = page.events("backlog.read")
    assert event.outcome == "failed" and event.error is not None and why in event.error


@pytest.mark.parametrize("body, why", [(b"{}", "one of above or below"), (b'{"project": "/", "ticket": "a", "above": "b", "below": "c"}', "never both")])
async def test_a_move_the_page_cannot_send_is_refused_and_its_event_says_why(page: Page, body: bytes, why: str) -> None:
    response = await page.client.post("/backlog/rank", data=body, headers=KEYED)
    text = await response.text()
    assert response.status == 400 and text.startswith("not moved: ") and why in text
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == text


@real_lit
async def test_a_read_asks_for_a_pass_only_while_something_is_unsaid(page: Page, workspace: Workspace) -> None:
    epic(workspace, "Voice", "Fix the mic")
    read = await sighted(page.store, workspace.root)
    assert isinstance(read, Sighted)
    assert await page.store.wanted()
    # Every ticket said, round by round, as the summarising task says them.
    while due := page.store.reckon(read.backlog.thing()).due:
        page.store.keep({each.digest: f"Said of {each.id}." for each in due})
    again = await sighted(page.store, workspace.root)
    assert isinstance(again, Sighted) and len(again.said) == 3
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(page.store.wanted(), 0.2)


@real_lit
async def test_a_level_moves_the_ticket_into_its_topics_epic_for_it_made_beside_its_epic_the_first_time(page: Page, workspace: Workspace) -> None:
    train, hey, custom, many = epic(workspace, "Train a new wake word", "Implement wake word mode", "Hey hands", "Many wake words", topic="wake-word")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 200
    [high] = [node for node in json.loads(body)["items"] if node["title"] == "wake word - high"]
    # Made directly above the epic the ticket was in, and the ticket moved into it with its id.
    assert titled(json.loads(body)["items"]) == [("wake word - high", [hey]), ("Train a new wake word", [custom, many])]
    assert high["children"][0]["standing"] == {"epic": high["id"], "level": "high"}
    status, body = await leveled(page, workspace, custom, "high")
    # The same epic, not a second.
    assert status == 200 and titled(json.loads(body)["items"]) == [("wake word - high", [hey, custom]), ("Train a new wake word", [many])]
    status, body = await leveled(page, workspace, hey, "low")
    # Made directly below the epic the ticket is in now: its topic's high one.
    assert status == 200 and titled(json.loads(body)["items"]) == [("wake word - high", [custom]), ("wake word - low", [hey]), ("Train a new wake word", [many])]
    low = next(node["id"] for node in json.loads(body)["items"] if node["title"] == "wake word - low")
    assert workspace.order(high["id"], low, train) == [high["id"], low, train]
    first, second, third = page.events("backlog.leveled")
    assert first.outcome == "ok" and (first.facts["ticket"], first.facts["level"], first.facts["epic"], first.facts["created"], first.facts["moved"]) == (hey, "high", high["id"], "yes", "yes")
    assert second.outcome == "ok" and (second.facts["ticket"], second.facts["epic"], second.facts["created"]) == (custom, high["id"], "no")
    # Ranked last in the epic it joined, below the ticket already there; alone in a new one, it is ranked nowhere.
    assert (first.facts["placed"], second.facts["placed"]) == ("no", "yes")
    assert third.outcome == "ok" and (third.facts["level"], third.facts["epic"], third.facts["created"]) == ("low", low, "yes")
    # What it showed after: three epics and three tickets.
    assert third.counts == {"tickets": 6, "summarised": 0}


@real_lit
async def test_a_level_the_ticket_is_at_already_is_refused_and_nothing_in_lit_moves(page: Page, workspace: Workspace) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    assert (await leveled(page, workspace, hey, "normal"))[0] == 200
    before = workspace.order(hey)
    status, body = await leveled(page, workspace, hey, "normal")
    assert status == 400 and body.startswith("not leveled: ") and "already in" in body
    assert workspace.order(hey) == before
    assert [event.outcome for event in page.events("backlog.leveled")] == ["ok", "failed"]


@real_lit
async def test_a_move_lit_refuses_after_making_the_epic_says_the_epic_was_made_and_the_ticket_was_not_moved(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(f'#!/bin/sh\nif [ "$1" = parent ]; then\necho "error (code=5, reason=workspace_busy): the store is locked" >&2\nexit 5\nfi\nexec {LIT} "$@"\n')
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 502 and body.startswith("not leveled: ") and "workspace_busy" in body
    [event] = page.events("backlog.leveled")
    assert event.outcome == "failed" and (event.facts["created"], event.facts["moved"]) == ("yes", "no")
    # The epic stands made and empty, and the next press puts the ticket in it rather than making another.
    monkeypatch.undo()
    status, body = await leveled(page, workspace, hey, "high")
    # The epic it was in holds nothing open now, and is drawn no more.
    assert status == 200 and titled(json.loads(body)["items"]) == [("wake word - high", [hey])]
    assert page.events("backlog.leveled")[1].facts["created"] == "no"


@real_lit
async def test_a_move_lit_stops_without_a_reason_is_said_to_be_unknown(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(f'#!/bin/sh\nif [ "$1" = parent ]; then\n{LIT} "$@" >/dev/null\nkill -9 $$\nfi\nexec {LIT} "$@"\n')
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    status, body = await leveled(page, workspace, hey, "low")
    assert status == 502 and body.startswith(f"whether lit moved {hey} into ") and "is not known" in body
    [event] = page.events("backlog.leveled")
    assert event.outcome == "failed" and event.facts["moved"] == "unknown"


@real_lit
async def test_a_level_epic_made_keeps_its_topics_levels_in_order_whatever_epic_the_ticket_is_in(page: Page, workspace: Workspace) -> None:
    _, first, second, third = epic(workspace, "Train a new wake word", "Hey hands", "Many wake words", "Train your own", topic="wake-word")
    assert (await leveled(page, workspace, first, "low"))[0] == 200
    # From the low epic, normal goes above it, not below.
    status, body = await leveled(page, workspace, first, "normal")
    assert status == 200 and [title for title, _ in titled(json.loads(body)["items"])] == ["Train a new wake word", "wake word - normal"]
    assert (await leveled(page, workspace, second, "low"))[0] == 200
    # From the topic's own epic, high goes above the normal one, not above the topic's epic.
    status, body = await leveled(page, workspace, third, "high")
    assert status == 200
    titles = [title for title, _ in titled(json.loads(body)["items"])]
    assert titles == ["wake word - high", "wake word - normal", "wake word - low"]
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    epics = [entry["title"] for entry in sorted(exported["issues"], key=lambda entry: entry["rank"]) if entry["issue_type"] == "epic"]
    assert epics == ["Train a new wake word", "wake word - high", "wake word - normal", "wake word - low"]


@real_lit
async def test_a_level_epic_made_from_a_topics_high_one_goes_between_it_and_the_topics_epic(page: Page, workspace: Workspace) -> None:
    train, first, second = epic(workspace, "Train a new wake word", "Hey hands", "Many wake words", topic="wake-word")
    assert (await leveled(page, workspace, first, "high"))[0] == 200
    assert (await leveled(page, workspace, second, "low"))[0] == 200
    status, body = await leveled(page, workspace, first, "normal")
    assert status == 200
    items = json.loads(body)["items"]
    assert [title for title, _ in titled(items)] == ["wake word - normal", "wake word - low"]
    ids = {node["title"]: node["id"] for node in items}
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    high = next(entry["id"] for entry in exported["issues"] if entry["title"] == "wake word - high")
    assert workspace.order(high, ids["wake word - normal"], ids["wake word - low"], train) == [high, ids["wake word - normal"], ids["wake word - low"], train]


@real_lit
async def test_a_ticket_leveled_out_of_its_epic_takes_the_emptied_epic_off_the_page(page: Page, workspace: Workspace) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    status, body = await leveled(page, workspace, hey, "normal")
    assert status == 200 and titled(json.loads(body)["items"]) == [("wake word - normal", [hey])]
    [event] = page.events("backlog.leveled")
    assert event.counts == {"tickets": 2, "summarised": 0}


@real_lit
@pytest.mark.parametrize(
    "before_rank, ranked, said",
    [
        ('echo "error (code=5, reason=workspace_busy): the store is locked" >&2\nexit 5', "no", "and left it at the bottom of its frame, so"),
        ('{lit} "$@" >/dev/null\nkill -9 $$', "unknown", "or left it at the bottom of its frame is not known, so"),
    ],
)
async def test_a_rank_that_fails_after_the_epic_is_made_says_the_epic_is_made_and_the_ticket_not_moved(
    page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_rank: str, ranked: str, said: str
) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    workspace.shim(tmp_path / "bin", before_rank.replace("{lit}", str(LIT)))
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:/bin:/usr/bin")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 502 and body.startswith("not leveled: lit made ") and "wake word - high" in body and said in body and f"{hey} was not moved" in body
    [event] = page.events("backlog.leveled")
    assert event.outcome == "failed" and event.error == body
    assert (event.facts["created"], event.facts["ranked"], event.facts["moved"]) == ("yes", ranked, "no")
    assert str(event.facts["epic"]) in body


@real_lit
async def test_a_ticket_whose_epic_is_inside_an_archived_one_is_refused_before_lit_makes_anything(page: Page, workspace: Workspace) -> None:
    room, *_ = epic(workspace, "Room", "Keep it", topic="room")
    wake = workspace.lit("new", "--type", "epic", "--parent", room, "--title", "Train a new wake word", "--topic", "wake-word")
    hey = workspace.lit("new", "--parent", wake, "--title", "Hey hands", "--topic", "wake-word")
    # Archived, the room still frames the wake word epic's rank, and the tree draws that epic as a root.
    workspace.lit("archive", room, "--reason", "Later.")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 400 and body.startswith("not leveled: ") and "under one it archived" in body
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    assert not [entry for entry in exported["issues"] if entry["title"] == "wake word - high"]
    [event] = page.events("backlog.leveled")
    assert (event.facts["created"], event.facts["ranked"], event.facts["moved"]) == ("no", "no", "no")
    # The page is given no level for it, so it draws no buttons for it.
    response = await page.client.get("/backlog/tickets", params={"project": str(workspace.root)}, headers=KEYED)
    [shelved] = [node for node in (await response.json())["items"] if node["id"] == wake]
    assert [(child["id"], child["standing"]) for child in shelved["children"]] == [(hey, None)]


@real_lit
async def test_a_level_epic_whose_every_ticket_is_closed_is_used_again_and_never_made_twice(page: Page, workspace: Workspace) -> None:
    _, hey, custom = epic(workspace, "Train a new wake word", "Hey hands", "Custom wake word", topic="wake-word")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 200
    high = next(node["id"] for node in json.loads(body)["items"] if node["title"] == "wake word - high")
    workspace.lit("close", hey, "--resolution", "obsolete", "--reason", "Dropped.")
    status, body = await leveled(page, workspace, custom, "high")
    assert status == 200 and titled(json.loads(body)["items"]) == [("wake word - high", [custom])]
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    assert [entry["id"] for entry in exported["issues"] if entry["title"] == "wake word - high"] == [high]
    assert page.events("backlog.leveled")[1].facts["created"] == "no"


def exported_epics(workspace: Workspace) -> list[dict[str, Any]]:
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    parents = {relation["src_id"]: relation["dst_id"] for relation in exported["relations"] if relation["type"] == "parent-child"}
    return [{**entry, "parent": parents.get(entry["id"])} for entry in sorted(exported["issues"], key=lambda entry: entry["rank"]) if entry["issue_type"] == "epic"]


@real_lit
async def test_a_ticket_leveled_into_an_epic_holding_others_is_ranked_last_there(page: Page, workspace: Workspace) -> None:
    _, old, second, third = epic(workspace, "Train a new wake word", "Implement wake word mode", "Hey hands", "Many wake words", topic="wake-word")
    assert (await leveled(page, workspace, second, "high"))[0] == 200
    assert (await leveled(page, workspace, third, "high"))[0] == 200
    # The oldest ticket ranks above both, and lit keeps that rank when it moves it: it is ranked below them instead.
    status, body = await leveled(page, workspace, old, "high")
    assert status == 200 and titled(json.loads(body)["items"]) == [("wake word - high", [second, third, old])]
    assert page.events("backlog.leveled")[2].facts["placed"] == "yes"


@real_lit
async def test_a_topic_in_two_frames_has_its_own_level_epics_in_each(page: Page, workspace: Workspace) -> None:
    rooms = [workspace.lit("new", "--type", "epic", "--title", title, "--topic", topic) for title, topic in [("Kitchen", "kitchen"), ("Garage", "garage")]]
    wakes = [workspace.lit("new", "--type", "epic", "--parent", room, "--title", "Wake word", "--topic", "wake-word") for room in rooms]
    tickets = [workspace.lit("new", "--parent", wake, "--title", "Hey hands", "--topic", "wake-word") for wake in wakes]
    for ticket in tickets:
        assert (await leveled(page, workspace, ticket, "high"))[0] == 200
    highs = [entry for entry in exported_epics(workspace) if entry["title"] == "wake word - high"]
    assert sorted(entry["parent"] for entry in highs) == sorted(rooms)
    [event, again] = page.events("backlog.leveled")
    assert (event.facts["created"], again.facts["created"]) == ("yes", "yes")


@real_lit
async def test_an_epic_made_that_the_read_after_does_not_hold_where_it_was_asked_is_a_failure(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other, *_ = epic(workspace, "Other", "Keep it", topic="other")
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    # lit files the new epic inside another one than it was asked to.
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(f'#!/bin/sh\nif [ "$1" = new ]; then\nexec {LIT} "$@" --parent {other}\nfi\nexec {LIT} "$@"\n')
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 502 and body == "not leveled: lit made the epic wake word - high, and the backlog read after it does not hold it"
    [event] = page.events("backlog.leveled")
    assert (event.facts["created"], event.facts["ranked"], event.facts["moved"]) == ("yes", "no", "no") and "epic" not in event.facts


@real_lit
async def test_a_project_lit_stops_tracking_once_the_epic_is_made_is_said_as_such(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    made = tmp_path / "made"
    bin = tmp_path / "bin"
    bin.mkdir()
    (bin / "lit").write_text(
        f'#!/bin/sh\nif [ "$1" = export ] && [ -e {made} ]; then\necho "error (code=3, reason=workspace_not_initialized): no workspace" >&2\nexit 3\nfi\n'
        f'if [ "$1" = new ]; then touch {made}; fi\nexec {LIT} "$@"\n'
    )
    (bin / "lit").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin}:/bin:/usr/bin")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 502 and body == f"not leveled: lit made the epic wake word - high, and lit no longer tracks {workspace.root}"


async def noted(page: Page, workspace: Workspace, ticket: str, text: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/note", json={"project": str(workspace.root), "ticket": ticket, "text": text}, headers=KEYED)
    return response.status, await response.text()


@real_lit
async def test_a_note_is_a_lit_comment_on_the_ticket_signed_by_the_person_not_a_session(page: Page, workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    # A hands started from inside a Claude Code session carries its id, which lit would sign the note with.
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "a-session")
    assert await noted(page, workspace, hey, "  Say which word wakes it \n") == (204, "")
    shown = subprocess.run([str(LIT), "show", hey], cwd=workspace.root, check=True, capture_output=True, text=True).stdout
    assert f"[{getpass.getuser()}] Say which word wakes it" in shown
    [event] = page.events("backlog.noted")
    assert event.outcome == "ok" and (event.facts["ticket"], event.facts["chars"], event.facts["noted"]) == (hey, 23, "yes")


@real_lit
async def test_a_note_on_a_ticket_lit_does_not_hold_is_refused_with_lits_verdict(page: Page, workspace: Workspace) -> None:
    status, body = await noted(page, workspace, "page-nope", "Split it")
    assert status == 502 and body.startswith("not noted: lit comment") and "entity_not_found" in body
    [event] = page.events("backlog.noted")
    assert event.outcome == "failed" and event.facts["noted"] == "no"
