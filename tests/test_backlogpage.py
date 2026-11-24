"""The backlog page: every open ticket under its epic with hands' summary for it, a drop that is a `lit rank`, and a
level that moves a ticket into its topic's epic for it."""

import asyncio
import errno
import getpass
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import web
from aiohttp.base_protocol import BaseProtocol
from aiohttp.streams import StreamReader
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from hands.clock import SYSTEM, Clock
from hands.sessions.audit import Entry, Record
from hands.sessions.child import Children, NeverStarted, Outcome as Ended, Ran
from hands.sessions.backlog import LEVELS, LIT_TIMEOUT_SECONDS, Inherited, Level, Lit, Leveling, Move, NewEpic, Place, Standing, Untracked, comment, parse_export, parse_level
from hands.sessions.payload import Rejected
from hands.sessions.summaries import Summaries
from hands.sessions.wide import WideEvent, begun, unit
from hands.virtualclock import VirtualClock
from hands.voice.backlogpage import backlog_routes, parse_asked, parse_leveling, parse_note, parse_project, shown
from hands.voice.summaries import Sighted, SummaryStore, sighted
from conftest import SYSTEM_PATH, Faked, children_on, events, executable, exited
from litcontract import LitContract, Refusal, answered, refusing

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
LIT = shutil.which("lit")
real_lit = pytest.mark.skipif(LIT is None, reason="the order is lit's, so it is lit that is driven")


def issue(id: str, rank: str, title: str, status: str | None = "open", kind: str = "task", topic: str = "page", **extra: str) -> dict[str, object]:
    made: dict[str, object] = {"id": id, "title": title, "description": f"The words of {title}.", "issue_type": kind, "topic": topic, "rank": rank, **extra}
    return made if status is None else {**made, "status": status, **({"closed_at": "2026-10-19T10:00:00Z"} if status == "closed" else {})}


def inside(parent: str, child: str) -> dict[str, object]:
    return {"src_id": child, "dst_id": parent, "type": "parent-child", "created_at": "", "created_by": ""}


def export(issues: Sequence[Mapping[str, object]], relations: Sequence[Mapping[str, object]], comments: Sequence[Mapping[str, object]] = ()) -> bytes:
    return json.dumps({"version": 5, "workspace_id": "w", "issues": list(issues), "relations": list(relations), "comments": list(comments), "labels": [], "events": [], "purges": [], "plan_reviews": []}).encode()


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


def test_the_page_is_every_open_ticket_under_its_epic_in_rank_order_with_hands_summary_and_never_its_description() -> None:
    backlog = parse_export(export(ISSUES, RELATIONS))
    recorded: list[Entry] = []
    with unit("backlog.read", recorded.append, begun(SYSTEM), ("tickets", "summarised")):
        page = shown(Sighted(backlog, {"e1.a": "A, said.", "e2": "Second, said."}))
    assert page["tracked"] is True
    items = cast(list[dict[str, Any]], page["items"])
    assert [(node["id"], [child["id"] for child in node["children"]]) for node in items] == [("e2", ["e2.a"]), ("e1", ["e1.a", "e1.b"]), ("e2.b.f", [])]
    second, first, follow_up = items
    assert second["summary"] == "Second, said." and first["children"][0]["summary"] == "A, said."
    # A ticket with no summary has none, rather than its description in its place.
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
    with unit("backlog.read", recorded.append, begun(SYSTEM), ("tickets", "summarised")):
        assert shown(Untracked(Path("/nowhere"))) == {"tracked": False, "why": "lit has no workspace there"}
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
    refused: list[tuple[str, Level, str]] = [("wh.a", "high", "already in wh"), ("w", "low", "is an epic"), ("nope", "low", "no ticket nope"), ("wl.a", "high", "wl.a is closed")]
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
    with unit("backlog.read", lambda _entry: None, begun(SYSTEM), ("tickets", "summarised")):
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


# --- the routes, with lit faked where hands runs it


@dataclass(frozen=True)
class Outcome:
    """How the fake lit answers a write: whether it makes it, what another writer does first, the reason it fails on
    where it fails, what every export after it answers where that is not the backlog, and whether lit is gone from PATH
    from this call on."""

    writes: bool = True
    meanwhile: Callable[["FakeLit"], object] = lambda _: None
    reason: str | None = None
    exports: "Ended | Refusal | None" = None
    # Set as lit is asked; lit answers only once `answer` is.
    asked: asyncio.Event | None = None
    answer: asyncio.Event | None = None
    gone: bool = False


class FakeLit:
    """lit as `child.run` hands it to `_lit`: a backlog held in memory, exported as lit exports it, and written as lit
    writes it, each write saying what lit says, held to lit by the contract in litcontract.py; each write of a
    subcommand answers as its `Outcome` scripts it."""

    def __init__(self, backlog: "Held", outcomes: Mapping[str, Outcome], exports: "Ended | Refusal | None" = None) -> None:
        issues, relations = backlog
        self.issues = list(issues)
        self.relations = list(relations)
        self.comments: list[dict[str, object]] = []
        self.outcomes = dict(outcomes)
        self.calls: list[str] = []
        self.exports = exports
        self.gone = False

    def order(self) -> list[str]:
        return [str(each["id"]) for each in self.issues]

    def parent(self, child: str) -> str | None:
        return next((str(each["dst_id"]) for each in self.relations if each["src_id"] == child), None)

    def note(self, ticket: str, body: str, by: str) -> str:
        id = f"cmt-{len(self.comments)}"
        self.comments.append({"id": id, "issue_id": ticket, "body": body, "created_at": f"2026-11-21T10:00:{len(self.comments):02d}Z", "created_by": by})
        return id

    def children(self, clock: Clock, record: Record = lambda _: None) -> Faked:
        """What runs lit for hands, on `clock`, answered by this lit."""
        return Faked(clock, record, self.run)

    async def run(self, *argv: str, limit: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Ended:
        _, command, *rest = argv
        self.calls.append(command)
        outcome = self.outcomes.get(command, Outcome())
        self.gone |= outcome.gone
        if self.gone:
            return NeverStarted(FileNotFoundError(errno.ENOENT, "No such file or directory", "lit"))
        assert env is not None and cwd is not None, "hands runs lit in a directory and an environment it names, its trace among it"
        if command == "export":
            if self.exports is not None:
                return answered(self.exports, env, cwd, f"lit {command}")
            ranked = [{**each, "rank": f"{at:04d}"} for at, each in enumerate(self.issues)]
            return exited(0, export(ranked, self.relations, self.comments))
        if outcome.asked is not None and outcome.answer is not None:
            outcome.asked.set()
            await outcome.answer.wait()
        outcome.meanwhile(self)
        ran = self.written(command, rest) if outcome.writes else said("")
        self.exports = outcome.exports or self.exports
        match outcome.reason:
            case None:
                return answered(ran, env, cwd, f"lit {command}")
            case reason:
                return answered(Refusal(reason, f"the fake's {command} stopped"), env, cwd, f"lit {command}")

    def written(self, command: str, rest: list[str]) -> "Ended | Refusal":
        """Make the write lit is asked for, and answer as lit answers it."""
        flags = {flag: value for flag, value in zip(rest, rest[1:]) if flag.startswith("--")}
        match command, rest:
            case "rank", [ticket, side, target]:
                moving = next(each for each in self.issues if each["id"] == ticket)
                self.issues.remove(moving)
                at = self.order().index(target) + (side == "--below")
                self.issues.insert(at, moving)
                return said(f"{ticket} [open/task/page/normal] {moving['title']}")
            case "new", _:
                id, kind, parent = f"made-{len(self.issues)}", flags["--type"], flags.get("--parent")
                # At the bottom of its frame, or with --top at the top of it.
                frame = [at for at, each in enumerate(self.issues) if self.parent(str(each["id"])) == parent]
                at = frame[0] if "--top" in rest and frame else len(self.issues)
                self.issues.insert(at, issue(id, "", flags["--title"], status=None if kind == "epic" else "open", kind=kind, topic=flags["--topic"]))
                self.relations += [] if parent is None else [inside(parent, id)]
                return said(f"{id} [open/{kind}/{flags['--topic']}/normal] {flags['--title']}")
            case "parent", ["set", *_]:
                child, epic = flags["--child"], flags["--parent"]
                if self.parent(child) == epic:
                    return Refusal("state_already_holds", f"relation src={child} dst={epic} type=parent-child already exists, so there is nothing to add")
                self.relations = [each for each in self.relations if each["src_id"] != child] + [inside(epic, child)]
                return said(f"{child} child of {epic}")
            case "comment", ["add", ticket, *_]:
                return said(f"{ticket} {self.note(ticket, flags['--body'], flags['--by'])}")
            case _:
                raise AssertionError(f"the fake lit makes no {command} {rest}")


def said(words: str) -> Ran:
    """lit's answer to a write it made: its words, and the guidance line after them."""
    return exited(0, f"{words}\ndeeper guidance: lit quickstart update\n".encode())


# What lit's export answers while another lit holds its store, and in a directory it has no workspace in.
BUSY = Refusal("workspace_busy", "another lit holds the store")
UNTRACKED = Refusal("workspace_not_initialized", "no lit workspace here")


@dataclass
class Routes:
    """The page's own app, its routes resolved by its router and answered in this process, with lit faked."""

    app: web.Application
    recorded: list[Entry]
    lit: FakeLit
    project: Path

    async def post(self, path: str, **sent: str) -> tuple[int, str]:
        payload = StreamReader(BaseProtocol(asyncio.get_running_loop()), 2**16, loop=asyncio.get_running_loop())
        payload.feed_data(json.dumps({"project": str(self.project), **sent}).encode())
        payload.feed_eof()
        request = make_mocked_request("POST", path, headers=KEYED, app=self.app, payload=payload)
        match = await self.app.router.resolve(request)
        response = await match.handler(request)
        assert isinstance(response, web.Response)
        return response.status, response.text or ""

    def fact(self, name: str, fact: str) -> object:
        [event] = events(self.recorded, name)
        return event.facts[fact]

    def event(self, name: str) -> WideEvent:
        [event] = events(self.recorded, name)
        return event


def routed(tmp_path: Path, lit: FakeLit) -> Routes:
    project = tmp_path / "project"
    project.mkdir()
    recorded: list[Entry] = []
    app = web.Application()
    app.add_routes(backlog_routes(Lit(lit.children(VirtualClock()), os.environ), SummaryStore(Summaries(tmp_path / "summaries.db")), KEY, recorded.append))
    return Routes(app, recorded, lit, project)


# A backlog as lit holds it: its issues, in rank order, and its relations.
Held = tuple[list[dict[str, object]], list[dict[str, object]]]
# A drop: B ranked above A in their epic.
DROP: Held = ([issue("e", "", "Voice", status=None, kind="epic"), issue("e.a", "", "A"), issue("e.b", "", "B")], [inside("e", "e.a"), inside("e", "e.b")])
# A leveling that makes its epic: the topic has no high epic yet, so it is made above the ticket's epic, and the ticket
# is alone in it.
MADE: Held = ([issue("w", "", "Train", status=None, kind="epic", topic="wake-word"), issue("w.a", "", "Hey hands", topic="wake-word")], [inside("w", "w.a")])
# A leveling into an epic that holds another ticket, which the leveled one is ranked below.
HELD: Held = (
    [
        issue("w", "", "Train", status=None, kind="epic", topic="wake-word"),
        issue("w.a", "", "Hey hands", topic="wake-word"),
        issue("wh", "", "wake word - high", status=None, kind="epic", topic="wake-word"),
        issue("wh.a", "", "Custom word", topic="wake-word"),
    ],
    [inside("w", "w.a"), inside("wh", "wh.a")],
)


def already_inside(lit: FakeLit) -> None:
    """Another writer moves the ticket into the high epic first, so lit's own `parent set` finds it there."""
    lit.relations = [each for each in lit.relations if each["src_id"] != "w.a"] + [inside("wh", "w.a")]


@dataclass(frozen=True)
class Case:
    """One of the page's lit writes: the event it is a fact on, its step, the lit subcommand it is, the backlog it is made
    in, the request, the answer where it lands, and how lit fails it having made it all the same, with the code it exits."""

    event: str
    step: str
    command: str
    backlog: Held
    path: str
    sent: Mapping[str, str]
    status: int
    landed: Outcome
    exits: int


WRITES = [
    pytest.param(Case("backlog.ranked", "ranked", "rank", DROP, "/backlog/rank", {"ticket": "e.b", "above": "e.a"}, 200, Outcome(reason="command_failed"), 1), id="rank"),
    pytest.param(Case("backlog.leveled", "created", "new", MADE, "/backlog/level", {"ticket": "w.a", "level": "high"}, 204, Outcome(reason="command_failed"), 1), id="epic made"),
    pytest.param(Case("backlog.leveled", "ranked", "rank", MADE, "/backlog/level", {"ticket": "w.a", "level": "high"}, 204, Outcome(reason="workspace_busy"), 1), id="epic ranked"),
    # lit itself answers 6 to a parent set the backlog already holds.
    pytest.param(Case("backlog.leveled", "moved", "parent", HELD, "/backlog/level", {"ticket": "w.a", "level": "high"}, 204, Outcome(meanwhile=already_inside), 6), id="moved"),
    pytest.param(Case("backlog.leveled", "placed", "rank", HELD, "/backlog/level", {"ticket": "w.a", "level": "high"}, 204, Outcome(reason="command_failed"), 1), id="ranked last"),
    pytest.param(Case("backlog.noted", "noted", "comment", MADE, "/backlog/note", {"ticket": "w.a", "text": "Split it"}, 204, Outcome(reason="command_failed"), 1), id="note"),
]


@pytest.mark.parametrize("write", WRITES)
async def test_a_write_lit_fails_having_made_it_is_one_that_landed(tmp_path: Path, write: Case) -> None:
    routes = routed(tmp_path, FakeLit(write.backlog, {write.command: write.landed}))
    answered, body = await routes.post(write.path, **write.sent)
    assert answered == write.status, body
    assert routes.fact(write.event, write.step) == "yes" and f"exited {write.exits}" in str(routes.fact(write.event, f"{write.step}_said"))


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("refused", [Outcome(writes=False, reason="workspace_busy"), Outcome(writes=False)], ids=["failed", "exited 0"])
async def test_a_write_the_backlog_after_does_not_have_is_not_done_whatever_lit_exited(tmp_path: Path, write: Case, refused: Outcome) -> None:
    routes = routed(tmp_path, FakeLit(write.backlog, {write.command: refused}))
    answered, body = await routes.post(write.path, **write.sent)
    assert answered == 409 and "the backlog lit holds after does not have" in body
    assert ("workspace_busy" in body) == (refused.reason == "workspace_busy")
    event = routes.event(write.event)
    assert event.outcome == "failed" and event.error == body and event.facts[write.step] == "no"


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize(("exports", "unseen"), [(BUSY, "could not be read: lit export"), (UNTRACKED, "lit no longer tracks")], ids=["unreadable", "untracked"])
async def test_a_write_whose_backlog_cannot_be_read_again_is_not_known(tmp_path: Path, write: Case, exports: Ran, unseen: str) -> None:
    routes = routed(tmp_path, FakeLit(write.backlog, {write.command: Outcome(reason="command_failed", exports=exports)}))
    answered, body = await routes.post(write.path, **write.sent)
    # lit may have written, so the answer says what is not known and never that nothing was done.
    assert answered == 502 and body.startswith("whether lit holds ") and "is not known" in body and "command_failed" in body and unseen in body
    event = routes.event(write.event)
    assert event.outcome == "failed" and event.error == body and event.facts[write.step] == "unknown"


@pytest.mark.parametrize("write", WRITES)
async def test_a_write_whose_request_is_cancelled_while_lit_is_asked_is_not_known(tmp_path: Path, write: Case) -> None:
    asked = asyncio.Event()
    routes = routed(tmp_path, FakeLit(write.backlog, {write.command: Outcome(asked=asked, answer=asyncio.Event())}))
    request = asyncio.create_task(routes.post(write.path, **write.sent))
    await asked.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    # lit may have written before the request ended, and no read after it settled whether it did.
    assert routes.fact(write.event, write.step) == "unknown"


@pytest.mark.parametrize("write", WRITES)
async def test_a_write_lit_cannot_be_started_for_is_not_done_and_no_read_is_made_after_it(tmp_path: Path, write: Case) -> None:
    lit = FakeLit(write.backlog, {write.command: Outcome(gone=True)})
    routes = routed(tmp_path, lit)
    answered, body = await routes.post(write.path, **write.sent)
    assert answered == 502 and "lit could not be started" in body and "the backlog" not in body
    assert lit.calls[-1] == write.command, f"a read was made after a write no lit ran for: {lit.calls}"
    assert routes.fact(write.event, write.step) == "no"


async def test_a_drop_the_page_cannot_send_is_refused_before_lit_is_asked(tmp_path: Path) -> None:
    lit = FakeLit(DROP, {})
    routes = routed(tmp_path, lit)
    status, body = await routes.post("/backlog/rank", ticket="e.b")
    assert status == 400 and body.startswith("not moved: a move is ") and lit.calls == []


async def test_a_drop_in_a_project_lit_does_not_track_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    lit = FakeLit(DROP, {}, exports=UNTRACKED)
    routes = routed(tmp_path, lit)
    assert await routes.post("/backlog/rank", ticket="e.b", above="e.a") == (404, f"not moved: {routes.project} has no backlog: lit has no workspace there")
    assert lit.calls == ["export"] and "ranked" not in routes.event("backlog.ranked").facts


def unranked_target(lit: FakeLit) -> None:
    """Another writer deletes the epic the made epic is to be ranked above."""
    lit.issues = [each for each in lit.issues if each["id"] != "w"]


async def test_an_epic_made_that_cannot_be_ranked_where_asked_is_said_made_at_the_bottom_of_its_frame(tmp_path: Path) -> None:
    lit = FakeLit(MADE, {"new": Outcome(meanwhile=unranked_target)})
    routes = routed(tmp_path, lit)
    status, body = await routes.post("/backlog/level", ticket="w.a", level="high")
    event = routes.event("backlog.leveled")
    assert (status, body) == (409, f"lit made {event.facts['epic']}, wake word - high, at the bottom of its frame, and it cannot be ranked above w: the backlog has no ticket w")
    assert event.facts["created"] == "yes" and not {"ranked", "moved", "placed"} & event.facts.keys() and lit.calls == ["export", "new", "export"]


async def test_a_drop_reads_the_backlog_once_after_its_rank_and_answers_with_it(tmp_path: Path) -> None:
    lit = FakeLit(DROP, {})
    routes = routed(tmp_path, lit)
    status, body = await routes.post("/backlog/rank", ticket="e.b", above="e.a")
    assert status == 200 and [child["id"] for child in json.loads(body)["items"][0]["children"]] == ["e.b", "e.a"]
    assert lit.calls == ["export", "rank", "export"]
    assert routes.fact("backlog.ranked", "ranked") == "yes" and str(routes.fact("backlog.ranked", "ranked_said")).startswith("lit exited 0: e.b ")


async def test_a_leveling_reads_the_backlog_once_after_each_write_and_says_a_rank_it_did_not_need(tmp_path: Path) -> None:
    lit = FakeLit(MADE, {})
    routes = routed(tmp_path, lit)
    assert await routes.post("/backlog/level", ticket="w.a", level="high") == (204, "")
    assert lit.calls == ["export", "new", "export", "rank", "export", "parent", "export"]
    event = routes.event("backlog.leveled")
    assert (event.facts["created"], event.facts["ranked"], event.facts["moved"], event.facts["placed"]) == ("yes", "yes", "yes", "unneeded")
    # The high epic made is ranked directly above the ticket's epic, and the ticket is alone in it.
    assert lit.order().index("made-2") == lit.order().index("w") - 1
    assert lit.parent("w.a") == "made-2" and event.facts["epic"] == "made-2"


async def test_a_leveling_into_an_epic_lit_holds_makes_none_and_ranks_the_ticket_last_there(tmp_path: Path) -> None:
    lit = FakeLit(HELD, {})
    routes = routed(tmp_path, lit)
    assert await routes.post("/backlog/level", ticket="w.a", level="high") == (204, "")
    event = routes.event("backlog.leveled")
    assert (event.facts["created"], event.facts["ranked"], event.facts["moved"], event.facts["placed"]) == ("unneeded", "unneeded", "yes", "yes")
    assert lit.parent("w.a") == "wh" and lit.order().index("wh.a") < lit.order().index("w.a")


async def test_a_ticket_lit_moves_in_last_among_its_epics_open_tickets_needs_no_rank(tmp_path: Path) -> None:
    # The ticket ranks below the high epic's one ticket, and lit keeps that rank when it moves it.
    below: Held = (
        [issue("wh", "", "wake word - high", status=None, kind="epic", topic="wake-word"), issue("wh.a", "", "Custom word", topic="wake-word"), issue("w", "", "Train", status=None, kind="epic", topic="wake-word"), issue("w.a", "", "Hey hands", topic="wake-word")],
        [inside("wh", "wh.a"), inside("w", "w.a")],
    )
    lit = FakeLit(below, {})
    routes = routed(tmp_path, lit)
    assert await routes.post("/backlog/level", ticket="w.a", level="high") == (204, "")
    assert lit.calls == ["export", "parent", "export"] and routes.fact("backlog.leveled", "placed") == "unneeded"


def filed_last_in_high(lit: FakeLit) -> None:
    """Another writer files an open ticket at the bottom of the high epic."""
    lit.issues.append(issue("wh.z", "", "Filed meanwhile", topic="wake-word"))
    lit.relations.append(inside("wh", "wh.z"))


async def test_a_ticket_ranked_below_the_last_and_not_last_after_is_not_placed(tmp_path: Path) -> None:
    lit = FakeLit(HELD, {"rank": Outcome(meanwhile=filed_last_in_high)})
    routes = routed(tmp_path, lit)
    status, body = await routes.post("/backlog/level", ticket="w.a", level="high")
    # Directly below the ticket that was last, and not last there.
    assert lit.order()[lit.order().index("wh.a") + 1] == "w.a"
    assert status == 409 and body.startswith("the backlog lit holds after does not have w.a last in wh; lit exited 0: ")
    assert routes.fact("backlog.leveled", "placed") == "no"


async def test_a_note_on_a_ticket_lit_does_not_hold_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    lit = FakeLit(MADE, {})
    routes = routed(tmp_path, lit)
    status, body = await routes.post("/backlog/note", ticket="page-nope", text="Split it")
    assert (status, body) == (404, f"not noted: lit holds no ticket page-nope in {routes.project}")
    assert lit.calls == ["export"] and "noted" not in routes.event("backlog.noted").facts


async def test_a_note_is_kept_only_where_the_page_signed_the_comment_that_says_it(tmp_path: Path) -> None:
    # lit refuses the page's comment while a session leaves the same words on the ticket.
    lit = FakeLit(MADE, {"comment": Outcome(writes=False, meanwhile=lambda lit: lit.note("w.a", "Split it", "claude_a-session"), reason="workspace_busy")})
    routes = routed(tmp_path, lit)
    status, body = await routes.post("/backlog/note", ticket="w.a", text="Split it")
    assert status == 409 and body.startswith("the backlog lit holds after does not have the note on w.a; ") and routes.fact("backlog.noted", "noted") == "no"


# --- the routes, against lit itself


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    store: SummaryStore
    recorded: list[Entry]
    clock: VirtualClock

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b"{", "a note is JSON"),
        (b'{"project": "/tmp", "ticket": "a"}', "a note is {project, ticket, and text"),
        (b'{"project": "/tmp", "ticket": "a", "text": "  \\n "}', "not blank"),
        (b'{"project": "/tmp", "ticket": "a", "text": "a\\u0000b"}', "holding a NUL"),
        (b'{"project": "/tmp", "ticket": "a\\u0000", "text": "ok"}', "holding a NUL"),
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
    store = SummaryStore(Summaries(tmp_path / "summaries.db"))
    app = web.Application()
    # [LAW:no-ambient-temporal-coupling] a clock nobody moves unless a test does, so no lit the page runs is ended by the machine being slow.
    clock = VirtualClock()
    # The page's lit is the one in the test's own bin, lit itself until a test stubs it (`stub_lit`), and none once a
    # stub removes itself: no other lit is on the page's PATH.
    unstub_lit(tmp_path)
    app.add_routes(backlog_routes(Lit(Children(clock, recorded.append), {**os.environ, "PATH": f"{tmp_path / 'bin'}:{SYSTEM_PATH}"}), store, KEY, recorded.append))
    client = TestClient(TestServer(app))
    await client.start_server()
    yield Page(client, store, recorded, clock)
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


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([str(LIT), "init"], cwd=root, check=True, capture_output=True)
    return Workspace(root)


def unstub_lit(tmp_path: Path) -> None:
    """Make the page's lit lit itself again."""
    (tmp_path / "bin").mkdir(exist_ok=True)
    executable(tmp_path / "bin" / "lit", f'#!/bin/sh\nexec {LIT} "$@"\n')


def stub_lit(tmp_path: Path, **before: str) -> None:
    """Make the page's lit one that runs each script in `before` ahead of the lit subcommand it is named for, then runs lit itself."""
    stub = tmp_path / "bin" / "lit"
    assert stub.read_text() == f'#!/bin/sh\nexec {LIT} "$@"\n', "a second stub lit would replace the first without a word"
    branches = "".join(f'if [ "$1" = {subcommand} ]; then\n{script}\nfi\n' for subcommand, script in before.items())
    executable(stub, f'#!/bin/sh\n{branches}exec {LIT} "$@"\n')


def epic(workspace: Workspace, title: str, *children: str, topic: str = "page") -> tuple[str, ...]:
    made = workspace.lit("new", "--type", "epic", "--title", title, "--topic", topic)
    return (made, *(workspace.lit("new", "--parent", made, "--title", child, "--topic", topic) for child in children))


async def moved(page: Page, workspace: Workspace, ticket: str, **place: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/rank", json={"project": str(workspace.root), "ticket": ticket, **place}, headers=KEYED)
    return response.status, await response.text()


async def leveled(page: Page, workspace: Workspace, ticket: str, level: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/level", json={"project": str(workspace.root), "ticket": ticket, "level": level}, headers=KEYED)
    return response.status, await response.text()


async def level(page: Page, workspace: Workspace, ticket: str, level: str) -> list[dict[str, Any]]:
    """The roots the page is given once `ticket` is put at `level`: the level is answered with nothing, since the page
    drew the press when it was made, and the backlog is read again."""
    assert await leveled(page, workspace, ticket, level) == (204, "")
    response = await page.client.get("/backlog/tickets", params={"project": str(workspace.root)}, headers=KEYED)
    return (await response.json())["items"]


def titled(items: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Each root the page is given, by title, over its children's ids."""
    return [(node["title"], [child["id"] for child in node["children"]]) for node in items]


@pytest.mark.outside
async def test_the_page_is_served_to_anyone_and_its_backlog_to_the_phones_key_alone(page: Page) -> None:
    shown_page = await page.client.get("/backlog")
    assert shown_page.status == 200 and "drag its card up or down" in await shown_page.text()
    assert (await page.client.get("/backlog/tickets?project=/tmp")).status == 401
    assert (await page.client.post("/backlog/rank", json={})).status == 401
    assert (await page.client.post("/backlog/level", json={})).status == 401
    assert (await page.client.post("/backlog/note", json={})).status == 401
    keyless = page.events("backlog.read") + page.events("backlog.ranked") + page.events("backlog.leveled") + page.events("backlog.noted")
    assert [event.error for event in keyless] == ["asked without the phone's key"] * 4


@pytest.mark.outside
@pytest.mark.parametrize(
    "query, why",
    [("", "absolute path"), ("?project=relative/path", "absolute path"), ("?project=/no/such/directory", "no directory"), ("?project=/a%00b", "is no path")],
)
async def test_a_read_that_names_no_directory_is_refused_rather_than_asking_lit(page: Page, query: str, why: str) -> None:
    response = await page.client.get(f"/backlog/tickets{query}", headers=KEYED)
    assert response.status == 400 and why in await response.text()


@pytest.mark.outside
@real_lit
async def test_a_directory_lit_has_no_workspace_in_is_drawn_as_untracked(page: Page, tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    response = await page.client.get("/backlog/tickets", params={"project": str(tmp_path)}, headers=KEYED)
    assert response.status == 200 and await response.json() == {"tracked": False, "why": "lit has no workspace there"}
    [event] = page.events("backlog.read")
    assert event.outcome == "ok" and event.facts["tracked"] is False


@pytest.mark.outside
@real_lit
async def test_the_read_is_lits_backlog_with_hands_summaries_and_asks_for_the_rest(page: Page, workspace: Workspace) -> None:
    voice, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    read = await sighted(Lit(children_on(page.clock), os.environ), page.store, workspace.root)
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


@pytest.mark.outside
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


@pytest.mark.outside
@real_lit
async def test_epics_are_dragged_among_themselves(page: Page, workspace: Workspace) -> None:
    voice, *_ = epic(workspace, "Voice", "Fix the mic")
    wire, *_ = epic(workspace, "Wire", "Tap it")
    status, _ = await moved(page, workspace, wire, above=voice)
    assert status == 200 and workspace.order(voice, wire) == [wire, voice]


@pytest.mark.outside
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


@pytest.mark.outside
@real_lit
async def test_a_deleted_epics_child_is_moved_among_the_roots(page: Page, workspace: Workspace) -> None:
    root, *_ = epic(workspace, "Root", "Keep it")
    gone, left = epic(workspace, "Gone", "Left behind")
    other, *_ = epic(workspace, "Other", "Another")
    workspace.lit("delete", gone, "--reason", "Dropped.")
    status, _ = await moved(page, workspace, left, below=root)
    assert status == 200 and workspace.order(root, left, other) == [root, left, other]


@pytest.mark.outside
@real_lit
async def test_an_archived_epics_child_is_not_moved_among_the_roots(page: Page, workspace: Workspace) -> None:
    root, *_ = epic(workspace, "Root", "Keep it")
    shelf, shelved = epic(workspace, "Shelf", "Shelved")
    workspace.lit("archive", shelf, "--reason", "Later.")
    before = workspace.order(root, shelf, shelved)
    status, body = await moved(page, workspace, shelved, above=root)
    assert status == 400 and "not inside the same parent" in body
    assert workspace.order(root, shelf, shelved) == before


@pytest.mark.outside
@real_lit
async def test_a_move_lit_makes_otherwise_because_the_backlog_changed_meanwhile_is_a_conflict(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    voice, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    wire, *_ = epic(workspace, "Wire", "Tap it")
    # An agent puts the parser in the Wire epic between the page's read and its rank: lit ranks the Wire epic instead.
    stub_lit(tmp_path, rank=f"{LIT} parent set --child {parser} --parent {wire} >/dev/null")
    status, body = await moved(page, workspace, parser, above=mic)
    # What lit ranked instead is said in lit's own words, to the page and on the event; nothing is undone.
    assert status == 409 and f"does not have {parser} above {mic}" in body and f"ranked the epic {wire} instead" in body
    assert workspace.order(voice, wire) == [wire, voice]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == body
    assert event.facts["ranked"] == "no" and f"ranked the epic {wire} instead" in str(event.facts["ranked_said"])


@pytest.mark.outside
@real_lit
async def test_a_rank_lit_fails_is_a_failure_carrying_lits_verdict(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    stub_lit(tmp_path, rank=refusing("workspace_busy", "the store is locked"))
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 409 and body.startswith(f"the backlog lit holds after does not have {parser} above {mic}; lit rank") and "workspace_busy" in body
    assert workspace.order(mic, parser) == [mic, parser]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error is not None and "workspace_busy" in event.error and event.facts["ranked"] == "no"


@pytest.mark.outside
async def test_a_lit_that_does_not_answer_in_its_bound_fails_the_read(page: Page, tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    stub_lit(tmp_path, export="exec sleep 600")
    read = asyncio.create_task(page.client.get("/backlog/tickets", params={"project": str(project)}, headers=KEYED))
    # The bound is waited out on the page's clock, so it comes due as the test moves the clock to it and no sooner.
    waiting = asyncio.create_task(page.clock.sleeping(page.clock.at + LIT_TIMEOUT_SECONDS))
    await asyncio.wait({read, waiting}, return_when=asyncio.FIRST_COMPLETED)
    waiting.cancel()
    assert not read.done(), f"the read ended before lit's bound was waited on: {read}"
    page.clock.advance(LIT_TIMEOUT_SECONDS)
    response = await read
    assert response.status == 502 and "was killed with SIGKILL, because it ran past" in await response.text()
    [event] = page.events("backlog.read")
    assert event.outcome == "failed" and event.error is not None and "was killed with SIGKILL, because it ran past" in event.error


@pytest.mark.outside
@real_lit
async def test_a_rank_in_a_project_lit_then_no_longer_tracks_is_not_known_and_says_why(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    # lit ranks, and the workspace is gone before the read after.
    stub_lit(tmp_path, rank=f'{LIT} "$@" || exit\nrm -rf .git\nexit 0')
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and body.startswith(f"whether lit holds {parser} above {mic} is not known; lit exited 0: ") and body.endswith(f"; and lit no longer tracks {workspace.root}: lit has no workspace there")
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == body and event.facts["ranked"] == "unknown"


@pytest.mark.outside
@real_lit
async def test_a_rank_whose_backlog_could_not_be_read_again_is_not_known(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    ranked = tmp_path / "ranked"
    # lit ranks, and every export after it fails.
    stub_lit(tmp_path, export=f'if [ -e {ranked} ]; then\n{refusing("workspace_busy", "the store is locked")}\nfi', rank=f"touch {ranked}")
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and body.startswith(f"whether lit holds {parser} above {mic} is not known; lit exited 0: ") and f"the backlog in {workspace.root} could not be read" in body and "workspace_busy" in body
    assert workspace.order(mic, parser) == [parser, mic]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == body and event.facts["ranked"] == "unknown"


@pytest.mark.outside
@real_lit
async def test_a_rank_lit_is_killed_on_after_writing_it_is_a_drop_that_landed(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    # lit writes the rank, then is killed before it can say so.
    stub_lit(tmp_path, rank=f'{LIT} "$@" >/dev/null\nkill -9 $$')
    status, _ = await moved(page, workspace, parser, above=mic)
    assert status == 200 and workspace.order(mic, parser) == [parser, mic]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "ok" and event.facts["ranked"] == "yes" and "was ended by SIGKILL" in str(event.facts["ranked_said"])


@pytest.mark.outside
@real_lit
@pytest.mark.parametrize("export, why", [(refusing("workspace_busy", "locked"), "workspace_busy"), ("echo not json", "JSON")])
async def test_a_read_lit_cannot_give_is_a_failure_on_the_page_and_its_event(page: Page, workspace: Workspace, tmp_path: Path, export: str, why: str) -> None:
    stub_lit(tmp_path, export=export)
    response = await page.client.get("/backlog/tickets", params={"project": str(workspace.root)}, headers=KEYED)
    body = await response.text()
    assert response.status == 502 and body.startswith(f"the backlog in {workspace.root} could not be read") and why in body
    [event] = page.events("backlog.read")
    assert event.outcome == "failed" and event.error is not None and why in event.error


@pytest.mark.outside
@pytest.mark.parametrize("body, why", [(b"{}", "one of above or below"), (b'{"project": "/", "ticket": "a", "above": "b", "below": "c"}', "never both")])
async def test_a_move_the_page_cannot_send_is_refused_and_its_event_says_why(page: Page, body: bytes, why: str) -> None:
    response = await page.client.post("/backlog/rank", data=body, headers=KEYED)
    text = await response.text()
    assert response.status == 400 and text.startswith("not moved: ") and why in text
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.error == text


@pytest.mark.outside
@real_lit
async def test_a_read_asks_for_a_pass_only_while_something_is_unsaid(page: Page, workspace: Workspace) -> None:
    epic(workspace, "Voice", "Fix the mic")
    read = await sighted(Lit(children_on(page.clock), os.environ), page.store, workspace.root)
    assert isinstance(read, Sighted)
    assert await page.store.wanted()
    # Every ticket said, round by round, as the summarising task says them.
    while due := page.store.reckon(read.backlog.thing()).due:
        page.store.keep({each.digest: f"Said of {each.id}." for each in due})
    again = await sighted(Lit(children_on(page.clock), os.environ), page.store, workspace.root)
    assert isinstance(again, Sighted) and len(again.said) == 3
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(page.store.wanted(), 0.2)


@pytest.mark.outside
@real_lit
async def test_a_level_moves_the_ticket_into_its_topics_epic_for_it_made_beside_its_epic_the_first_time(page: Page, workspace: Workspace) -> None:
    train, hey, custom, many = epic(workspace, "Train a new wake word", "Implement wake word mode", "Hey hands", "Many wake words", topic="wake-word")
    items = await level(page, workspace, hey, "high")
    [high] = [node for node in items if node["title"] == "wake word - high"]
    # Made directly above the epic the ticket was in, and the ticket moved into it with its id.
    assert titled(items) == [("wake word - high", [hey]), ("Train a new wake word", [custom, many])]
    assert high["children"][0]["standing"] == {"epic": high["id"], "level": "high"}
    items = await level(page, workspace, custom, "high")
    # The same epic, not a second.
    assert titled(items) == [("wake word - high", [hey, custom]), ("Train a new wake word", [many])]
    items = await level(page, workspace, hey, "low")
    # Made directly below the epic the ticket is in now: its topic's high one.
    assert titled(items) == [("wake word - high", [custom]), ("wake word - low", [hey]), ("Train a new wake word", [many])]
    low = next(node["id"] for node in items if node["title"] == "wake word - low")
    assert workspace.order(high["id"], low, train) == [high["id"], low, train]
    first, second, third = page.events("backlog.leveled")
    assert first.outcome == "ok" and (first.facts["ticket"], first.facts["level"], first.facts["epic"], first.facts["created"], first.facts["moved"]) == (hey, "high", high["id"], "yes", "yes")
    assert second.outcome == "ok" and (second.facts["ticket"], second.facts["epic"], second.facts["created"]) == (custom, high["id"], "unneeded")
    # Alone in a new epic, or ranked below the ticket already there, it is last where it joins and needs no rank.
    assert (first.facts["placed"], second.facts["placed"]) == ("unneeded", "unneeded")
    assert third.outcome == "ok" and (third.facts["level"], third.facts["epic"], third.facts["created"]) == ("low", low, "yes")


@pytest.mark.outside
@real_lit
@pytest.mark.quarantined("promptctl-testing-c3u.m9b", until="2026-11-26")
async def test_a_level_the_ticket_is_at_already_is_refused_and_nothing_in_lit_moves(page: Page, workspace: Workspace) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    await level(page, workspace, hey, "normal")
    before = workspace.order(hey)
    status, body = await leveled(page, workspace, hey, "normal")
    assert status == 400 and body.startswith("not leveled: ") and "already in" in body
    assert workspace.order(hey) == before
    assert [event.outcome for event in page.events("backlog.leveled")] == ["ok", "failed"]


@pytest.mark.outside
@real_lit
async def test_a_move_lit_refuses_after_making_the_epic_says_the_epic_was_made_and_the_ticket_was_not_moved(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    stub_lit(tmp_path, parent=refusing("workspace_busy", "the store is locked"))
    status, body = await leveled(page, workspace, hey, "high")
    [event] = page.events("backlog.leveled")
    assert status == 409 and body.startswith(f"the backlog lit holds after does not have {hey} in {event.facts['epic']}; lit parent") and "workspace_busy" in body
    assert event.outcome == "failed" and (event.facts["created"], event.facts["moved"]) == ("yes", "no")
    # The epic stands made and empty, and the next press puts the ticket in it rather than making another.
    unstub_lit(tmp_path)
    items = await level(page, workspace, hey, "high")
    # The epic it was in holds nothing open now, and is drawn no more.
    assert titled(items) == [("wake word - high", [hey])]
    assert page.events("backlog.leveled")[1].facts["created"] == "unneeded"


@pytest.mark.outside
@real_lit
async def test_a_move_lit_is_killed_on_after_writing_it_finishes_the_leveling(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    stub_lit(tmp_path, parent=f'{LIT} "$@" >/dev/null\nkill -9 $$')
    assert await leveled(page, workspace, hey, "low") == (204, "")
    [event] = page.events("backlog.leveled")
    assert event.outcome == "ok" and (event.facts["created"], event.facts["moved"], event.facts["placed"]) == ("yes", "yes", "unneeded")


@pytest.mark.outside
@real_lit
async def test_a_level_epic_made_keeps_its_topics_levels_in_order_whatever_epic_the_ticket_is_in(page: Page, workspace: Workspace) -> None:
    _, first, second, third = epic(workspace, "Train a new wake word", "Hey hands", "Many wake words", "Train your own", topic="wake-word")
    await level(page, workspace, first, "low")
    # From the low epic, normal goes above it, not below.
    items = await level(page, workspace, first, "normal")
    assert [title for title, _ in titled(items)] == ["Train a new wake word", "wake word - normal"]
    await level(page, workspace, second, "low")
    # From the topic's own epic, high goes above the normal one, not above the topic's epic.
    items = await level(page, workspace, third, "high")
    titles = [title for title, _ in titled(items)]
    assert titles == ["wake word - high", "wake word - normal", "wake word - low"]
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    epics = [entry["title"] for entry in sorted(exported["issues"], key=lambda entry: entry["rank"]) if entry["issue_type"] == "epic"]
    assert epics == ["Train a new wake word", "wake word - high", "wake word - normal", "wake word - low"]


@pytest.mark.outside
@real_lit
async def test_a_level_epic_made_from_a_topics_high_one_goes_between_it_and_the_topics_epic(page: Page, workspace: Workspace) -> None:
    train, first, second = epic(workspace, "Train a new wake word", "Hey hands", "Many wake words", topic="wake-word")
    await level(page, workspace, first, "high")
    await level(page, workspace, second, "low")
    items = await level(page, workspace, first, "normal")
    assert [title for title, _ in titled(items)] == ["wake word - normal", "wake word - low"]
    ids = {node["title"]: node["id"] for node in items}
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    high = next(entry["id"] for entry in exported["issues"] if entry["title"] == "wake word - high")
    assert workspace.order(high, ids["wake word - normal"], ids["wake word - low"], train) == [high, ids["wake word - normal"], ids["wake word - low"], train]


@pytest.mark.outside
@real_lit
async def test_a_ticket_leveled_out_of_its_epic_takes_the_emptied_epic_off_the_page(page: Page, workspace: Workspace) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    items = await level(page, workspace, hey, "normal")
    assert titled(items) == [("wake word - normal", [hey])]
    [event] = page.events("backlog.leveled")
    assert event.outcome == "ok" and event.counts == {}


@pytest.mark.outside
@real_lit
@pytest.mark.parametrize(
    "before_rank, status, ranked, moved",
    [
        (refusing("workspace_busy", "the store is locked"), 409, "no", None),
        (f'{LIT} "$@" >/dev/null\nkill -9 $$', 204, "yes", "yes"),
    ],
    ids=["refused", "killed after writing"],
)
async def test_a_rank_lit_fails_after_the_epic_is_made_is_settled_by_the_backlog_after_it(
    page: Page, workspace: Workspace, tmp_path: Path, before_rank: str, status: int, ranked: str, moved: str | None
) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    stub_lit(tmp_path, rank=before_rank)
    answered, body = await leveled(page, workspace, hey, "high")
    [event] = page.events("backlog.leveled")
    assert answered == status, body
    # The move is never asked of lit where the epic's rank does not hold.
    assert (event.facts["created"], event.facts["ranked"], event.facts.get("moved")) == ("yes", ranked, moved)
    assert status == 204 or body.startswith(f"the backlog lit holds after does not have {event.facts['epic']} above ")


@pytest.mark.outside
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
    assert not {"created", "ranked", "moved", "placed"} & event.facts.keys()
    # The page is given no level for it, so it draws no buttons for it.
    response = await page.client.get("/backlog/tickets", params={"project": str(workspace.root)}, headers=KEYED)
    [shelved] = [node for node in (await response.json())["items"] if node["id"] == wake]
    assert [(child["id"], child["standing"]) for child in shelved["children"]] == [(hey, None)]


@pytest.mark.outside
@real_lit
async def test_a_level_epic_whose_every_ticket_is_closed_is_used_again_and_never_made_twice(page: Page, workspace: Workspace) -> None:
    _, hey, custom = epic(workspace, "Train a new wake word", "Hey hands", "Custom wake word", topic="wake-word")
    items = await level(page, workspace, hey, "high")
    high = next(node["id"] for node in items if node["title"] == "wake word - high")
    workspace.lit("close", hey, "--resolution", "obsolete", "--reason", "Dropped.")
    items = await level(page, workspace, custom, "high")
    assert titled(items) == [("wake word - high", [custom])]
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    assert [entry["id"] for entry in exported["issues"] if entry["title"] == "wake word - high"] == [high]
    assert page.events("backlog.leveled")[1].facts["created"] == "unneeded"


def exported_epics(workspace: Workspace) -> list[dict[str, Any]]:
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    parents = {relation["src_id"]: relation["dst_id"] for relation in exported["relations"] if relation["type"] == "parent-child"}
    return [{**entry, "parent": parents.get(entry["id"])} for entry in sorted(exported["issues"], key=lambda entry: entry["rank"]) if entry["issue_type"] == "epic"]


@pytest.mark.outside
@real_lit
async def test_a_ticket_leveled_into_an_epic_holding_others_is_ranked_last_there(page: Page, workspace: Workspace) -> None:
    _, old, second, third = epic(workspace, "Train a new wake word", "Implement wake word mode", "Hey hands", "Many wake words", topic="wake-word")
    await level(page, workspace, second, "high")
    await level(page, workspace, third, "high")
    # The oldest ticket ranks above both, and lit keeps that rank when it moves it: it is ranked below them instead.
    items = await level(page, workspace, old, "high")
    assert titled(items) == [("wake word - high", [second, third, old])]
    assert page.events("backlog.leveled")[2].facts["placed"] == "yes"


@pytest.mark.outside
@real_lit
async def test_a_topic_in_two_frames_has_its_own_level_epics_in_each(page: Page, workspace: Workspace) -> None:
    rooms = [workspace.lit("new", "--type", "epic", "--title", title, "--topic", topic) for title, topic in [("Kitchen", "kitchen"), ("Garage", "garage")]]
    wakes = [workspace.lit("new", "--type", "epic", "--parent", room, "--title", "Wake word", "--topic", "wake-word") for room in rooms]
    tickets = [workspace.lit("new", "--parent", wake, "--title", "Hey hands", "--topic", "wake-word") for wake in wakes]
    for ticket in tickets:
        await level(page, workspace, ticket, "high")
    highs = [entry for entry in exported_epics(workspace) if entry["title"] == "wake word - high"]
    assert sorted(entry["parent"] for entry in highs) == sorted(rooms)
    [event, again] = page.events("backlog.leveled")
    assert (event.facts["created"], again.facts["created"]) == ("yes", "yes")


@pytest.mark.outside
@real_lit
async def test_an_epic_made_that_the_read_after_does_not_hold_where_it_was_asked_is_a_failure(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    other, *_ = epic(workspace, "Other", "Keep it", topic="other")
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    # lit files the new epic inside another one than it was asked to.
    stub_lit(tmp_path, new=f'exec {LIT} "$@" --parent {other}')
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 409 and body.startswith("the backlog lit holds after does not have the epic wake word - high; lit exited 0: ")
    [event] = page.events("backlog.leveled")
    assert event.facts["created"] == "no" and not {"ranked", "moved"} & event.facts.keys() and "epic" not in event.facts


@pytest.mark.outside
@real_lit
async def test_a_project_lit_stops_tracking_once_the_epic_is_made_is_said_as_such(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    made = tmp_path / "made"
    stub_lit(tmp_path, export=f'if [ -e {made} ]; then\n{refusing("workspace_not_initialized", "no workspace")}\nfi', new=f"touch {made}")
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 502 and body.startswith("whether lit holds the epic wake word - high is not known; lit exited 0: ") and body.endswith(f"; and lit no longer tracks {workspace.root}: lit has no workspace there")
    [event] = page.events("backlog.leveled")
    assert event.facts["created"] == "unknown"


async def noted(page: Page, workspace: Workspace, ticket: str, text: str) -> tuple[int, str]:
    response = await page.client.post("/backlog/note", json={"project": str(workspace.root), "ticket": ticket, "text": text}, headers=KEYED)
    return response.status, await response.text()


@pytest.mark.outside
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


@pytest.mark.outside
@real_lit
async def test_a_note_on_a_ticket_lit_does_not_hold_is_refused_before_lit_is_asked_to_keep_it(page: Page, workspace: Workspace) -> None:
    status, body = await noted(page, workspace, "page-nope", "Split it")
    assert (status, body) == (404, f"not noted: lit holds no ticket page-nope in {workspace.root}")
    [event] = page.events("backlog.noted")
    assert event.outcome == "failed" and "noted" not in event.facts


@pytest.mark.outside
@real_lit
async def test_a_note_for_a_flag_is_refused_before_lit_is_asked(page: Page, workspace: Workspace) -> None:
    # lit would read it as its own flag, and `--help` exits 0 keeping nothing; no ticket lit holds is one.
    assert await noted(page, workspace, "--help", "Split it") == (404, f"not noted: lit holds no ticket --help in {workspace.root}")
    [event] = page.events("backlog.noted")
    assert event.outcome == "failed" and "noted" not in event.facts


def actors(workspace: Workspace) -> list[tuple[str, str]]:
    """Each change lit recorded, as what it did and who it says did it."""
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    return [(event["action"], event["actor"]) for event in exported["events"]]


@pytest.mark.outside
@real_lit
async def test_a_level_press_is_signed_by_the_person_even_inside_a_session(page: Page, workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    before = len(actors(workspace))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", str(uuid.uuid4()))
    await level(page, workspace, hey, "high")
    # The epic it made and the move into it are the person's.
    made = actors(workspace)[before:]
    assert made and {actor for _, actor in made} == {getpass.getuser()}


@pytest.mark.outside
@real_lit
async def test_a_comment_signed_by_no_one_is_the_sessions_hands_runs_in(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    session = str(uuid.uuid4())
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", session)
    # `hands file-tickets` comments as the session it is run from.
    await comment(Lit(children_on(VirtualClock()), os.environ), workspace.root, hey, "Seen again.", Inherited())
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    [said] = [each for each in exported["comments"] if each["body"] == "Seen again."]
    assert session in said["created_by"] and getpass.getuser() not in said["created_by"]


@pytest.mark.outside
@real_lit
async def test_a_note_lit_is_killed_on_after_keeping_it_is_kept(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    # lit keeps the comment, then dies before it can say so.
    stub_lit(tmp_path, comment=f'{LIT} "$@" >/dev/null\nkill -9 $$')
    assert await noted(page, workspace, hey, "Split it") == (204, "")
    [event] = page.events("backlog.noted")
    assert event.outcome == "ok" and event.facts["noted"] == "yes"


def lit_gone_after_export(tmp_path: Path) -> None:
    """lit answers the page's read of the backlog and is gone from PATH for whatever the page asks it next."""
    stub_lit(tmp_path, export='rm -f "$0"')


@pytest.mark.outside
@real_lit
async def test_a_note_whose_lit_is_gone_for_the_write_says_nothing_was_noted(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    lit_gone_after_export(tmp_path)
    status, body = await noted(page, workspace, hey, "Split it")
    assert status == 502 and body.startswith(f"not noted: lit could not be started in {workspace.root}: ")
    [event] = page.events("backlog.noted")
    assert event.outcome == "failed" and event.facts["noted"] == "no"


@pytest.mark.outside
@real_lit
async def test_a_rank_whose_lit_is_gone_for_the_write_says_nothing_was_moved(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    lit_gone_after_export(tmp_path)
    status, body = await moved(page, workspace, parser, above=mic)
    assert status == 502 and body.startswith(f"not moved: lit could not be started in {workspace.root}: ")
    assert workspace.order(mic, parser) == [mic, parser]
    [event] = page.events("backlog.ranked")
    assert event.outcome == "failed" and event.facts["ranked"] == "no"


@pytest.mark.outside
@real_lit
async def test_a_level_whose_lit_is_gone_for_its_first_write_says_nothing_was_leveled(page: Page, workspace: Workspace, tmp_path: Path) -> None:
    _, hey = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    lit_gone_after_export(tmp_path)
    status, body = await leveled(page, workspace, hey, "high")
    assert status == 502 and body.startswith(f"not leveled: lit could not be started in {workspace.root}: ")
    [event] = page.events("backlog.leveled")
    assert event.outcome == "failed" and event.facts["created"] == "no" and "moved" not in event.facts


@pytest.mark.outside
@real_lit
async def test_a_drop_from_the_page_ranks_outside_the_session_hands_was_started_in(page: Page, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, mic, parser = epic(workspace, "Voice", "Fix the mic", "Fix the parser")
    seen = tmp_path / "seen"
    # lit rank takes no --by, and still runs in the person's environment, not the session's.
    stub_lit(tmp_path, rank=f'echo "session=$CLAUDE_CODE_SESSION_ID" > {seen}')
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", str(uuid.uuid4()))
    status, _ = await moved(page, workspace, parser, above=mic)
    assert status == 200 and seen.read_text() == "session=\n"


@pytest.mark.outside
@real_lit
async def test_a_note_saved_while_a_ticket_is_leveled_waits_its_turn_and_is_kept(page: Page, workspace: Workspace) -> None:
    _, hey, custom = epic(workspace, "Train a new wake word", "Hey hands", "Custom wake word", topic="wake-word")
    leveling, noting = await asyncio.gather(leveled(page, workspace, hey, "high"), noted(page, workspace, custom, "Say which word"))
    assert (leveling, noting) == ((204, ""), (204, ""))
    shown = subprocess.run([str(LIT), "show", custom], cwd=workspace.root, check=True, capture_output=True, text=True).stdout
    assert "Say which word" in shown


# --- what hands relies on of lit, held to the fake and to lit (litcontract.py)


class TestTheFakeLit(LitContract):
    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.fixture
    def children(self) -> Children:
        return FakeLit(([], []), {}).children(VirtualClock())


@pytest.mark.outside
@real_lit
class TestLit(LitContract):
    @pytest.fixture
    def project(self, workspace: Workspace) -> Path:
        return workspace.root

    @pytest.fixture
    def children(self) -> Children:
        return children_on(VirtualClock())
