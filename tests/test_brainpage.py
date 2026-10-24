"""The brain on the phone's page: a read of what it is doing that waits for that to change, and one Stop that names the
turn it was shown."""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.core.brainwork import AlreadyEnded, BrainWork, Epoch, Idle, Running, StopOutcome, Stopping, Working
from hands.core.wire import Unreadable
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import longpoll
from hands.voice.brainpage import brain_routes
from conftest import events

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
SLEEPING = Working("turn-1", Epoch(1000.0), (Running("Bash", {"command": "sleep 30", "description": "Wait"}, Epoch(1002.0)),))


@dataclass
class Brain:
    """The brain as the page sees it: its work, as the test sets it, and each turn a Stop named."""

    work_now: BrainWork = field(default_factory=Idle)
    stopped: list[str] = field(default_factory=list[str])

    def work(self) -> BrainWork:
        return self.work_now

    def stop(self, turn: str) -> StopOutcome:
        self.stopped.append(turn)
        match self.work_now:
            case Working(turn=shown) if shown == turn:
                return Stopping()
            case _:
                return AlreadyEnded()


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    brain: Brain
    recorded: list[Entry]

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


@pytest.fixture
async def page(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Page]:
    # A read that has seen the work looks again soon and gives up soon, so a test waits on neither.
    monkeypatch.setattr(longpoll, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(longpoll, "WAIT_SECONDS", 0.5)
    brain = Brain()
    recorded: list[Entry] = []
    app = web.Application()
    app.add_routes(brain_routes(brain, KEY, recorded.append))
    async with TestClient(TestServer(app)) as client:
        yield Page(client, brain, recorded)


async def test_the_read_shows_the_call_the_brain_runs_with_its_input_and_the_macs_clock(page: Page) -> None:
    page.brain.work_now = SLEEPING
    answered = await page.client.get("/brain", headers=KEYED)
    body = await answered.json()
    assert body["work"] == {
        "doing": "working", "turn": "turn-1", "since": 1000.0,
        "calls": [{"tool": "Bash", "since": 1002.0, "input": {"command": "sleep 30", "description": "Wait"}}],
    }
    assert isinstance(body["now"], float) and isinstance(body["seen"], str)
    [read] = page.events("brain.read")
    assert (read.outcome, read.facts["doing"], read.counts["calls"]) == ("ok", "Working", 1)


async def test_an_input_that_could_not_be_read_is_shown_as_such(page: Page) -> None:
    page.brain.work_now = Working("turn-1", Epoch(1000.0), (Running("Bash", Unreadable("tool call Bash's input is not JSON"), Epoch(1002.0)),))
    body = await (await page.client.get("/brain", headers=KEYED)).json()
    assert body["work"]["calls"] == [{"tool": "Bash", "since": 1002.0, "unreadable": "tool call Bash's input is not JSON"}]


async def test_a_read_that_has_seen_the_work_answers_once_the_call_ends(page: Page) -> None:
    page.brain.work_now = SLEEPING
    seen = (await (await page.client.get("/brain", headers=KEYED)).json())["seen"]
    waiting = asyncio.ensure_future(page.client.get(f"/brain?seen={seen}", headers=KEYED))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    page.brain.work_now = Working("turn-1", Epoch(1000.0), ())
    body = await (await waiting).json()
    assert body["work"]["calls"] == [] and body["seen"] != seen
    _, read = page.events("brain.read")
    assert read.facts["changed"] is True and read.counts["calls"] == 0


async def test_stop_names_the_turn_it_was_shown_and_says_what_came_of_it(page: Page) -> None:
    page.brain.work_now = SLEEPING
    answered = await page.client.post("/brain/stop", json={"turn": "turn-1"}, headers=KEYED)
    assert (answered.status, await answered.json()) == (200, {"readback": "Stopping the brain's turn."})
    late = await page.client.post("/brain/stop", json={"turn": "turn-0"}, headers=KEYED)
    assert await late.json() == {"readback": "That turn had already ended."}
    assert page.brain.stopped == ["turn-1", "turn-0"]
    assert [(stop.outcome, stop.facts["turn"], stop.facts["outcome"]) for stop in page.events("brain.stop")] == [
        ("ok", "turn-1", "Stopping"),
        ("ok", "turn-0", "AlreadyEnded"),
    ]


async def test_a_stop_that_names_no_turn_is_refused_and_stops_nothing(page: Page) -> None:
    for body in ("not json", json.dumps({}), json.dumps({"turn": ""})):
        refused = await page.client.post("/brain/stop", data=body, headers=KEYED)
        assert refused.status == 400
    assert page.brain.stopped == []
    assert [stop.outcome for stop in page.events("brain.stop")] == ["failed"] * 3


async def test_neither_route_answers_without_the_phones_key(page: Page) -> None:
    page.brain.work_now = SLEEPING
    assert (await page.client.get("/brain")).status == 401
    assert (await page.client.post("/brain/stop", json={"turn": "turn-1"})).status == 401
    assert page.brain.stopped == []
