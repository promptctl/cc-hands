"""The sessions on the phone's page: each live session as the brain's listing says it, a Stop that presses Escape at the
one session it is beside, and a read that waits for what a row shows to change."""

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.core.effects import Fritter, Input, Key, Type
from hands.core.events import Displayed, Joined, Prompted, StatusReported
from hands.core.session import Membership, PromptId, SessionId
from hands.core.status import Busy, Idle, Report, Stamp
from hands.core.tmux import Keyboard, NotInTmux
from hands.sessions.audit import Entry
from hands.sessions.registry import Sessions
from hands.sessions.wide import WideEvent
from hands.voice import longpoll, sessionspage
from hands.voice.sessionspage import session_routes
from conftest import events

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
WORKING, RESTING = SessionId("s1"), SessionId("s2")
TURN = PromptId("p1")


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    sessions: Sessions
    socket: Path
    # What the routes recorded, and what was typed into each session's keyboard.
    recorded: list[Entry] = field(default_factory=list[Entry])
    typed: list[Type[Input]] = field(default_factory=list[Type[Input]])

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


async def no_pane(pids: Sequence[int]) -> list[Keyboard]:
    return [NotInTmux()] * len(pids)


@pytest.fixture
async def page(tmp_path: Path) -> AsyncGenerator[Page]:
    typed: list[Type[Input]] = []

    async def typist(effect: Type[Input]) -> None:
        typed.append(effect)

    sessions = Sessions(clock=lambda: 0.0, record=lambda _: None, typist=typist, keyboards=no_pane)
    socket = tmp_path / "f.sock"
    # Both under fritter, so either could be typed into; only the one Stop is beside ever is.
    await sessions.apply(Joined(Membership(WORKING, 4242, Path("/code/cc-hands"), tmp_path / "s1.jsonl", socket), "startup"))
    await sessions.apply(Joined(Membership(RESTING, 4343, Path("/code/lit"), tmp_path / "s2.jsonl", socket), "startup"))
    await sessions.apply(Prompted(WORKING, at=1.0, mode=None, prompt=TURN))
    await sessions.apply(StatusReported(WORKING, Report(Busy(), Stamp(1)), 1.0))
    await sessions.apply(StatusReported(RESTING, Report(Idle(), Stamp(1)), 1.0))
    recorded: list[Entry] = []
    app = web.Application()
    app.add_routes(session_routes(sessions, KEY, recorded.append))
    client = TestClient(TestServer(app))
    await client.start_server()
    yield Page(client, sessions, socket, recorded, typed)
    await client.close()


async def read(page: Page, seen: str | None = None) -> dict[str, object]:
    answered = await page.client.get("/sessions", params={} if seen is None else {"seen": seen}, headers=KEYED)
    assert answered.status == 200
    return await answered.json()


async def stop(page: Page, session: str) -> dict[str, object]:
    answered = await page.client.post(f"/sessions/{session}/interrupt", headers=KEYED)
    assert answered.status == 200
    return await answered.json()


async def test_the_sessions_are_read_as_the_brain_lists_them(page: Page) -> None:
    body = await read(page)
    assert body["sessions"] == [
        {"id": "s1", "name": "cc-hands", "state": "working", "mode": "not reported yet"},
        {"id": "s2", "name": "lit", "state": "idle", "mode": "not reported yet"},
    ]
    [event] = page.events("sessions.read")
    assert event.outcome == "ok" and event.counts == {"sessions": 2}
    assert (event.facts["seen"], event.facts["looks"], event.facts["changed"]) == (None, 1, True)


async def test_stop_presses_escape_at_the_one_session_it_is_beside_and_says_so(page: Page) -> None:
    assert await stop(page, WORKING) == {"readback": "Typed Escape into cc-hands."}
    assert page.typed == [Type(WORKING, Fritter(page.socket, 4242), Key("escape"))]
    [event] = page.events("sessions.interrupt")
    assert event.outcome == "ok" and event.facts["session"] == "s1" and event.facts["readback"] == "Typed Escape into cc-hands."


async def test_stop_at_a_session_at_its_prompt_presses_nothing_and_says_there_was_nothing_to_stop(page: Page) -> None:
    assert await stop(page, RESTING) == {"readback": "lit is at its prompt, so there is nothing to interrupt."}
    assert await stop(page, "gone") == {"readback": "There is no session gone."}
    assert page.typed == []


async def test_a_stop_hands_refuses_is_said_and_fails_its_event(page: Page, monkeypatch: pytest.MonkeyPatch) -> None:
    async def refused(_sessions: Sessions, _session: str) -> dict[str, object]:
        return {"error": "the keyboard could not be read"}

    monkeypatch.setattr(sessionspage, "interrupt", refused)
    assert await stop(page, WORKING) == {"error": "the keyboard could not be read"}
    [event] = page.events("sessions.interrupt")
    assert event.outcome == "failed" and event.error == "the keyboard could not be read"


@pytest.mark.parametrize(("method", "path"), [("GET", "/sessions"), ("POST", "/sessions/s1/interrupt")])
async def test_a_request_without_the_phones_key_is_refused_and_presses_nothing(page: Page, method: str, path: str) -> None:
    refused = await page.client.request(method, path)
    assert refused.status == 401
    assert page.typed == []
    [event] = [entry for entry in page.recorded if isinstance(entry, WideEvent)]
    assert event.outcome == "failed"


async def test_a_read_that_has_seen_the_rows_waits_through_streamed_text_and_answers_once_the_stopped_session_is_idle(page: Page) -> None:
    first = await read(page)
    waiting = asyncio.create_task(read(page, str(first["seen"])))
    before = page.sessions.live_sessions()
    await page.sessions.apply(Displayed(WORKING, (TURN,), "A line Claude wrote.\n", at=2.0))
    # The text moved the registry, and moved nothing a row shows.
    assert page.sessions.live_sessions() != before
    await asyncio.sleep(longpoll.POLL_SECONDS * 3)
    assert not waiting.done()
    await page.sessions.apply(StatusReported(WORKING, Report(Idle(), Stamp(2)), 2.0))
    body = await waiting
    assert [session["state"] for session in cast(list[dict[str, str]], body["sessions"])] == ["idle", "idle"]
    assert body["seen"] != first["seen"]
    _, waited = page.events("sessions.read")
    assert isinstance(looks := waited.facts["looks"], int) and looks > 3


async def test_a_read_that_has_seen_the_rows_answers_once_a_session_is_renamed(page: Page, tmp_path: Path) -> None:
    first = await read(page)
    waiting = asyncio.create_task(read(page, str(first["seen"])))
    await asyncio.sleep(longpoll.POLL_SECONDS * 3)
    assert not waiting.done()
    (tmp_path / "s2.jsonl").write_text(json.dumps({"type": "custom-title", "customTitle": "the backlog"}, separators=(",", ":")) + "\n")
    body = await waiting
    assert [session["name"] for session in cast(list[dict[str, str]], body["sessions"])] == ["cc-hands", "lit, the backlog"]
    _, woken = page.events("sessions.read")
    assert woken.facts["seen"] == first["seen"] and woken.facts["changed"] is True


async def test_a_read_whose_rows_never_change_is_answered_with_them_as_they_are(page: Page, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(longpoll, "WAIT_SECONDS", 0.3)
    first = await read(page)
    again = await read(page, str(first["seen"]))
    assert again == first
    _, waited = page.events("sessions.read")
    assert isinstance(ms := waited.facts["waited_ms"], float) and ms >= 300
    assert waited.facts["changed"] is False
