"""Each phone page's own code, run in a headless Chrome (tests/browser.py) against the routes hands serves the phone
(hands.voice.phonepage.phone_pages) on the route tests' fakes: a page opened with its key, a person's choice made on it,
and what the page shows of it read off the page."""

import json
import os
import subprocess
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pytest
from aiohttp.test_utils import TestServer

from hands.clock import SYSTEM
from hands.core.brainwork import StopOutcome
from hands.sessions.audit import AuditLog, Entry, Replied, Transcribed
from hands.sessions.home import Home
from hands.sessions.summaries import Summaries
from hands.virtualclock import VirtualClock
from hands.voice import longpoll
from hands.voice.gallerypage import Gallery
from hands.voice.phone import Phone
from hands.voice.phoneaddress import phone_key
from hands.voice.phonepage import phone_pages
from hands.voice.ptt import PushToTalk
from hands.voice.summaries import SummaryStore
from hands.voice.tool import tool
from hands.voice.tools import audited
from hands.voice.voices import VOICES
from hands.voice.wakeword import Pretrained
from browser import Browser, Tab, chrome
from test_backlogpage import LIT, a_workspace, epic, real_lit
from test_brainpage import SLEEPING, Brain
from test_conversationpage import a_log, set_trigger
from test_gallerypage import MODELS, Desk, Listening, Own, Speaker, Voicing

pytestmark = pytest.mark.outside


@dataclass
class Served:
    """The pages as hands serves them, on fakes the test sets and reads, and the browser that opens them."""

    server: TestServer
    key: str
    browser: Browser
    brain: Brain
    log: AuditLog
    voices: Voicing
    typed: list[str]

    def address(self, path: str) -> str:
        """`path` as the phone opens it: its address carrying the phone's key in the fragment."""
        return f"{self.server.make_url(path)}#{self.key}"


@pytest.fixture
async def served(tmp_path: Path) -> AsyncGenerator[Served]:
    recorded: list[Entry] = []
    phone = Phone(SYSTEM, PushToTalk(SYSTEM, lambda _: None), heard_rate=16000, played_rate=24000, record=recorded.append)
    brain, voices = Brain(work_now=SLEEPING), Voicing()
    gallery = Gallery(voices=voices, levels=Listening(), desk=Desk(), own=Own(), speaker=Speaker(VirtualClock()), wake=Pretrained(), models=MODELS)
    typed: list[str] = []

    async def typing(text: str) -> None:
        typed.append(text)

    home = Home(tmp_path / "hands")
    store = SummaryStore(Summaries(home.summaries))
    server = TestServer(phone_pages(SYSTEM, phone, brain, typing, store, home, recorded.append, os.environ, gallery), host="127.0.0.1")
    await server.start_server()
    try:
        async with chrome(tmp_path / "chrome") as browser:
            yield Served(server, phone_key(SYSTEM, home), browser, brain, a_log(home.audit), voices, typed)
    finally:
        await server.close()
        await phone.stop()


async def shown(tab: Tab, selector: str, text: str) -> None:
    """Once the element `selector` finds says `text`."""
    await tab.until(f"document.querySelector({json.dumps(selector)})?.textContent.includes({json.dumps(text)})")


async def test_the_phone_page_draws_the_brains_call_and_a_stop_ends_the_turn_it_shows(served: Served) -> None:
    async with served.browser.tab() as tab:
        await tab.open(served.address("/"))
        await shown(tab, "#calls", "sleep 30")
        await shown(tab, "#doing", "Running Bash")
        await tab.until("!document.getElementById('stop').disabled")
        await tab.click("#stop")
        await shown(tab, "#brain-said", "Stopping")
        assert served.brain.stopped == ["turn-1"]


async def test_a_stop_hands_refuses_says_it_was_not_stopped(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    def refusing(turn: str) -> StopOutcome:
        raise RuntimeError(f"no stopping {turn}")

    monkeypatch.setattr(served.brain, "stop", refusing)
    async with served.browser.tab() as tab:
        await tab.open(served.address("/"))
        await tab.until("!document.getElementById('stop').disabled")
        await tab.click("#stop")
        await shown(tab, "#brain-said", "Could not stop it: 500")


async def test_a_stop_whose_answer_is_lost_says_whether_it_stopped_is_not_known(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    # hands closes once the page's read waiting at it is answered, so the read waits only briefly.
    monkeypatch.setattr(longpoll, "WAIT_SECONDS", 0.5)
    async with served.browser.tab() as tab:
        await tab.open(served.address("/"))
        await tab.until("!document.getElementById('stop').disabled")
        await served.server.close()
        await tab.click("#stop")
        await shown(tab, "#brain-said", "Whether the brain stopped is not known: hands is out of reach")


async def test_a_page_whose_phone_another_page_took_says_so_and_calls_no_more(served: Served) -> None:
    async with served.browser.tab() as first, served.browser.tab() as second:
        for tab in (first, second):
            await tab.open(served.address("/"))
        await first.click("#talk")
        await shown(first, "#state", "Connected")
        await second.click("#talk")
        await shown(second, "#state", "Connected")
        # The first page's call dropped, and its call made again is declined: the second page has the phone.
        await shown(first, "#state", "Another page has hands' phone")
        await shown(first, "#talk", "Connect")


async def test_the_conversation_page_sends_what_is_typed_and_keeps_an_opened_call_open_as_it_is_redrawn(served: Served) -> None:
    served.log.record(Transcribed("switch to the wake word"))
    await audited(SYSTEM, tool(set_trigger), served.log.record).body(trigger="wake word")
    async with served.browser.tab() as tab:
        await tab.open(served.address("/conversation"))
        await shown(tab, "#moments", "called set_trigger")
        await tab.click("#moments summary")
        served.log.record(Replied("Okay. Say Hey Jarvis, then what you want.", False))
        await shown(tab, "#moments", "Say Hey Jarvis")
        assert await tab.evaluate("document.querySelector('#moments details').open")
        await tab.type("#text", "what time is it")
        await tab.click("#send")
        await tab.until("document.getElementById('text').value === ''")
        assert served.typed == ["what time is it"]


async def test_the_gallery_page_draws_the_voices_and_speaks_in_the_one_used(served: Served) -> None:
    async with served.browser.tab() as tab:
        await tab.open(served.address("/gallery"))
        await shown(tab, "#speaking", "hands speaks in Charles.")
        row = "#voices > li:nth-child(2)"
        name = await tab.evaluate(f"document.querySelector('{row} > .name').textContent")
        await tab.click(f"{row} > button:last-of-type")
        await shown(tab, "#status", f"Speaking in {name} from now on.")
        await shown(tab, "#speaking", f"hands speaks in {name}.")
        assert served.voices.used_asked == [VOICES[1]]


@real_lit
async def test_the_backlog_page_draws_a_ticket_under_its_epic_and_lit_keeps_the_note_made_on_it(served: Served, tmp_path: Path) -> None:
    workspace = a_workspace(tmp_path / "project")
    made, ticket = epic(workspace, "Phone", "Stop")
    project = workspace.root
    async with served.browser.tab() as tab:
        await tab.open(served.address(f"/backlog?project={quote(str(project))}"))
        await shown(tab, "#tree", "Phone")
        await tab.click(f'li[data-id="{made}"] > .card')
        await tab.type(f'textarea[aria-label="Note on {ticket}"]', "Split it")
        await tab.click(f'li[data-id="{ticket}"] .noting button')
        await tab.until(f'document.querySelector(\'textarea[aria-label="Note on {ticket}"]\')?.placeholder === "Noted: Split it"')
        exported = json.loads(subprocess.run([str(LIT), "export"], cwd=project, check=True, capture_output=True).stdout)
        assert [comment["body"] for comment in exported["comments"] if comment["issue_id"] == ticket] == ["Split it"]
