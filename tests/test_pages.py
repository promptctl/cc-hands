"""Each phone page's own code, run in the installed Google Chrome, driven by Playwright, against the routes hands serves
the phone (hands.voice.phonepage.phone_pages) on the route tests' fakes: a page opened with the phone's key, a choice made
on it as a person makes it, and what the page shows of it read off the page."""

import json
import os
import subprocess
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit

import pytest
from aiohttp.test_utils import TestServer
from playwright.async_api import Browser, ConsoleMessage, Page, Route, async_playwright, expect

from hands.clock import SYSTEM
from hands.core.brainwork import StopOutcome
from hands.sessions.audit import AuditLog, Entry, Replied, Transcribed
from hands.sessions.home import Home
from hands.sessions.summaries import Summaries
from hands.virtualclock import VirtualClock
from hands.voice.gallerypage import Gallery
from hands.voice.phone import Phone
from hands.voice.phoneaddress import Untailed, page_urls, phone_key
from hands.voice.phonepage import phone_pages
from hands.voice.ptt import PushToTalk
from hands.voice.summaries import SummaryStore
from hands.voice.tool import tool
from hands.voice.tools import audited
from hands.voice.voices import VOICES
from hands.voice.wakeword import Pretrained
from test_backlogpage import LIT, Workspace, epic, real_lit
from test_backlogpage import workspace as workspace  # the fixture the backlog test takes, a lit workspace of its own
from test_brainpage import SLEEPING, Brain
from test_conversationpage import a_log, set_trigger
from test_gallerypage import MODELS, Desk, Listening, Own, Speaker, Voicing

pytestmark = pytest.mark.outside
# How long what a test waits on in a page is given: a page draws what hands answers within a second or two, and a call
# that drops is made again within a few.
expect.set_options(timeout=10_000)


@dataclass
class Served:
    """The pages as hands serves them, on fakes the test sets and reads, and the browser that opens them; what every page
    opened in it threw."""

    server: TestServer
    key: str
    browser: Browser
    brain: Brain
    log: AuditLog
    voices: Voicing
    typed: list[str]
    thrown: list[str] = field(default_factory=list[str])

    async def opened(self, path: str) -> Page:
        """The page at `path`, opened as the phone opens it: its address carrying the phone's key in the fragment
        `phoneaddress.page_urls` gives it."""
        page = await self.browser.new_page()
        page.on("pageerror", lambda error: self.thrown.append(f"{page.url}: {error}"))
        page.on("console", lambda message: self._said(page, message))
        [phones] = page_urls(Untailed("served on a test's own port"), ["127.0.0.1"], self.key)
        await page.goto(f"{self.server.make_url(path)}#{urlsplit(phones).fragment}")
        return page

    def _said(self, page: Page, message: ConsoleMessage) -> None:
        # A console.error of a page's own script carries what it was called with; a page's own fault reaches `pageerror`,
        # thrown or reported. Chrome's own line for each fetch answered with a 4xx or 5xx carries nothing, and the pages
        # are answered so on purpose: a 409 when another page takes the phone, a 500 from a hands that failed.
        if message.type == "error" and message.args:
            self.thrown.append(f"{page.url}: {message.text}")


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
    server = TestServer(phone_pages(SYSTEM, phone, brain, typing, SummaryStore(Summaries(home.summaries)), home, recorded.append, os.environ, gallery), host="127.0.0.1")
    await server.start_server()
    try:
        async with async_playwright() as playwright:
            # A missing Chrome fails the launch, and the test with it: the pages are run in nothing else.
            browser = await playwright.chromium.launch(channel="chrome", args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])
            pages = Served(server, phone_key(SYSTEM, home), browser, brain, a_log(home.audit), voices, typed)
            yield pages
    finally:
        await server.close()
        await phone.stop()
    # [LAW:single-enforcer] the one check that no page threw, for every test, run whether the test passed or failed: on
    # one that failed it is reported beside that failure.
    assert pages.thrown == []


async def test_the_phone_page_draws_the_brains_call_and_a_stop_ends_the_turn_it_shows(served: Served) -> None:
    page = await served.opened("/")
    await expect(page.locator("#calls")).to_contain_text("sleep 30")
    await expect(page.locator("#doing")).to_contain_text("Running Bash")
    await page.click("#stop")
    await expect(page.locator("#brain-said")).to_contain_text("Stopping")
    assert served.brain.stopped == ["turn-1"]


async def test_a_stop_hands_fails_at_says_it_could_not_stop_it(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(turn: str) -> StopOutcome:
        raise RuntimeError(f"the stage could not stop {turn}")

    monkeypatch.setattr(served.brain, "stop", failing)
    page = await served.opened("/")
    await page.click("#stop")
    await expect(page.locator("#brain-said")).to_have_text("Could not stop it: hands failed at brain.stop: RuntimeError: the stage could not stop turn-1")


async def test_a_stop_hands_took_whose_answer_is_lost_says_whether_the_brain_stopped_is_not_known(served: Served) -> None:
    page = await served.opened("/")

    async def lost(route: Route) -> None:
        # hands takes the Stop, and its answer never reaches the page.
        await route.fetch()
        await route.abort("connectionreset")

    await page.route("**/brain/stop", lost)
    await page.click("#stop")
    await expect(page.locator("#brain-said")).to_contain_text("Whether the brain stopped is not known: hands is out of reach")
    assert served.brain.stopped == ["turn-1"]


async def unreadable(route: Route) -> None:
    """hands' answer as no page reads it: JSON, and none of what any route answers."""
    await route.fulfill(status=200, content_type="application/json", body="null")


async def test_a_page_whose_own_code_throws_fails_the_check_every_page_test_ends_with(served: Served) -> None:
    page = await served.opened("/")
    await expect(page.locator("#doing")).to_contain_text("Running Bash")
    await page.route("**/brain/stop", unreadable)
    async with page.expect_event("pageerror"):
        await page.click("#stop")
    [thrown] = served.thrown
    assert "Cannot read properties of null (reading 'readback')" in thrown
    # What the check would fail this test on, seen: the test passes on it.
    served.thrown.clear()


async def test_a_page_that_reports_a_fault_of_its_own_fails_the_check_every_page_test_ends_with(served: Served) -> None:
    page = await served.opened("/")
    await page.route("**/brain", unreadable)
    async with page.expect_event("pageerror"):
        await page.reload()
    [thrown] = served.thrown
    assert "hands' answer is not a read" in thrown
    served.thrown.clear()


async def test_a_page_whose_phone_another_page_took_says_so_and_calls_no_more(served: Served) -> None:
    first, second = await served.opened("/"), await served.opened("/")
    await first.click("#talk")
    await expect(first.locator("#state")).to_contain_text("Connected")
    await second.click("#talk")
    await expect(second.locator("#state")).to_contain_text("Connected")
    # The first page's call dropped, and its call made again is declined: the second page has the phone.
    await expect(first.locator("#state")).to_contain_text("Another page has hands' phone")
    await expect(first.locator("#talk")).to_have_text("Connect")


async def test_the_conversation_page_sends_what_is_typed_and_keeps_an_opened_call_open_as_it_is_redrawn(served: Served) -> None:
    served.log.record(Transcribed("switch to the wake word"))
    await audited(SYSTEM, tool(set_trigger), served.log.record).body(trigger="wake word")
    page = await served.opened("/conversation")
    moments = page.locator("#moments")
    await expect(moments).to_contain_text("called set_trigger")
    await moments.locator("summary").click()
    served.log.record(Replied("Okay. Say Hey Jarvis, then what you want.", False))
    await expect(moments).to_contain_text("Say Hey Jarvis")
    await expect(moments.locator("details")).to_have_attribute("open", "")
    await page.fill("#text", "what time is it")
    await page.click("#send")
    await expect(page.locator("#text")).to_have_value("")
    assert served.typed == ["what time is it"]


async def test_the_gallery_page_draws_the_voices_and_speaks_in_the_one_used(served: Served) -> None:
    page = await served.opened("/gallery")
    await expect(page.locator("#speaking")).to_contain_text("hands speaks in Charles.")
    row = page.locator("#voices > li").nth(1)
    name = await row.locator(".name").text_content()
    await row.get_by_role("button", name="Use").click()
    await expect(page.locator("#status")).to_contain_text(f"Speaking in {name} from now on.")
    await expect(page.locator("#speaking")).to_contain_text(f"hands speaks in {name}.")
    assert served.voices.used_asked == [VOICES[1]]


@real_lit
async def test_the_backlog_page_draws_a_ticket_under_its_epic_and_lit_keeps_the_note_made_on_it(served: Served, workspace: Workspace) -> None:
    made, ticket = epic(workspace, "Phone", "Stop")
    page = await served.opened(f"/backlog?project={quote(str(workspace.root))}")
    await expect(page.locator("#tree")).to_contain_text("Phone")
    await page.locator(f'li[data-id="{made}"] > .card').click()
    note = page.get_by_label(f"Note on {ticket}")
    await note.fill("Split it")
    await page.locator(f'li[data-id="{ticket}"] .noting').get_by_role("button", name="Save").click()
    await expect(note).to_have_attribute("placeholder", "Noted: Split it")
    exported = json.loads(subprocess.run([str(LIT), "export"], cwd=workspace.root, check=True, capture_output=True).stdout)
    assert [comment["body"] for comment in exported["comments"] if comment["issue_id"] == ticket] == ["Split it"]
