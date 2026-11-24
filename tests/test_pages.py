"""The phone's pages, each its own code run in the installed Google Chrome against the routes hands serves the phone, on
the route tests' fakes: a choice made on each page as a person makes it, and what the page shows of it."""

import base64
import os
import subprocess
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import BaseTestServer, TestServer
from playwright.async_api import Browser, BrowserContext, ConsoleMessage, Page, Route, async_playwright, expect

from hands.clock import SYSTEM
from hands.sessions.audit import AuditLog, Replied, Transcribed
from hands.sessions.home import Home
from hands.sessions.summaries import Summaries
from hands.virtualclock import VirtualClock
from hands.voice.clips import KEPT_SECONDS
from hands.voice.gallerypage import Gallery
from hands.voice.keyed import UNKEYED, Handler
from hands.voice.ortweb import Runtime
from hands.voice.phone import Phone
from hands.voice.phoneaddress import keyed_address, phone_key
from hands.voice.phonepage import phone_pages
from hands.voice.phonewake import MODELS as WAKE_MODELS
from hands.voice.ptt import PushToTalk
from hands.voice.summaries import SummaryStore
from hands.voice.tool import tool
from hands.voice.tools import audited
from hands.voice.voicechoice import VOICES, Voice
from hands.voice.wake import THRESHOLD, Models, WakeWord
from hands.voice.wakeword import Pretrained
from speech import RATE, quiet, spoken
from test_backlogpage import LIT, Workspace, epic, real_lit, workspace  # pyright: ignore[reportUnusedImport]  (a fixture, found by its name)
from test_brainpage import SLEEPING, Brain
from test_conversationpage import a_log, set_trigger
from test_gallerypage import MODELS, Desk, Listening, Own, Speaker, Voicing

pytestmark = pytest.mark.outside

# A microphone Chrome makes up, granted without asking: the phone's call takes one, and no test has a person to allow it.
FAKE_MICROPHONE = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]

# What starts every line a page's own code says with console.error, set before any of the page's code runs: Chrome
# says its own lines, a failed fetch's among them, on the same console.
FAULT_MARK = "page fault:"
MARKED = f"const said = console.error; console.error = (...args) => said({FAULT_MARK!r}, ...args);"


@dataclass
class Hands:
    """hands as its pages reach it: the routes it serves the phone, on a port of the test's own, and the fakes behind them."""

    server: BaseTestServer
    key: str
    phone: Phone
    brain: Brain
    typed: list[str]
    log: AuditLog
    voices: Voicing
    # The path of every request hands took, in the order it took them.
    requests: list[str]

    def address(self, path: str, **query: str) -> str:
        """`path` on this server, carrying the phone's key as `hands phone` prints it."""
        return keyed_address(str(self.server.make_url(path).with_query(query)), self.key)


@pytest.fixture
async def hands(tmp_path: Path, models: Path, onnx_runtime: Path) -> AsyncGenerator[Hands]:
    home = Home(tmp_path / "hands")
    key = phone_key(SYSTEM, home)
    phone = Phone(SYSTEM, PushToTalk(SYSTEM, lambda _: None), heard_rate=16000, played_rate=24000, record=lambda _: None)
    brain, typed, voices = Brain(), list[str](), Voicing()

    async def typing(text: str) -> None:
        typed.append(text)

    gallery = Gallery(voices=voices, levels=Listening(), desk=Desk(), own=Own(), speaker=Speaker(VirtualClock()), wake=Pretrained(), models=MODELS)
    # The environment hands was started in, as it stood: one a test changes after does not reach the page's writes.
    pages = phone_pages(SYSTEM, phone, brain, typing, SummaryStore(Summaries(home.summaries)), home.audit, key, lambda _: None, dict(os.environ), gallery, Models(models), Runtime(onnx_runtime))
    requests = list[str]()

    @web.middleware
    async def taken(request: web.Request, handler: Handler) -> web.StreamResponse:
        requests.append(request.path)
        return await handler(request)

    pages.middlewares.append(taken)
    async with TestServer(pages) as server:
        yield Hands(server, key, phone, brain, typed, a_log(home.audit), voices, requests)
        await phone.stop()


@dataclass
class Opened:
    """The pages a test opened, and every fault their own code made: an error it never caught, or one it said with
    console.error."""

    context: BrowserContext
    faults: list[str] = field(default_factory=list[str])

    @classmethod
    async def made(cls, chrome: Browser) -> "Opened":
        return await cls.watching(await chrome.new_context())

    @classmethod
    async def watching(cls, context: BrowserContext) -> "Opened":
        opened = cls(context)
        await opened.context.add_init_script(MARKED)
        # Every page the context opens, those a page's link opens among them.
        opened.context.on("page", opened._watched)
        return opened

    def _watched(self, page: Page) -> None:
        page.on("pageerror", lambda error: self.faults.append(f"{page.url}: uncaught {error}"))

        def said(message: ConsoleMessage) -> None:
            if message.text.startswith(FAULT_MARK):
                self.faults.append(f"{page.url}: console.error {message.text.removeprefix(FAULT_MARK)}")

        page.on("console", said)

    async def open(self, address: str) -> Page:
        page = await self.context.new_page()
        await page.goto(address)
        return page

    async def followed(self, page: Page, link: str) -> Page:
        """The page `page`'s link named `link` opens beside it."""
        async with page.expect_popup() as popup:
            await page.get_by_role("link", name=link).click()
        return await popup.value

    async def closed(self) -> list[str]:
        """Every fault the pages made, once each is closed: [LAW:no-ambient-temporal-coupling] Playwright delivers a page's
        events before its close resolves, so none is still on its way."""
        for page in self.context.pages:
            await page.close()
        return self.faults


@pytest.fixture
async def chrome(hands: Hands) -> AsyncGenerator[Browser]:
    """The installed Google Chrome. [LAW:no-ambient-temporal-coupling] it takes `hands` so it ends first, its pages with
    it: a page still reading hands would hold hands' server open as it closes."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(channel="chrome", args=FAKE_MICROPHONE)
        yield browser
        await browser.close()


@pytest.fixture
async def browser(chrome: Browser) -> AsyncGenerator[Opened]:
    """The pages a test opens, checked once it is over, passed or failed: a page fault fails it."""
    opened = await Opened.made(chrome)
    yield opened
    assert await opened.closed() == []


async def test_a_fault_in_a_pages_own_code_is_caught_by_the_check(chrome: Browser, hands: Hands) -> None:
    # The check's own pages, not the fixture's, so the faults made here are this test's to assert.
    opened = await Opened.made(chrome)
    page = await opened.open(hands.address("/"))
    # A console.error with nothing to say is still the page's.
    async with page.expect_console_message(lambda message: message.text.startswith(FAULT_MARK)):
        await page.evaluate("console.error()")
    # And an error nothing catches.
    async with page.expect_event("pageerror"):
        await page.evaluate("setTimeout(() => { throw new Error('a fault of the page') })")
    faults = await opened.closed()
    assert any("console.error" in fault for fault in faults)
    assert any("uncaught" in fault and "a fault of the page" in fault for fault in faults)


async def test_connect_calls_hands_and_the_page_says_it_is_on_the_call(browser: Opened, hands: Hands) -> None:
    page = await browser.open(hands.address("/"))
    await page.get_by_role("button", name="Connect").click()
    await expect(page.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    await expect(page.locator("#talk")).to_have_text("Hold to talk")


async def test_a_second_page_takes_the_phone_and_the_first_says_so(browser: Opened, hands: Hands) -> None:
    first = await browser.open(hands.address("/"))
    await first.get_by_role("button", name="Connect").click()
    await expect(first.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    second = await browser.open(hands.address("/"))
    await second.get_by_role("button", name="Connect").click()
    await expect(second.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    # The first calls again on its own, and hands declines it: the second is the page in the user's hand.
    await expect(first.locator("#state")).to_have_text("Another page has hands' phone. Press Connect to take it here.")
    await expect(first.locator("#talk")).to_have_text("Connect")


async def test_stop_stops_the_turn_the_page_shows_and_says_what_came_of_it(browser: Opened, hands: Hands) -> None:
    hands.brain.work_now = SLEEPING
    page = await browser.open(hands.address("/"))
    await expect(page.locator("#doing")).to_contain_text("Running Bash")
    await expect(page.locator("#calls")).to_contain_text("sleep 30")
    await page.get_by_role("button", name="Stop").click()
    await expect(page.locator("#brain-said")).to_have_text("Stopping the brain's turn.")
    assert hands.brain.stopped == ["turn-1"]


async def test_words_typed_on_the_conversation_page_are_sent_to_hands(browser: Opened, hands: Hands) -> None:
    phone = await browser.open(hands.address("/"))
    page = await browser.followed(phone, "Conversation")
    await page.get_by_label("Type to hands").fill("what is on the backlog")
    await page.get_by_label("Type to hands").press("Enter")
    # Emptied once hands has the words.
    await expect(page.get_by_label("Type to hands")).to_have_value("")
    await expect(page.locator("#state")).to_have_text("")
    assert hands.typed == ["what is on the backlog"]


async def test_a_tool_call_opened_on_the_conversation_page_stays_open_as_the_conversation_is_drawn_again(browser: Opened, hands: Hands) -> None:
    hands.log.record(Transcribed("switch to the wake word"))
    await audited(SYSTEM, tool(set_trigger), hands.log.record).body(trigger="wake word")
    page = await browser.open(hands.address("/conversation"))
    await page.get_by_text("called set_trigger").click()
    hands.log.record(Replied("Okay. Say Hey Jarvis, then what you want.", False))
    await expect(page.locator("#moments")).to_contain_text("Okay. Say Hey Jarvis")
    await expect(page.locator("details")).to_have_attribute("open", "")


async def test_use_on_the_gallery_makes_hands_speak_in_that_voice_and_the_page_says_so(browser: Opened, hands: Hands) -> None:
    voice = VOICES[1]
    shown = voice.replace("_", " ").title()
    phone = await browser.open(hands.address("/"))
    page = await browser.followed(phone, "Voices and options")
    await page.locator("#voices li", has_text=shown).get_by_role("button", name="Use").click()
    await expect(page.locator("#status")).to_have_text(f"Speaking in {shown} from now on.")
    await expect(page.locator("#speaking")).to_have_text(f"hands speaks in {shown}.")
    assert hands.voices.used_asked == [Voice(voice)]


@real_lit
async def test_a_note_saved_on_the_backlog_page_is_kept_by_lit_on_its_ticket(browser: Opened, hands: Hands, workspace: Workspace) -> None:
    _, ticket = epic(workspace, "Train a new wake word", "Hey hands", topic="wake-word")
    page = await browser.open(hands.address("/backlog", project=str(workspace.root)))
    await page.get_by_text("Train a new wake word").click()
    await page.get_by_label(f"Note on {ticket}").fill("Say which word wakes it")
    await page.locator(f'li[data-id="{ticket}"] > .card').get_by_role("button", name="Save").click()
    await expect(page.get_by_label(f"Note on {ticket}")).to_have_attribute("placeholder", "Noted: Say which word wakes it")
    shown = subprocess.run([str(LIT), "show", ticket], cwd=workspace.root, check=True, capture_output=True, text=True).stdout
    assert "Say which word wakes it" in shown


# The wake word on the phone: a page's own code runs the Mac's models on the audio it is given. What it is given is
# recordings of speech, as a phone's microphone would give them, at 48 kHz in pieces of 20 ms.
PHRASE = "Hey Jarvis. What is the status of the build?"
IN_PASSING = "I was talking with Jarvis earlier about the build, and it was fine."
PHONE_RATE = 48_000
PIECE = PHONE_RATE // 50

START_DETECTOR = """
async () => {
  const { startDetector } = await import("/detector.js");
  window.events = [];
  window.detector = startDetector({
    onReady: ready => events.push({ kind: "ready", ...ready }),
    onFailed: why => events.push({ kind: "failed", why }),
    onHeard: (score, at) => events.push({ kind: "heard", score, at }),
  });
}
"""

FEED = """
({ encoded, rate, piece }) => {
  const bytes = Uint8Array.from(atob(encoded), each => each.charCodeAt(0));
  const heard = new Int16Array(bytes.buffer);
  for (let at = 0; at < heard.length; at += piece) detector.feed(Float32Array.from(heard.subarray(at, at + piece), each => each / 32768), rate);
}
"""


def at_the_phones_rate(audio: bytes) -> str:
    """`audio`, 16 kHz as the desk hears it, as a phone's microphone would give it: 48 kHz, as base64 for the page."""
    heard = np.frombuffer(audio, np.int16)
    upsampled = np.interp(np.arange(len(heard) * 3) / 3, np.arange(len(heard)), heard).astype(np.int16)
    return base64.b64encode(upsampled.tobytes()).decode()


async def detecting(page: Page) -> None:
    """The detector started on `page`, and said whether it can listen."""
    await page.evaluate(START_DETECTOR)
    await page.wait_for_function("events.length > 0")


def mac_hears_at(models: Path, audio: bytes) -> float:
    """The seconds of `audio` the Mac's own detector has been given when it first hears the wake word, as the desk gives it audio, 20 ms at a time."""
    word = WakeWord(models, Pretrained())
    step = 640
    for start in range(0, len(audio), step):
        if word.score(audio[start : start + step]) >= THRESHOLD:
            return (start + step) / (RATE * 2)
    raise AssertionError("the Mac's detector never heard the wake word in this audio")


async def test_the_phone_hears_the_wake_word_where_the_mac_does_and_asks_nothing_of_any_site_but_hands(browser: Opened, hands: Hands, models: Path) -> None:
    asked_of = list[str]()
    browser.context.on("request", lambda request: asked_of.append(request.url))
    page = await browser.open(hands.address("/"))
    await detecting(page)
    assert await page.evaluate("events") == [{"kind": "ready", "phrase": "Hey Jarvis", "threshold": THRESHOLD, "keptSeconds": KEPT_SECONDS}]
    # Jarvis named in passing comes first, with a second of quiet either side: the word is in what comes after it.
    before = quiet(1.0) + spoken(IN_PASSING) + quiet(1.0)
    said = quiet(1.0) + spoken(PHRASE) + quiet(1.0)
    await page.evaluate(FEED, {"encoded": at_the_phones_rate(before + said), "rate": PHONE_RATE, "piece": PIECE})
    await page.wait_for_function("events.some(each => each.kind === 'heard')")
    heard = next(each for each in await page.evaluate("events") if each["kind"] == "heard")
    assert heard["score"] >= THRESHOLD
    # Not in the sentence that names Jarvis in passing, but in the one that says the word: and where the Mac hears it, to
    # within what the phone's chunks and its own resampling move it by.
    assert len(before) / (RATE * 2) < heard["at"]
    assert abs(heard["at"] - mac_hears_at(models, before + said)) < 0.5
    # Everything the page and its worker loaded came from hands, the runtime and the models among it.
    origin = str(hands.server.make_url("/"))
    assert [url for url in asked_of if not url.startswith(origin)] == []
    wanted = {"/detector.js", "/detector-worker.js", "/wakestream.js", "/hands.js", "/wake/settings", "/ort/1.31.1/ort.wasm.bundle.min.mjs", "/ort/1.31.1/ort-wasm-simd-threaded.wasm", *(f"/wake/models/{name}" for name in WAKE_MODELS)}
    assert wanted <= set(hands.requests)


async def test_a_second_page_runs_the_runtime_from_the_browsers_copy(hands: Hands, tmp_path: Path) -> None:
    def runtime() -> list[str]:
        return [path for path in hands.requests if path.startswith("/ort/")]

    # A profile on disk, as a phone's browser has: a throwaway profile keeps its cache in memory, which holds no file of 15 MB.
    async with async_playwright() as playwright:
        opened = await Opened.watching(await playwright.chromium.launch_persistent_context(tmp_path / "profile", channel="chrome", args=FAKE_MICROPHONE))
        await detecting(await opened.open(hands.address("/")))
        assert len(runtime()) == 2
        await detecting(await opened.open(hands.address("/")))
        assert len(runtime()) == 2
        assert await opened.closed() == []
        await opened.context.close()


async def test_a_model_route_that_fails_reaches_the_page_as_why_it_cannot_listen(browser: Opened, hands: Hands) -> None:
    async def broken(route: Route) -> None:
        await route.fulfill(status=503, body="hands cannot read the word's model")

    await browser.context.route("**/wake/models/word.onnx", broken)
    page = await browser.open(hands.address("/"))
    await detecting(page)
    assert await page.evaluate("events") == [{"kind": "failed", "why": "hands cannot read the word's model"}]


async def test_a_page_opened_without_the_phones_key_is_told_so_by_hands_and_says_it_cannot_listen(browser: Opened, hands: Hands) -> None:
    page = await browser.open(keyed_address(str(hands.server.make_url("/")), "not-the-phones-key"))
    await detecting(page)
    assert await page.evaluate("events") == [{"kind": "failed", "why": UNKEYED}]
