"""The phone's pages, each its own code run in the installed Google Chrome against the routes hands serves the phone, on
the route tests' fakes: a choice made on each page as a person makes it, and what the page shows of it."""

import asyncio
import base64
import os
import subprocess
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import BaseTestServer, TestServer
from playwright.async_api import Browser, BrowserContext, ConsoleMessage, Page, async_playwright, expect

from hands.clock import SYSTEM
from hands.sessions.audit import AuditLog, Entry, Replied, Transcribed
from hands.sessions.home import Home
from hands.sessions.summaries import Summaries
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice.modelswitch import Switches
from hands.voice.clips import KEPT_SECONDS, Clip
from hands.voice.gallerypage import Gallery
from hands.voice.keyed import UNKEYED, Handler
from hands.voice.milestone import ClipDiscarded, ClipOpened, TurnUnopened
from hands.voice.phone import CallLeft, Phone
from hands.voice.phoneaddress import keyed_address, phone_key
from hands.voice.phonepage import phone_pages
from hands.voice.pinnedfile import Fetch, Kept, Mismatch, Pin, Release
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.voice.summaries import SummaryStore
from hands.voice.tool import audited, tool
from hands.voice.voicechoice import VOICES, Voice
from hands.voice.wake import WakeWord
from hands.voice.wakemodels import PRETRAINED, ROLES, Files, WakeModels
from hands.voice.wakeword import THRESHOLD, Pretrained
from speech import RATE, quiet, spoken
from test_backlogpage import LIT, Workspace, epic, real_lit, workspace  # pyright: ignore[reportUnusedImport]  (a fixture, found by its name)
from test_brainpage import SLEEPING, Brain
from test_conversationpage import a_log, set_trigger
from test_gallerypage import MODELS, Desk, Listening, Own, Speaker, Voicing

pytestmark = pytest.mark.outside

# A model of two words, as openWakeWord scores them, which no wake word is: tests/test_trigger.py has it too.
TWO_WORDS = Path(__file__).parent / "fixtures" / "two_words.onnx"

# A microphone Chrome makes up, granted without asking: the phone's call takes one, and no test has a person to allow it.
FAKE_MICROPHONE = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]

# The choice of a person who turned the phone wake word off, as the page keeps it: a page of the button alone, which the tests of the
# button run on.
WAKE_OFF = 'if (location.protocol.startsWith("http")) localStorage.setItem("wake word", "off")'

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
    # What the phone recorded, as its calls end.
    recorded: list[Entry]
    # The gate the phone moves, which a test moves as the desk's key or the pipeline would.
    gate: PushToTalk

    def address(self, path: str, **query: str) -> str:
        """`path` on this server, carrying the phone's key as `hands phone` prints it."""
        return keyed_address(str(self.server.make_url(path).with_query(query)), self.key)


@pytest.fixture
def wake_models(request: pytest.FixtureRequest, models: Path, tmp_path: Path) -> WakeModels:
    """The models hands serves the wake word's page: the run's own, already fetched, unless a test parametrizes this with
    the fetch it wants given the run's models, which then fetches into a directory that holds none."""
    fetch: Callable[[Path], Fetch] | None = getattr(request, "param", None)
    return WakeModels(models, SYSTEM) if fetch is None else WakeModels(tmp_path / "unfetched", SYSTEM, fetch(models))


@pytest.fixture
async def hands(tmp_path: Path, wake_models: WakeModels, onnx_runtime: Path) -> AsyncGenerator[Hands]:
    home = Home(tmp_path / "hands")
    key = phone_key(SYSTEM, home)
    recorded = list[Entry]()
    gate = PushToTalk(SYSTEM, lambda _: None)
    phone = Phone(SYSTEM, gate, heard_rate=16000, played_rate=24000, record=recorded.append, speaking=lambda: False)
    brain, typed, voices = Brain(), list[str](), Voicing()

    async def typing(text: str) -> None:
        typed.append(text)

    gallery = Gallery(voices=voices, levels=Listening(), desk=Desk(), own=Own(), switches=Switches(Speaker(VirtualClock())), wake=Pretrained(), models=MODELS)
    # The environment hands was started in, as it stood: one a test changes after does not reach the page's writes.
    pages = phone_pages(SYSTEM, phone, brain, typing, SummaryStore(Summaries(home.summaries)), home.audit, key, lambda _: None, dict(os.environ), gallery, wake_models, Release(onnx_runtime, SYSTEM))
    requests = list[str]()

    @web.middleware
    async def taken(request: web.Request, handler: Handler) -> web.StreamResponse:
        requests.append(request.path)
        return await handler(request)

    pages.middlewares.append(taken)
    async with TestServer(pages) as server:
        yield Hands(server, key, phone, brain, typed, a_log(home.audit), voices, requests, recorded, gate)
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


@pytest.fixture
async def keeping_browser(hands: Hands, tmp_path: Path) -> AsyncGenerator[Opened]:
    """As `browser`, with a profile on disk, as a phone's browser has: a throwaway profile keeps its cache in memory, which
    holds no file of 15 MB. It takes `hands` so it ends first, as `chrome` does."""
    async with async_playwright() as playwright:
        opened = await Opened.watching(await playwright.chromium.launch_persistent_context(tmp_path / "profile", channel="chrome", args=FAKE_MICROPHONE))
        yield opened
        faults = await opened.closed()
        await opened.context.close()
    assert faults == []


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


@pytest.mark.quarantined("promptctl-phone-app-9d9y.yrr", until="2026-11-29")
async def test_connect_calls_hands_and_the_page_says_it_is_on_the_call(browser: Opened, hands: Hands) -> None:
    await browser.context.add_init_script(WAKE_OFF)
    page = await browser.open(hands.address("/"))
    await page.get_by_role("button", name="Connect").click()
    await expect(page.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    await expect(page.locator("#talk")).to_have_text("Hold to talk")


async def test_a_second_page_takes_the_phone_and_the_first_says_so(browser: Opened, hands: Hands) -> None:
    await browser.context.add_init_script(WAKE_OFF)
    first = await browser.open(hands.address("/"))
    await first.get_by_role("button", name="Connect").click()
    await expect(first.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    second = await browser.open(hands.address("/"))
    await second.get_by_role("button", name="Connect").click()
    await expect(second.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    # The first calls again on its own, and hands declines it: the second is the page in the user's hand.
    await expect(first.locator("#state")).to_have_text("Another page has hands' phone. Press Connect to take it here.")
    await expect(first.locator("#talk")).to_have_text("Connect")


# A microphone that is never silent, and a record, in the order the page lived it, of the buttons it sent and the messages
# of its microphone. Each block the tap posts reaches the page as one number, its own, so a message of audio says which
# blocks it holds, and a mark says how many the tap had posted before it. While `holding`, what the tap posts waits in the
# order it came, as it does for a main thread that is busy, until `drain` hands it to the page.
NEVER_SILENT = """
if (window.isSecureContext) {
  window.timeline = [];
  window.arrived = 0;
  window.holding = false;
  const backlog = [];
  let deliver = null;
  window.drain = () => { holding = false; while (backlog.length) deliver(backlog.shift()); };
  navigator.mediaDevices.getUserMedia = async () => {
    const context = new AudioContext();
    await context.resume();
    const tone = context.createOscillator(), level = context.createGain(), out = context.createMediaStreamDestination();
    level.gain.value = 0.5;
    tone.connect(level).connect(out);
    tone.start();
    return out.stream;
  };
  const NativeNode = AudioWorkletNode;
  window.AudioWorkletNode = class extends NativeNode {
    constructor(...made) {
      super(...made);
      let handler = null;
      deliver = (data) => handler({ data });
      Object.defineProperty(this.port, "onmessage", { set(each) { handler = each; } });
      this.port.addEventListener("message", ({ data }) => {
        if (typeof data === "number") timeline.push({ mark: data, after: arrived });
        else data = new Float32Array(data.length).fill(++arrived / 32767);
        if (holding || backlog.length) backlog.push(data); else deliver(data);
      });
      this.port.start();
    }
  };
  const sendOnChannel = RTCDataChannel.prototype.send;
  RTCDataChannel.prototype.send = function (message) {
    if (typeof message !== "string") {
      const blocks = new Int16Array(message);
      timeline.push({ audio: blocks.length, first: blocks[0], last: blocks[blocks.length - 1] });
    } else if (message === "press" || message === "release") timeline.push({ button: message });
    else { const { kind, samples } = JSON.parse(message); timeline.push({ [kind]: samples }); }
    return sendOnChannel.call(this, message);
  };
}
"""


async def until_sent(page: Page, kind: str, more: int) -> None:
    """Until the page has sent `more` messages of `kind` beyond those it had when this was called."""
    seen = await page.evaluate("timeline.length")
    await page.wait_for_function("([seen, kind, more]) => timeline.slice(seen).filter(each => kind in each).length >= more", arg=[seen, kind, more])


async def at_the_edge(page: Page, act: Callable[[], Awaitable[None]]) -> None:
    """`act`, a press or a release at the button, while the page's main thread is busy: blocks the tap posted before it are
    still on their way, and the mark the edge posts comes behind them."""
    await page.evaluate("holding = true")
    posted = await page.evaluate("arrived")
    await page.wait_for_function("posted => arrived >= posted + 3", arg=posted)
    seen = await page.evaluate("timeline.length")
    await act()
    await page.wait_for_function("seen => timeline.slice(seen).some(each => 'mark' in each)", arg=seen)
    await page.evaluate("drain()")


async def test_the_page_sends_its_microphone_only_inside_a_turn_and_counts_what_it_held_back_outside_one(browser: Opened, hands: Hands) -> None:
    await browser.context.add_init_script(WAKE_OFF)
    await browser.context.add_init_script(NEVER_SILENT)
    page = await browser.open(hands.address("/"))
    await page.get_by_role("button", name="Connect").click()
    await expect(page.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    talk = (await page.locator("#talk").bounding_box()) or {}
    await page.mouse.move(talk["x"] + talk["width"] / 2, talk["y"] + talk["height"] / 2)
    for _ in range(2):
        await until_sent(page, "quiet", 3)
        await at_the_edge(page, page.mouse.down)
        await until_sent(page, "audio", 3)
        await at_the_edge(page, page.mouse.up)
    await until_sent(page, "quiet", 3)
    timeline: list[dict[str, int | str]] = await page.evaluate("timeline")
    # Each button's mark says how many blocks the tap had posted when the button moved. A turn holds exactly the blocks after
    # its press's mark, through its release's: none posted before the press rides in it, and none after the release.
    after_mark, opened, carried = 0, 0, list[dict[str, int | str]]()
    inside = False
    for each in timeline:
        match each:
            case {"mark": _, "after": int(after)}:
                after_mark = after
            case {"button": "press"}:
                opened, carried, inside = after_mark, [], True
            case {"button": "release"}:
                assert carried and all(int(sent["first"]) > opened for sent in carried), "a block posted before the press went in the turn"
                assert max(int(sent["last"]) for sent in carried) == after_mark, "a block posted before the release was held back from the turn"
                inside = False
            case {"quiet": _}:
                assert not inside, "a count was sent inside a turn"
            case {"audio": _}:
                assert inside, "microphone audio was sent outside a turn"
                carried.append(each)
            case unknown:
                pytest.fail(f"the page sent what this test does not know: {unknown}")
    assert [each["button"] for each in timeline if "button" in each] == ["press", "release"] * 2
    # What hands heard: the tone inside the turns, and the second clip ended by the silence that follows it.
    frames: list[KeyedAudio] = []
    async with asyncio.timeout(10):
        while not frames or len(frames[-1].ended) < 2:
            await asyncio.sleep(0.05)
            while not hands.phone.heard.empty():
                frames.append(hands.phone.heard.get_nowait())
    assert any(any(frame.audio) for frame in frames if frame.key == "down")
    # And hands could read every message the page sent: it counted what the page held back and rejected nothing.
    await hands.phone.stop()
    [event] = [entry for entry in hands.recorded if isinstance(entry, WideEvent) and entry.event == "phone.call"]
    ended = event.facts["ended"]
    assert isinstance(ended, CallLeft) and ended.rejected == 0 and ended.held_back > 0



async def test_the_page_says_it_heard_nothing_from_a_press_hands_opened_no_clip_for_and_from_a_clip_hands_threw_away(browser: Opened, hands: Hands) -> None:
    await browser.context.add_init_script(WAKE_OFF)
    page = await browser.open(hands.address("/"))
    await page.get_by_role("button", name="Connect").click()
    await expect(page.locator("#state")).to_have_text("Connected. Hold the button and speak.")
    talk = (await page.locator("#talk").bounding_box()) or {}
    await page.mouse.move(talk["x"] + talk["width"] / 2, talk["y"] + talk["height"] / 2)

    async def until_moved(count: int) -> None:
        """Until hands has taken `count` moves of the page's button, so what it tells of a press finds the press."""
        async with asyncio.timeout(10):
            while hands.phone.moves.qsize() < count:
                await asyncio.sleep(0.01)

    # hands numbers the call's presses as the page does: the first press opened the gate's turn 0, and says no clip came of it.
    await page.mouse.down()
    await until_moved(1)
    hands.phone.tell(TurnUnopened(0))
    await expect(page.locator("#state")).to_have_text("Heard nothing from that press.")
    await page.mouse.up()
    await until_moved(2)
    # The page keeps what hands said of the press once the button is let go.
    await expect(page.locator("#state")).to_have_text("Heard nothing from that press.")
    # The second press opened turn 1 and a clip of its own, which hands then threw away.
    await page.mouse.down()
    await until_moved(3)
    thrown = Clip(9, "phone button")
    hands.phone.tell(ClipOpened(thrown, 1))
    await expect(page.locator("#state")).to_have_text("Listening…")
    hands.phone.tell(ClipDiscarded(thrown))
    await expect(page.locator("#state")).to_have_text("Heard nothing from that press.")
    await page.mouse.up()


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
async rate => {
  const { startDetector } = await import("/detector.js");
  window.events = [];
  window.detector = startDetector({
    rate,
    onReady: ready => events.push({ kind: "ready", ...ready }),
    onFailed: why => events.push({ kind: "failed", why }),
    onHeard: (score, at) => events.push({ kind: "heard", score, at }),
    onWaiting: why => events.push({ kind: "waiting", why }),
  });
}
"""

FEED = """
({ encoded, piece }) => {
  const bytes = Uint8Array.from(atob(encoded), each => each.charCodeAt(0));
  const heard = new Int16Array(bytes.buffer);
  for (let at = 0; at < heard.length; at += piece) detector.feed(Float32Array.from(heard.subarray(at, at + piece), each => each / 32768));
}
"""


def at_the_phones_rate(audio: bytes) -> str:
    """`audio`, 16 kHz as the desk hears it, as a phone's microphone would give it: 48 kHz, as base64 for the page."""
    heard = np.frombuffer(audio, np.int16)
    upsampled = np.interp(np.arange(len(heard) * 3) / 3, np.arange(len(heard)), heard).astype(np.int16)
    return base64.b64encode(upsampled.tobytes()).decode()


async def detecting(page: Page, rate: float = PHONE_RATE) -> None:
    """The detector started on `page` for audio at `rate`, and said whether it can listen."""
    await page.evaluate(START_DETECTOR, rate)
    await page.wait_for_function("events.length > 0")


def mac_hears_at(files: Files, audio: bytes) -> float:
    """The seconds of `audio` the Mac's own detector has been given when it first hears the wake word, as the desk gives it audio, 20 ms at a time."""
    word = WakeWord(files)
    step = 640
    for start in range(0, len(audio), step):
        if word.score(audio[start : start + step]) >= THRESHOLD:
            return (start + step) / (RATE * 2)
    raise AssertionError("the Mac's detector never heard the wake word in this audio")


async def test_the_phone_hears_the_wake_word_where_the_mac_does_and_asks_nothing_of_any_site_but_hands(browser: Opened, hands: Hands, models: Path) -> None:
    asked_of = list[str]()
    browser.context.on("request", lambda request: asked_of.append(request.url))
    page = await browser.open(hands.address("/"))
    # Jarvis named in passing comes first, with a second of quiet either side: the word is in what comes after it.
    before = quiet(1.0) + spoken(IN_PASSING) + quiet(1.0)
    said = quiet(1.0) + spoken(PHRASE) + quiet(1.0)
    # All of it is fed as the detector starts, before its models have loaded: it is heard once they have, and in the order it came.
    await page.evaluate(START_DETECTOR, PHONE_RATE)
    await page.evaluate(FEED, {"encoded": at_the_phones_rate(before + said), "piece": PIECE})
    await page.wait_for_function("events.some(each => each.kind === 'heard')")
    [ready, *_, heard] = await page.evaluate("events")
    assert ready == {"kind": "ready", "phrase": "Hey Jarvis", "threshold": THRESHOLD, "keptSeconds": KEPT_SECONDS}
    assert heard["kind"] == "heard" and heard["score"] >= THRESHOLD
    # Not in the sentence that names Jarvis in passing, but in the one that says the word: and where the Mac hears it, to
    # within what the phone's chunks and its own resampling move it by.
    assert len(before) / (RATE * 2) < heard["at"]
    assert abs(heard["at"] - mac_hears_at((await WakeModels(models, SYSTEM).ensured(Pretrained())).files, before + said)) < 0.5
    # Everything the page and its worker loaded came from hands, the runtime and the models among it.
    origin = str(hands.server.make_url("/"))
    assert [url for url in asked_of if not url.startswith(origin)] == []
    wanted = {"/detector.js", "/detector-worker.js", "/wakestream.js", "/hands.js", "/wake/settings", "/ort/1.31.1/ort.wasm.bundle.min.mjs", "/ort/1.31.1/ort-wasm-simd-threaded.wasm", *(f"/wake/models/{role}" for role in ROLES)}
    assert wanted <= set(hands.requests)


async def test_a_second_page_runs_the_runtime_from_the_browsers_copy(keeping_browser: Opened, hands: Hands) -> None:
    def runtime() -> list[str]:
        return [path for path in hands.requests if path.startswith("/ort/")]

    await detecting(await keeping_browser.open(hands.address("/")))
    assert len(runtime()) == 2
    await detecting(await keeping_browser.open(hands.address("/")))
    assert len(runtime()) == 2


class Unreachable:
    """A fetch that cannot reach the release."""

    async def __call__(self, directory: Path, pin: Pin, stop: threading.Event) -> Kept:
        raise OSError("the network is down")


class WithTwoWords:
    """A fetch that gives the run's own models, but for Hey Jarvis's, which it gives a model that scores two words."""

    def __init__(self, models: Path) -> None:
        self._models = models

    async def __call__(self, directory: Path, pin: Pin, stop: threading.Event) -> Kept:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / pin.name).write_bytes((TWO_WORDS if pin == PRETRAINED["Hey Jarvis"] else self._models / pin.name).read_bytes())
        return Kept(directory / pin.name, "fetched")


def unreachable(_models: Path) -> Fetch:
    return Unreachable()


class NotPinned:
    """A fetch whose bytes are not the ones pinned."""

    async def __call__(self, directory: Path, pin: Pin, stop: threading.Event) -> Kept:
        raise Mismatch(f"{pin.name} at {pin.at} is aaa, not the {pin.digest} pinned")


def not_pinned(_models: Path) -> Fetch:
    return NotPinned()


@pytest.mark.parametrize("wake_models", [unreachable], indirect=True)
async def test_models_hands_cannot_fetch_reach_the_page_as_why_it_waits_in_hands_words(browser: Opened, hands: Hands) -> None:
    page = await browser.open(hands.address("/"))
    await detecting(page)
    # Waited on, not given up: hands fetches again as it is next asked (`patiently`, tests/test_handsjs.py).
    assert await page.evaluate("events") == [{"kind": "waiting", "why": "hands cannot get openWakeWord v0.5.1's melspectrogram.onnx: the network is down"}]


@pytest.mark.quarantined("promptctl-testing-c3u.sm9", until="2026-11-29")
@pytest.mark.parametrize("wake_models", [WithTwoWords], indirect=True)
async def test_a_model_that_scores_two_words_is_refused_by_the_phone_as_the_desk_refuses_it(browser: Opened, hands: Hands) -> None:
    page = await browser.open(hands.address("/"))
    await detecting(page)
    assert await page.evaluate("events") == [{"kind": "failed", "why": "the wake word model scores 2 words, and a wake word is one"}]


async def test_a_rate_that_is_not_a_rate_is_refused_by_the_detector_in_words(browser: Opened, hands: Hands) -> None:
    page = await browser.open(hands.address("/"))
    await detecting(page, rate=0)
    assert await page.evaluate("events") == [{"kind": "failed", "why": "a stream's rate is a number of samples a second above zero, not 0"}]


async def test_a_page_opened_without_the_phones_key_is_told_so_by_hands_and_says_it_cannot_listen(browser: Opened, hands: Hands) -> None:
    page = await browser.open(keyed_address(str(hands.server.make_url("/")), "not-the-phones-key"))
    await detecting(page)
    assert await page.evaluate("events") == [{"kind": "failed", "why": UNKEYED}]
