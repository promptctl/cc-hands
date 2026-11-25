"""The phone page listening for its wake word, run as a person has it: the page's own code in the installed Google Chrome, on hands'
real routes, the real runtime and the real models, its microphone a recording of speech (tests/speech.py) that the test plays when
it means the page to hear it. What leaves the phone is read where it leaves, on the call's data channel; what hands made of it,
at hands: the gate the phone moves and the clips a real clip owner cuts from the audio it was sent."""

import asyncio
import base64
import re
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import numpy as np
import pytest
from conftest import events, running
from playwright.async_api import Page, Route, expect
from speech import spoken
from test_clipowner import Heeding, Scripted
from test_pages import IN_PASSING, PHRASE, Hands, Opened, browser, chrome, hands, unreachable, wake_models  # pyright: ignore[reportUnusedImport]  (fixtures, found by their names)

from hands.clock import SYSTEM
from hands.voice.clipowner import ClipOwner
from hands.voice.clips import KEPT_SECONDS, Clip, Sound
from hands.voice.latency import LatencyObserver
from hands.voice.ptt import Ending
from hands.voice.trigger import PHONE_WAKE

pytestmark = pytest.mark.outside

LISTENING = 'Connected. Say "Hey Jarvis", or hold the button.'
LOADING = "Connected. Hold the button and speak."

# The page's microphone is a recording the test plays when it means the page to hear it, and silence otherwise; and everything the
# page sends on its call's data channel is written down as it goes, audio by its length and its loudest sample. `beep` is a tone
# louder than any speech, for a marker that only what was made in a place can carry.
SPEECH = """
if (window.isSecureContext) {
  window.sent = [];
  const sendOnChannel = RTCDataChannel.prototype.send;
  RTCDataChannel.prototype.send = function (message) {
    if (typeof message !== "string") {
      const samples = new Int16Array(message);
      let peak = 0;
      for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
      sent.push({ audio: samples.length, peak });
    } else if (message === "press" || message === "release") sent.push({ button: message });
    else sent.push(JSON.parse(message));
    return sendOnChannel.call(this, message);
  };
  let context = null, destination = null;
  navigator.mediaDevices.getUserMedia = async () => {
    context = new AudioContext();
    await context.resume();
    window.rate = context.sampleRate;
    destination = context.createMediaStreamDestination();
    return destination.stream;
  };
  const played = (source) => new Promise((done) => { source.onended = done; source.start(); });
  window.say = (encoded) => {
    const bytes = Uint8Array.from(atob(encoded), each => each.charCodeAt(0));
    const heard = new Int16Array(bytes.buffer);
    const buffer = context.createBuffer(1, heard.length, 16000);
    buffer.copyToChannel(Float32Array.from(heard, each => each / 32768), 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(destination);
    return played(source);
  };
  window.beep = (seconds) => {
    const tone = context.createOscillator(), loud = context.createGain();
    loud.gain.value = 0.99;
    tone.connect(loud).connect(destination);
    const done = played(tone);
    tone.stop(context.currentTime + seconds);
    return done;
  };
}
"""

# The screen wake lock the browser gives, counted: `locks` is how many the page holds now. `takeBack()` is the browser letting
# go of the one it gave last, as it does on a low battery, which the lock says as it does: with a `release` event.
WAKE_LOCK = """
if (window.isSecureContext) {
  window.locks = 0;
  let latest = null;
  Object.defineProperty(navigator, "wakeLock", { value: { request: async () => {
    locks += 1;
    latest = new EventTarget();
    latest.release = async () => { locks -= 1; };
    return latest;
  } } });
  window.takeBack = () => { locks -= 1; latest.dispatchEvent(new Event("release")); };
}
"""


async def speak(page: Page, audio: bytes) -> None:
    """The microphone says `audio`, 16 kHz as the desk hears speech, and this returns once it has been said."""
    await page.evaluate("encoded => say(encoded)", base64.b64encode(audio).decode())


async def sent(page: Page) -> list[dict[str, Any]]:
    """Everything the page has sent on its call, in the order it sent it."""
    return await page.evaluate("sent")


async def until_sent(page: Page, kind: str, count: int = 1) -> None:
    """Until the page has sent `count` messages of `kind` on its call, from the page's first."""
    await page.wait_for_function("([kind, count]) => sent.filter(each => each.kind === kind).length >= count", arg=[kind, count])


async def until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(15):
        while not condition():
            await asyncio.sleep(0.02)


async def held_open(held: asyncio.Event, route: Route) -> None:
    """A request kept waiting until `held` is set, and then let through."""
    await held.wait()
    await route.continue_()


async def button_down(page: Page) -> None:
    """The button goes down and stays down."""
    talk = (await page.locator("#talk").bounding_box()) or {}
    await page.mouse.move(talk["x"] + talk["width"] / 2, talk["y"] + talk["height"] / 2)
    await page.mouse.down()


def detector_states(hands: Hands) -> list[object]:
    """What the page told hands of its detector on each call, in order: loaded, waiting, or failed."""
    return [event.facts["state"] for event in events(hands.recorded, "phone.detector")]


def hands_ends_the_turn(hands: Hands, how: Ending) -> None:
    """hands ends the turn the gate has open, as its driver does by Smart Turn's verdict or by dropping what nothing was asked after."""
    hands.phone.took(hands.gate.end(hands.gate.gate.turn, how, PHONE_WAKE))


@asynccontextmanager
async def hearing(hands: Hands) -> AsyncGenerator[Heeding]:
    """The audio hands is sent cut into clips by a real clip owner, whose observer tells the page what became of each, as the
    daemon tells it. The listener writes down every clip as it was opened and what it was given."""
    heeding = Heeding("kept")
    owner = ClipOwner(SYSTEM, Scripted([None] * 8), (heeding,), hands.recorded.append)
    async with running([owner], [LatencyObserver(SYSTEM, owner, hands.phone.tell)]) as run:

        async def feed() -> None:
            while True:
                await run.worker.queue_frame(await hands.phone.heard.get())

        feeding = asyncio.create_task(feed())
        try:
            yield heeding
        finally:
            feeding.cancel()


def opened(heeding: Heeding) -> list[tuple[Clip, Sound]]:
    """The clips the clip owner opened, each with the sound before it, in the order it opened them."""
    return [given for kind, given in heeding.given if kind == "opened" and isinstance(given, tuple) and isinstance(given[0], Clip) and isinstance(given[1], Sound)]


async def connected(browser: Opened, hands: Hands, *init: str) -> Page:
    """The page, opened with the speech microphone (and `init`'s scripts), and Connect pressed."""
    for script in (SPEECH, *init):
        await browser.context.add_init_script(script)
    page = await browser.open(hands.address("/"))
    await page.get_by_role("button", name="Connect").click()
    return page


async def listening(browser: Opened, hands: Hands, *init: str) -> Page:
    """As `connected`, once the detector is ready and says so."""
    page = await connected(browser, hands, *init)
    await expect(page.locator("#state")).to_have_text(LISTENING)
    return page


def after(messages: list[dict[str, Any]], mark: dict[str, Any]) -> list[dict[str, Any]]:
    """What the page sent after the last message equal to `mark`."""
    at = max(index for index, each in enumerate(messages) if each == mark)
    return messages[at + 1 :]


def wake_at(messages: list[dict[str, Any]], number: int = 0) -> int:
    """Where in `messages` the page's `number`th wake is."""
    return [index for index, each in enumerate(messages) if each.get("kind") == "wake"][number]


async def test_with_the_wake_word_on_nothing_leaves_the_phone_until_the_word_is_said_and_then_a_turn_opens_with_the_second_before_it(browser: Opened, hands: Hands) -> None:
    async with hearing(hands) as heard:
        page = await listening(browser, hands)
        rate = await page.evaluate("rate")
        # Speech that does not hold the word: hands is sent a count of what was held back and none of it.
        await speak(page, spoken(IN_PASSING))
        said = await sent(page)
        assert [each for each in said if "audio" in each or each.get("kind") == "wake"] == []
        assert {"kind": "level", "level": "wake word"} in said
        # The word, and a question after it.
        await speak(page, spoken(PHRASE))
        await until_sent(page, "wake")
        said = await sent(page)
        wake = wake_at(said)
        kept = [each for each in said[:wake] if "audio" in each]
        # The second before the word goes as audio, just ahead of the wake: the quiet comes before it, and nothing else.
        assert said[wake - len(kept) : wake] == kept and all("audio" not in each for each in said[: wake - len(kept)])
        assert 0 < sum(each["audio"] for each in kept) <= KEPT_SECONDS * rate + 128
        assert any(each["peak"] > 0 for each in kept)
        assert said[wake]["score"] >= 0.5
        # From the wake on, the microphone as it comes: audio and no count.
        await until(lambda: hands.gate.gate.turn_open)
        assert hands.gate.gate.opened == PHONE_WAKE
        await until(lambda: bool(opened(heard)))
        [(clip, before)] = opened(heard)
        assert clip == Clip(1, PHONE_WAKE) and any(before.heard)
        await expect(page.locator("#state")).to_have_text("Listening…")
        assert all(each.get("kind") != "quiet" for each in (await sent(page))[wake + 1 :])
        # hands says the turn is over: the page sends only a count again, and the next saying is a turn of its own.
        hands_ends_the_turn(hands, "stop")
        await expect(page.locator("#state")).not_to_have_text("Listening…")
        mark = len(await sent(page))
        await page.wait_for_function("mark => sent.slice(mark).filter(each => each.kind === 'quiet').length >= 5", arg=mark)
        assert [each for each in (await sent(page))[mark:] if "audio" in each] == []
        await speak(page, spoken(PHRASE))
        await until_sent(page, "wake", 2)
        await until(lambda: len(opened(heard)) == 2)
        assert [clip for clip, _ in opened(heard)] == [Clip(1, PHONE_WAKE), Clip(2, PHONE_WAKE)]


async def test_a_browser_that_never_opened_the_page_has_the_wake_word_on_and_the_line_says_so_once_the_detector_is_ready(browser: Opened, hands: Hands) -> None:
    held = asyncio.Event()
    await browser.context.route("**/wake/settings", partial(held_open, held))
    page = await connected(browser, hands)
    await expect(page.get_by_role("button", name="Phone wake word")).to_have_attribute("aria-pressed", "true")
    # While the detector loads, the page says what it says with the wake word off, and hands has been told nothing of the phone listening.
    await expect(page.locator("#state")).to_have_text(LOADING)
    assert hands.gate.gate.phone_level == "off"
    held.set()
    await expect(page.locator("#state")).to_have_text(LISTENING)
    [loaded] = events(hands.recorded, "phone.detector")
    assert loaded.facts["state"] == "loaded" and "Chrome" in str(loaded.facts["browser"])
    await until(lambda: hands.gate.gate.phone_level == "wake word")


async def test_turned_off_the_choice_survives_a_reload_and_a_page_off_before_connect_asks_hands_for_none_of_the_detector(browser: Opened, hands: Hands) -> None:
    page = await browser.open(hands.address("/"))
    switch = page.get_by_role("button", name="Phone wake word")
    await switch.click()
    await expect(switch).to_have_attribute("aria-pressed", "false")
    await page.reload()
    await expect(switch).to_have_attribute("aria-pressed", "false")
    asked_before = len(hands.requests)
    await page.get_by_role("button", name="Connect").click()
    await expect(page.locator("#state")).to_have_text(LOADING)
    # The button works, which hands hears only once the call is up and the page would have asked for the detector by now.
    await holds_and_opens(page, hands)
    assert [path for path in hands.requests[asked_before:] if path.startswith(("/detector", "/wake", "/ort"))] == []
    assert events(hands.recorded, "phone.detector") == [] and hands.gate.gate.phone_level == "off"


async def holds_and_opens(page: Page, hands: Hands) -> None:
    """The button held opens a turn at the phone, which hands says it has, and let go ends it."""
    await button_down(page)
    await until(lambda: hands.gate.gate.turn_open and hands.gate.gate.opened == "phone button")
    await page.mouse.up()
    await until(lambda: not hands.gate.gate.turn_open)


async def test_turning_the_wake_word_off_and_on_in_a_call_stops_and_starts_the_detector_and_tells_hands_each(browser: Opened, hands: Hands) -> None:
    page = await listening(browser, hands)
    await until(lambda: hands.gate.gate.phone_level == "wake word")
    await page.get_by_role("button", name="Phone wake word").click()
    await expect(page.locator("#state")).to_have_text(LOADING)
    await until(lambda: hands.gate.gate.phone_level == "off")
    # Off, a word is heard by nothing.
    await speak(page, spoken(PHRASE))
    assert [each for each in await sent(page) if "audio" in each or each.get("kind") == "wake"] == []
    await page.get_by_role("button", name="Phone wake word").click()
    await expect(page.locator("#state")).to_have_text(LISTENING)
    await until(lambda: hands.gate.gate.phone_level == "wake word")
    assert detector_states(hands) == ["loaded", "loaded"]


async def test_a_word_said_after_a_button_turn_sends_the_kept_second_without_what_was_heard_before_the_button_went_down(browser: Opened, hands: Hands) -> None:
    speech_peak = int(np.abs(np.frombuffer(spoken(PHRASE), np.int16)).max())
    assert speech_peak < 0.9 * 32767, "the tone below is no marker if speech is as loud"
    page = await listening(browser, hands)
    rate = await page.evaluate("rate")
    # A tone, in the second the page keeps before a press; then the press and its release; then the word.
    await page.evaluate("beep(0.5)")
    await holds_and_opens(page, hands)
    await speak(page, spoken(PHRASE))
    await until_sent(page, "wake")
    said = await sent(page)
    sent_after_release = after(said, {"button": "release"})
    kept = [each for each in sent_after_release[: wake_at(sent_after_release)] if "audio" in each]
    assert kept and max(each["peak"] for each in kept) <= speech_peak * 1.1
    assert sum(each["audio"] for each in kept) <= KEPT_SECONDS * rate + 128


async def test_a_word_said_inside_a_button_turn_opens_nothing_and_is_a_hearing_in_hands_record_with_its_score(browser: Opened, hands: Hands) -> None:
    page = await listening(browser, hands)
    await button_down(page)
    await until(lambda: hands.gate.gate.turn_open)
    await speak(page, spoken(PHRASE))
    await page.mouse.up()
    await until(lambda: not hands.gate.gate.turn_open)
    assert [each for each in await sent(page) if each.get("kind") == "wake"] == []
    assert events(hands.recorded, "phone.wake") == []
    await until(lambda: bool(events(hands.recorded, "phone.ignored")))
    [ignored] = events(hands.recorded, "phone.ignored")
    assert isinstance(ignored.facts["score"], float) and ignored.facts["score"] >= 0.5


async def test_turned_off_during_a_wakes_turn_the_turn_is_sent(browser: Opened, hands: Hands) -> None:
    async with hearing(hands):
        page = await listening(browser, hands)
        await speak(page, spoken(PHRASE))
        await until(lambda: hands.gate.gate.turn_open)
        await page.get_by_role("button", name="Phone wake word").click()
        await until(lambda: hands.gate.gate.ended == ("sent",))
        await expect(page.locator("#state")).not_to_have_text("Listening…")
        assert hands.gate.gate.phone_level == "off"


async def test_holding_the_button_opens_a_turn_while_the_detector_loads(browser: Opened, hands: Hands) -> None:
    held = asyncio.Event()
    await browser.context.route("**/wake/settings", partial(held_open, held))
    page = await connected(browser, hands)
    await expect(page.locator("#state")).to_have_text(LOADING)
    await holds_and_opens(page, hands)
    held.set()


async def test_holding_the_button_during_a_wakes_turn_sends_it_and_opens_the_buttons_own(browser: Opened, hands: Hands) -> None:
    async with hearing(hands):
        page = await listening(browser, hands)
        await speak(page, spoken(PHRASE))
        await until(lambda: hands.gate.gate.turn_open and hands.gate.gate.opened == PHONE_WAKE)
        await button_down(page)
        await until(lambda: hands.gate.gate.opened == "phone button")
        assert hands.gate.gate.ended == ("sent",) and hands.gate.gate.turn_open
        await page.mouse.up()
        await until(lambda: not hands.gate.gate.turn_open)


@pytest.mark.parametrize("wake_models", [unreachable], indirect=True)
async def test_a_detector_that_cannot_load_is_said_with_why_and_told_to_hands_and_the_button_works(browser: Opened, hands: Hands) -> None:
    page = await connected(browser, hands)
    await expect(page.locator("#wake-note")).to_have_text("The wake word is not listening: hands cannot get openWakeWord v0.5.1's melspectrogram.onnx: the network is down.")
    await expect(page.locator("#state")).to_have_text(LOADING)
    [failed] = events(hands.recorded, "phone.detector")
    assert failed.outcome == "failed" and failed.facts["state"] == "failed" and "melspectrogram.onnx" in str(failed.error)
    assert hands.gate.gate.phone_level == "off"
    await holds_and_opens(page, hands)


async def test_a_hands_that_does_not_answer_for_the_detector_is_waited_on_and_the_wake_word_listens_once_it_does(browser: Opened, hands: Hands) -> None:
    refused = list[str]()

    async def lose_the_first(route: Route) -> None:
        if refused:
            await route.continue_()
        else:
            refused.append(route.request.url)
            await route.abort()

    models_come_in = asyncio.Event()
    await browser.context.route("**/wake/settings", lose_the_first)
    await browser.context.route("**/wake/models/melspectrogram", partial(held_open, models_come_in))
    page = await connected(browser, hands)
    # Lost, as a request past the page's bound is too, and waited on the same (`patiently`, tests/test_handsjs.py).
    await expect(page.locator("#wake-note")).to_have_text(re.compile(r"The wake word is waiting on hands: \S"))
    # hands answered: the models still coming in is no wait on hands, and the page does not go on saying it is.
    await expect(page.locator("#wake-note")).to_have_text("")
    await expect(page.locator("#state")).to_have_text(LOADING)
    models_come_in.set()
    await expect(page.locator("#state")).to_have_text(LISTENING)
    assert len(refused) == 1
    # hands' record has the wait as its own fact, and its end, ahead of the load: a detector waiting is not one that never ran,
    # and one that hands answered is not waiting still.
    assert detector_states(hands) == ["waiting", "answered", "loaded"]


async def test_a_wake_hands_answers_unopened_at_the_desk_says_so_and_leaves_the_page_waiting_for_the_word(browser: Opened, hands: Hands) -> None:
    page = await listening(browser, hands)
    await until(lambda: hands.gate.gate.phone_level == "wake word")
    # A hand at the desk takes hands there; the call and the phone's listening stay as they were.
    hands.gate.move("start", "shortcut")
    hands.gate.end(hands.gate.gate.turn, "stop", "shortcut")
    assert hands.gate.gate.place == "desk"
    await speak(page, spoken(PHRASE))
    await expect(page.locator("#state")).to_have_text("hands is at the desk. Hold the button to bring it here.")
    [refused] = events(hands.recorded, "phone.wake")
    assert refused.facts["opened"] is False and refused.outcome == "ok"
    # The button brings hands here, as the line says; and the page, waiting for the word, hears it: the next saying is a turn.
    await holds_and_opens(page, hands)
    await speak(page, spoken(PHRASE))
    await until(lambda: hands.gate.gate.turn_open and hands.gate.gate.opened == PHONE_WAKE)


async def test_a_wake_hands_refuses_for_a_phone_not_listening_says_hands_opened_no_turn_and_is_the_pages_mistake_in_hands_record(browser: Opened, hands: Hands) -> None:
    page = await listening(browser, hands)
    await until(lambda: hands.gate.gate.phone_level == "wake word")
    # hands has the phone deaf to the wake word, as a page whose detector hears a word it was not to would find it.
    hands.gate.move("deafen", PHONE_WAKE)
    await speak(page, spoken(PHRASE))
    await expect(page.locator("#state")).to_have_text("hands opened no turn for that: the phone is not listening for the wake word.")
    [refused] = events(hands.recorded, "phone.wake")
    assert refused.facts["opened"] is False and refused.outcome == "failed" and "not listening for the wake word" in str(refused.error)


async def test_a_wakes_turn_hands_threw_away_says_nothing_was_heard_after_the_phrase_and_leaves_the_page_waiting(browser: Opened, hands: Hands) -> None:
    async with hearing(hands):
        page = await listening(browser, hands)
        await speak(page, spoken(PHRASE))
        await until(lambda: hands.gate.gate.turn_open)
        hands_ends_the_turn(hands, "drop")
        await expect(page.locator("#state")).to_have_text('Heard nothing after "Hey Jarvis".')
        await speak(page, spoken(PHRASE))
        await until(lambda: hands.gate.gate.turn_open)
        assert hands.gate.gate.ended == ("dropped",)


async def test_a_call_that_drops_and_is_made_again_by_the_page_listens_for_the_wake_word_again_without_a_tap(browser: Opened, hands: Hands) -> None:
    page = await listening(browser, hands)
    await until(lambda: hands.gate.gate.phone_level == "wake word")
    await hands.phone.hang_up("hung up")
    await expect(page.locator("#state")).to_contain_text("Calling again")
    await expect(page.locator("#state")).to_have_text(LISTENING)
    await until(lambda: hands.gate.gate.phone_level == "wake word")
    assert detector_states(hands) == ["loaded", "loaded"]


async def test_a_call_that_ends_while_the_detectors_script_comes_in_starts_no_detector(browser: Opened, hands: Hands) -> None:
    held = asyncio.Event()
    arrived = asyncio.Event()

    async def hold_the_script(route: Route) -> None:
        arrived.set()
        await held_open(held, route)

    await browser.context.route("**/detector.js", hold_the_script)
    page = await connected(browser, hands)
    await asyncio.wait_for(arrived.wait(), 15)
    await hands.phone.hang_up("hung up")
    await expect(page.locator("#state")).to_contain_text("Calling again")
    await expect(page.locator("#state")).to_have_text(LOADING)
    held.set()
    await expect(page.locator("#state")).to_have_text(LISTENING)
    # One detector asked hands for its settings: the first call's script arrived to a call that was over, and started none.
    assert [path for path in hands.requests if path.startswith("/wake/settings")] == ["/wake/settings"]
    assert detector_states(hands) == ["loaded"]


async def test_a_browser_that_will_not_keep_the_choice_says_so_beside_the_button(browser: Opened, hands: Hands) -> None:
    await browser.context.add_init_script('if (window.isSecureContext) Object.defineProperty(window, "localStorage", { get() { throw new Error("storage is off"); } })')
    page = await browser.open(hands.address("/"))
    await expect(page.locator("#wake-note")).to_have_text("This browser will not keep the wake word's choice: storage is off.")
    await expect(page.get_by_role("button", name="Phone wake word")).to_have_attribute("aria-pressed", "true")


async def test_the_page_holds_a_screen_wake_lock_while_the_wake_word_is_on_and_the_call_is_up_and_none_otherwise(browser: Opened, hands: Hands) -> None:
    for script in (SPEECH, WAKE_LOCK):
        await browser.context.add_init_script(script)
    page = await browser.open(hands.address("/"))
    assert await page.evaluate("locks") == 0
    await page.get_by_role("button", name="Connect").click()
    await page.wait_for_function("locks === 1")
    switch = page.get_by_role("button", name="Phone wake word")
    await switch.click()
    await page.wait_for_function("locks === 0")
    await switch.click()
    await page.wait_for_function("locks === 1")
    # The call drops: no lock while it is down, and one again when the page has called again.
    await hands.phone.hang_up("hung up")
    await page.wait_for_function("locks === 0")
    await page.wait_for_function("locks === 1")
    # Each lock the page got is in hands' record.
    assert {event.facts["state"] for event in events(hands.recorded, "phone.screen")} == {"kept"}


async def test_a_screen_lock_the_browser_takes_back_is_said_and_told_to_hands(browser: Opened, hands: Hands) -> None:
    for script in (SPEECH, WAKE_LOCK):
        await browser.context.add_init_script(script)
    page = await browser.open(hands.address("/"))
    await page.get_by_role("button", name="Connect").click()
    await page.wait_for_function("locks === 1")
    await until(lambda: len(events(hands.recorded, "phone.screen")) == 1)
    await page.evaluate("takeBack()")
    await expect(page.locator("#wake-note")).to_have_text("The screen may go to sleep, which stops the wake word: the browser let go of it.")
    await until(lambda: len(events(hands.recorded, "phone.screen")) == 2)
    kept, unkept = events(hands.recorded, "phone.screen")
    assert (kept.facts["state"], unkept.facts["state"], unkept.error) == ("kept", "unkept", "the browser let go of it")


async def test_a_browser_with_no_screen_wake_lock_says_the_screen_may_sleep_and_tells_hands(browser: Opened, hands: Hands) -> None:
    await browser.context.add_init_script('if (window.isSecureContext) Object.defineProperty(navigator, "wakeLock", { value: undefined })')
    page = await connected(browser, hands)
    await expect(page.locator("#wake-note")).to_have_text("The screen may go to sleep, which stops the wake word: this browser has no screen wake lock.")
    await until(lambda: bool(events(hands.recorded, "phone.screen")))
    [unkept] = events(hands.recorded, "phone.screen")
    assert unkept.outcome == "failed" and unkept.facts["state"] == "unkept" and unkept.error == "this browser has no screen wake lock"
