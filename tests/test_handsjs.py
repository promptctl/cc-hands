"""The module every phone page asks hands through (src/hands/voice/hands.js), run in QuickJS against a fake hands: its
requests are held until the test answers them, and its timers until the test fires them, so no time passes but the test's.
What hands answers a read is built as hands builds it (hands.voice.longpoll.Read).
"""

import json
from dataclasses import dataclass
from typing import cast

import pytest
from jsmodule import JsModule

from hands.voice.longpoll import WAIT_SECONDS, Read

# The browser as the module meets it: an address whose fragment is the key, a page's clock that stands still, a fetch
# hands answers when the test says, a timer that fires when the test says, and the bound each request was given.
BROWSER = """
var console = {error() {}};
Date.now = () => 1000000;
var requests = [], timers = [], drawn = [], unread = [];
function fetch(path, init) { return new Promise((resolve, reject) => requests.push({path, init, resolve, reject})); }
function setTimeout(fire, ms) { timers.push({fire, ms}); }
var AbortSignal = {timeout: ms => ({bound: ms})};
function answer(status, text) {
  requests.shift().resolve({ok: status >= 200 && status < 300, status, text: async () => text});
}
function answerBytes(status, bytes) {
  requests.shift().resolve({ok: status >= 200 && status < 300, status, text: async () => "refused in words", arrayBuffer: async () => new Uint8Array(bytes).buffer});
}
function cutOffBytes(message) {
  requests.shift().resolve({ok: true, status: 200, arrayBuffer: async () => { throw new Error(message); }});
}
function cutOff(message) {
  requests.shift().resolve({ok: true, status: 200, text: async () => { throw new Error(message); }});
}
function lose(name) { const error = new Error("lost"); error.name = name; requests.shift().reject(error); }
function fire() { timers.shift().fire(); }
function asking() {
  return JSON.stringify(requests.map(({path, init}) => ({method: init.method, path, key: init.headers.Authorization, body: init.body ?? null, bound: init.signal.bound})));
}
"""


@dataclass(frozen=True)
class Asked:
    method: str
    path: str
    key: str
    body: str | None
    bound: int


class Page:
    """A page holding the script, and the fake hands it asks."""

    def __init__(self, fragment: str = "#the%20key") -> None:
        self._js = JsModule(
            "hands.js", "asked, askedForBytes, unanswered, patiently, reading, handsNow, carriesKey, ANSWER_MS, CROSSING_MS", f"var location = {{hash: {json.dumps(fragment)}}};{BROWSER}"
        )
        self.answer_ms = cast(int, self._js.eval("ANSWER_MS"))
        self.crossing_ms = cast(int, self._js.eval("CROSSING_MS"))

    def run(self, code: str) -> None:
        self._js.run(code)

    def settle(self) -> None:
        """Every job the script has to run before it next waits on hands or a timer."""
        self._js.settle()

    def asking(self) -> list[Asked]:
        return [Asked(**each) for each in json.loads(cast(str, self._js.eval("asking()")))]

    def answer(self, status: int, text: str) -> None:
        self.run(f"answer({status}, {json.dumps(text)})")

    def read(self, seen: str | int, shown: object, now: float = 5000.0) -> None:
        self.answer(200, json.dumps(Read(seen=seen, waits=WAIT_SECONDS, now=now, shown=shown)))

    def timers(self) -> list[int]:
        return cast(list[int], self._js.value("timers.map(timer => timer.ms)"))

    def seen(self, name: str) -> list[object]:
        return cast(list[object], self._js.value(name))


@pytest.fixture
def page() -> Page:
    return Page()


@pytest.fixture
def reading(page: Page) -> Page:
    page.run('reading("/state", shown => drawn.push(shown), why => unread.push(why))')
    return page


def test_a_request_carries_the_key_from_the_address_and_a_choice_as_json(page: Page) -> None:
    page.run('var outcome; asked("POST", "/choose", {voice: "mary"}).then(said => { outcome = said; })')
    assert page.asking() == [Asked("POST", "/choose", "Bearer the key", '{"voice":"mary"}', page.answer_ms)]
    page.answer(200, '{"said": "Mary it is."}')
    assert page.seen("outcome") == {"kind": "answered", "body": {"said": "Mary it is."}}


@pytest.mark.parametrize(("settled", "outcome"), [
    ('answer(202, "")', {"kind": "answered", "body": None}),
    ('answer(400, "the page sends {voice: ...}")', {"kind": "refused", "status": 400, "transient": False, "text": "the page sends {voice: ...}"}),
    ('answer(503, "")', {"kind": "refused", "status": 503, "transient": True, "text": "hands refused with 503"}),
    ('cutOff("network reset")', {"kind": "lost", "transient": True, "why": "hands' answer was cut off: network reset"}),
    ('lose("TimeoutError")', {"kind": "lost", "transient": True, "why": "hands did not answer within 60 s"}),
    ('lose("TypeError")', {"kind": "lost", "transient": True, "why": "hands is out of reach: lost"}),
])
def test_what_came_of_a_request_is_one_of_three_outcomes(page: Page, settled: str, outcome: dict[str, object]) -> None:
    page.run('var outcome; asked("GET", "/state").then(said => { outcome = said; })')
    page.run(settled)
    assert page.seen("outcome") == outcome


def test_a_file_is_asked_for_with_the_key_and_answered_as_its_bytes(page: Page) -> None:
    page.run('var outcome; askedForBytes("/wake/models/word.onnx").then(said => { outcome = said; })')
    assert page.asking() == [Asked("GET", "/wake/models/word.onnx", "Bearer the key", None, page.answer_ms)]
    page.run("answerBytes(200, [1, 2, 255])")
    page.run("outcome.body = Array.from(outcome.body)")
    assert page.seen("outcome") == {"kind": "answered", "body": [1, 2, 255]}


@pytest.mark.parametrize(("settled", "outcome"), [
    ("answerBytes(503, [1, 2])", {"kind": "refused", "status": 503, "transient": True, "text": "refused in words"}),
    ('cutOffBytes("network reset")', {"kind": "lost", "transient": True, "why": "hands' answer was cut off: network reset"}),
    ('lose("TimeoutError")', {"kind": "lost", "transient": True, "why": "hands did not answer within 60 s"}),
    ('lose("TypeError")', {"kind": "lost", "transient": True, "why": "hands is out of reach: lost"}),
])
def test_what_came_of_a_request_for_a_file_is_one_of_the_same_three_outcomes(page: Page, settled: str, outcome: dict[str, object]) -> None:
    page.run('var outcome; askedForBytes("/wake/models/word.onnx").then(said => { outcome = said; })')
    page.run(settled)
    assert page.seen("outcome") == outcome


PATIENTLY = 'var outcome, waits = []; patiently(() => asked("GET", "/wake/settings"), why => waits.push(why)).then(said => { outcome = said; })'


def test_a_request_hands_does_not_answer_in_time_is_asked_again_after_a_pause_saying_why_until_hands_answers(page: Page) -> None:
    page.run(PATIENTLY)
    page.run('lose("TimeoutError")')
    # Nothing is asked until the pause is over.
    assert (page.seen("waits"), page.asking(), page.timers()) == (["hands did not answer within 60 s"], [], [2000])
    page.run("fire()")
    assert [asked.path for asked in page.asking()] == ["/wake/settings"]
    page.answer(200, '{"phrase": "hey jarvis"}')
    # The wait is over once hands answers, and said to be.
    assert (page.seen("waits"), page.seen("outcome")) == (["hands did not answer within 60 s", None], {"kind": "answered", "body": {"phrase": "hey jarvis"}})


def test_a_request_hands_could_not_answer_now_is_asked_again_after_a_pause_saying_hands_words(page: Page) -> None:
    page.run(PATIENTLY)
    # As a cold first run whose fetch of the models failed: hands fetches again as it is next asked.
    page.answer(503, "hands could not fetch the word's model")
    assert (page.seen("waits"), page.asking(), page.timers()) == (["hands could not fetch the word's model"], [], [2000])
    page.run("fire()")
    assert [asked.path for asked in page.asking()] == ["/wake/settings"]


@pytest.mark.parametrize(("status", "text"), [(401, "this page's address is missing the phone's key"), (404, "Not Found")])
def test_a_request_refused_as_the_pages_own_mistake_is_hands_word_and_is_not_asked_again(page: Page, status: int, text: str) -> None:
    page.run(PATIENTLY)
    page.answer(status, text)
    assert (page.seen("waits"), page.asking(), page.timers()) == ([], [], [])
    assert page.seen("outcome") == {"kind": "refused", "status": status, "transient": False, "text": text}


def test_an_answer_that_is_not_json_is_lost_with_why_and_would_come_again_the_same(page: Page) -> None:
    page.run('var outcome; asked("GET", "/state").then(said => { outcome = said; })')
    page.answer(200, "<html>")
    outcome = cast(dict[str, object], page.seen("outcome"))
    # The rest of why is the browser's own words for what it could not parse.
    assert (outcome["kind"], outcome["transient"], str(outcome["why"]).startswith("hands' answer is not one this page reads: ")) == ("lost", False, True)


def test_an_answer_patiently_cannot_read_is_not_asked_again(page: Page) -> None:
    page.run(PATIENTLY)
    page.answer(200, "<html>")
    assert (page.seen("waits"), page.asking(), page.timers()) == ([], [], [])


def test_a_read_asks_at_once_then_says_what_it_saw_bounded_by_the_wait_hands_said(reading: Page) -> None:
    assert reading.asking() == [Asked("GET", "/state", "Bearer the key", None, reading.answer_ms)]
    reading.read("v1", {"voice": "mary"})
    assert reading.seen("drawn") == [{"voice": "mary"}]
    assert reading.asking() == [Asked("GET", "/state?seen=v1", "Bearer the key", None, round(WAIT_SECONDS * 1000) + reading.crossing_ms)]


def test_a_read_answered_unchanged_draws_nothing_and_one_changed_draws_it(reading: Page) -> None:
    reading.read("v1", {"voice": "mary"})
    reading.read("v1", {"voice": "mary"})
    assert reading.seen("drawn") == [{"voice": "mary"}]
    reading.read("v2", {"voice": "bill"})
    assert reading.seen("drawn") == [{"voice": "mary"}, {"voice": "bill"}]
    assert [asked.path for asked in reading.asking()] == ["/state?seen=v2"]


def test_a_lost_read_is_said_and_read_again_after_a_pause_with_nothing_seen(reading: Page) -> None:
    reading.read("v1", {"voice": "mary"})
    reading.run('lose("TypeError")')
    assert reading.seen("unread") == ["hands is out of reach: lost; reading it again…"]
    # Nothing is asked until the pause is over.
    assert (reading.asking(), reading.timers()) == ([], [2000])
    reading.run("fire()")
    assert reading.asking() == [Asked("GET", "/state", "Bearer the key", None, reading.answer_ms)]
    # The run hands is back with answers it at once, the same as the page last drew or not, and it is drawn.
    reading.read("v1", {"voice": "mary"})
    assert reading.seen("drawn") == [{"voice": "mary"}, {"voice": "mary"}]


def test_a_read_refused_for_any_other_reason_than_the_key_is_read_again(reading: Page) -> None:
    reading.answer(503, "hands is restarting")
    assert (reading.seen("unread"), reading.timers()) == (["hands is restarting; reading it again…"], [2000])


@pytest.mark.parametrize(("status", "text"), [(401, "this page's address is missing the phone's key"), (404, "Not Found")])
def test_a_read_refused_as_the_pages_own_mistake_stops_with_hands_words(reading: Page, status: int, text: str) -> None:
    reading.answer(status, text)
    assert reading.seen("unread") == [text]
    assert (reading.asking(), reading.timers()) == ([], [])


def test_a_read_whose_answer_is_not_json_stops_the_reading_and_says_why(reading: Page) -> None:
    reading.answer(200, "<html>")
    [why] = cast(list[str], reading.seen("unread"))
    assert (why.startswith("hands' answer is not one this page reads: "), reading.asking(), reading.timers()) == (True, [], [])


@pytest.mark.parametrize("answer", [
    {"voice": "mary"},
    {"seen": None, "waits": WAIT_SECONDS, "now": 5000.0, "shown": {}},
    {"seen": "v1", "waits": WAIT_SECONDS, "now": 5000.0},
    None,
])
def test_an_answer_that_is_not_a_read_stops_the_reading_and_says_so(reading: Page, answer: object) -> None:
    reading.answer(200, json.dumps(answer))
    assert reading.seen("unread") == ["This page stopped reading hands: hands' answer is not a read. Reload it to read again."]
    assert (reading.seen("drawn"), reading.asking(), reading.timers()) == ([], [], [])


def test_a_read_the_page_cannot_draw_stops_the_reading_and_says_why(page: Page) -> None:
    page.run('reading("/state", () => { throw new Error("no such row"); }, why => unread.push(why))')
    page.read("v1", {})
    assert (page.seen("unread"), page.asking()) == (["This page stopped reading hands: no such row. Reload it to read again."], [])


def test_hands_clock_is_as_every_read_last_said_it_changed_or_not(reading: Page) -> None:
    reading.read("v1", {}, now=5000.0)
    assert reading.seen("handsNow()") == 5000.0
    reading.read("v1", {}, now=5025.0)
    assert (reading.seen("handsNow()"), reading.seen("drawn")) == (5025.0, [{}])


@pytest.mark.parametrize(("fragment", "carries"), [("#the%20key", True), ("", False), ("#", False)])
def test_a_page_knows_whether_its_address_carries_a_key(fragment: str, carries: bool) -> None:
    assert Page(fragment).seen("carriesKey") is carries
