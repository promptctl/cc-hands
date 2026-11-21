"""The script every phone page asks hands through (src/hands/voice/hands.js), run in QuickJS against a fake hands: its
requests are held until the test answers them, and its timers until the test fires them, so no time passes but the test's.
"""

import json
from dataclasses import dataclass
from importlib import resources
from typing import cast

import pytest
import quickjs

# The browser as the script meets it: an address whose fragment is the key, a fetch hands answers when the test says, a
# timer that fires when the test says, and the bound each request was given.
BROWSER = """
var location = {hash: "#the%20key"};
var requests = [], timers = [], drawn = [], unread = [];
function fetch(path, init) { return new Promise((resolve, reject) => requests.push({path, init, resolve, reject})); }
function setTimeout(fire, ms) { timers.push({fire, ms}); }
var AbortSignal = {timeout: ms => ({bound: ms})};
function answer(status, text) {
  requests.shift().resolve({ok: status >= 200 && status < 300, status, text: async () => text});
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

    def __init__(self) -> None:
        self._js = quickjs.Context()
        self._js.eval(BROWSER)
        self._js.eval(resources.files("hands.voice").joinpath("hands.js").read_text())
        self.answer_ms = cast(int, self._js.eval("ANSWER_MS"))

    def run(self, code: str) -> None:
        self._js.eval(code)
        self.settle()

    def settle(self) -> None:
        """Every job the script has to run before it next waits on hands or a timer."""
        while self._js.execute_pending_job():
            pass

    def asking(self) -> list[Asked]:
        return [Asked(**each) for each in json.loads(cast(str, self._js.eval("asking()")))]

    def answer(self, status: int, text: str) -> None:
        self.run(f"answer({status}, {json.dumps(text)})")

    def read(self, seen: object, shown: object, waits: float = 25.0) -> None:
        self.answer(200, json.dumps({"seen": seen, "waits": waits, "shown": shown}))

    def timers(self) -> list[int]:
        return json.loads(cast(str, self._js.eval("JSON.stringify(timers.map(timer => timer.ms))")))

    def seen(self, name: str) -> list[object]:
        return json.loads(cast(str, self._js.eval(f"JSON.stringify({name})")))


@pytest.mark.parametrize("name", ["phone.html", "conversation.html", "gallery.html", "backlog.html"])
def test_every_phone_page_asks_hands_through_the_script_alone(name: str) -> None:
    html = resources.files("hands.voice").joinpath(name).read_text()
    assert ('<script src="/hands.js"></script>' in html, "fetch(" in html) == (True, False)


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
    ('answer(400, "the page sends {voice: ...}")', {"kind": "refused", "status": 400, "text": "the page sends {voice: ...}"}),
    ('answer(503, "")', {"kind": "refused", "status": 503, "text": "hands refused with 503"}),
    ('cutOff("network reset")', {"kind": "lost", "why": "hands' answer was cut off: network reset"}),
    ('lose("TimeoutError")', {"kind": "lost", "why": "hands did not answer within 60 s"}),
    ('lose("TypeError")', {"kind": "lost", "why": "hands is out of reach"}),
])
def test_what_came_of_a_request_is_one_of_three_outcomes(page: Page, settled: str, outcome: dict[str, object]) -> None:
    page.run('var outcome; asked("GET", "/state").then(said => { outcome = said; })')
    page.run(settled)
    assert page.seen("outcome") == outcome


def test_an_answer_that_is_not_json_is_lost_with_why(page: Page) -> None:
    page.run('var outcome; asked("GET", "/state").then(said => { outcome = said; })')
    page.answer(200, "<html>")
    outcome = cast(dict[str, str], page.seen("outcome"))
    # The rest of why is the browser's own words for what it could not parse.
    assert (outcome["kind"], outcome["why"].startswith("hands' answer is not one this page reads: ")) == ("lost", True)


def test_a_read_asks_at_once_then_says_what_it_saw_bounded_by_the_wait_hands_said(reading: Page) -> None:
    assert reading.asking() == [Asked("GET", "/state", "Bearer the key", None, reading.answer_ms)]
    reading.read("v1", {"voice": "mary"}, waits=25.0)
    assert reading.seen("drawn") == [{"voice": "mary"}]
    assert reading.asking() == [Asked("GET", "/state?seen=v1", "Bearer the key", None, 25000 + reading.answer_ms)]


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
    assert reading.seen("unread") == ["hands is out of reach; reading it again…"]
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


def test_a_read_refused_for_the_key_stops_with_hands_words(reading: Page) -> None:
    reading.answer(401, "this page's address is missing the phone's key")
    assert reading.seen("unread") == ["this page's address is missing the phone's key"]
    assert (reading.asking(), reading.timers()) == ([], [])


def test_an_answer_that_is_not_a_read_stops_the_reading_and_says_so(reading: Page) -> None:
    reading.answer(200, '{"voice": "mary"}')
    assert reading.seen("unread") == ["This page stopped reading hands: hands' answer is not a read. Reload it to read again."]
    assert (reading.seen("drawn"), reading.asking(), reading.timers()) == ([], [], [])


def test_a_read_of_a_path_with_a_query_says_what_it_saw_after_it(page: Page) -> None:
    page.run('reading("/state?project=hands", () => {}, () => {})')
    page.read(3, {})
    assert [asked.path for asked in page.asking()] == ["/state?project=hands&seen=3"]
