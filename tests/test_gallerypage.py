"""The gallery on the phone's page: the voices heard and chosen, the listening level, the modality, and the model, each
choice the one its tool makes, and each refused where the tool refuses it."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.clock import SYSTEM
from hands.core.place import Modality
from hands.core.playback import Labelled
from hands.sessions.audit import Entry
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent
from hands.voice.desk import Level
from hands.voice.gallerypage import Gallery, gallery_routes
from hands.voice.library import Found
from hands.voice.voices import Chosen
from hands.voice.trigger import described
from hands.voice.voices import VOICES, Voice
from hands.voice.wakeword import Pretrained
from conftest import events
from pagescript import Click, Era, Reply, Submit, Wait, run_page

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")
# The run the page is served by.
STARTED = datetime(2026, 11, 20, 9, tzinfo=UTC)


@dataclass
class Voicing:
    """The voices as the test sets them: the one spoken, and what each call was asked."""

    spoken: Chosen = Voice("charles")
    heard_asked: list[tuple[Chosen, ...]] = field(default_factory=list[tuple[Chosen, ...]])
    used_asked: list[Chosen] = field(default_factory=list[Chosen])
    failure: Exception | None = None

    async def speaking_in(self) -> Chosen:
        return self.spoken

    async def hear(self, voices: tuple[Chosen, ...]) -> Chosen:
        self.heard_asked.append(voices)
        if self.failure is not None:
            raise self.failure
        return self.spoken

    async def use(self, voice: Chosen) -> None:
        self.used_asked.append(voice)
        self.spoken = voice

    async def search(self, query: str, limit: int) -> Found:
        return Found(tuple(name for name in ("vctk/p225", "alba-mackenna/casual") if query in name)[:limit], 3.0, True)


@dataclass
class Listening:
    level: Level = "off"
    chosen: list[Level] = field(default_factory=list[Level])

    async def choose(self, level: Level) -> Level:
        was, self.level = self.level, level
        self.chosen.append(level)
        return was


@dataclass
class Desk:
    modality: Modality = "screen"
    switched: list[Modality] = field(default_factory=list[Modality])

    def switch(self, to: Modality) -> None:
        self.modality = to
        self.switched.append(to)


@dataclass
class Speaker:
    """Hands' speech as the model switch reads it: a line, and whether the user heard it to its end."""

    heard_in_full: bool = True
    hold: float = 0.0
    said: list[str] = field(default_factory=list[str])

    async def heard(self, line: Labelled) -> bool:
        self.said.append(line.text)
        await asyncio.sleep(self.hold)
        return self.heard_in_full


@dataclass
class Own:
    """Hands' own model: the one it runs on, and a weighing that either refuses or gives the keep the edit waits on."""

    running_on: str = MODELS[0]
    refused: Exception | None = None
    kept: list[str] = field(default_factory=list[str])
    changing: asyncio.Lock = field(default_factory=asyncio.Lock)
    open_weighings: int = 0
    most_open: int = 0

    def running(self) -> str:
        return self.running_on

    def weigh(self, model: str) -> Callable[[], None]:
        if self.refused is not None:
            raise self.refused
        self.open_weighings += 1
        self.most_open = max(self.most_open, self.open_weighings)

        def keep() -> None:
            self.open_weighings -= 1
            self.kept.append(model)

        return keep


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    voices: Voicing
    levels: Listening
    desk: Desk
    speaker: Speaker
    own: Own
    recorded: list[Entry]
    gallery: Gallery

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


@pytest.fixture
async def page() -> AsyncGenerator[Page]:
    voices, levels, desk, speaker, own = Voicing(), Listening(), Desk(), Speaker(), Own()
    recorded: list[Entry] = []
    gallery = Gallery(voices=voices, levels=levels, desk=desk, own=own, speaker=speaker, wake=Pretrained(), models=MODELS, started_at=STARTED)
    async with _served(gallery, recorded) as client:
        yield Page(client, voices, levels, desk, speaker, own, recorded, gallery)


@asynccontextmanager
async def _served(gallery: Gallery, recorded: list[Entry]) -> AsyncGenerator[TestClient[web.Request, web.Application]]:
    app = web.Application()
    app.add_routes(gallery_routes(SYSTEM, gallery, KEY, recorded.append))
    async with TestClient(TestServer(app)) as client:
        yield client


@pytest.mark.outside
async def test_the_page_is_served_and_its_reads_and_choices_need_the_phones_key(page: Page) -> None:
    shown = await page.client.get("/gallery")
    assert shown.status == 200 and "hands' voices and options" in await shown.text()
    assert shown.headers["Cache-Control"] == "no-store"

    refused = await page.client.get("/gallery/state")
    assert refused.status == 401
    refused = await page.client.post("/gallery/use", json={"voice": "bill_boerst"})
    assert refused.status == 401
    assert page.voices.used_asked == []
    assert [event.outcome for event in page.events("gallery.read")] == ["failed"]


@pytest.mark.outside
async def test_the_read_lists_every_voice_the_speaking_one_and_the_options(page: Page) -> None:
    page.levels.level = "engaged"
    response = await page.client.get("/gallery/state", headers=KEYED)
    body = await response.json()
    assert body == {
        "voices": list(VOICES), "speaking_in": "charles",
        "levels": ["off", "wake word", "engaged"], "level": "engaged",
        "modalities": ["screen", "audio-only"], "modality": "screen",
        "models": list(MODELS), "running_on": MODELS[0], "started_at": STARTED.isoformat(),
    }
    [read] = page.events("gallery.read")
    assert (read.outcome, read.facts["speaking_in"], read.facts["level"], read.facts["modality"]) == ("ok", "charles", "engaged", "screen")


@pytest.mark.outside
async def test_hearing_a_voice_says_it_once_and_names_the_voice_in_the_event(page: Page) -> None:
    response = await page.client.post("/gallery/hear", headers=KEYED, json={"voice": "Bill Boerst"})
    assert response.status == 200
    assert await response.json() == {"heard": "bill_boerst", "speaking_in": "charles"}
    assert page.voices.heard_asked == [("bill_boerst",)]
    [hear] = page.events("gallery.hear")
    assert (hear.outcome, hear.facts["heard"], hear.facts["speaking_in"]) == ("ok", "bill_boerst", "charles")


@pytest.mark.outside
async def test_a_voice_name_hands_cannot_read_is_refused_and_nothing_is_heard(page: Page) -> None:
    response = await page.client.post("/gallery/hear", headers=KEYED, json={"voice": "Zed!"})
    assert response.status == 400
    assert "hands has no voice called 'Zed!'" in await response.text()
    assert page.voices.heard_asked == []
    assert [hear.outcome for hear in page.events("gallery.hear")] == ["failed"]


@pytest.mark.outside
async def test_a_voice_that_cannot_be_fetched_is_a_503_and_says_why(page: Page) -> None:
    page.voices.failure = OSError("no network for the voice's embedding")
    response = await page.client.post("/gallery/hear", headers=KEYED, json={"voice": "charles"})
    assert response.status == 503
    assert "no network" in await response.text()
    assert [hear.outcome for hear in page.events("gallery.hear")] == ["failed"]


@pytest.mark.outside
async def test_choosing_a_voice_keeps_it_as_the_one_hands_speaks_in(page: Page) -> None:
    response = await page.client.post("/gallery/use", headers=KEYED, json={"voice": "mary"})
    assert response.status == 200
    assert await response.json() == {"speaking_in": "mary"}
    assert page.voices.used_asked == ["mary"]
    [use] = page.events("gallery.use")
    assert (use.outcome, use.facts["voice"]) == ("ok", "mary")


@pytest.mark.outside
async def test_a_body_that_is_not_a_request_is_refused_with_what_it_should_be(page: Page) -> None:
    response = await page.client.post("/gallery/use", headers=KEYED, data=b"not json")
    assert response.status == 400
    response = await page.client.post("/gallery/use", headers=KEYED, json={"name": "mary"})
    assert response.status == 400
    assert "{voice: ...}" in await response.text()
    assert page.voices.used_asked == []


@pytest.mark.outside
async def test_a_listening_level_is_chosen_and_its_readback_is_the_one_the_voice_tool_gives(page: Page) -> None:
    response = await page.client.post("/gallery/level", headers=KEYED, json={"level": "wake word"})
    body = await response.json()
    assert body == {"level": "wake word", "was": "off", "readback": described("wake word", Pretrained())}
    assert page.levels.chosen == ["wake word"]
    [choice] = page.events("gallery.level")
    assert (choice.outcome, choice.facts["level"], choice.facts["was"]) == ("ok", "wake word", "off")


@pytest.mark.outside
async def test_a_level_hands_does_not_listen_at_is_refused(page: Page) -> None:
    response = await page.client.post("/gallery/level", headers=KEYED, json={"level": "loud"})
    assert response.status == 400
    assert page.levels.chosen == []


@pytest.mark.outside
async def test_a_modality_is_switched_and_one_outside_the_two_is_refused(page: Page) -> None:
    response = await page.client.post("/gallery/modality", headers=KEYED, json={"modality": "audio-only"})
    assert await response.json() == {"modality": "audio-only"}
    assert page.desk.switched == ["audio-only"] and page.desk.modality == "audio-only"
    refused = await page.client.post("/gallery/modality", headers=KEYED, json={"modality": "braille"})
    assert refused.status == 400
    assert page.desk.switched == ["audio-only"]
    assert [choice.outcome for choice in page.events("gallery.modality")] == ["ok", "failed"]


@pytest.mark.outside
async def test_a_model_is_switched_once_its_line_is_heard_and_names_the_switch(page: Page) -> None:
    response = await page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-opus-5-5"})
    assert response.status == 200
    body = await response.json()
    assert body["switching_to"] == "claude-opus-5-5"
    assert page.speaker.said == ["Switching to claude-opus-5-5. I'll be back in a few seconds."]
    assert page.own.kept == ["claude-opus-5-5"]
    [switch] = page.events("gallery.model")
    assert (switch.outcome, switch.facts["model"], switch.facts["switching"]) == ("ok", "claude-opus-5-5", True)


@pytest.mark.outside
async def test_a_model_switch_talked_over_is_refused_and_hands_stays_on_its_model(page: Page) -> None:
    # The restart waits on the line being heard: a barge-in over it leaves the edit unkept, as the voice tool does.
    page.speaker.heard_in_full = False
    response = await page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-opus-5-5"})
    assert response.status == 400
    assert "spoke over hands" in await response.text()
    assert page.own.kept == []
    [switch] = page.events("gallery.model")
    assert (switch.outcome, switch.facts["switching"]) == ("failed", False)


@pytest.mark.outside
async def test_two_model_switches_at_once_are_made_one_after_the_other(page: Page) -> None:
    # Each weighs the file as the one before it kept it: never two weighed and open at once.
    page.speaker.hold = 0.05
    first, second = await asyncio.gather(
        page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-opus-5-5"}),
        page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-sonnet-5-5"}),
    )
    assert (first.status, second.status) == (200, 200)
    assert page.own.most_open == 1
    assert sorted(page.own.kept) == ["claude-opus-5-5", "claude-sonnet-5-5"]


@pytest.mark.outside
async def test_a_model_the_weighing_refuses_is_a_400_and_nothing_is_said(page: Page) -> None:
    page.own.refused = Rejected("hands runs on claude-sonnet-5-5 already")
    response = await page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-sonnet-5-5"})
    assert response.status == 400
    assert "already" in await response.text()
    assert page.speaker.said == []
    assert page.own.kept == []


@pytest.mark.outside
async def test_a_model_the_machine_cannot_keep_is_a_503_and_says_why(page: Page) -> None:
    page.own.refused = OSError("config.toml is not writable")
    response = await page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-opus-5-5"})
    assert response.status == 503
    assert "not writable" in await response.text()
    assert page.speaker.said == []
    [switch] = page.events("gallery.model")
    assert (switch.outcome, switch.facts["switching"]) == ("failed", False)


@pytest.mark.outside
async def test_the_library_search_returns_matches_and_records_the_listing(page: Page) -> None:
    response = await page.client.get("/gallery/library?q=vctk&limit=5", headers=KEYED)
    assert await response.json() == {"voices": ["vctk/p225"], "listed_at": 3.0, "from_cache": True}
    [search] = page.events("gallery.library")
    assert (search.outcome, search.facts["query"], search.facts["matches"], search.facts["from_cache"]) == ("ok", "vctk", 1, True)


def test_the_routes_are_the_gallery_and_its_five_choices_and_a_read() -> None:
    # The page's routes are named in one place, so the page's requests and these routes cannot drift apart.
    gallery = Gallery(voices=Voicing(), levels=Listening(), desk=Desk(), own=Own(), speaker=Speaker(), wake=Pretrained(), models=MODELS, started_at=STARTED)
    routes: list[Any] = [route for route in gallery_routes(SYSTEM, gallery, KEY, lambda _entry: None)]
    assert sorted(f"{route.method} {route.path}" for route in routes) == sorted([
        "GET /gallery", "GET /gallery/state", "GET /gallery/library", "POST /gallery/hear", "POST /gallery/use",
        "POST /gallery/level", "POST /gallery/modality", "POST /gallery/model",
    ])


async def _reply(client: TestClient[web.Request, web.Application], method: str, path: str, body: object | None = None) -> Reply:
    response = await client.request(method, path, headers=KEYED, json=body)
    return Reply(response.status, await response.text())


async def _run_after(page: Page, running_on: str) -> Reply:
    """hands' state as the run a switch restarts into answers it: started later, on `running_on`."""
    page.own.running_on = running_on
    async with _served(replace(page.gallery, started_at=STARTED + timedelta(minutes=1)), []) as client:
        return await _reply(client, "GET", "/gallery/state")


STATE, SWITCH = "GET /gallery/state", "POST /gallery/model"
# The person switches at 1 s; hands restarts from 2 s, out of reach until 9 s.
SWITCHED = Submit("model-form", 1000, {"model": "claude-opus-5-5"})


@pytest.mark.outside
async def test_a_switch_is_waited_out_and_the_page_comes_back_on_the_run_after_it(page: Page) -> None:
    before, switched = await _reply(page.client, "GET", "/gallery/state"), await _reply(page.client, "POST", "/gallery/model", {"model": "claude-opus-5-5"})
    after = await _run_after(page, "claude-opus-5-5")
    opened, _, waiting, back = run_page("gallery.html", [Era(0, {STATE: before, SWITCH: switched}), Era(2000, {STATE: None}), Era(9000, {STATE: after})], [Wait(0), SWITCHED, Wait(5000), Wait(20000)])
    assert (opened.running, opened.disabled) == ("hands runs on claude-sonnet-5-5.", 0)
    # While hands restarts the page says so, and its controls stay live.
    assert (waiting.reading, waiting.status, waiting.disabled) == ("hands is out of reach; reading hands again…", page.speaker.said[0], 0)
    assert (back.running, back.reading, back.errors) == ("hands runs on claude-opus-5-5.", "hands is back, on claude-opus-5-5.", [])
    assert back.controls == opened.controls and back.disabled == 0


@pytest.mark.outside
async def test_a_run_after_a_switch_on_another_model_is_said_with_the_model_it_runs_on(page: Page) -> None:
    before, switched = await _reply(page.client, "GET", "/gallery/state"), await _reply(page.client, "POST", "/gallery/model", {"model": "claude-opus-5-5"})
    after = await _run_after(page, "claude-sonnet-5-5")
    *_, back = run_page("gallery.html", [Era(0, {STATE: before, SWITCH: switched}), Era(2000, {STATE: None}), Era(9000, {STATE: after})], [Wait(0), SWITCHED, Wait(20000)])
    assert (back.running, back.reading, back.errors) == ("hands runs on claude-sonnet-5-5.", "hands came back on claude-sonnet-5-5, not claude-opus-5-5.", ["reading"])


@pytest.mark.outside
async def test_the_run_before_a_switch_answering_is_not_taken_for_hands_back(page: Page) -> None:
    before, switched = await _reply(page.client, "GET", "/gallery/state"), await _reply(page.client, "POST", "/gallery/model", {"model": "claude-opus-5-5"})
    _, still = run_page("gallery.html", [Era(0, {STATE: before, SWITCH: switched})], [Wait(0), SWITCHED])
    assert (still.running, still.reading) == ("hands runs on claude-sonnet-5-5.", "hands still runs on claude-sonnet-5-5; the page reads it until it restarts on claude-opus-5-5.")
    assert still.disabled == 0


@pytest.mark.outside
async def test_a_switch_whose_answer_is_lost_is_waited_out_as_one_that_may_have_been_made(page: Page) -> None:
    before = await _reply(page.client, "GET", "/gallery/state")
    after = await _run_after(page, "claude-opus-5-5")
    _, lost, back = run_page("gallery.html", [Era(0, {STATE: before}), Era(500, {STATE: None, SWITCH: None}), Era(9000, {STATE: after})], [Wait(500), SWITCHED, Wait(20000)])
    assert lost.status == "hands is out of reach, so the switch to claude-opus-5-5 may still be made; the page reads hands until it is."
    assert (back.reading, back.disabled) == ("hands is back, on claude-opus-5-5.", 0)


@pytest.mark.outside
async def test_a_page_opened_while_hands_is_out_of_reach_reads_it_until_it_answers(page: Page) -> None:
    state = await _reply(page.client, "GET", "/gallery/state")
    away, opened = run_page("gallery.html", [Era(0, {STATE: None}), Era(7000, {STATE: state})], [Wait(3000), Wait(20000)])
    assert (away.errors, away.running) == (["reading"], "")
    assert (opened.running, opened.reading, opened.errors) == ("hands runs on claude-sonnet-5-5.", "", [])
    assert opened.controls > 0 and opened.disabled == 0


@pytest.mark.outside
async def test_a_page_whose_key_hands_refuses_stops_and_says_where_to_open_it(page: Page) -> None:
    refused = await page.client.get("/gallery/state")
    [shown] = run_page("gallery.html", [Era(0, {STATE: Reply(refused.status, await refused.text())})], [Wait(60000)])
    assert (shown.reading, shown.errors) == ("Open this page from the address `hands phone` prints, with gallery on its path.", ["reading"])
    assert shown.asked == [STATE]


@pytest.mark.outside
async def test_a_choice_that_does_not_restart_hands_is_said_back_and_leaves_the_controls_live(page: Page) -> None:
    before = await _reply(page.client, "GET", "/gallery/state")
    used = await _reply(page.client, "POST", "/gallery/use", {"voice": VOICES[0]})
    _, chosen = run_page("gallery.html", [Era(0, {STATE: before}), Era(1000, {STATE: None, "POST /gallery/use": used})], [Wait(1000), Click("voices", "Use", 2000)])
    # Said back once hands answers the choice, whatever the read after it: here, hands gone out of reach.
    assert (chosen.status, chosen.errors, chosen.disabled) == (f"Speaking in {VOICES[0].replace('_', ' ').title()} from now on.", ["reading"], 0)
