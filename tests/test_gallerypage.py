"""The gallery on the phone's page: the voices heard and chosen, the listening level, the modality, and the model, each
choice the one its tool makes, and each refused where the tool refuses it."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.core.place import Modality
from hands.core.playback import Labelled
from hands.sessions.audit import Entry
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent
from hands.voice.modelswitch import Switches
from hands.voice.listening import Asker, Level
from hands.virtualclock import VirtualClock
from hands.voice.longpoll import POLL_SECONDS, WAIT_SECONDS
from hands.voice.gallerypage import Gallery, gallery_routes
from hands.voice.library import Found
from hands.voice.voicechoice import Chosen, LibraryVoice, Saved
from hands.voice.trigger import described
from hands.voice.voices import said
from hands.voice.voicechoice import VOICES, Voice
from hands.voice.wakeword import Pretrained
from conftest import events

KEY = "the-phones-key"
KEYED = {"Authorization": f"Bearer {KEY}"}
MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")


@dataclass
class Voicing:
    """The voices as the test sets them: the one spoken, the lines written for them, and what each call was asked."""

    spoken: Chosen = Voice("charles")
    written: dict[Chosen, str] = field(default_factory=dict[Chosen, str])
    heard_asked: list[tuple[Chosen, ...]] = field(default_factory=list[tuple[Chosen, ...]])
    previewed: list[tuple[Chosen, str]] = field(default_factory=list[tuple[Chosen, str]])
    used_asked: list[Chosen] = field(default_factory=list[Chosen])
    failure: Exception | None = None

    def speaking_in(self) -> Chosen:
        return self.spoken

    async def hear(self, voices: tuple[Chosen, ...]) -> Chosen:
        self.heard_asked.append(voices)
        if self.failure is not None:
            raise self.failure
        return self.spoken

    async def said_by(self, voices: tuple[Chosen, ...]) -> tuple[str, ...]:
        return tuple(said(voice, self.written.get(voice, "")) for voice in voices)

    async def preview(self, voice: Chosen, line: str) -> Chosen:
        self.previewed.append((voice, line))
        return self.spoken

    async def keep_line(self, voice: Chosen, line: str) -> Saved:
        self.written[voice] = line
        [saying] = await self.said_by((voice,))
        return Saved(str(voice), saying)

    async def use(self, voice: Chosen) -> None:
        self.used_asked.append(voice)
        self.spoken = voice

    async def search(self, query: str, limit: int) -> Found:
        return Found(tuple(name for name in ("vctk/p225", "alba-mackenna/casual") if query in name)[:limit], 3.0, True)


@dataclass
class Listening:
    level: Level = "off"
    chosen: list[tuple[Level, Asker]] = field(default_factory=list[tuple[Level, Asker]])

    async def choose(self, level: Level, by: Asker) -> Level:
        was, self.level = self.level, level
        self.chosen.append((level, by))
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
    """Hands' speech as the model switch reads it: a line, heard for `hold` on the page's clock, and whether the user
    heard it to its end."""

    clock: VirtualClock
    heard_in_full: bool = True
    hold: float = 0.0
    said: list[str] = field(default_factory=list[str])

    async def heard(self, line: Labelled) -> bool:
        self.said.append(line.text)
        await self.clock.sleep(self.hold)
        return self.heard_in_full


@dataclass
class Own:
    """Hands' own model: the one it runs on, a weighing that may refuse, and what was kept, or the keep refused."""

    running_on: str = MODELS[0]
    refused: Exception | None = None
    unkept: Exception | None = None
    kept: list[str] = field(default_factory=list[str])

    def running(self) -> str:
        return self.running_on

    def weigh(self, model: str) -> "Weighing":
        if self.refused is not None:
            raise self.refused
        return Weighing(model, self)


@dataclass(frozen=True)
class Weighing:
    """A model the page's own weighed, kept into it, or refused as it is kept."""

    to: str
    own: Own

    def keep(self) -> None:
        if self.own.unkept is not None:
            raise self.own.unkept
        self.own.kept.append(self.to)


@dataclass
class Page:
    client: TestClient[web.Request, web.Application]
    clock: VirtualClock
    voices: Voicing
    levels: Listening
    desk: Desk
    speaker: Speaker
    own: Own
    recorded: list[Entry]
    arrived: asyncio.Event

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)

    async def ended(self, name: str, count: int = 1) -> list[WideEvent]:
        """The events `name` once `count` of them are recorded: a request's event is recorded as its handler returns,
        which for a model switch is after its answer is written and its keep run."""
        while len(self.events(name)) < count:
            self.arrived.clear()
            await self.arrived.wait()
        return self.events(name)


@pytest.fixture
async def page() -> AsyncGenerator[Page]:
    # A read's wait passes only as a test moves the page's clock.
    clock = VirtualClock()
    voices, levels, desk, speaker, own = Voicing(), Listening(), Desk(), Speaker(clock), Own()
    recorded: list[Entry] = []
    arrived = asyncio.Event()

    def record(entry: Entry) -> None:
        recorded.append(entry)
        arrived.set()

    gallery = Gallery(voices=voices, levels=levels, desk=desk, own=own, switches=Switches(speaker), wake=Pretrained(), models=MODELS)
    app = web.Application()
    app.add_routes(gallery_routes(clock, gallery, KEY, record))
    async with TestClient(TestServer(app)) as client:
        yield Page(client, clock, voices, levels, desk, speaker, own, recorded, arrived)


@pytest.mark.outside
async def test_the_page_is_served_and_its_reads_and_choices_need_the_phones_key(page: Page) -> None:
    shown = await page.client.get("/gallery")
    body = await shown.text()
    assert shown.status == 200 and "hands' voices and options" in body
    # The level control is the desk's own: the phone page has its wake word button beside Connect.
    assert "Listening at the desk" in body
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
    assert (isinstance(body["seen"], str), body["waits"]) == (True, WAIT_SECONDS)
    assert body["shown"] == {
        "voices": list(VOICES), "speaking_in": "charles",
        "levels": ["off", "wake word", "engaged"], "level": "engaged",
        "modalities": ["screen", "audio-only"], "modality": "screen",
        "models": list(MODELS), "running_on": MODELS[0],
    }
    [read] = page.events("gallery.read")
    assert (read.outcome, read.facts["speaking_in"], read.facts["level"], read.facts["modality"], read.facts["running_on"]) == ("ok", "charles", "engaged", "screen", MODELS[0])


async def _seen(page: Page) -> str:
    return (await (await page.client.get("/gallery/state", headers=KEYED)).json())["seen"]


@pytest.mark.outside
async def test_a_read_that_has_seen_the_state_answers_once_a_choice_changes_it(page: Page) -> None:
    seen = await _seen(page)
    waiting = asyncio.create_task(page.client.get(f"/gallery/state?seen={seen}", headers=KEYED))
    await page.clock.sleeping(POLL_SECONDS)
    assert not waiting.done()
    await page.client.post("/gallery/use", headers=KEYED, json={"voice": "bill_boerst"})
    page.clock.advance(POLL_SECONDS)
    body = await (await waiting).json()
    assert (body["shown"]["speaking_in"], body["seen"] != seen) == ("bill_boerst", True)
    assert page.events("gallery.read")[-1].facts["changed"] is True


@pytest.mark.outside
async def test_a_read_that_has_seen_the_state_answers_it_unchanged_at_the_end_of_its_wait(page: Page) -> None:
    seen = await _seen(page)
    waiting = asyncio.create_task(page.client.get(f"/gallery/state?seen={seen}", headers=KEYED))
    await page.clock.sleeping(POLL_SECONDS)
    page.clock.advance(WAIT_SECONDS)
    body = await (await waiting).json()
    assert body["seen"] == seen
    assert page.events("gallery.read")[-1].facts["changed"] is False


@pytest.mark.outside
async def test_a_run_on_another_model_answers_a_read_that_saw_the_run_before_at_once(page: Page) -> None:
    # A restart's run is told apart by what it runs on: a page that saw the run before is answered without waiting.
    seen = await _seen(page)
    page.own.running_on = MODELS[1]
    body = await (await page.client.get(f"/gallery/state?seen={seen}", headers=KEYED)).json()
    assert (body["shown"]["running_on"], body["seen"] != seen) == (MODELS[1], True)
    assert page.events("gallery.read")[-1].facts["looks"] == 1


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
async def test_a_voices_line_is_read_as_the_file_holds_it_now_or_hands_own_where_it_holds_none(page: Page) -> None:
    page.voices.written[Voice("mary")] = "Mary here."
    response = await page.client.get("/gallery/line?voice=Mary", headers=KEYED)
    assert await response.json() == {"voice": "mary", "line": "Mary here."}
    response = await page.client.get("/gallery/line?voice=vctk/p225", headers=KEYED)
    assert await response.json() == {"voice": "vctk/p225", "line": said(LibraryVoice("vctk/p225"), "")}
    refused = await page.client.get("/gallery/line?voice=Zed!", headers=KEYED)
    assert refused.status == 400
    assert [event.outcome for event in page.events("gallery.line.read")] == ["ok", "ok", "failed"]


@pytest.mark.outside
async def test_a_line_previewed_is_heard_in_its_voice_and_its_event_says_its_length_not_its_words(page: Page) -> None:
    response = await page.client.post("/gallery/preview", headers=KEYED, json={"voice": "Mary", "line": "Not saved yet."})
    assert await response.json() == {"heard": "mary", "speaking_in": "charles"}
    blank = await page.client.post("/gallery/preview", headers=KEYED, json={"voice": "mary", "line": ""})
    assert blank.status == 200
    assert page.voices.previewed == [("mary", "Not saved yet."), ("mary", "")]
    assert page.voices.written == {}
    first, _ = page.events("gallery.preview")
    assert (first.outcome, first.facts["heard"], first.facts["line_chars"]) == ("ok", "mary", 14)
    assert "Not saved yet." not in str(first.facts)


@pytest.mark.outside
async def test_a_line_saved_is_kept_for_its_voice_and_a_blank_one_is_hands_own_again(page: Page) -> None:
    response = await page.client.post("/gallery/line", headers=KEYED, json={"voice": "bill_boerst", "line": "Bill here."})
    assert await response.json() == {"voice": "bill_boerst", "line": "Bill here.", "default": False}
    response = await page.client.post("/gallery/line", headers=KEYED, json={"voice": "bill_boerst", "line": " "})
    assert await response.json() == {"voice": "bill_boerst", "line": said(Voice("bill_boerst"), ""), "default": True}
    refused = await page.client.post("/gallery/line", headers=KEYED, json={"voice": "bill_boerst"})
    assert refused.status == 400 and "{line: ...}" in await refused.text()
    assert [(event.outcome, event.facts.get("default"), event.facts.get("key")) for event in page.events("gallery.line")] == [("ok", False, "bill_boerst"), ("ok", True, "bill_boerst"), ("failed", None, None)]


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
    assert page.levels.chosen == [("wake word", "page")]
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
    [switch] = await page.ended("gallery.model")
    assert page.own.kept == ["claude-opus-5-5"]
    assert (switch.outcome, switch.facts["model"], switch.facts["switching"]) == ("ok", "claude-opus-5-5", True)


@pytest.mark.outside
async def test_a_model_switch_heard_but_refused_as_it_is_kept_is_said_aloud_and_its_event_fails(page: Page) -> None:
    page.own.unkept = Rejected("config.toml was saved as the choice was kept")
    response = await page.client.post("/gallery/model", headers=KEYED, json={"model": "claude-opus-5-5"})
    # Answered before the keep: the page is told the switch, and the user hears that it did not happen.
    assert response.status == 200
    [switch] = await page.ended("gallery.model")
    assert page.speaker.said == ["Switching to claude-opus-5-5. I'll be back in a few seconds.", "The switch to claude-opus-5-5 did not happen: config.toml was saved as the choice was kept"]
    assert (switch.outcome, switch.error, switch.facts["switching"]) == ("failed", "the model's switch failed once answered: Rejected: config.toml was saved as the choice was kept", True)
    assert page.own.kept == []


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


def test_the_routes_are_the_gallery_and_its_seven_choices_and_two_reads() -> None:
    # The page's routes are named in one place, so the page's requests and these routes cannot drift apart.
    clock = VirtualClock()
    gallery = Gallery(voices=Voicing(), levels=Listening(), desk=Desk(), own=Own(), switches=Switches(Speaker(clock)), wake=Pretrained(), models=MODELS)
    routes: list[Any] = [route for route in gallery_routes(clock, gallery, KEY, lambda _entry: None)]
    assert sorted(f"{route.method} {route.path}" for route in routes) == sorted([
        "GET /gallery", "GET /gallery/state", "GET /gallery/line", "GET /gallery/library", "POST /gallery/hear", "POST /gallery/use",
        "POST /gallery/preview", "POST /gallery/line", "POST /gallery/level", "POST /gallery/modality", "POST /gallery/model",
    ])
