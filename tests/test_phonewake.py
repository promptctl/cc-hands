"""What hands serves the phone's page to hear the wake word on the phone: the detector's scripts, the runtime it runs on, and
the Mac's own wake word models with the word's settings: each the one thing it is, as the page asks for it. The models
the real page runs are the Chrome test's (tests/test_pages.py). The fetches are the test's: no socket opens."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from fakefetch import Fetching

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import ortweb
from hands.voice.clips import KEPT_SECONDS
from hands.voice.keyed import UNKEYED
from hands.voice.ortweb import FILES
from hands.voice.phonewake import KEPT_A_YEAR, SCRIPTS, wake_routes
from hands.voice.pinnedfile import Mismatch, Release
from hands.voice.wakemodels import EMBEDDING, MELSPECTROGRAM, PRETRAINED, ROLES, WakeModels
from hands.voice.wakeword import THRESHOLD, Pretrained, Trained, Word
from conftest import events

pytestmark = pytest.mark.outside

KEY = "the-phones-key"
JARVIS = PRETRAINED["Hey Jarvis"]
AUTHORISED = {"Authorization": f"Bearer {KEY}"}
# What each file is, as the stand-in fetch writes it: the bytes say which file it is.
RUNTIME = {file.pin.name: f"the contents of {file.pin.name}".encode() for file in FILES}
SCRIPT, WASM = (file.pin.name for file in FILES)


@dataclass
class Served:
    client: TestClient[web.Request, web.Application]
    runtime: Fetching
    models: Fetching
    recorded: list[Entry]

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


async def serving(tmp_path: Path, word: Word) -> AsyncGenerator[Served]:
    recorded: list[Entry] = []
    runtime, models = Fetching(), Fetching()
    app = web.Application()
    app.add_routes(wake_routes(VirtualClock(), word, WakeModels(tmp_path / "models", VirtualClock(), models), Release(tmp_path / "runtime", VirtualClock(), runtime), KEY, recorded.append))
    async with TestClient(TestServer(app)) as client:
        yield Served(client, runtime, models, recorded)


@pytest.fixture
async def served(tmp_path: Path) -> AsyncGenerator[Served]:
    async for each in serving(tmp_path, Pretrained()):
        yield each


async def test_the_detectors_scripts_are_served_to_anyone_and_kept_by_no_browser(served: Served) -> None:
    for name in SCRIPTS:
        response = await served.client.get(f"/{name}")
        assert (response.status, response.content_type, response.headers["Cache-Control"]) == (200, "text/javascript", "no-store")
        assert await response.text() == resources.files("hands.voice").joinpath(name).read_text()


async def test_a_file_of_the_runtime_is_served_to_anyone_for_a_year_as_what_it_is(served: Served) -> None:
    script = await served.client.get(f"/ort/{ortweb.VERSION}/{SCRIPT}")
    wasm = await served.client.get(f"/ort/{ortweb.VERSION}/{WASM}")
    assert (script.status, script.content_type, script.headers["Cache-Control"], await script.read()) == (200, "text/javascript", KEPT_A_YEAR, RUNTIME[SCRIPT])
    assert (wasm.status, wasm.content_type, wasm.headers["Cache-Control"], await wasm.read()) == (200, "application/wasm", KEPT_A_YEAR, RUNTIME[WASM])
    # [LAW:nothing-unseen] each request says it made hands fetch the file.
    assert [(event.facts["file"], event.facts["source"]) for event in served.events("phone.wake.runtime")] == [(SCRIPT, "fetched"), (WASM, "fetched")]


@pytest.mark.parametrize("path", [f"/ort/1.0.0/{SCRIPT}", f"/ort/{ortweb.VERSION}/anything-else.js", f"/ort/{ortweb.VERSION}/..%2Fsecret", "/wake/models/passwd"])
async def test_a_version_or_a_file_hands_does_not_serve_is_not_found_and_nothing_is_fetched_for_it(served: Served, path: str) -> None:
    response = await served.client.get(path, headers=AUTHORISED)
    assert response.status == 404
    assert (served.runtime.calls, served.models.calls) == ([], [])


async def test_a_runtime_hands_cannot_fetch_is_refused_in_hands_words_and_never_served(served: Served) -> None:
    served.runtime.failure = Mismatch("ort.wasm.bundle.min.mjs at onnxruntime-web 1.31.1 is aaa, not the bbb pinned")
    response = await served.client.get(f"/ort/{ortweb.VERSION}/{SCRIPT}")
    assert response.status == 503
    assert "is aaa, not the bbb pinned" in await response.text()
    [event] = served.events("phone.wake.runtime")
    assert (event.outcome, "is aaa, not the bbb pinned" in (event.error or "")) == ("failed", True)


async def test_the_settings_and_the_models_are_the_phones_alone(served: Served) -> None:
    for path in ("/wake/settings", *(f"/wake/models/{role}" for role in ROLES)):
        response = await served.client.get(path)
        assert (response.status, await response.text()) == (401, UNKEYED)
    assert [event.outcome for event in served.events("phone.wake.settings")] == ["failed"]
    assert [event.outcome for event in served.events("phone.wake.model")] == ["failed"] * 3
    assert (served.runtime.calls, served.models.calls) == ([], [])


async def test_the_settings_are_the_macs_word_and_threshold_the_second_kept_and_where_the_files_are(served: Served) -> None:
    response = await served.client.get("/wake/settings", headers=AUTHORISED)
    assert (response.status, response.headers["Cache-Control"]) == (200, "no-store")
    assert await response.json() == {
        "phrase": "Hey Jarvis",
        "threshold": THRESHOLD,
        "kept_seconds": KEPT_SECONDS,
        "runtime": f"/ort/{ortweb.VERSION}/{SCRIPT}",
        "models": {"melspectrogram": "/wake/models/melspectrogram", "embedding": "/wake/models/embedding", "word": "/wake/models/word"},
    }
    # The models are held to their digests before the page is told where they are; the runtime is fetched when the page asks
    # for it, which is the one ask that waits as long as the file takes.
    assert (served.runtime.names, served.models.names) == ([], [MELSPECTROGRAM.name, EMBEDDING.name, JARVIS.name])


async def test_each_model_is_the_file_the_mac_runs_and_is_kept_by_no_browser(served: Served) -> None:
    for role, pin in zip(ROLES, (MELSPECTROGRAM, EMBEDDING, JARVIS), strict=True):
        response = await served.client.get(f"/wake/models/{role}", headers=AUTHORISED)
        assert (response.status, response.headers["Cache-Control"], await response.read()) == (200, "no-store", f"the contents of {pin.name}".encode())
    # Fetched once for the three asks, and not again for the asks after.
    assert served.models.names == [MELSPECTROGRAM.name, EMBEDDING.name, JARVIS.name]


async def test_a_word_the_user_trained_is_served_from_where_they_keep_it_under_the_same_name(tmp_path: Path) -> None:
    mine = tmp_path / "mine" / "hey_hands.onnx"
    mine.parent.mkdir()
    mine.write_bytes(b"my model")
    async for each in serving(tmp_path, Trained("Hey Hands", mine)):
        settings = await each.client.get("/wake/settings", headers=AUTHORISED)
        assert (await settings.json())["phrase"] == "Hey Hands"
        response = await each.client.get("/wake/models/word", headers=AUTHORISED)
        assert (response.status, await response.read()) == (200, b"my model")


async def test_a_model_that_is_not_there_is_refused_in_hands_words_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "mine" / "hey_hands.onnx"
    async for each in serving(tmp_path, Trained("Hey Hands", missing)):
        for path, unit in (("/wake/settings", "phone.wake.settings"), ("/wake/models/word", "phone.wake.model")):
            response = await each.client.get(path, headers=AUTHORISED)
            assert (response.status, await response.text()) == (503, f"no wake word model at {missing}")
            assert [(event.outcome, event.error) for event in each.events(unit)] == [("failed", f"no wake word model at {missing}")]


async def test_models_hands_cannot_fetch_are_refused_in_hands_words_naming_why(served: Served) -> None:
    served.models.failure = OSError("the network is down")
    response = await served.client.get("/wake/models/word", headers=AUTHORISED)
    said = await response.text()
    assert (response.status, "the network is down" in said, said.startswith("hands cannot get openWakeWord")) == (503, True, True)
    [event] = served.events("phone.wake.model")
    assert (event.outcome, event.error) == ("failed", said)

