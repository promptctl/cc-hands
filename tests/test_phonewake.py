"""What hands serves the phone's page to hear the wake word on the phone: the detector's scripts, the runtime it runs on, and
the Mac's own wake word models with the word's settings: each the one thing it is, as the page asks for it. The models
the real page runs are the Chrome test's (tests/test_pages.py)."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import ortweb
from hands.voice.clips import KEPT_SECONDS
from hands.voice.keyed import UNKEYED
from hands.voice.ortweb import FILES, Runtime
from hands.voice.phonewake import KEPT_A_YEAR, MODELS, SCRIPTS, UNMODELLED, wake_routes
from hands.voice.wake import EMBEDDING, MELSPECTROGRAM, THRESHOLD, Models
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained, Word
from conftest import events
from test_wake import release

pytestmark = pytest.mark.outside

KEY = "the-phones-key"
JARVIS = PRETRAINED["Hey Jarvis"]
AUTHORISED = {"Authorization": f"Bearer {KEY}"}
# What each runtime file is, as the stand-in fetch writes it.
RUNTIME = {name: f"the contents of {name}".encode() for name in FILES}


@dataclass
class Fetching:
    """The runtime's fetch as the test answers it: it writes each file's bytes, or raises what the test says."""

    failure: Exception | None = None
    calls: int = 0

    async def __call__(self, directory: Path, name: str, url: str, digest: str, at: str) -> Path:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(RUNTIME[name])
        return directory / name


@dataclass
class Served:
    client: TestClient[web.Request, web.Application]
    models: Path
    fetching: Fetching
    recorded: list[Entry]

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


def kept(directory: Path) -> Path:
    """`directory` holding the three files the Mac's wake word runs from, each saying which it is."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in (MELSPECTROGRAM, EMBEDDING, JARVIS):
        (directory / name).write_bytes(f"the contents of {name}".encode())
    return directory


async def serving(tmp_path: Path, word: Word, models: Models, fetching: Fetching) -> AsyncGenerator[Served]:
    recorded: list[Entry] = []
    app = web.Application()
    app.add_routes(wake_routes(VirtualClock(), word, models, Runtime(tmp_path / "runtime", FILES, fetching), KEY, recorded.append))
    async with TestClient(TestServer(app)) as client:
        yield Served(client, models.directory, fetching, recorded)


@pytest.fixture
async def served(tmp_path: Path) -> AsyncGenerator[Served]:
    async for each in serving(tmp_path, Pretrained(), Models(kept(tmp_path / "models")), Fetching()):
        yield each


async def test_the_detectors_scripts_are_served_to_anyone_and_kept_by_no_browser(served: Served) -> None:
    for name in SCRIPTS:
        response = await served.client.get(f"/{name}")
        assert (response.status, response.content_type, response.headers["Cache-Control"]) == (200, "text/javascript", "no-store")
        assert await response.text() == resources.files("hands.voice").joinpath(name).read_text()


async def test_a_file_of_the_runtime_is_served_to_anyone_for_a_year_as_what_it_is(served: Served) -> None:
    script = await served.client.get(f"/ort/{ortweb.VERSION}/ort.wasm.bundle.min.mjs")
    wasm = await served.client.get(f"/ort/{ortweb.VERSION}/ort-wasm-simd-threaded.wasm")
    assert (script.status, script.content_type, script.headers["Cache-Control"], await script.read()) == (200, "text/javascript", KEPT_A_YEAR, RUNTIME["ort.wasm.bundle.min.mjs"])
    assert (wasm.status, wasm.content_type, wasm.headers["Cache-Control"], await wasm.read()) == (200, "application/wasm", KEPT_A_YEAR, RUNTIME["ort-wasm-simd-threaded.wasm"])


@pytest.mark.parametrize("path", ["/ort/1.0.0/ort.wasm.bundle.min.mjs", f"/ort/{ortweb.VERSION}/anything-else.js", f"/ort/{ortweb.VERSION}/..%2Fsecret"])
async def test_a_version_or_a_file_hands_does_not_serve_is_told_there_is_none(served: Served, path: str) -> None:
    response = await served.client.get(path)
    assert response.status == 404
    assert ortweb.VERSION in await response.text()
    assert served.fetching.calls == 0
    assert [event.outcome for event in served.events("phone.wake.runtime")] == ["failed"]


async def test_a_runtime_hands_cannot_fetch_is_refused_in_hands_words_and_never_served(served: Served) -> None:
    served.fetching.failure = ValueError("ort.wasm.bundle.min.mjs at onnxruntime-web 1.31.1 is aaa, not the bbb pinned")
    response = await served.client.get(f"/ort/{ortweb.VERSION}/ort.wasm.bundle.min.mjs")
    assert response.status == 503
    assert "is aaa, not the bbb pinned" in await response.text()
    [event] = served.events("phone.wake.runtime")
    assert (event.outcome, "is aaa, not the bbb pinned" in (event.error or "")) == ("failed", True)


async def test_the_settings_and_the_models_are_the_phones_alone(served: Served) -> None:
    for path in ("/wake/settings", *(f"/wake/models/{name}" for name in MODELS)):
        response = await served.client.get(path)
        assert (response.status, await response.text()) == (401, UNKEYED)
    assert [event.outcome for event in served.events("phone.wake.settings")] == ["failed"]
    assert [event.outcome for event in served.events("phone.wake.model")] == ["failed"] * 3


async def test_the_settings_are_the_macs_word_and_threshold_the_second_kept_and_where_the_files_are(served: Served) -> None:
    response = await served.client.get("/wake/settings", headers=AUTHORISED)
    assert (response.status, response.headers["Cache-Control"]) == (200, "no-store")
    assert await response.json() == {
        "phrase": "Hey Jarvis",
        "threshold": THRESHOLD,
        "kept_seconds": KEPT_SECONDS,
        "runtime": f"/ort/{ortweb.VERSION}/",
        "models": {name: f"/wake/models/{name}" for name in MODELS},
    }
    # Both files of the runtime are held to their digests before the page is told where they are.
    assert served.fetching.calls == len(FILES)


async def test_each_model_is_the_file_the_mac_runs_and_is_kept_by_no_browser(served: Served) -> None:
    for name, file in zip(MODELS, (MELSPECTROGRAM, EMBEDDING, JARVIS), strict=True):
        response = await served.client.get(f"/wake/models/{name}", headers=AUTHORISED)
        assert (response.status, response.headers["Cache-Control"], await response.read()) == (200, "no-store", f"the contents of {file}".encode())


async def test_a_model_hands_does_not_have_is_told_there_is_none(served: Served) -> None:
    response = await served.client.get("/wake/models/passwd", headers=AUTHORISED)
    assert response.status == 404
    assert "melspectrogram.onnx" in await response.text()


async def test_a_word_the_user_trained_is_served_from_where_they_keep_it_under_the_same_name(tmp_path: Path) -> None:
    mine = tmp_path / "mine" / "hey_hands.onnx"
    mine.parent.mkdir()
    mine.write_bytes(b"my model")
    word = Trained("Hey Hands", mine)
    async for each in serving(tmp_path, word, Models(kept(tmp_path / "models")), Fetching()):
        settings = await each.client.get("/wake/settings", headers=AUTHORISED)
        assert (await settings.json())["phrase"] == "Hey Hands"
        response = await each.client.get("/wake/models/word.onnx", headers=AUTHORISED)
        assert (response.status, await response.read()) == (200, b"my model")


async def test_a_model_that_is_not_there_is_refused_in_hands_words_naming_it(tmp_path: Path) -> None:
    missing = tmp_path / "mine" / "hey_hands.onnx"
    async for each in serving(tmp_path, Trained("Hey Hands", missing), Models(kept(tmp_path / "models")), Fetching()):
        response = await each.client.get("/wake/settings", headers=AUTHORISED)
        assert (response.status, await response.text()) == (503, UNMODELLED)
        [event] = each.events("phone.wake.settings")
        assert (event.outcome, str(missing) in (event.error or "")) == ("failed", True)


async def test_models_hands_cannot_fetch_are_refused_in_hands_words_naming_why(tmp_path: Path) -> None:
    async def gone(request: web.Request) -> web.StreamResponse:
        raise web.HTTPNotFound()

    async with release(gone) as address:
        async for each in serving(tmp_path, Pretrained(), Models(tmp_path / "models", address), Fetching()):
            response = await each.client.get("/wake/models/word.onnx", headers=AUTHORISED)
            assert (response.status, await response.text()) == (503, UNMODELLED)
            [event] = each.events("phone.wake.model")
            assert (event.outcome, "cannot fetch the wake word's models" in (event.error or "")) == ("failed", True)


async def test_the_settings_are_refused_when_the_runtime_cannot_be_had(served: Served) -> None:
    served.fetching.failure = OSError("the network is down")
    response = await served.client.get("/wake/settings", headers=AUTHORISED)
    assert response.status == 503
    assert "the network is down" in await response.text()
