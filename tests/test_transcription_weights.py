"""Whisper's weights from a snapshot a fetch cut short: the hub's cache holding some of the files of MODEL at REVISION,
beside the list of the commit's files the hub keeps once it starts a fetch, and a hub served in this process."""

import json
from collections.abc import Generator
from pathlib import Path

import httpx
import huggingface_hub.constants
import pytest
from huggingface_hub import set_client_factory
from huggingface_hub.errors import IncompleteSnapshotError
from huggingface_hub.utils import _http  # pyright: ignore[reportPrivateUsage]  (the hub exports no name for its default client factory)
from huggingface_hub.utils.sha import git_hash

from hands.sessions.wide import chain
from hands.voice import transcription

# The commit's files: the one a cut-short fetch left in the snapshot, and the one it never linked there.
FETCHED = ("weights.safetensors", b"the weights")
MISSING = ("config.json", b'{"n_mels": 128}')


@pytest.fixture(autouse=True)
def no_whisper_weights() -> None:
    """These tests are about the weights: the suite's stand-in for them is not put in their place here."""


@pytest.fixture(autouse=True)
def cut_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """The hub's cache in `tmp_path`, holding a snapshot of MODEL at REVISION with FETCHED in it and MISSING not."""
    repo = tmp_path / f"models--{transcription.MODEL.replace('/', '--')}"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / transcription.REVISION).mkdir(parents=True)
    (repo / "trees").mkdir()
    name, content = FETCHED
    (repo / "blobs" / git_hash(content)).write_bytes(content)
    (repo / "snapshots" / transcription.REVISION / name).symlink_to(f"../../blobs/{git_hash(content)}")
    files = {name: {"size": len(content), "blob_id": git_hash(content)} for name, content in (FETCHED, MISSING)}
    (repo / "trees" / f"{transcription.REVISION}.json").write_text(json.dumps({"format_version": 1, "files": files}))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    transcription.weights.cache_clear()
    yield
    transcription.weights.cache_clear()


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Generator[list[str]]:
    """A hub serving the commit's files to the hub's client, online; what it was asked for, in order."""
    asked: list[str] = []
    contents: dict[str, bytes] = dict((FETCHED, MISSING))

    def serve(request: httpx.Request) -> httpx.Response:
        asked.append(f"{request.method} {request.url.path}")
        content = contents[request.url.path.rsplit("/", 1)[-1]]
        headers = {"X-Repo-Commit": transcription.REVISION, "ETag": f'"{git_hash(content)}"', "Content-Length": str(len(content))}
        return httpx.Response(200, headers=headers, content=b"" if request.method == "HEAD" else content)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(serve)))
    yield asked
    set_client_factory(_http.default_client_factory)


def test_a_snapshot_a_fetch_cut_short_is_fetched_whole_from_the_hub(hub: list[str]) -> None:
    weights = Path(transcription.weights())

    assert {name: (weights / name).read_bytes() for name in (FETCHED[0], MISSING[0])} == dict((FETCHED, MISSING))
    # Only what the snapshot was missing is fetched.
    assert hub
    assert [asked for asked in hub if not asked.endswith(f"/{transcription.REVISION}/{MISSING[0]}")] == []


def test_offline_a_snapshot_a_fetch_cut_short_fails_naming_the_file_it_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    with pytest.raises(IncompleteSnapshotError) as raised:
        transcription.weights()

    # The chain is what the failed load's event carries as its stacktrace; the hub's own error heads it.
    assert MISSING[0] in chain(raised.value)[0]
