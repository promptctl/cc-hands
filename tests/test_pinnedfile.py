"""A file hands fetches whole and holds to the digest it pins. The network is replaced where hands reads it
(`urllib.request.urlopen`), so no socket opens."""

import hashlib
import io
from pathlib import Path

import pytest

from hands.voice.pinnedfile import fetched

BODY = b"the pinned file"
DIGEST = hashlib.sha256(BODY).hexdigest()
URL = "https://example.test/pinned"


class Network:
    """What the file's server answers, and every URL asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
        self.asked = list[str]()
        self._body = body
        monkeypatch.setattr("urllib.request.urlopen", self._open)

    def _open(self, url: str, timeout: float) -> io.BytesIO:
        self.asked.append(url)
        return io.BytesIO(self._body)


def test_a_file_that_is_not_there_is_fetched_whole_from_its_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    network = Network(monkeypatch, BODY)
    path = fetched(tmp_path, "model.bin", URL, DIGEST, "revision 1")
    assert (path, path.read_bytes(), network.asked, [each.name for each in tmp_path.iterdir()]) == (tmp_path / "model.bin", BODY, [URL], ["model.bin"])


def test_a_file_there_with_the_digest_is_not_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.bin").write_bytes(BODY)
    network = Network(monkeypatch, b"never read")
    fetched(tmp_path, "model.bin", URL, DIGEST, "revision 1")
    assert network.asked == []


def test_a_file_there_without_the_digest_is_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.bin").write_bytes(b"changed on disk")
    network = Network(monkeypatch, BODY)
    assert (fetched(tmp_path, "model.bin", URL, DIGEST, "revision 1").read_bytes(), network.asked) == (BODY, [URL])


def test_a_download_that_is_not_the_pinned_file_is_refused_with_its_name_where_it_is_from_and_both_digests_and_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Network(monkeypatch, b"other bytes")
    with pytest.raises(ValueError) as refused:
        fetched(tmp_path, "model.bin", URL, DIGEST, "revision 1")
    found = hashlib.sha256(b"other bytes").hexdigest()
    assert str(refused.value) == f"model.bin at revision 1 is {found}, not the {DIGEST} pinned"
    assert list(tmp_path.iterdir()) == []
