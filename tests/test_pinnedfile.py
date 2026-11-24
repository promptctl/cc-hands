"""A file hands fetches whole and holds to the digest it pins, and a release's files kept in a directory. The network is
replaced where hands reads it (`urllib.request.urlopen`), so no socket opens: what a real server does to a fetch is
tests/test_pinnedfile_served.py's."""

import asyncio
import hashlib
import urllib.error
from http.client import IncompleteRead
from pathlib import Path

import pytest
from fakefetch import Fetching, Network

from hands.voice.pinnedfile import Mismatch, Pin, Release, Unserved, fetched

BODY = b"the pinned file"
DIGEST = hashlib.sha256(BODY).hexdigest()
URL = "https://example.test/pinned"
PIN = Pin("model.bin", URL, DIGEST, "revision 1")


def test_a_file_that_is_not_there_is_fetched_whole_from_its_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    network = Network(monkeypatch, BODY)
    path = fetched(tmp_path, PIN)
    assert (path, path.read_bytes(), network.asked, [each.name for each in tmp_path.iterdir()]) == (tmp_path / "model.bin", BODY, [URL], ["model.bin"])


def test_a_file_there_with_the_digest_is_not_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.bin").write_bytes(BODY)
    network = Network(monkeypatch, b"never read")
    fetched(tmp_path, PIN)
    assert network.asked == []


def test_a_file_there_without_the_digest_is_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.bin").write_bytes(b"changed on disk")
    network = Network(monkeypatch, BODY)
    assert (fetched(tmp_path, PIN).read_bytes(), network.asked) == (BODY, [URL])


def test_a_download_that_is_not_the_pinned_file_is_refused_with_its_name_where_it_is_from_and_both_digests_and_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Network(monkeypatch, b"other bytes")
    with pytest.raises(Mismatch) as refused:
        fetched(tmp_path, PIN)
    found = hashlib.sha256(b"other bytes").hexdigest()
    assert str(refused.value) == f"model.bin at revision 1 is {found}, not the {DIGEST} pinned"
    assert list(tmp_path.iterdir()) == []


PINS = [Pin("first.bin", f"{URL}/first.bin", "digest of the first", "release 2"), Pin("second.bin", f"{URL}/second.bin", "digest of the second", "release 2")]


async def test_a_file_of_a_release_is_fetched_as_its_pin_says_on_the_first_ask_and_held_there_after(tmp_path: Path) -> None:
    fetching = Fetching()
    release = Release(tmp_path, PINS, fetching)
    first = await release.path("second.bin")
    assert (first, fetching.calls) == (tmp_path / "second.bin", [(tmp_path, PINS[1])])
    assert await release.path("second.bin") == first
    assert len(fetching.calls) == 1


async def test_ensured_fetches_each_file_of_the_release_for_itself(tmp_path: Path) -> None:
    fetching = Fetching()
    await Release(tmp_path, PINS, fetching).ensured()
    assert fetching.names == ["first.bin", "second.bin"]


@pytest.mark.parametrize("failure", [
    Mismatch("second.bin at release 2 is aaa, not the bbb pinned"),
    urllib.error.URLError("the network is down"),
    IncompleteRead(b"", 10),
    PermissionError("the directory is read-only"),
])
async def test_a_fetch_that_fails_is_unserved_with_the_reason_and_the_next_ask_fetches_again(tmp_path: Path, failure: Exception) -> None:
    fetching = Fetching()
    release = Release(tmp_path, PINS, fetching)
    fetching.failure = failure
    with pytest.raises(Unserved) as unserved:
        await release.path("second.bin")
    assert (str(unserved.value).startswith("hands cannot get release 2's second.bin: "), str(failure) in str(unserved.value)) == (True, True)
    fetching.failure = None
    assert await release.path("second.bin") == tmp_path / "second.bin"
    assert len(fetching.calls) == 2


async def test_a_directory_that_cannot_be_made_is_unserved_naming_why(tmp_path: Path) -> None:
    blocked = tmp_path / "a file"
    blocked.write_bytes(b"in the way")
    with pytest.raises(Unserved, match="hands cannot get release 2's first.bin: "):
        await Release(blocked / "under it", PINS, Fetching()).path("first.bin")


async def test_a_defect_in_a_fetch_is_raised_as_itself(tmp_path: Path) -> None:
    fetching = Fetching()
    fetching.failure = ValueError("unknown url type: 'htps'")
    with pytest.raises(ValueError, match="unknown url type"):
        await Release(tmp_path, PINS, fetching).path("first.bin")


async def test_asks_that_arrive_while_a_file_is_being_fetched_wait_for_that_fetch(tmp_path: Path) -> None:
    fetching = Fetching()
    fetching.release.clear()
    release = Release(tmp_path, PINS, fetching)
    first = asyncio.create_task(release.path("first.bin"))
    # The fetch is under way, held open by the test; the other asks arrive in it.
    await fetching.begun.wait()
    others = [asyncio.create_task(release.path("first.bin")) for _ in range(3)]
    await asyncio.sleep(0)
    assert not any(ask.done() for ask in others)
    fetching.release.set()
    assert set(await asyncio.gather(first, *others)) == {tmp_path / "first.bin"}
    assert len(fetching.calls) == 1
