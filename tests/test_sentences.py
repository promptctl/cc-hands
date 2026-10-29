"""An NLTK data package at a pin (tests/sentences.py): unzipped whole under the pin's name once its bytes hash to it, and
fetched once; bytes that do not leave nothing behind."""

import hashlib
import io
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from hands.brain import pinned
from sentences import Pin, Unpinned, install

URL = "https://nltk.example/packages/tokenizers/punkt_tab.zip"


def package() -> bytes:
    """An archive laid out as NLTK's punkt_tab is: one directory named for the package."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as written:
        written.writestr("punkt_tab/english/sent_starters.txt", "however\n")
    return archive.getvalue()


def serving(body: bytes, fetched: list[str]) -> Callable[[str, pinned.Write], Awaitable[None]]:
    """A fetch that serves `body` at any URL, recording each URL it was asked for in `fetched`."""

    async def fetch(url: str, write: pinned.Write) -> None:
        fetched.append(url)
        write(body)

    return fetch


async def test_a_package_that_hashes_to_its_pin_is_unzipped_under_the_pin_s_name_and_fetched_once(tmp_path: Path) -> None:
    body = package()
    pin = Pin(URL, hashlib.sha256(body).hexdigest(), "tokenizers")
    fetched: list[str] = []

    data = await install(pin, tmp_path, serving(body, fetched))

    assert data == pin.data(tmp_path)
    assert (data / "tokenizers" / "punkt_tab" / "english" / "sent_starters.txt").read_text() == "however\n"
    assert await install(pin, tmp_path, serving(body, fetched)) == data
    assert fetched == [URL]
    assert [path.name for path in tmp_path.iterdir()] == [pin.sha256]


async def test_a_package_that_does_not_hash_to_its_pin_leaves_nothing(tmp_path: Path) -> None:
    pin = Pin(URL, "0" * 64, "tokenizers")

    with pytest.raises(Unpinned, match=f"{URL} hashes to "):
        await install(pin, tmp_path, serving(package(), []))

    assert list(tmp_path.iterdir()) == []


async def test_a_package_another_run_installed_while_this_one_fetched_stands(tmp_path: Path) -> None:
    body = package()
    pin = Pin(URL, hashlib.sha256(body).hexdigest(), "tokenizers")

    async def racing(url: str, write: pinned.Write) -> None:
        (pin.data(tmp_path) / "tokenizers" / "punkt_tab").mkdir(parents=True)
        write(body)

    assert await install(pin, tmp_path, racing) == pin.data(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == [pin.sha256]
