"""The fetch of a pinned file (`hands.voice.pinnedfile.Fetch`) as a test answers it: no socket opens and no thread runs. The
digest a file is held to is `hands.voice.pinnedfile`'s, and tested there."""

import asyncio
import io
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from hands.voice.pinnedfile import Kept, Pin


class Fetching:
    """Each call's arguments, and what the call does: it writes the file, with the bytes the test gave it for that name or
    ones that say which file it is, or it raises what the test says."""

    def __init__(self, contents: Mapping[str, bytes] | None = None) -> None:
        self.calls = list[tuple[Path, Pin]]()
        # The event each call was given, which is set once the fetch is given up on.
        self.stops = list[threading.Event]()
        self.failure: Exception | None = None
        self._contents = contents or {}
        # Set as a call begins, and the call finishes only once `release` is: so a test holds a fetch open.
        self.begun, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()

    @property
    def names(self) -> list[str]:
        return [pin.name for _, pin in self.calls]

    async def __call__(self, directory: Path, pin: Pin, stop: threading.Event) -> Kept:
        self.calls.append((directory, pin))
        self.stops.append(stop)
        self.begun.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        directory.mkdir(parents=True, exist_ok=True)
        (directory / pin.name).write_bytes(self._contents.get(pin.name, f"the contents of {pin.name}".encode()))
        return Kept(directory / pin.name, "fetched")


class Whole(io.BytesIO):
    """A response that arrives whole: it owes no bytes (`length`, as an HTTP response's own says what it still owes)."""

    length = 0


class Network:
    """What a file's server answers `urllib.request.urlopen`, and every URL asked for: the network as `hands.voice.pinnedfile.fetched` reads it, replaced where it reads it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
        self.asked = list[str]()
        self._body = body
        monkeypatch.setattr("urllib.request.urlopen", self._open)

    def _open(self, url: str, timeout: float) -> Whole:
        self.asked.append(url)
        return Whole(self._body)
