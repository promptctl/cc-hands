"""A file hands fetches whole from a URL and holds to the sha256 digest it pins it to: DTLN-aec's models
(`hands.voice.dtln`), ONNX Runtime Web's files (`hands.voice.ortweb`) and openWakeWord's (`hands.voice.wakemodels`) are
fetched this way.
"""

import asyncio
import hashlib
import os
import threading
import urllib.request
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from functools import partial
from http.client import HTTPException
from pathlib import Path

from hands.threads import off_loop

# How long a read of the file's server may wait for bytes, in seconds, before the fetch fails.
FETCH_TIMEOUT = 30


@dataclass(frozen=True)
class Pin:
    """A file and the one thing it must be: where it is fetched from, and the sha256 digest its bytes are held to. `at`
    says which release or revision `url` names."""

    name: str
    url: str
    digest: str
    at: str


class Mismatch(Exception):
    """A download that is not the file pinned."""


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetched(directory: Path, pin: Pin) -> Path:
    """`pin`'s file in `directory`, fetched from its URL unless a file there already has its digest.

    [LAW:no-silent-failure] a file that is not the one pinned raises Mismatch naming it and both digests: a model or a
    runtime with other bytes would fail, or worse run, with nothing to say why. A file is fetched whole and held to its
    digest before it is written beside its place, by a name of this process's and thread's own, and moved to it: a fetch
    cut short leaves none to be taken for the file, and two fetches at once do not write the same one.
    """
    path = directory / pin.name
    if not (path.is_file() and digest_of(path) == pin.digest):
        with urllib.request.urlopen(pin.url, timeout=FETCH_TIMEOUT) as response:  # noqa: S310
            body = response.read()
        if (found := hashlib.sha256(body).hexdigest()) != pin.digest:
            raise Mismatch(f"{pin.name} at {pin.at} is {found}, not the {pin.digest} pinned")
        beside = directory / f"{pin.name}.{os.getpid()}.{threading.get_ident()}.part"
        beside.write_bytes(body)
        beside.replace(path)
    return path


# A pin's fetch, as `Release` makes it: the directory the file goes in, and the file; it answers where the file is.
type Fetch = Callable[[Path, Pin], Awaitable[Path]]


async def fetched_off_the_loop(directory: Path, pin: Pin) -> Path:
    """The fetch hands makes: a read waits for the network on a thread of its own, so it holds nothing else up."""
    return await off_loop(partial(fetched, directory, pin), f"fetching {pin.name}")


class Unserved(Exception):
    """A pinned file that cannot be had, and why: it could not be fetched, or is not the one pinned."""


class Release:
    """A release's pinned files in `directory`, each fetched on the first ask for it and handed out only once its bytes
    have matched its digest.

    [LAW:single-enforcer] the one place a file's digest is held: `path` is the only way to a file's place. Each file is
    checked once for as long as this runs, when it is first asked for; what is in the directory after that is hands' own,
    as the rest of its home is. One ask is in the fetch at a time, so asks that arrive together fetch a file once.
    """

    def __init__(self, directory: Path, pins: Iterable[Pin], fetch: Fetch = fetched_off_the_loop) -> None:
        self._directory = directory
        self._pins = {pin.name: pin for pin in pins}
        self._fetch = fetch
        self._lock = asyncio.Lock()
        self._held = dict[str, Path]()

    async def path(self, name: str) -> Path:
        """Where the file called `name`, one of the pins', is, fetched if it is not there; raises Unserved naming why it cannot be had."""
        pin = self._pins[name]
        async with self._lock:
            if name not in self._held:
                try:
                    self._directory.mkdir(parents=True, exist_ok=True)
                    self._held[name] = await self._fetch(self._directory, pin)
                # A fetch's failures are its own: the directory's, the network's, a read cut short, and a file that is not the one pinned.
                except (OSError, HTTPException, Mismatch) as error:
                    raise Unserved(f"hands cannot get {pin.at}'s {name}: {error}") from error
            return self._held[name]

    async def ensured(self) -> None:
        """Every file of the release, fetched and held to its digest."""
        for name in self._pins:
            await self.path(name)
