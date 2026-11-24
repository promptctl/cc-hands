"""A file hands fetches whole from a URL and holds to the sha256 digest it pins it to: DTLN-aec's models
(`hands.voice.dtln`), ONNX Runtime Web's files (`hands.voice.ortweb`) and openWakeWord's (`hands.voice.wakemodels`) are
fetched this way.
"""

import asyncio
import hashlib
import os
import threading
import urllib.request
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from http.client import HTTPException
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from hands.clock import Clock
from hands.threads import off_loop

# How long a read of the file's server may wait for bytes, in seconds, before the fetch fails.
FETCH_TIMEOUT = 30
# How long a whole fetch is given, in seconds, however steadily the server sends: long enough for 15 MB on a slow
# connection; a stalled one fails the ask rather than hang it.
FETCH_DEADLINE = 180


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


# What a fetch found: a file already there with the digest, one fetched because none was, or one written over because the
# bytes there were not the pinned ones.
type How = Literal["kept", "fetched", "replaced"]


@dataclass(frozen=True)
class Kept:
    """A pinned file, held to its digest, where it is, and how it came to be there."""

    path: Path
    how: How


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetched(directory: Path, pin: Pin) -> Kept:
    """`pin`'s file in `directory`, fetched from its URL unless a file there already has its digest.

    [LAW:no-silent-failure] a download that is not the one pinned raises Mismatch naming it and both digests: a model or a
    runtime with other bytes would fail, or worse run, with nothing to say why. A file is fetched whole and held to its
    digest before it is written beside its place, by a name of this process's and thread's own, and moved to it: a fetch
    cut short leaves none to be taken for the file, and two fetches at once do not write the same one. A file there with
    other bytes is written over, and said to be, so a run that replaced one is not taken for one that only read it.
    """
    path = directory / pin.name
    if path.is_file() and digest_of(path) == pin.digest:
        return Kept(path, "kept")
    how: How = "replaced" if path.exists() else "fetched"
    with urllib.request.urlopen(pin.url, timeout=FETCH_TIMEOUT) as response:  # noqa: S310
        body = response.read()
    if (found := hashlib.sha256(body).hexdigest()) != pin.digest:
        raise Mismatch(f"{pin.name} at {pin.at} is {found}, not the {pin.digest} pinned")
    beside = directory / f"{pin.name}.{os.getpid()}.{threading.get_ident()}.part"
    try:
        beside.write_bytes(body)
        beside.replace(path)
    finally:
        # A write or a move that failed leaves nothing beside the place; after a move there is nothing to remove.
        beside.unlink(missing_ok=True)
    return Kept(path, how)


# A pin's fetch, as `Release` makes it: the directory the file goes in, and the file; it answers where the file is.
type Fetch = Callable[[Path, Pin], Awaitable[Kept]]


async def fetched_off_the_loop(directory: Path, pin: Pin) -> Kept:
    """The fetch hands makes: a read waits for the network on a thread of its own, so it holds nothing else up."""
    return await off_loop(partial(fetched, directory, pin), f"fetching {pin.name}")


class Unserved(Exception):
    """A file hands runs or serves that cannot be had, and why: it could not be fetched, is not the one pinned, or is not
    there."""


class Release:
    """Pinned files in `directory`, each fetched on the first ask for it and handed out only once its bytes have matched
    its digest.

    [LAW:single-enforcer] the one way to a pinned file the daemon runs or serves is `path`, which is asked for by the pin
    itself, so there is no name that is not a pin. A file's bytes are checked once for as long as this runs, when it is
    first asked for; what is in the directory after that is hands' own, as the rest of its home is, and one that has gone
    from it is fetched again. One ask is in the fetch at a time, so asks that arrive together fetch a file once.
    """

    def __init__(self, directory: Path, clock: Clock, fetch: Fetch = fetched_off_the_loop) -> None:
        self._directory = directory
        self._clock = clock
        self._fetch = fetch
        self._lock = asyncio.Lock()
        self._held = dict[Pin, Kept]()

    @property
    def how(self) -> Mapping[str, How]:
        """How each file asked for so far came to be there, as it was first asked for."""
        return MappingProxyType({pin.name: kept.how for pin, kept in self._held.items()})

    async def path(self, pin: Pin) -> Path:
        """Where `pin`'s file is, fetched if it is not there; raises Unserved naming why it cannot be had."""
        async with self._lock:
            if (held := self._held.get(pin)) is None or not held.path.is_file():
                try:
                    self._directory.mkdir(parents=True, exist_ok=True)
                    held = self._held[pin] = await self._within_deadline(pin)
                # A fetch's failures are its own: the directory's, the network's, a read cut short, one that took too long, and a file that is not the one pinned.
                except (OSError, HTTPException, Mismatch) as error:
                    raise Unserved(f"hands cannot get {pin.at}'s {pin.name}: {error}") from error
            return held.path

    async def _within_deadline(self, pin: Pin) -> Kept:
        """[LAW:no-silent-failure] the whole fetch is given FETCH_DEADLINE, however steadily the server sends: the lock is held
        across it, so one that never ended would hold every ask for the release behind it, with nothing said."""
        deadline = self._clock.timeout(FETCH_DEADLINE)
        try:
            async with deadline:
                return await self._fetch(self._directory, pin)
        except TimeoutError as error:
            if deadline.expired():
                raise TimeoutError(f"it took more than {FETCH_DEADLINE} s") from error
            raise

    async def ensured(self, pins: Iterable[Pin]) -> None:
        """Every one of `pins`, fetched and held to its digest."""
        for pin in pins:
            await self.path(pin)
