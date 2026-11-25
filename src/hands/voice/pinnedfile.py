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
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from typing import Literal

from hands.clock import Clock
from hands.threads import off_loop

# How long a read of the file's server may wait for bytes, in seconds, before the fetch fails.
FETCH_TIMEOUT = 30
# How long a whole fetch is given, in seconds, however steadily the server sends: long enough for 15 MB on a slow
# connection; a stalled one fails the ask rather than hang it.
FETCH_DEADLINE = 180
_CHUNK = 1 << 16


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


class Stopped(Exception):
    """A fetch that was told to give up and did."""


# How an ask came by a file: one this run already held, one already there with the digest, one fetched because none was,
# or one written over because the bytes there were not the pinned ones.
type How = Literal["held", "kept", "fetched", "replaced"]


@dataclass(frozen=True)
class Kept:
    """A pinned file, held to its digest, where it is, and how this ask came by it."""

    path: Path
    how: How


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetched(directory: Path, pin: Pin, stop: threading.Event | None = None) -> Kept:
    """`pin`'s file in `directory`, fetched from its URL unless a file there already has its digest.

    [LAW:single-enforcer] the one place bytes are held to a digest before they are a pinned file. A download that is not
    the one pinned raises Mismatch naming it and both digests: a model or a runtime with other bytes would fail, or worse
    run, with nothing to say why. A file is fetched whole and held to its digest before it is written beside its place, by a
    name of this process's and thread's own, and moved to it: a fetch cut short leaves none to be taken for the file, and
    two fetches at once do not write the same one. A file there with other bytes is written over, and said to be, so a run
    that replaced one is not taken for one that only read it. `stop`, once set, ends the fetch at the next bytes it reads
    (raising Stopped), which is how a fetch that cannot be cancelled is given up on.
    """
    path = directory / pin.name
    if path.is_file() and digest_of(path) == pin.digest:
        return Kept(path, "kept")
    how: How = "replaced" if path.exists() else "fetched"
    received = bytearray()
    with urllib.request.urlopen(pin.url, timeout=FETCH_TIMEOUT) as response:  # noqa: S310
        # `read1`, so a server sending a few bytes at a time is heard between them, and not waited on for a whole chunk.
        while chunk := response.read1(_CHUNK):
            if stop is not None and stop.is_set():
                raise Stopped(f"the fetch of {pin.name} from {pin.url} was told to stop")
            received += chunk
        # `read1` ends quietly where the server closed early; the bytes a response still owed say it was cut short.
        if response.length:
            raise IncompleteRead(bytes(received), response.length)
    body = bytes(received)
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


# A pin's fetch, as `Release` makes it: the directory the file goes in, the file, and the event that is set when the fetch is
# given up on; it answers where the file is.
type Fetch = Callable[[Path, Pin, threading.Event], Awaitable[Kept]]


async def fetched_off_the_loop(directory: Path, pin: Pin, stop: threading.Event) -> Kept:
    """The fetch hands makes: a read waits for the network on a thread of its own, so it holds nothing else up."""
    return await off_loop(partial(fetched, directory, pin, stop), f"fetching {pin.name}")


class Unserved(Exception):
    """A file hands runs or serves that cannot be had, and why: it could not be fetched, is not the one pinned, or is not
    there. `passing` says whether asking again may change it: a fetch that failed on its way may come the next time, and
    bytes that are not the ones pinned, or a file that is not there, would be the same."""

    def __init__(self, why: str, passing: bool) -> None:
        super().__init__(why)
        self.passing = passing


class Release:
    """Pinned files in `directory`, each fetched on the first ask for it and handed out only once its bytes have matched
    its digest.

    [LAW:single-enforcer] the one way to a pinned file the daemon serves is `path`, which is asked for by the pin itself,
    so there is no name that is not a pin. The bytes are held to the digest by `fetched`, which the fetch it is given is
    expected to be, and once for as long as this runs: what is in the directory after that is hands' own, as the rest of its
    home is, and one that has gone from it is fetched again. Asks that arrive together fetch a file once: one is in the
    fetch at a time, though one given up on may still be ending, within a read's timeout, as it is told to.
    """

    def __init__(self, directory: Path, clock: Clock, fetch: Fetch = fetched_off_the_loop) -> None:
        self._directory = directory
        self._clock = clock
        self._fetch = fetch
        self._lock = asyncio.Lock()
        self._held = dict[Pin, Path]()

    async def path(self, pin: Pin) -> Kept:
        """`pin`'s file, fetched if it is not there, and how this ask came by it; raises Unserved naming why it cannot be had."""
        async with self._lock:
            if (held := self._held.get(pin)) is not None and held.is_file():
                return Kept(held, "held")
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
                kept = await self._within_deadline(pin)
            # A fetch's failures are its own: the directory's, the network's, a read cut short, and one that took too long,
            # each of which may pass; and a file that is not the one pinned, which fetched again would be the same.
            except (OSError, HTTPException) as error:
                raise Unserved(f"hands cannot get {pin.at}'s {pin.name}: {error}", passing=True) from error
            except Mismatch as error:
                raise Unserved(f"hands cannot get {pin.at}'s {pin.name}: {error}", passing=False) from error
            self._held[pin] = kept.path
            return kept

    async def _within_deadline(self, pin: Pin) -> Kept:
        """[LAW:no-silent-failure] the whole fetch is given FETCH_DEADLINE, however steadily the server sends: the lock is held
        across it, so one that never ended would hold every ask for the release behind it, with nothing said."""
        deadline = self._clock.timeout(FETCH_DEADLINE)
        stop = threading.Event()
        try:
            async with deadline:
                return await self._fetch(self._directory, pin, stop)
        except TimeoutError as error:
            if deadline.expired():
                raise TimeoutError(f"it took more than {FETCH_DEADLINE} s") from error
            raise
        finally:
            # However it ended, a fetch still running is told to give up, so none outlives its ask.
            stop.set()

    async def ensured(self, pins: Iterable[Pin]) -> None:
        """Every one of `pins`, fetched and held to its digest."""
        for pin in pins:
            await self.path(pin)
