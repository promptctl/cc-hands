"""ONNX Runtime Web, the runtime the phone page runs the wake word's models on (`hands.voice.phonewake`): its WebAssembly
build, which runs on one thread, so the page need not be cross-origin isolated. The build is two files, a script that
carries its own glue and the WebAssembly file it fetches from beside itself.

hands fetches both at VERSION, holds each to the digest it pins, and serves them: nothing the phone page runs comes from
another site.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from http.client import HTTPException
from pathlib import Path

from hands.voice import pinnedfile

VERSION = "1.31.1"
RELEASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist"
# [LAW:one-source-of-truth] the files the page loads, each by the digest of its file.
FILES: Mapping[str, str] = {
    "ort.wasm.bundle.min.mjs": "51bd7168175767324ab0acfcfc4ca94a096a3826e3ebac9ccc4f195eb865f8dd",
    "ort-wasm-simd-threaded.wasm": "4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d",
}
# Where the page asks for them. The version is in the name, so a browser keeps the 14.8 MB file for as long as hands runs
# this version, and a page that asks for another version is told there is none.
PREFIX = f"/ort/{VERSION}/"

# Where the files are kept. A test run points it at a directory of its own (tests/conftest.py), fetched once for the run,
# because each test has a HOME of its own that holds nothing.
CACHE = Path.home() / ".cache" / "hands" / "onnxruntime-web" / VERSION


# A file's fetch: the directory it goes in, its name, the URL it is fetched from, the digest it is held to, and where the
# URL is from, as `hands.voice.pinnedfile.fetched` takes them, and where the file is once it is fetched.
type Fetch = Callable[[Path, str, str, str, str], Awaitable[Path]]


async def fetched_off_the_loop(directory: Path, name: str, url: str, digest: str, at: str) -> Path:
    """The fetch hands makes: a 14.8 MB read waits for the network on a thread of its own, so it holds nothing else up."""
    return await asyncio.to_thread(pinnedfile.fetched, directory, name, url, digest, at)


class Unserved(Exception):
    """A file of the runtime that cannot be served, and why: it could not be fetched, or is not the one pinned."""


class Runtime:
    """The runtime's files in `directory`, each fetched on the first ask for it and served only once its bytes have matched
    its digest in `files`.

    [LAW:single-enforcer] the one place a file's digest is held: `path` is the only way to a file's place, so a route that
    serves what `path` returned serves a file that matched, never one read beside the check. Each file is checked once
    for as long as this runs. One ask is in the fetch at a time, so a page's asks that arrive together fetch a file once.
    """

    def __init__(self, directory: Path = CACHE, files: Mapping[str, str] = FILES, fetch: Fetch = fetched_off_the_loop) -> None:
        self.files = files
        self._directory = directory
        self._fetch = fetch
        self._lock = asyncio.Lock()
        self._held = dict[str, Path]()

    async def path(self, name: str) -> Path:
        """Where the file called `name` is, fetched from RELEASE if it is not there; raises Unserved naming why it cannot be served. `name` is one of `files`."""
        async with self._lock:
            if name not in self._held:
                self._directory.mkdir(parents=True, exist_ok=True)
                try:
                    self._held[name] = await self._fetch(self._directory, name, f"{RELEASE}/{name}", self.files[name], f"onnxruntime-web {VERSION}")
                # A fetch's failures are its own: the network's, a read cut short, and a file that is not the one pinned.
                except (OSError, HTTPException, ValueError) as error:
                    raise Unserved(f"hands cannot serve onnxruntime-web {VERSION}'s {name}: {error}") from error
            return self._held[name]

    async def ensured(self) -> None:
        """Every file of the runtime, fetched and held to its digest."""
        for name in self.files:
            await self.path(name)
