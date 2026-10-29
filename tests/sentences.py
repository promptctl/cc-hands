"""NLTK's punkt_tab, the sentence tokenizer pipecat splits a reply with, at a pin, fetched before the run (conftest):
unfound, pipecat downloads whatever NLTK's index serves that day into the HOME of whichever test first splits a sentence."""

import hashlib
import io
import tempfile
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from hands.brain import pinned


class Unpinned(Exception):
    """A package's fetched bytes did not hash to its pin: nothing is put in its place."""


@dataclass(frozen=True)
class Pin:
    """An NLTK data package as NLTK's index describes one: the archive at `url`, whose bytes hash to `sha256`, unzipped
    under its `subdir` of an NLTK data directory."""

    url: str
    sha256: str
    subdir: str

    def data(self, cache: Path) -> Path:
        """The NLTK data directory under `cache` that holds the package once it is fetched: named by the pin, so a moved
        pin is fetched afresh and never taken for the one before it."""
        return cache / self.sha256


# [LAW:one-source-of-truth] the pin: the archive at a commit of NLTK's data, and the SHA-256 NLTK's index gives it.
PUNKT_TAB = Pin(
    url="https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip",
    sha256="e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
    subdir="tokenizers",
)

# What install raises when the package cannot be put in place: a fetch that failed, bytes not the pin's, or the disk's.
INSTALL_FAILURES = (*pinned.DOWNLOAD_FAILURES, Unpinned, OSError)


async def install(pin: Pin, cache: Path, fetch: Callable[[str, pinned.Write], Awaitable[None]]) -> Path:
    """`pin`'s NLTK data directory under `cache`: fetched and checked against the pin unless it is there already, and
    whole or not there at all."""
    data = pin.data(cache)
    if data.is_dir():
        return data
    archive = io.BytesIO()
    digest = hashlib.sha256()

    def write(chunk: bytes) -> None:
        archive.write(chunk)
        digest.update(chunk)

    await fetch(pin.url, write)
    # [LAW:single-enforcer] the bytes are the pin's before anything of them is unzipped.
    if digest.hexdigest() != pin.sha256:
        raise Unpinned(f"{pin.url} hashes to {digest.hexdigest()}, not the {pin.sha256} it is pinned to")
    with tempfile.TemporaryDirectory(dir=cache, prefix=".partial-") as staging:
        unzipped = Path(staging) / "data"
        zipfile.ZipFile(archive).extractall(unzipped / pin.subdir)
        # A directory under the pin's name is the whole package: an unzip that was interrupted is left in staging.
        unzipped.rename(data)
    return data
