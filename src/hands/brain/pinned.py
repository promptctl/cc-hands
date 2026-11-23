"""The Claude Code release the brain runs: one version, hands' own copy of it under the brain's directory, so what hands
has measured of Claude Code (each claim names the release it was measured on) stays true of the brain until the pin is
moved on purpose, and a move re-measures those claims.

The person's own claude is never it: their sessions run whatever release their install holds, and its updater moves
~/.local/bin/claude and prunes ~/.local/share/claude/versions whenever a release lands.

    hands install-brain     # fetches PIN's release into the brain's directory, checked against its checksum
"""

import functools
import hashlib
import os
import platform
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiohttp

RELEASES = "https://downloads.claude.ai/claude-code-releases"
INSTALL = "hands install-brain"
# How long the fetch waits on a connection, or on the next bytes of the download, before failing the install rather than
# hang it: a stall, never a slow link, which takes as long as 230 MB takes on it.
STALL_SECONDS = 60
_CHUNK = 1 << 20

# [LAW:one-source-of-truth] how this release marks a side question's cache rests on two variables the brain runs under,
# pinned in its --settings env (`hands.brain.process`), which outranks its settings.json env and its process environment,
# and followed by the side questions hands forks (`hands.brain.fork`), never read from the brain's setup
# (docs/claude-code-behavior.md, "brain stellar drift" and "brain prompt-cache ttl").
# Off: text of whitespace alone carries a cache marker.
STELLAR_DRIFT: Literal[False] = False
# Five minutes, of the two a marker can give.
CACHE_TTL: Literal["5m", "1h"] = "5m"
ENV = {"CLAUDE_CODE_STELLAR_DRIFT": str(int(STELLAR_DRIFT)), "CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL": CACHE_TTL}


class Unpinned(Exception):
    """The brain's copy of the pinned release is not there, or is not that release, so there is no claude to run as the
    brain: never another in its place."""


class Uninstallable(Exception):
    """The pinned release could not be put in place: no build of it for this machine, or a fetch that failed or did not
    hash to the pin. Nothing is left in its place."""


# What a fetch hands each chunk of a build's bytes to, in order.
Write = Callable[[bytes], object]


@dataclass(frozen=True)
class Source:
    """Where a build of a pin is fetched from: the URL of each build of each version, the fetch that writes the bytes at
    that URL, and the failures that fetch raises when the URL cannot give them. What it wrote is the pin only if it
    hashes to the pin's checksum."""

    url: Callable[[str, str], str]
    fetch: Callable[[str, Write], Awaitable[None]]
    failures: tuple[type[Exception], ...]


@dataclass(frozen=True)
class Installed:
    """The pinned claude, where it is, and, where this install fetched it, why the copy there was not the pin: None when it
    found the pin there already."""

    path: Path
    unpinned: str | None

    @property
    def fetched(self) -> bool:
        return self.unpinned is not None


@dataclass(frozen=True)
class Pin:
    """A Claude Code release the brain runs: its version, and the SHA-256 each macOS build of it hashes to, by build.

    [LAW:no-shared-mutable-globals] a value each start is handed, so any number of pins can be weighed in one process and
    nothing rebinds the one hands runs, PIN."""

    version: str
    checksums: Mapping[str, str]

    def build(self, machine: str) -> str:
        """The build of the pin a Mac with this `platform.machine()` runs; raises Uninstallable for any other."""
        match machine:
            case "arm64":
                return "darwin-arm64"
            case "x86_64":
                return "darwin-x64"
            case other:
                raise Uninstallable(f"Claude Code {self.version} has no macOS build for a {other!r} machine")

    def claude(self, config_dir: Path) -> Path:
        """Where the pinned claude of the brain on `config_dir` is, whether or not it has been installed there."""
        # Beside the brain's login and settings: everything the brain is, in the one directory hands owns for it.
        return config_dir / "pinned-claude" / self.version / "claude"

    def verified(self, config_dir: Path) -> Path:
        """The brain's pinned claude, once its bytes are the pin's and it can be run; raises Unpinned, naming the command
        that installs it, when it is not there, cannot be read, is not the pin, or cannot be run."""
        found = self.claude(config_dir)
        try:
            expected = self.checksums[self.build(platform.machine())]
        except Uninstallable as error:
            raise Unpinned(str(error)) from None
        try:
            held = found.stat()
            got = _sha256(found, (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns, held.st_ctime_ns))
        except FileNotFoundError:
            raise Unpinned(f"Claude Code {self.version}, the brain's, is not installed at {found}: run `{INSTALL}`") from None
        except OSError as error:
            raise Unpinned(f"{found} cannot be read as Claude Code {self.version}, the brain's ({error}): run `{INSTALL}` to fetch it again") from None
        # [LAW:parse-dont-validate] only bytes that hash to the pin, in a file this process can run, are ever the brain.
        if got != expected:
            raise Unpinned(f"{found} hashes to {got}, not the {expected} of Claude Code {self.version}, the brain's: run `{INSTALL}` to fetch it again")
        if not os.access(found, os.X_OK):
            raise Unpinned(f"{found}, Claude Code {self.version}, the brain's, cannot be run: run `{INSTALL}` to fetch it again")
        return found


# [LAW:one-source-of-truth] the pin hands runs: the version, and the SHA-256 Anthropic's manifest for it gives each macOS
# build (the manifest of the release the version names). Moving the pin is one commit that changes both, after measuring
# again what docs/claude-code-behavior.md and the comments that name the old version say. CI reads the version off this
# first line, as its cache key for the pinned Claude Code the end-to-end tests run (.github/workflows/test.yml).
PIN = Pin("2.1.308", {
    "darwin-arm64": "ca923c86ac84b30574a82331a05b6e066c27477a926ceaffe46a845a33e4dd37",
    "darwin-x64": "9a0bc140f5fb06e6366624d770371ef290e6731d1e84c4def5daf4df946d1b59",
})


@functools.cache
def _sha256(path: Path, identity: tuple[int, int, int, int, int]) -> str:
    """The SHA-256 of the file at `path` while its `identity` holds: one copy is hashed once a process, not at every side
    question, and one written over since is hashed again, its change time moved however its modified time is set back."""
    del identity  # the cache's key alone
    with path.open("rb") as held:
        return hashlib.file_digest(held, "sha256").hexdigest()


async def install(config_dir: Path, pin: Pin, source: Source) -> Installed:
    """`pin`'s claude in the brain's directory: fetched from `source` and checked against its checksum unless the pin is
    there already, and whole or not there at all."""
    try:
        return Installed(pin.verified(config_dir), unpinned=None)
    except Unpinned as error:
        # [LAW:nothing-unseen] why the copy there is fetched over: missing, unreadable, unrunnable, or bytes not the pin's.
        unpinned = str(error)
    built = pin.build(platform.machine())
    target = pin.claude(config_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Uninstallable(f"could not make {target.parent} for Claude Code {pin.version}: {error!r}") from error
    url = source.url(pin.version, built)
    try:
        # A name of this install's own: two at once never write, rename, or remove each other's download.
        handle, name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".partial")
    except OSError as error:
        raise Uninstallable(f"could not write {url} to {target}: {error!r}") from error
    os.close(handle)
    partial = Path(name)
    digest = hashlib.sha256()
    try:
        with partial.open("wb") as written:

            def write(chunk: bytes) -> None:
                written.write(chunk)
                digest.update(chunk)

            await source.fetch(url, write)
        # [LAW:single-enforcer] whatever the source, the bytes are the pin's before they are put in its place.
        if digest.hexdigest() != pin.checksums[built]:
            raise Uninstallable(f"{url} hashes to {digest.hexdigest()}, not the {pin.checksums[built]} Claude Code {pin.version} is pinned to")
        os.chmod(partial, 0o755)
        # [LAW:one-source-of-truth] a file under its own name is the whole of it: a copy that is not the pin is replaced whole.
        partial.replace(target)
    except source.failures as error:
        # [LAW:types-are-the-program] the source names its own failures: those are the fetch's, any other OSError is the
        # disk's, and anything else is a defect, raised as itself.
        raise Uninstallable(f"could not fetch {url}: {error!r}") from error
    except OSError as error:
        raise Uninstallable(f"could not write {url} to {target}: {error!r}") from error
    finally:
        # Whole or not there at all: a fetch that failed, did not hash to the pin, or was interrupted leaves none of itself.
        partial.unlink(missing_ok=True)
    # The bytes renamed into place are the ones hashed to the pin as they were written.
    return Installed(target, unpinned)


# What download raises when the URL cannot give its body: refused, unreachable, or stalled.
DOWNLOAD_FAILURES = (aiohttp.ClientError, TimeoutError)


async def download(url: str, write: Write) -> None:
    """The body at `url`, written as it arrives, failing on a stall rather than hanging."""
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=STALL_SECONDS, sock_read=STALL_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, raise_for_status=True) as http, http.get(url) as response:
        async for chunk in response.content.iter_chunked(_CHUNK):
            write(chunk)


def released(base: str) -> Source:
    """The releases served under `base` as Anthropic serves its own."""
    return Source(url=lambda version, built: f"{base}/{version}/{built}/claude", fetch=download, failures=DOWNLOAD_FAILURES)


# The release as Anthropic serves it, the one `hands install-brain` fetches.
RELEASE = released(RELEASES)
