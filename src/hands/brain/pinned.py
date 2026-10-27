"""The Claude Code release the brain runs: one version, hands' own copy of it under the brain's directory, so what hands
has measured of Claude Code (each claim names the release it was measured on) stays true of the brain until the pin is
moved on purpose, and a move re-measures those claims.

The person's own claude is never it: their sessions run whatever release their install holds, and its updater moves
~/.local/bin/claude and prunes ~/.local/share/claude/versions whenever a release lands.

    hands install-brain     # fetches this release into the brain's directory, checked against its checksum below
"""

import functools
import hashlib
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

import aiohttp

# [LAW:one-source-of-truth] the pin: the version, and the SHA-256 Anthropic's manifest for it gives each macOS build
# (the manifest of the release VERSION names). Moving the pin is one commit that changes both, after
# measuring again what the comments that name the old version say.
VERSION = "2.1.308"
CHECKSUMS = {
    "darwin-arm64": "ca923c86ac84b30574a82331a05b6e066c27477a926ceaffe46a845a33e4dd37",
    "darwin-x64": "9a0bc140f5fb06e6366624d770371ef290e6731d1e84c4def5daf4df946d1b59",
}
RELEASES = "https://downloads.claude.ai/claude-code-releases"
INSTALL = "hands install-brain"
# How long the fetch waits on a connection, or on the next bytes of the download, before failing the install rather than
# hang it: a stall, never a slow link, which takes as long as 230 MB takes on it.
STALL_SECONDS = 60
_CHUNK = 1 << 20


class Unpinned(Exception):
    """The brain's copy of the pinned release is not there, or is not that release, so there is no claude to run as the
    brain: never another in its place."""


class Uninstallable(Exception):
    """The pinned release could not be put in place: no build of it for this machine, or a fetch that failed or did not
    hash to the pin. Nothing is left in its place."""


@dataclass(frozen=True)
class Installed:
    """The pinned claude, where it is, and, where this install fetched it, why the copy there was not the pin: None when it
    found the pin there already."""

    path: Path
    unpinned: str | None

    @property
    def fetched(self) -> bool:
        return self.unpinned is not None


def claude(config_dir: Path) -> Path:
    """Where the pinned claude of the brain on `config_dir` is, whether or not it has been installed there."""
    # Beside the brain's login and settings: everything the brain is, in the one directory hands owns for it.
    return config_dir / "pinned-claude" / VERSION / "claude"


def build(machine: str) -> str:
    """The build of the pin a Mac with this `platform.machine()` runs; raises Uninstallable for any other."""
    match machine:
        case "arm64":
            return "darwin-arm64"
        case "x86_64":
            return "darwin-x64"
        case other:
            raise Uninstallable(f"Claude Code {VERSION} has no macOS build for a {other!r} machine")


def verified(config_dir: Path) -> Path:
    """The brain's pinned claude, once its bytes are the pin's and it can be run; raises Unpinned, naming the command
    that installs it, when it is not there, cannot be read, is not the pin, or cannot be run."""
    found = claude(config_dir)
    try:
        expected = CHECKSUMS[build(platform.machine())]
    except Uninstallable as error:
        raise Unpinned(str(error)) from None
    try:
        held = found.stat()
        got = _sha256(found, (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns, held.st_ctime_ns))
    except FileNotFoundError:
        raise Unpinned(f"Claude Code {VERSION}, the brain's, is not installed at {found}: run `{INSTALL}`") from None
    except OSError as error:
        raise Unpinned(f"{found} cannot be read as Claude Code {VERSION}, the brain's ({error}): run `{INSTALL}` to fetch it again") from None
    # [LAW:parse-dont-validate] only bytes that hash to the pin, in a file this process can run, are ever the brain.
    if got != expected:
        raise Unpinned(f"{found} hashes to {got}, not the {expected} of Claude Code {VERSION}, the brain's: run `{INSTALL}` to fetch it again")
    if not os.access(found, os.X_OK):
        raise Unpinned(f"{found}, Claude Code {VERSION}, the brain's, cannot be run: run `{INSTALL}` to fetch it again")
    return found


@functools.cache
def _sha256(path: Path, identity: tuple[int, int, int, int, int]) -> str:
    """The SHA-256 of the file at `path` while its `identity` holds: one copy is hashed once a process, not at every side
    question, and one written over since is hashed again, its change time moved however its modified time is set back."""
    del identity  # the cache's key alone
    with path.open("rb") as held:
        return hashlib.file_digest(held, "sha256").hexdigest()


async def install(config_dir: Path) -> Installed:
    """The pinned claude in the brain's directory: fetched and checked against its checksum unless the pin is there
    already, and whole or not there at all."""
    try:
        return Installed(verified(config_dir), unpinned=None)
    except Unpinned as error:
        # [LAW:nothing-unseen] why the copy there is fetched over: missing, unreadable, unrunnable, or bytes not the pin's.
        unpinned = str(error)
    built = build(platform.machine())
    target = claude(config_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Uninstallable(f"could not make {target.parent} for Claude Code {VERSION}: {error!r}") from error
    url = f"{RELEASES}/{VERSION}/{built}/claude"
    try:
        # A name of this install's own: two at once never write, rename, or remove each other's download.
        handle, name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".partial")
    except OSError as error:
        raise Uninstallable(f"could not write {url} to {target}: {error!r}") from error
    os.close(handle)
    partial = Path(name)
    try:
        try:
            digest = await _fetched(url, partial)
        except (aiohttp.ClientError, TimeoutError) as error:
            raise Uninstallable(f"could not fetch {url}: {error!r}") from error
        if digest != CHECKSUMS[built]:
            raise Uninstallable(f"{url} hashes to {digest}, not the {CHECKSUMS[built]} Claude Code {VERSION} is pinned to")
        os.chmod(partial, 0o755)
        # [LAW:one-source-of-truth] a file under its own name is the whole of it: a copy that is not the pin is replaced whole.
        partial.replace(target)
    except OSError as error:
        raise Uninstallable(f"could not write {url} to {target}: {error!r}") from error
    finally:
        # Whole or not there at all: a fetch that failed, did not hash to the pin, or was interrupted leaves none of itself.
        partial.unlink(missing_ok=True)
    # The bytes renamed into place are the ones hashed to the pin as they were written.
    return Installed(target, unpinned)


async def _fetched(url: str, partial: Path) -> str:
    """Download `url` into `partial`, and the SHA-256 of what it wrote."""
    digest = hashlib.sha256()
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=STALL_SECONDS, sock_read=STALL_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout, raise_for_status=True) as http:
        async with http.get(url) as response:
            with partial.open("wb") as written:
                async for chunk in response.content.iter_chunked(_CHUNK):
                    digest.update(chunk)
                    written.write(chunk)
    return digest.hexdigest()
