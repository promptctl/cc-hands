"""The pinned Claude Code as npm's registry serves it: each build of a release is a package of its own, whose
`package/claude` is the bytes of Anthropic's release, so the tests take the pin without a connection to Anthropic."""

import asyncio
import tarfile
import tempfile
from typing import IO

from hands.brain import pinned


def claude_of(tarball: IO[bytes], write: pinned.Write) -> None:
    """The `package/claude` of the npm package in `tarball`, written out; raises naming it when the package has none."""
    with tarfile.open(fileobj=tarball, mode="r:gz") as package:
        extracted = package.extractfile("package/claude")
        if extracted is None:
            raise tarfile.ExtractError("the package's package/claude is not a file")
        while chunk := extracted.read(1 << 20):
            write(chunk)


async def unpacked(url: str, write: pinned.Write) -> None:
    """The `package/claude` of npm's package at `url`: the tarball downloaded as the release is, then read out of it."""
    with tempfile.TemporaryFile() as tarball:
        await pinned.download(url, tarball.write)
        tarball.seek(0)
        # Off the event loop: the gunzip of a local file, bounded, never a wait on the network.
        await asyncio.to_thread(claude_of, tarball, write)


REGISTRY = pinned.Source(
    url=lambda built: f"https://registry.npmjs.org/@anthropic-ai/claude-code-{built}/-/claude-code-{built}-{pinned.VERSION}.tgz",
    fetch=unpacked,
)
