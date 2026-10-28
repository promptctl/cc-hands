"""The pin as npm's registry serves it: read out of its package, held to the pin like every source."""

import hashlib
import io
import tarfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from aiohttp import web

from hands.brain import pinned
from npm_registry import REGISTRY

RELEASE = b"#!/bin/sh\necho 'the pinned release'\n"


def package(**members: bytes) -> bytes:
    """An npm package's tarball: each of `members` under package/, gzipped."""
    held = io.BytesIO()
    with tarfile.open(fileobj=held, mode="w:gz") as packed:
        for name, content in members.items():
            member = tarfile.TarInfo(f"package/{name}")
            member.size = len(content)
            packed.addfile(member, io.BytesIO(content))
    return held.getvalue()


@pytest.fixture
async def registry() -> AsyncIterator[Callable[[str], pinned.Source]]:
    """A stand-in for npm's registry, serving a package with the pin and one without: the source of each by its name."""
    packages = {"whole": package(**{"package.json": b"{}", "claude": RELEASE}), "without": package(**{"package.json": b"{}"})}

    async def tarball(request: web.Request) -> web.Response:
        return web.Response(body=packages[request.match_info["name"]])

    app = web.Application()
    app.router.add_get("/{name}.tgz", tarball)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    [(_, port)] = runner.addresses
    yield lambda name: pinned.Source(url=lambda built: f"http://127.0.0.1:{port}/{name}.tgz", fetch=REGISTRY.fetch)
    await runner.cleanup()


async def test_the_pin_is_read_out_of_its_package_and_a_package_without_it_is_refused_leaving_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: Callable[[str], pinned.Source]) -> None:
    monkeypatch.setattr(pinned, "CHECKSUMS", dict.fromkeys(pinned.CHECKSUMS, hashlib.sha256(RELEASE).hexdigest()))
    with pytest.raises(pinned.Uninstallable, match=r"^could not fetch http://127\.0\.0\.1:\d+/without\.tgz: KeyError\(\"filename 'package/claude' not found\"\)$"):
        await pinned.install(tmp_path, registry("without"))
    assert list(pinned.claude(tmp_path).parent.iterdir()) == []
    installed = await pinned.install(tmp_path, registry("whole"))
    assert installed.fetched and pinned.verified(tmp_path).read_bytes() == RELEASE

