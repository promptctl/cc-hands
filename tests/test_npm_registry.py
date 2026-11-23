"""The pin as npm's registry serves it: read out of its package, held to the pin like every source."""

import io
import tarfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from aiohttp import web

from hands.brain import pinned
from fixtures import pins
from npm_registry import REGISTRY


pytestmark = pytest.mark.outside

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
    yield lambda name: pinned.Source(url=lambda _version, _built: f"http://127.0.0.1:{port}/{name}.tgz", fetch=REGISTRY.fetch, failures=REGISTRY.failures)
    await runner.cleanup()


async def test_the_pin_is_read_out_of_its_package_and_a_package_without_it_is_refused_leaving_nothing(tmp_path: Path, registry: Callable[[str], pinned.Source]) -> None:
    pin = pins.of(RELEASE)
    with pytest.raises(pinned.Uninstallable, match=r"^could not fetch http://127\.0\.0\.1:\d+/without\.tgz: ExtractError\('the package holds no package/claude file'\)$"):
        await pinned.install(tmp_path, pin, registry("without"))
    assert list(pin.claude(tmp_path).parent.iterdir()) == []
    installed = await pinned.install(tmp_path, pin, registry("whole"))
    assert installed.fetched and pin.verified(tmp_path).read_bytes() == RELEASE

