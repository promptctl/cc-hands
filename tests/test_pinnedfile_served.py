"""A file hands fetches, from a server of the test's own: what a real server does to a fetch that the network replaced in
tests/test_pinnedfile.py cannot show."""

import asyncio
import hashlib
import socket
import threading
import urllib.request
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from http.client import IncompleteRead
from pathlib import Path

import pytest
from aiohttp import web

from hands.clock import SYSTEM
from hands.voice import pinnedfile
from hands.voice.pinnedfile import Pin, Release, Unserved, fetched

pytestmark = pytest.mark.outside

BODY = b"the pinned file"
DIGEST = hashlib.sha256(BODY).hexdigest()


@pytest.fixture(autouse=True)
def unproxied(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each fetch from the test's own loopback server goes direct. The environment names a proxy, so urllib ignores the
    machine's own proxy settings, as a Mac's system proxy, and `no_proxy` exempts 127.0.0.1. The proxy named is a listener
    nobody accepts on, and the test fails at teardown if any fetch connected to it."""
    with socket.socket() as proxy:
        proxy.bind(("127.0.0.1", 0))
        proxy.listen()
        proxy.setblocking(False)
        address = f"http://127.0.0.1:{proxy.getsockname()[1]}"
        for name in ("http_proxy", "https_proxy"):
            monkeypatch.setenv(name, address)
        monkeypatch.setenv("no_proxy", "127.0.0.1")
        # urlopen keeps the opener it builds first, with the proxies it read then: each test's is its own.
        monkeypatch.setattr(urllib.request, "_opener", None)
        yield
        try:
            proxy.accept()[0].close()
        except BlockingIOError:
            return
        pytest.fail("a fetch from the test's own server went through the proxy")


@asynccontextmanager
async def serving(handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> AsyncGenerator[Pin]:
    """A server of this machine's that answers every ask with `handler`: the pin of the file it is fetched as."""
    app = web.Application()
    app.router.add_get("/model.bin", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield Pin("model.bin", f"http://127.0.0.1:{runner.addresses[0][1]}/model.bin", DIGEST, "revision 1")
    finally:
        await runner.cleanup()


async def test_a_file_is_fetched_from_a_server_whose_header_lines_are_far_longer_than_githubs(tmp_path: Path) -> None:
    async def answer(_request: web.Request) -> web.StreamResponse:
        # GitHub's content-security-policy alone was 9243 bytes on 2026-10-23, and aiohttp's own client refuses a line of
        # 8190; this is three and a half times GitHub's.
        return web.Response(body=BODY, headers={"Content-Security-Policy": "x" * (32 * 1024)})

    async with serving(answer) as pin:
        kept = await asyncio.to_thread(fetched, tmp_path, pin)
    assert kept.path.read_bytes() == BODY


async def test_a_fetch_cut_short_leaves_no_file_to_be_taken_for_the_file(tmp_path: Path) -> None:
    async def cut(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Length": "100"})
        await response.prepare(request)
        await response.write(b"half")
        if request.transport:
            request.transport.close()
        return response

    async with serving(cut) as pin:
        with pytest.raises(IncompleteRead):
            await asyncio.to_thread(fetched, tmp_path, pin)
    assert list(tmp_path.iterdir()) == []


async def test_a_fetch_that_never_finishes_is_given_up_on_whatever_the_server_keeps_sending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stalled = asyncio.Event()

    async def drip(request: web.Request) -> web.StreamResponse:
        # Bytes keep coming, so no read waits long enough to fail on its own, and the response never ends.
        response = web.StreamResponse()
        await response.prepare(request)
        while not stalled.is_set():
            await response.write(b".")
            await asyncio.sleep(0.01)
        return response

    monkeypatch.setattr(pinnedfile, "FETCH_DEADLINE", 0.3)
    async with serving(drip) as pin:
        try:
            with pytest.raises(Unserved, match=r"hands cannot get revision 1's model.bin: it took more than 0.3 s"):
                await Release(tmp_path, SYSTEM).path(pin)
            # Given up on, the fetch's own thread ends at the next bytes it reads, while the server is still sending: no fetch
            # goes on after the ask that wanted it said it could not be had.
            for _ in range(300):
                if not any(thread.name == "fetching model.bin" for thread in threading.enumerate()):
                    break
                await asyncio.sleep(0.01)
            assert [thread.name for thread in threading.enumerate() if thread.name == "fetching model.bin"] == []
        finally:
            stalled.set()
    assert list(tmp_path.iterdir()) == []
