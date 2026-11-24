"""A file hands fetches, from a server of the test's own: what a real server does to a fetch that the network replaced in
tests/test_pinnedfile.py cannot show."""

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http.client import IncompleteRead
from pathlib import Path

import pytest
from aiohttp import web

from hands.voice.pinnedfile import Pin, fetched

pytestmark = pytest.mark.outside

BODY = b"the pinned file"
DIGEST = hashlib.sha256(BODY).hexdigest()


@asynccontextmanager
async def serving(headers: dict[str, str]) -> AsyncGenerator[str]:
    """A server of this machine's that answers every ask with the pinned file and `headers`: the address it is fetched from."""

    async def answer(_request: web.Request) -> web.Response:
        return web.Response(body=BODY, headers=headers)

    app = web.Application()
    app.router.add_get("/model.bin", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/model.bin"
    finally:
        await runner.cleanup()


async def test_a_file_is_fetched_from_a_server_whose_header_lines_are_far_longer_than_githubs(tmp_path: Path) -> None:
    # GitHub's content-security-policy alone was 9243 bytes on 2026-10-23, and aiohttp's own client refuses a line of 8190;
    # this is three and a half times GitHub's.
    async with serving({"Content-Security-Policy": "x" * (32 * 1024)}) as url:
        path = await asyncio.to_thread(fetched, tmp_path, Pin("model.bin", url, DIGEST, "revision 1"))
    assert path.read_bytes() == BODY


async def test_a_fetch_cut_short_leaves_no_file_to_be_taken_for_the_file(tmp_path: Path) -> None:
    async def cut(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Length": "100"})
        await response.prepare(request)
        await response.write(b"half")
        if request.transport:
            request.transport.close()
        return response

    app = web.Application()
    app.router.add_get("/model.bin", cut)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        url = f"http://127.0.0.1:{runner.addresses[0][1]}/model.bin"
        with pytest.raises(IncompleteRead):
            await asyncio.to_thread(fetched, tmp_path, Pin("model.bin", url, DIGEST, "revision 1"))
    finally:
        await runner.cleanup()
    assert list(tmp_path.iterdir()) == []


