"""The proxy as a client's HTTPS proxy: a CONNECT to the API's host is answered by the proxy itself, which ends its TLS and
serves what comes inside it as it serves a request to its own origin; a CONNECT to any other host is a tunnel."""

import asyncio
import ssl
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

from hands.clock import SYSTEM
from hands.core.session import SessionId
from hands.core.wire import API_HOST, UPSTREAM, Exchanged, MainTurn, Reached, Send, Sent, Synthesize, Synthetic, Tail
from hands.sessions.authority import mint
from hands.sessions.connect import ESTABLISHED
from hands.sessions.proxy import Proxy, serve_proxy
from hands.sessions.wide import root
from test_proxy import HEADERS, REQUEST, STREAM, Handler, Upstream, Wire, only_exchange, serve, streamed  # pyright: ignore[reportUnusedImport]

pytestmark = pytest.mark.outside

Serve = Callable[[Handler], Awaitable[tuple[Upstream, Wire]]]


def trusting(proxy: Proxy) -> ssl.SSLContext:
    """What a client given the proxy's trust file trusts, as Node does: the system's authorities and the file's."""
    context = ssl.create_default_context(cafile=str(proxy.through.trust))
    context.set_alpn_protocols(["h2", "http/1.1"])
    return context


async def through(proxy: Proxy, path: str = "/v1/messages?beta=true", body: bytes = REQUEST) -> tuple[int, dict[str, str], bytes]:
    """A request to https://api.anthropic.com, the proxy the client's HTTPS proxy."""
    async with aiohttp.ClientSession(auto_decompress=False) as client:
        async with client.post(UPSTREAM + path, data=body, headers=HEADERS, proxy=proxy.url, ssl=trusting(proxy)) as response:
            return response.status, dict(response.headers), await response.read()


@asynccontextmanager
async def tunnel(proxy: Proxy, authority: str) -> AsyncGenerator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    """One connection to the proxy, its CONNECT to `authority` granted."""
    reader, writer = await asyncio.open_connection(URL(proxy.url).host, URL(proxy.url).port)
    try:
        writer.write(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
        assert await reader.readuntil(b"\r\n\r\n") == ESTABLISHED
        yield reader, writer
    finally:
        writer.close()


async def answered(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    """One HTTP/1.1 answer read off `reader`: its status line, and its body, by its Content-Length."""
    head = await reader.readuntil(b"\r\n\r\n")
    status, *lines = head.decode().split("\r\n")
    [length] = [int(line.split(":", 1)[1]) for line in lines if line.lower().startswith("content-length:")]
    return status.encode(), await reader.readexactly(length)


async def test_requests_inside_one_connect_are_served_by_the_proxy_kept_alive_and_each_recorded_once(serve: Serve) -> None:
    async def echoed(request: web.Request) -> web.Response:
        return web.json_response({"n": len(await request.read())})

    upstream, wire = await serve(echoed)
    async with tunnel(wire.proxy, "api.anthropic.com:443") as (reader, writer):
        await writer.start_tls(trusting(wire.proxy), server_hostname="api.anthropic.com")
        # Only the HTTP the proxy serves is offered, whatever the client asks for.
        assert writer.get_extra_info("ssl_object").selected_alpn_protocol() == "http/1.1"
        for _ in range(3):
            headers = "".join(f"{name}: {value}\r\n" for name, value in HEADERS.items())
            writer.write(f"POST /v1/messages?beta=true HTTP/1.1\r\nHost: api.anthropic.com\r\n{headers}Content-Length: {len(REQUEST)}\r\n\r\n".encode() + REQUEST)
            assert await answered(reader) == (b"HTTP/1.1 200 OK", b'{"n": %d}' % len(REQUEST))
    assert [path for path, _, _ in upstream.asked] == ["/v1/messages?beta=true"] * 3
    ended = [seen for seen in wire.seen if isinstance(seen, Exchanged)]
    sent = [seen for seen in wire.seen if isinstance(seen, Sent)]
    assert len({each.exchange for each in ended}) == len(ended) == 3 and {each.exchange for each in sent} == {each.exchange for each in ended}


async def test_a_stream_inside_tls_streams_and_is_one_classified_exchange(serve: Serve) -> None:
    # The API holds the rest of its answer until the client has had the first frame: only a reply relayed as it comes
    # reaches the client at all.
    first = asyncio.Event()

    async def held(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "request-id": "req_1"})
        await response.prepare(request)
        await response.write(STREAM[0])
        await first.wait()
        for chunk in STREAM[1:]:
            await response.write(chunk)
        await response.write_eof()
        return response

    upstream, wire = await serve(held)
    async with aiohttp.ClientSession(auto_decompress=False) as client:
        async with client.post(UPSTREAM + "/v1/messages?beta=true", data=REQUEST, headers=HEADERS, proxy=wire.proxy.url, ssl=trusting(wire.proxy)) as response:
            got = await response.content.readany()
            first.set()
            got += await response.read()
    assert got == b"".join(STREAM)
    [(path, asked_headers, asked_body)] = upstream.asked
    assert (path, asked_body, asked_headers["Authorization"]) == ("/v1/messages?beta=true", REQUEST, "Bearer the-clients-own")
    exchange = only_exchange(wire)
    assert (exchange.session, exchange.kind, exchange.path) == (SessionId("s1"), MainTurn(None), "/v1/messages?beta=true")
    assert isinstance(exchange.reply, Reached) and exchange.reply.reply_bytes == len(got)


async def test_a_route_changes_answers_and_refuses_inside_tls_as_at_the_proxys_origin(serve: Serve) -> None:
    async def refusing(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(status=529, body=b'{"type":"error","error":{"type":"overloaded_error"}}', headers={"Content-Type": "application/json"})

    upstream, wire = await serve(refusing)
    # Changed: the API is asked the request with hands' tail on it.
    wire.route = lambda _sent: Send((Tail("a note"),), span=root(SYSTEM))
    await through(wire.proxy)
    assert b"a note" in upstream.asked[-1][2]
    # Held final: the client is told the proxy's own refusal, never to ask again.
    wire.route = lambda _sent: Send(refusal="final", span=root(SYSTEM))
    status, headers, _ = await through(wire.proxy)
    assert (status, {name.lower(): value for name, value in headers.items()}.get("x-should-retry")) == (502, "false")
    # Answered by hands itself, the API not asked.
    asked = len(upstream.asked)
    wire.route = lambda _sent: Synthesize("(stayed silent)", root(SYSTEM))
    status, _, body = await through(wire.proxy)
    assert status == 200 and b"(stayed silent)" in body and len(upstream.asked) == asked
    assert isinstance([seen for seen in wire.seen if isinstance(seen, Exchanged)][-1].reply, Synthetic)


@pytest.fixture
async def elsewhere() -> AsyncIterator[tuple[str, ssl.SSLContext]]:
    """A TLS server on loopback that is not the API: its authority, as `localhost:<port>`, and how a client trusts it."""
    signed = mint("localhost", SYSTEM)
    app = web.Application()

    async def hello(request: web.Request) -> web.Response:
        return web.Response(text=f"hello from {request.host}")

    app.router.add_get("/", hello)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0, ssl_context=signed.answer).start()
    trust = ssl.create_default_context(cadata=signed.certificate.decode())
    try:
        yield f"localhost:{runner.addresses[0][1]}", trust
    finally:
        await runner.cleanup()


async def test_a_connect_to_another_host_is_a_tunnel_to_it_the_proxy_reads_nothing_of(serve: Serve, elsewhere: tuple[str, ssl.SSLContext]) -> None:
    authority, trust = elsewhere
    upstream, wire = await serve(streamed)
    async with aiohttp.ClientSession() as client:
        async with client.get(f"https://{authority}/", proxy=wire.proxy.url, ssl=trust) as response:
            assert (response.status, await response.text()) == (200, f"hello from {authority}")
    assert (upstream.asked, wire.seen) == ([], [])


@pytest.fixture
async def outer() -> AsyncIterator[tuple[str, list[bytes]]]:
    """An outer HTTPS proxy on loopback: its url, and the head of each CONNECT it was asked, which it grants."""
    asked: list[bytes] = []

    async def granted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        asked.append(head)
        host, port = head.split(b" ")[1].decode().rsplit(":", 1)
        far_reader, far_writer = await asyncio.open_connection(host, int(port))
        writer.write(ESTABLISHED)

        async def pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
            while chunk := await source.read(65536):
                sink.write(chunk)
                await sink.drain()
            sink.close()

        await asyncio.gather(pipe(reader, far_writer), pipe(far_reader, writer))

    server = await asyncio.start_server(granted, "127.0.0.1", 0)
    try:
        yield f"http://u:p@127.0.0.1:{server.sockets[0].getsockname()[1]}", asked
    finally:
        server.close()


async def test_a_tunnel_goes_through_the_outer_proxy_hands_environment_names(tmp_path: Path, elsewhere: tuple[str, ssl.SSLContext], outer: tuple[str, list[bytes]]) -> None:
    authority, trust = elsewhere
    url, asked = outer
    proxy = await serve_proxy("http://127.0.0.1:1", lambda _: None, lambda _: Send(span=root(SYSTEM)), SYSTEM, {"HTTPS_PROXY": url}, mint(API_HOST, SYSTEM), tmp_path / "trusted.pem")
    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(f"https://{authority}/", proxy=proxy.url, ssl=trust) as response:
                assert (response.status, await response.text()) == (200, f"hello from {authority}")
    finally:
        await proxy.close()
    # Its own CONNECT, with the credentials the outer proxy's url carries.
    assert asked == [f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\nProxy-Authorization: Basic dTpw\r\n\r\n".encode()]


async def test_a_host_the_proxy_cannot_reach_is_refused_saying_so(serve: Serve) -> None:
    _, wire = await serve(streamed)
    reader, writer = await asyncio.open_connection(URL(wire.proxy.url).host, URL(wire.proxy.url).port)
    try:
        # Port 1 on loopback: nothing listens there.
        writer.write(b"CONNECT 127.0.0.1:1 HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n")
        status, body = await answered(reader)
    finally:
        writer.close()
    assert status == b"HTTP/1.1 502 Bad Gateway" and body.startswith(b"hands' proxy could not reach 127.0.0.1:1: ")


async def test_a_request_for_another_origin_in_absolute_form_is_relayed_there_and_not_recorded(serve: Serve) -> None:
    other = web.Application()

    async def there(request: web.Request) -> web.Response:
        return web.Response(text=f"{request.method} {request.path_qs} {await request.text()}")

    other.router.add_route("*", "/{path:.*}", there)
    runner = web.AppRunner(other)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    upstream, wire = await serve(streamed)
    try:
        async with aiohttp.ClientSession() as client:
            # A client given the proxy as its HTTP proxy asks it in absolute form, as aiohttp does here.
            async with client.post(f"http://127.0.0.1:{runner.addresses[0][1]}/mcp?x=1", data=b"hi", proxy=wire.proxy.url) as response:
                assert (response.status, await response.text()) == (200, "POST /mcp?x=1 hi")
    finally:
        await runner.cleanup()
    assert (upstream.asked, wire.seen) == ([], [])


async def test_a_connect_the_proxy_cannot_read_is_refused_saying_why(serve: Serve) -> None:
    _, wire = await serve(streamed)
    reader, writer = await asyncio.open_connection(URL(wire.proxy.url).host, URL(wire.proxy.url).port)
    try:
        writer.write(b"CONNECT nowhere HTTP/1.1\r\n\r\n")
        status, body = await answered(reader)
    finally:
        writer.close()
    assert (status, body) == (b"HTTP/1.1 400 Bad Request", b"a CONNECT names a host and its port, not 'nowhere'")
