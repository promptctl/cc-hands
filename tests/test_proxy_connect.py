"""The proxy as a client's HTTPS proxy: a CONNECT to the API's host is answered by the proxy itself, which ends its TLS and
serves what comes inside it as it serves a request to its own origin; a CONNECT to any other host is a tunnel, through
the outer proxy hands' environment names or directly. Each CONNECT is one event, recorded as its connection ends."""

import asyncio
import contextlib
import ssl
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from cryptography.hazmat.primitives import serialization
from loguru import logger
from yarl import URL

from hands.clock import SYSTEM, Clock
from hands.core.session import SessionId
from hands.core.wire import API_HOST, UPSTREAM, Exchanged, MainTurn, Reached, Send, Sent, Synthesize, Synthetic, Tail
from hands.sessions.audit import Entry
from hands.sessions.authority import mint
from hands.sessions.connect import REACH_SECONDS
from hands.sessions.proxy import Proxy, serve_proxy
from hands.sessions.wide import WideEvent, root
from hands.virtualclock import VirtualClock
from test_proxy import HEADERS, REQUEST, STREAM, Serve, Wire, only_exchange, serving, streamed

pytestmark = pytest.mark.outside

ESTABLISHED = b"HTTP/1.1 200 Connection Established\r\n\r\n"


async def established(reader: asyncio.StreamReader) -> None:
    """The proxy's grant of a CONNECT read off `reader`: its 200, with no framing, as a tunnel has none."""
    status, *lines = (await reader.readuntil(b"\r\n\r\n")).decode().split("\r\n")
    assert status == "HTTP/1.1 200 Connection Established" and not [line for line in lines if line.lower().startswith(("content-length:", "transfer-encoding:"))], lines


@pytest.fixture
async def serve(tmp_path: Path) -> AsyncIterator[Serve]:
    # On the machine's time: the client verifies the certificate the proxy's authority signs against it.
    async with serving(tmp_path, SYSTEM) as start:
        yield start


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


async def connected(proxy: Proxy) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    url = URL(proxy.url)
    return await asyncio.open_connection(url.host, url.port)


@contextlib.asynccontextmanager
async def tunnel(proxy: Proxy, authority: str) -> AsyncGenerator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    """One connection to the proxy, its CONNECT to `authority` granted."""
    reader, writer = await connected(proxy)
    try:
        writer.write(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
        await established(reader)
        yield reader, writer
    finally:
        writer.close()


def posted(path: str = "/v1/messages?beta=true", host: str = API_HOST) -> bytes:
    headers = "".join(f"{name}: {value}\r\n" for name, value in HEADERS.items())
    return f"POST {path} HTTP/1.1\r\nHost: {host}\r\n{headers}Content-Length: {len(REQUEST)}\r\n\r\n".encode() + REQUEST


def parsed(got: bytes) -> tuple[bytes, bytes] | None:
    """The status line and body of the HTTP/1.1 answer `got` begins with, by its Content-Length; None until it is whole."""
    head, ended, rest = got.partition(b"\r\n\r\n")
    if not ended:
        return None
    status, *lines = head.decode().split("\r\n")
    [length] = [int(line.split(":", 1)[1]) for line in lines if line.lower().startswith("content-length:")]
    return (status.encode(), rest[:length]) if len(rest) >= length else None


async def answered(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    """One HTTP/1.1 answer read off `reader`."""
    got = await reader.readuntil(b"\r\n\r\n")
    while (whole := parsed(got)) is None:
        got += await reader.read(65536)
    return whole


class ByHand:
    """A client's TLS to the API's host, run by hand over a connection to the proxy, so a test says which of its bytes go
    out in one write."""

    def __init__(self, proxy: Proxy) -> None:
        self._in, self._out = ssl.MemoryBIO(), ssl.MemoryBIO()
        self._tls = trusting(proxy).wrap_bio(self._in, self._out, server_hostname=API_HOST)

    def hello(self) -> bytes:
        with contextlib.suppress(ssl.SSLWantReadError):
            self._tls.do_handshake()
        return self._out.read()

    async def finished(self, reader: asyncio.StreamReader) -> bytes:
        """The handshake done on what the proxy answers, and the client's last flight of it, not yet sent."""
        while True:
            try:
                self._tls.do_handshake()
                return self._out.read()
            except ssl.SSLWantReadError:
                self._in.write(await _more(reader))

    def sealed(self, plain: bytes) -> bytes:
        self._tls.write(plain)
        return self._out.read()

    async def answered(self, reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
        """One HTTP/1.1 answer read inside the TLS."""
        got = b""
        while (whole := parsed(got)) is None:
            try:
                got += self._tls.read(65536)
            except ssl.SSLWantReadError:
                self._in.write(await _more(reader))
        return whole


async def _more(reader: asyncio.StreamReader) -> bytes:
    more = await reader.read(65536)
    assert more, "the proxy closed the connection"
    return more


async def connect_event(recorded: asyncio.Queue[Entry]) -> WideEvent:
    """The next event the proxy recorded, a CONNECT's, once its connection ended."""
    entry = await recorded.get()
    assert isinstance(entry, WideEvent) and entry.event == "proxy.connect", entry
    return entry


def exchanges(wire: Wire) -> list[Exchanged]:
    return [seen for seen in wire.seen if isinstance(seen, Exchanged)]


async def whole(request: web.Request) -> web.Response:
    """The API's answer, sent whole: what it was asked for."""
    await request.read()
    return web.json_response({"asked": request.path_qs})


async def test_requests_inside_one_connect_are_served_by_the_proxy_kept_alive_and_each_recorded_once(serve: Serve) -> None:
    async def echoed(request: web.Request) -> web.Response:
        return web.json_response({"n": len(await request.read())})

    upstream, wire = await serve(echoed)
    async with tunnel(wire.proxy, "api.anthropic.com:443") as (reader, writer):
        await writer.start_tls(trusting(wire.proxy), server_hostname=API_HOST)
        # Only the HTTP the proxy serves is offered, whatever the client asks for.
        assert writer.get_extra_info("ssl_object").selected_alpn_protocol() == "http/1.1"
        for _ in range(3):
            writer.write(posted())
            assert await answered(reader) == (b"HTTP/1.1 200 OK", b'{"n": %d}' % len(REQUEST))
    assert [path for path, _, _ in upstream.asked] == ["/v1/messages?beta=true"] * 3
    sent = [seen for seen in wire.seen if isinstance(seen, Sent)]
    assert len({each.exchange for each in exchanges(wire)}) == len(exchanges(wire)) == 3 and {each.exchange for each in sent} == {each.exchange for each in exchanges(wire)}
    # The CONNECT is one event of its own, said once its connection ended: where it asked for, and that the proxy ended
    # its TLS.
    event = await connect_event(wire.recorded)
    assert (event.outcome, event.facts) == ("ok", {"host": API_HOST, "port": 443, "intercepted": True, "tls": "made"})


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


async def test_a_request_inside_tls_that_is_not_a_message_reaches_the_api_as_sent(serve: Serve) -> None:
    async def usage(request: web.Request) -> web.Response:
        return web.json_response({"asked": request.path_qs})

    upstream, wire = await serve(usage)
    async with aiohttp.ClientSession() as client:
        async with client.get(UPSTREAM + "/api/oauth/usage", headers={"Authorization": "Bearer the-clients-own"}, proxy=wire.proxy.url, ssl=trusting(wire.proxy)) as response:
            assert (response.status, await response.json()) == (200, {"asked": "/api/oauth/usage"})
    [(path, asked_headers, asked_body)] = upstream.asked
    assert (path, asked_headers["Authorization"], asked_body) == ("/api/oauth/usage", "Bearer the-clients-own", b"")


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
    assert isinstance(exchanges(wire)[-1].reply, Synthetic)


async def test_what_a_client_sends_in_one_flight_with_its_connect_and_its_handshakes_end_is_all_served(serve: Serve) -> None:
    # Its TLS hello behind the CONNECT's head, and its first request behind its handshake's last flight, each in one
    # write: the proxy has them past what it was reading for, and loses none.
    upstream, wire = await serve(whole)
    reader, writer = await connected(wire.proxy)
    try:
        tls = ByHand(wire.proxy)
        writer.write(f"CONNECT {API_HOST}:443 HTTP/1.1\r\nHost: {API_HOST}:443\r\n\r\n".encode() + tls.hello())
        await established(reader)
        writer.write(await tls.finished(reader) + tls.sealed(posted()))
        status, body = await tls.answered(reader)
    finally:
        writer.close()
    assert (status, body) == (b"HTTP/1.1 200 OK", b'{"asked": "/v1/messages?beta=true"}')
    assert [path for path, _, _ in upstream.asked] == ["/v1/messages?beta=true"]


async def test_a_connect_after_a_request_on_the_same_connection_is_answered_and_both_reach_the_api(serve: Serve) -> None:
    upstream, wire = await serve(whole)
    reader, writer = await connected(wire.proxy)
    try:
        # A side question to the proxy's own origin, its connection kept alive, then a CONNECT on it.
        writer.write(posted(host=URL(wire.proxy.url).authority))
        assert (await answered(reader))[0] == b"HTTP/1.1 200 OK"
        tls = ByHand(wire.proxy)
        writer.write(f"CONNECT {API_HOST}:443 HTTP/1.1\r\nHost: {API_HOST}:443\r\n\r\n".encode())
        await established(reader)
        writer.write(tls.hello())
        writer.write(await tls.finished(reader) + tls.sealed(posted()))
        assert (await tls.answered(reader))[0] == b"HTTP/1.1 200 OK"
    finally:
        writer.close()
    assert [path for path, _, _ in upstream.asked] == ["/v1/messages?beta=true"] * 2


async def test_whose_a_request_is_is_said_by_its_connection_and_its_form_never_by_how_its_path_reads(serve: Serve) -> None:
    upstream, wire = await serve(whole)
    reader, writer = await connected(wire.proxy)
    try:
        # Origin form, though its path begins as a scheme-relative URL would: the API's.
        writer.write(posted("//v1/messages", host=URL(wire.proxy.url).authority))
        assert (await answered(reader))[0] == b"HTTP/1.1 200 OK"
    finally:
        writer.close()
    async with tunnel(wire.proxy, f"{API_HOST}:443") as (reader, writer):
        await writer.start_tls(trusting(wire.proxy), server_hostname=API_HOST)
        # Inside the TLS the proxy ended, absolute form too: every request there is the API's.
        writer.write(posted(f"https://{API_HOST}/v1/messages?beta=true"))
        assert (await answered(reader))[0] == b"HTTP/1.1 200 OK"
    assert [path for path, _, _ in upstream.asked] == ["//v1/messages", "/v1/messages?beta=true"]


async def test_a_client_that_refuses_the_proxys_tls_is_said_in_the_log_and_its_connects_event(serve: Serve) -> None:
    upstream, wire = await serve(streamed)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="ERROR")
    try:
        # A client not given the proxy's trust file: it trusts the system's authorities alone, and refuses the proxy's.
        async with aiohttp.ClientSession() as client:
            with pytest.raises(aiohttp.ClientConnectorCertificateError):
                await client.post(UPSTREAM + "/v1/messages", data=REQUEST, proxy=wire.proxy.url, ssl=ssl.create_default_context())
        event = await connect_event(wire.recorded)
    finally:
        logger.remove(sink)
    said = f"a client of hands' proxy did not finish TLS for {API_HOST}, so none of its requests reach the API: "
    assert event.outcome == "failed" and event.error is not None and event.error.startswith(said) and "tls" not in event.facts
    assert [line for line in logged if line.startswith(said)] == [event.error]
    assert (upstream.asked, wire.seen) == ([], [])


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
    trust = ssl.create_default_context()
    trust.load_verify_locations(cadata=signed.certificate.public_bytes(serialization.Encoding.PEM).decode())
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
    event = await connect_event(wire.recorded)
    assert event.outcome == "ok" and event.facts == {"host": "localhost", "port": int(authority.rsplit(":", 1)[1]), "intercepted": False, "through": "directly"}
    assert event.counts["tunnelled_to_host"] > 0 and event.counts["tunnelled_to_client"] > 0


async def test_closing_the_proxy_ends_each_tunnel_still_open_and_says_so(serve: Serve) -> None:
    # A host that holds every connection open until it is ended.
    held: list[asyncio.StreamWriter] = []

    async def holding(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    server = await asyncio.start_server(holding, "127.0.0.1", 0)
    _, wire = await serve(streamed)
    try:
        async with tunnel(wire.proxy, f"127.0.0.1:{server.sockets[0].getsockname()[1]}") as (reader, _writer):
            await wire.proxy.close()
            assert await reader.read() == b""
        assert (await connect_event(wire.recorded)).outcome == "cancelled"
    finally:
        for writer in held:
            writer.close()
        server.close()


async def test_a_host_the_proxy_cannot_reach_is_refused_saying_so(serve: Serve) -> None:
    _, wire = await serve(streamed)
    reader, writer = await connected(wire.proxy)
    try:
        # Port 1 on loopback: nothing listens there.
        writer.write(b"CONNECT 127.0.0.1:1 HTTP/1.1\r\nHost: 127.0.0.1:1\r\n\r\n")
        status, body = await answered(reader)
    finally:
        writer.close()
    assert status == b"HTTP/1.1 502 Bad Gateway" and body.startswith(b"hands' proxy could not reach 127.0.0.1:1: ")
    event = await connect_event(wire.recorded)
    assert event.outcome == "failed" and event.error == body.decode()


@pytest.mark.parametrize("target", ["nowhere", "example.com:0"])
async def test_a_connect_naming_no_host_and_port_to_reach_is_refused_and_reaches_nothing(serve: Serve, target: str) -> None:
    _, wire = await serve(streamed)
    reader, writer = await connected(wire.proxy)
    try:
        writer.write(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
        status, _ = await answered(reader)
    finally:
        writer.close()
    # aiohttp's parser refuses it, before any handler of the proxy's is asked.
    assert status.endswith(b" 400 Bad Request") and wire.recorded.empty()


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


@dataclass(frozen=True)
class Grants:
    """An outer proxy that answers a CONNECT with `head` and tunnels it."""

    head: bytes = ESTABLISHED


@dataclass(frozen=True)
class Says:
    """An outer proxy that answers a CONNECT with `said` alone, granting nothing."""

    said: bytes


@dataclass(frozen=True)
class Closes:
    """An outer proxy that closes a CONNECT's connection unanswered."""


@dataclass(frozen=True)
class Holds:
    """An outer proxy that never answers a CONNECT."""


@dataclass
class Outer:
    """An outer HTTP proxy on loopback, its url carrying credentials, and the head of each request it was asked: a
    CONNECT's answered as `answer` says, and any other's by itself."""

    answer: Grants | Says | Closes | Holds
    url: str = ""
    asked: list[bytes] = field(default_factory=lambda: [])

    async def served(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        self.asked.append(head)
        if not head.startswith(b"CONNECT "):
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{"outer": true}')
            return
        match self.answer:
            case Grants(head=granted):
                host, port = head.split(b" ")[1].decode().rsplit(":", 1)
                far_reader, far_writer = await asyncio.open_connection(host, int(port))
                writer.write(granted)
                await asyncio.gather(_piped(reader, far_writer), _piped(far_reader, writer))
            case Says(said=said):
                writer.write(said)
            case Closes():
                writer.close()
            case Holds():
                await reader.read()


@contextlib.asynccontextmanager
async def outer_proxy(answer: Grants | Says | Closes | Holds = Grants()) -> AsyncGenerator[Outer]:
    outer = Outer(answer)
    connections: list[asyncio.StreamWriter] = []

    async def served(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connections.append(writer)
        await outer.served(reader, writer)

    server = await asyncio.start_server(served, "127.0.0.1", 0)
    outer.url = f"http://u:p@127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        yield outer
    finally:
        for writer in connections:
            writer.close()
        server.close()


async def _piped(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
    while chunk := await source.read(65536):
        sink.write(chunk)
        await sink.drain()
    sink.close()


@contextlib.asynccontextmanager
async def proxied(tmp_path: Path, environ: Mapping[str, str], clock: Clock = SYSTEM, upstream: str = "http://127.0.0.1:1") -> AsyncGenerator[tuple[Proxy, asyncio.Queue[Entry]]]:
    """A proxy started in `environ`, hands' own environment, and what it recorded."""
    recorded: asyncio.Queue[Entry] = asyncio.Queue()
    proxy = await serve_proxy(upstream, lambda _: None, lambda _: Send(span=root(SYSTEM)), clock, environ, tmp_path / "trusted.pem", recorded.put_nowait)
    try:
        yield proxy, recorded
    finally:
        await proxy.close()


@pytest.mark.parametrize("granted", [ESTABLISHED, b"HTTP/1.0 299 Fine\r\nVia: corp\r\n\r\n"], ids=["200", "any-2xx"])
async def test_a_tunnel_goes_through_the_outer_proxy_hands_environment_names(tmp_path: Path, elsewhere: tuple[str, ssl.SSLContext], granted: bytes) -> None:
    authority, trust = elsewhere
    async with outer_proxy(Grants(granted)) as outer, proxied(tmp_path, {"https_proxy": outer.url}) as (proxy, recorded):
        async with aiohttp.ClientSession() as client:
            async with client.get(f"https://{authority}/", proxy=proxy.url, ssl=trust) as response:
                assert (response.status, await response.text()) == (200, f"hello from {authority}")
        assert (await connect_event(recorded)).facts["through"] == outer.url.replace("u:p@", "")
    # Its own CONNECT, with the credentials the outer proxy's url carries.
    assert outer.asked == [f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\nProxy-Authorization: Basic dTpw\r\n\r\n".encode()]


async def test_a_host_no_proxy_names_is_reached_directly_past_the_outer_proxy(tmp_path: Path, elsewhere: tuple[str, ssl.SSLContext]) -> None:
    authority, trust = elsewhere
    async with outer_proxy() as outer, proxied(tmp_path, {"HTTPS_PROXY": outer.url, "NO_PROXY": "example.com,localhost"}) as (proxy, recorded):
        async with aiohttp.ClientSession() as client:
            async with client.get(f"https://{authority}/", proxy=proxy.url, ssl=trust) as response:
                assert (response.status, await response.text()) == (200, f"hello from {authority}")
        assert (await connect_event(recorded)).facts["through"] == "directly"
    assert outer.asked == []


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (Says(b"HTTP/1.1 403 Forbidden\r\n\r\n"), "answered its CONNECT 'HTTP/1.1 403 Forbidden'"),
        (Closes(), "closed the connection without answering its CONNECT, 0 bytes in"),
        (Says(b"HTTP/1.1 200 " + b"x" * 70000), "answered its CONNECT with a head longer than the"),
    ],
    ids=["refused", "unanswered", "too-long"],
)
async def test_an_outer_proxy_that_does_not_grant_a_tunnel_is_a_502_saying_how(tmp_path: Path, answer: Says | Closes, said: str) -> None:
    async with outer_proxy(answer) as outer, proxied(tmp_path, {"HTTPS_PROXY": outer.url}) as (proxy, recorded):
        reader, writer = await connected(proxy)
        try:
            writer.write(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
            status, body = await answered(reader)
        finally:
            writer.close()
        event = await connect_event(recorded)
    told = f"hands' proxy could not reach example.com:443: OuterRefused: the outer proxy at {outer.url.replace('u:p@', '')} {said}"
    assert status == b"HTTP/1.1 502 Bad Gateway" and body.decode().startswith(told)
    assert event.outcome == "failed" and event.error == body.decode()


async def test_an_outer_proxy_that_never_answers_is_given_up_on_at_the_deadline(tmp_path: Path) -> None:
    clock = VirtualClock()
    async with outer_proxy(Holds()) as outer, proxied(tmp_path, {"HTTPS_PROXY": outer.url}, clock) as (proxy, _):
        reader, writer = await connected(proxy)
        try:
            writer.write(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
            await clock.scheduled(clock.at + REACH_SECONDS)
            clock.advance(REACH_SECONDS)
            status, body = await answered(reader)
        finally:
            writer.close()
    assert (status, body) == (b"HTTP/1.1 502 Bad Gateway", b"hands' proxy could not reach example.com:443: TimeoutError: ")


async def test_a_request_to_the_api_goes_through_the_outer_proxy_hands_environment_names(tmp_path: Path) -> None:
    async with outer_proxy() as outer, proxied(tmp_path, {"http_proxy": outer.url}, upstream="http://api.example:8080") as (proxy, _):
        async with aiohttp.ClientSession() as client:
            async with client.post(proxy.url + "/v1/messages?beta=true", data=REQUEST, headers=HEADERS) as response:
                assert (response.status, await response.json()) == (200, {"outer": True})
    [head] = outer.asked
    assert head.startswith(b"POST http://api.example:8080/v1/messages?beta=true HTTP/1.1\r\n") and b"\r\nProxy-Authorization: Basic dTpw\r\n" in head

