"""How hands' proxy answers a CONNECT, the method a client asks its HTTPS proxy for a tunnel to a host with, and how the
proxy reaches every host it does not answer for.

A CONNECT is a request aiohttp's parser reads like any other, and its handler here takes the connection it came on over
from aiohttp's once it has answered it. To the API's host, the proxy ends the client's TLS itself, with its authority's
certificate for the host (`hands.sessions.authority`), and the HTTP inside it is served by the API's own server, every
request of it the API's. To any other host, the connection is a tunnel, its bytes piped both ways as they came. Each
CONNECT is one unit of work, its event emitted as its connection ends.

The world past the proxy is reached as hands' own environment says, read as Python's urllib reads it: through the outer
proxy its https_proxy or http_proxy names, for a host its no_proxy does not, or directly.
"""

import asyncio
import ssl
import urllib.request
from asyncio import sslproto
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import aiohttp
from aiohttp import web
from loguru import logger
from yarl import URL

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.authority import Authority, Uninterceptable
from hands.sessions.wide import annotate, begun, count, fail, unit
from hands.tasks import cancel_and_wait

if TYPE_CHECKING:
    from aiohttp.http_websocket import WebSocketReader

# The port of the TLS the proxy ends for the API's host: HTTPS's, which a client reaching https://<host> opens.
HTTPS_PORT = 443
# How long reaching a host a tunnel is asked to may take, through an outer proxy's CONNECT or directly.
REACH_SECONDS = 30.0
# What the outer proxy is told, and answers, a CONNECT in: HTTP/1.1, whose head ends on a blank line.
_HEAD_ENDS = b"\r\n\r\n"
_PIPED = 1 << 16
# urllib's reading of no_proxy against a host, which typeshed leaves undeclared: True for a host reached directly.
_bypassed = cast("Callable[[str, dict[str, str]], bool]", getattr(urllib.request, "proxy_bypass_environment"))


def host_port(host: str, port: int) -> str:
    """[LAW:one-source-of-truth] the one writing of a host and its port: an IPv6 address in brackets, as a URL has it."""
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


@dataclass(frozen=True)
class Outward:
    """How the proxy reaches a host it does not answer for, as hands' environment says in urllib's reading of it:
    `proxies`, each variable it read by the scheme it names, no_proxy's among them, and `through`, the outer proxy of
    each scheme a client of the proxy reaches."""

    proxies: Mapping[str, str]
    through: Mapping[str, URL]

    def onward(self, scheme: str, host: str, port: int) -> URL | None:
        """The outer proxy a host is reached through by `scheme`, or None, aiohttp's own word for directly: for a scheme
        no outer proxy is named for, or a host no_proxy names, which urllib reads with its port, as it reads a request's."""
        exempt = _bypassed(host_port(host, port), dict(self.proxies))
        return None if exempt else self.through.get(scheme)


def outward(world: Mapping[str, str]) -> Outward:
    """How the proxy reaches the world from `world`, hands' own environment; raises Uninterceptable for an outer proxy
    it cannot reach a host through."""
    proxies = _proxies(world)
    return Outward(proxies, {scheme: _outer(scheme, named) for scheme, named in proxies.items() if scheme in ("http", "https")})


def _proxies(world: Mapping[str, str]) -> dict[str, str]:
    """urllib.request.getproxies_environment's reading, of `world` in place of os.environ: each <scheme>_proxy set, by
    its scheme, its lower-case spelling over any other, and one set empty in lower case naming none."""
    # [LAW:one-source-of-truth] the standard library's own two passes, which it runs only on os.environ: any spelling,
    # then the lower case alone, which an empty value takes away.
    spelled = {name.lower().removesuffix("_proxy"): value for name, value in world.items() if value and name.lower().endswith("_proxy")}
    lower = {name.lower().removesuffix("_proxy"): value for name, value in world.items() if name.endswith("_proxy")}
    return {scheme: value for scheme, value in (spelled | lower).items() if value}


def _outer(scheme: str, named: str) -> URL:
    """The outer proxy hands' environment names for `scheme`, parsed; raises Uninterceptable for one aiohttp cannot
    reach a host through: any but http://, and one without a host or with port 0."""
    proxy = URL(named)
    if proxy.scheme != "http" or not proxy.host or proxy.port == 0:
        said = f"its scheme {proxy.scheme!r}, its host {proxy.host!r} and its port {proxy.port}"
        raise Uninterceptable(f"{scheme}_proxy or {scheme.upper()}_PROXY names an outer proxy of {said}: hands' proxy reaches one at an http:// URL with a host and a port that is not 0")
    return proxy


class OuterRefused(Exception):
    """The outer proxy hands' environment names did not open a tunnel to a host, saying why."""


class Connects:
    """Every CONNECT the proxy answers, from its request to its connection's end; closing ends each and waits for it.

    [LAW:single-enforcer] the one place a CONNECT is answered: to the API's host by `inside`, the API's own server, and
    to every other host by a tunnel."""

    def __init__(self, authority: Authority, outward: Outward, inside: web.Server, clock: Clock, record: Record) -> None:
        self._authority = authority
        self._outward = outward
        self._inside = inside
        self._clock = clock
        self._record = record
        # [LAW:no-shared-mutable-globals] each CONNECT answering, added and taken away only here.
        self._open: set[asyncio.Task[web.StreamResponse]] = set()

    async def close(self) -> None:
        """End every CONNECT's connection, and return once each has ended."""
        await asyncio.gather(*(cancel_and_wait(answering) for answering in tuple(self._open)))

    async def answer(self, request: web.BaseRequest) -> web.StreamResponse:
        """The CONNECT `request`, answered, and served until its connection ends."""
        answering = asyncio.ensure_future(self._answered(request))
        self._open.add(answering)
        try:
            return await answering
        finally:
            self._open.discard(answering)

    async def _answered(self, request: web.BaseRequest) -> web.StreamResponse:
        # [LAW:nothing-unseen] one event for each CONNECT, held open as long as its connection is: where it asked for,
        # whether the proxy ended its TLS or tunnelled it, and how it ended.
        with unit("proxy.connect", self._record, begun(self._clock), ("tunnelled_to_host", "tunnelled_to_client")):
            # aiohttp's reading of its authority-form target (RFC 9110 §9.3.6): a host and its port, which is not 0, as
            # its parser refuses any other before here.
            match request.url.host, request.url.port:
                case (str() as host, int() as port) if port > 0:
                    pass
                case _:
                    refused = f"a CONNECT names a host and a port that is not 0, not {request.raw_path!r}"
                    fail(refused)
                    return web.Response(status=400, text=refused)
            intercepted = (host, port) == (self._authority.host, HTTPS_PORT)
            annotate(host=host, port=port, intercepted=intercepted)
            return await (self._intercepted(request) if intercepted else self._tunnelled(request, host, port))

    async def _intercepted(self, request: web.BaseRequest) -> web.StreamResponse:
        """The client's TLS to the API's host ended here, and the HTTP inside it served by the API's server."""
        established = await _established(request)
        loop = asyncio.get_running_loop()
        handshake = loop.create_future()
        inside = _Inside(self._inside())
        # [LAW:no-ambient-temporal-coupling] the TLS asyncio's start_tls would make, made here so that what the client
        # sent past its CONNECT goes into it first; and the inside's handler told its connection was made before any
        # byte of it is read, as SSLProtocol makes it only once its handshake is done.
        tls = sslproto.SSLProtocol(loop, inside, self._authority.answer, handshake, server_side=True)
        with _taken(request, tls) as (_, tail):
            for at in range(0, len(tail), _PIPED):
                part = tail[at : at + _PIPED]
                tls.get_buffer(len(part))[: len(part)] = part
                tls.buffer_updated(len(part))
            try:
                await _made(handshake, inside)
            except TlsRefused as error:
                # [LAW:no-silent-failure] a client refusing the proxy's TLS leaves its every request to the API unmade,
                # and nothing answers it that: said here, as fritter says its HandshakeRefused.
                said = f"a client of hands' proxy did not finish TLS for {self._authority.host}, so none of its requests reach the API: {error}"
                fail(said)
                logger.error(said)
                return established
            annotate(tls="made")
            await inside.ended
        return established

    async def _tunnelled(self, request: web.BaseRequest, host: str, port: int) -> web.StreamResponse:
        """Bytes piped both ways between the client and `host`, as each side sent them, until both have ended."""
        # A CONNECT's onward is HTTPS's: a client asks for a tunnel to reach a host by TLS.
        onward = self._outward.onward("https", host, port)
        annotate(through="directly" if onward is None else str(onward.origin()))
        try:
            far_reader, far_writer = await _reached(self._clock, host, port, onward)
        except (OuterRefused, OSError, TimeoutError) as error:
            unreached = f"hands' proxy could not reach {host_port(host, port)}: {type(error).__name__}: {error}"
            fail(unreached)
            logger.warning(unreached)
            return web.Response(status=502, text=unreached)
        try:
            established = await _established(request)
            reader = asyncio.StreamReader()
            streamed = asyncio.StreamReaderProtocol(reader)
            with _taken(request, streamed) as (transport, tail):
                client = asyncio.StreamWriter(transport, streamed, reader, asyncio.get_running_loop())
                far_writer.write(tail)
                count(tunnelled_to_host=len(tail))
                await asyncio.gather(_piped(reader, far_writer, lambda n: count(tunnelled_to_host=n)), _piped(far_reader, client, lambda n: count(tunnelled_to_client=n)))
            return established
        finally:
            far_writer.close()


class TlsRefused(Exception):
    """A client that did not finish the TLS the proxy ends, saying how."""


async def _made(handshake: asyncio.Future[None], inside: "_Inside") -> None:
    """Returns once the client's TLS is made; raises TlsRefused when it never is: refused, or its connection gone first."""
    try:
        await handshake
    except (ssl.SSLError, OSError) as error:
        raise TlsRefused(f"{type(error).__name__}: {error}") from error
    # SSLProtocol's handshake ends without an error on a connection lost before it was made.
    if not inside.made:
        raise TlsRefused("its connection ended before its TLS was made")


class _Inside(asyncio.Protocol):
    """The HTTP inside the TLS the proxy ends, served by `served`, a handler of the API's server: `made` once the TLS is,
    and `ended` once its connection is gone."""

    def __init__(self, served: asyncio.Protocol) -> None:
        self._served = served
        self.made = False
        self.ended: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.made = True
        self._served.connection_made(transport)

    def data_received(self, data: bytes) -> None:
        self._served.data_received(data)

    def eof_received(self) -> bool | None:
        return self._served.eof_received()

    def pause_writing(self) -> None:
        self._served.pause_writing()

    def resume_writing(self) -> None:
        self._served.resume_writing()

    def connection_lost(self, exc: Exception | None) -> None:
        self._served.connection_lost(exc)
        self.ended.set_result(None)


async def _established(request: web.BaseRequest) -> web.StreamResponse:
    """The CONNECT `request` granted: aiohttp's own 200, which it writes with no body and no framing (RFC 9110 §9.3.6)."""
    established = web.StreamResponse(status=200, reason="Connection Established")
    await established.prepare(request)
    await established.write_eof()
    return established


class _Tail:
    """What a client sent past its CONNECT's head, as aiohttp's handler gives an upgraded connection's bytes to the
    parser it is handed."""

    def __init__(self) -> None:
        self.data = b""

    def feed_data(self, data: bytes) -> tuple[bool, bytes]:
        self.data += data
        return False, b""

    def feed_eof(self) -> None:
        pass


@contextmanager
def _taken(request: web.BaseRequest, protocol: asyncio.BaseProtocol) -> Generator[tuple[asyncio.Transport, bytes]]:
    """The connection `request` came on, taken over from aiohttp's handler of it by `protocol`, and the bytes its
    client sent past the request's head, which that handler read: once the body is done with it, the connection is
    closed, and the handler told it ended, as the transport now tells `protocol` alone."""
    transport = request.transport
    if transport is None:
        raise ConnectionResetError("the client hung up before its CONNECT was answered")
    handler = request.protocol
    tail = _Tail()
    # aiohttp reads a CONNECT as a request that upgrades its connection, and keeps what came past its head for the
    # parser of what it upgrades to, which it types as its WebSocketReader and asks only to be fed.
    handler.set_parser(cast("WebSocketReader", tail))
    transport.set_protocol(protocol)
    protocol.connection_made(transport)
    # Whatever aiohttp's handler left the reading at, it is `protocol`'s from here.
    transport.resume_reading()
    try:
        yield transport, tail.data
    finally:
        transport.close()
        handler.connection_lost(None)


async def _reached(clock: Clock, host: str, port: int, onward: URL | None) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A connection to `host`:`port`, through `onward`, an outer proxy, or directly, made within REACH_SECONDS."""
    async with clock.timeout(REACH_SECONDS):
        match onward:
            case None:
                return await asyncio.open_connection(host, port)
            case URL() as outer:
                return await _through(outer, host, port)


async def _through(outer: URL, host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A tunnel to `host`:`port` the outer proxy at `outer` opened at its CONNECT, with the Proxy-Authorization its
    URL's credentials make; closed on any failure, and raising OuterRefused when the proxy did not grant it."""
    reader, writer = await asyncio.open_connection(outer.host, outer.port)
    told = outer.origin()
    try:
        authority = host_port(host, port)
        credentials = aiohttp.BasicAuth.from_url(outer)
        authorization = "" if credentials is None else f"Proxy-Authorization: {credentials.encode()}\r\n"
        writer.write(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n{authorization}\r\n".encode())
        try:
            answer = await reader.readuntil(_HEAD_ENDS)
        except asyncio.IncompleteReadError as error:
            raise OuterRefused(f"the outer proxy at {told} closed the connection without answering its CONNECT, {len(error.partial)} bytes in") from error
        except asyncio.LimitOverrunError as error:
            raise OuterRefused(f"the outer proxy at {told} answered its CONNECT with a head longer than the {error.consumed} bytes read") from error
        status = answer.split(b"\r\n", 1)[0]
        # Any 2xx grants the tunnel (RFC 9110 §9.3.6).
        match status.split(b" ", 2):
            case [version, code, *_] if version.startswith(b"HTTP/") and code.isdigit() and 200 <= int(code) < 300:
                return reader, writer
            case _:
                raise OuterRefused(f"the outer proxy at {told} answered its CONNECT {status.decode(errors='replace')!r}")
    except BaseException:
        writer.close()
        raise


async def _piped(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, moved: Callable[[int], None]) -> None:
    """What `reader` reads written on as it came, each part's size told to `moved`, until it ends; then the other side's
    writing ends too."""
    try:
        while chunk := await reader.read(_PIPED):
            moved(len(chunk))
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except ConnectionError:
        # The other side hung up: the tunnel ends, which closing both sides says to each.
        writer.close()
