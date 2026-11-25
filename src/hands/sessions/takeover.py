"""A connection taken over from aiohttp's server once a CONNECT on it is answered: a tunnel's, as a stream, or the TLS
hands' proxy ends, as that TLS's server side, the HTTP inside it served by a protocol of the API's server.

The order every taking keeps:

1. The CONNECT is answered, with aiohttp's own 200, before its connection is taken.
2. aiohttp's handler of the connection is handed a parser that keeps what it read past the CONNECT's head, as it hands
   an upgraded connection's bytes to the parser it is given, and the transport is given the protocol that takes it
   over, told it is made, and its reading resumed.
3. Those bytes reach the new protocol first: handed it at once, with nothing awaited between, so no byte the transport
   reads after can come before them.
4. Once the connection is done with, the transport is closed, aiohttp's handler is told to close the connection once
   the CONNECT's handler returns, and told it lost the connection only once its own task has ended. So the CONNECT's
   handler ends by what it returns, never by the cancellation aiohttp sends a handler whose connection is lost.

[LAW:locality-or-seam] exception: no public API carries the bytes a server already read past a request's head into
another protocol. aiohttp has none that hands its connection over, so set_parser is handed a parser that is no
WebSocketReader; asyncio's start_tls starts TLS only on bytes still to be read, so asyncio.sslproto's SSLProtocol, a
private class, is built here and fed them through its buffered-protocol methods. tests/test_proxy_connect.py runs each
on every path, the bytes past a CONNECT's head and past a handshake's last flight among them, so an upgrade of Python or
aiohttp that breaks one fails there.
"""

import asyncio
import ssl
from asyncio import sslproto
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from aiohttp import web

if TYPE_CHECKING:
    from aiohttp.http_websocket import WebSocketReader

# How much of what came past a CONNECT's head is handed to the TLS at once: a buffered protocol's buffer may be smaller.
_PART = 1 << 16


class TlsRefused(Exception):
    """A client that did not finish the TLS the proxy ends, saying how."""


class Tls:
    """The TLS the proxy ends on a connection it took over: `made` once its handshake is, and `ended` once the
    connection inside it is gone."""

    def __init__(self, handshake: asyncio.Future[None], inside: "_Inside") -> None:
        self._handshake = handshake
        self._inside = inside

    async def made(self) -> None:
        """Returns once the client's TLS is made; raises TlsRefused when it never is: refused, or its connection gone first."""
        try:
            await self._handshake
        except (ssl.SSLError, OSError) as error:
            raise TlsRefused(f"{type(error).__name__}: {error}") from error
        # SSLProtocol's handshake ends without an error on a connection lost before it was made.
        if not self._inside.made:
            raise TlsRefused("its connection ended before its TLS was made")

    @property
    def ended(self) -> asyncio.Future[None]:
        return self._inside.ended


@contextmanager
def tls_ended(request: web.BaseRequest, served: asyncio.Protocol, context: ssl.SSLContext) -> Generator[Tls]:
    """The connection the answered CONNECT `request` came on, taken over as the server side of TLS made with `context`,
    the HTTP inside it served by `served`; closed once the body is done with it."""
    loop = asyncio.get_running_loop()
    handshake = loop.create_future()
    inside = _Inside(served)
    # The TLS start_tls would make, made here so that what the client sent past its CONNECT goes into it first; the
    # inside told its connection was made before any byte of it is read, as SSLProtocol makes it only once its handshake
    # is done.
    tls = sslproto.SSLProtocol(loop, inside, context, handshake, server_side=True)
    with _taken(request, tls) as (_, tail):
        for at in range(0, len(tail), _PART):
            part = tail[at : at + _PART]
            tls.get_buffer(len(part))[: len(part)] = part
            tls.buffer_updated(len(part))
        yield Tls(handshake, inside)


@contextmanager
def streamed(request: web.BaseRequest) -> Generator[tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]]:
    """The connection the answered CONNECT `request` came on, taken over as a stream, and what its client sent past the
    CONNECT's head, which is to go on before anything read from the stream; closed once the body is done with it."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    with _taken(request, protocol) as (transport, tail):
        yield reader, asyncio.StreamWriter(transport, protocol, reader, asyncio.get_running_loop()), tail


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
    """The connection `request` came on, taken over by `protocol`, and the bytes its client sent past the request's
    head, in the order the module says."""
    transport = request.transport
    if transport is None:
        raise ConnectionResetError("the client hung up before its CONNECT was answered")
    handler = request.protocol
    # aiohttp's own task for the connection, which runs the CONNECT's handler: read now, as aiohttp lets go of it once
    # the handler returns.
    handling = request.task
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
        # The connection's end as aiohttp has it: closed once the CONNECT's handler returns (RequestHandler.close), then
        # lost, which cancels a handler still running, once aiohttp's task for it has ended.
        handler.close()
        handling.add_done_callback(lambda _: handler.connection_lost(None))
