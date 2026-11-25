"""How hands' proxy takes each connection made to it: HTTP for its app, or a CONNECT, the method a client asks its HTTPS
proxy for a tunnel to a host with.

A CONNECT to the API's host is answered by the proxy itself: it ends the client's TLS there, with its authority's
certificate for the host (`hands.sessions.authority`), and serves the HTTP inside it with the same app, the same handler,
as a request made to its own origin. A CONNECT to any other host is a tunnel, its bytes piped both ways as they came, to
the host directly or through the outer proxy hands' own environment names for it. One listening socket, one app.
"""

import asyncio
import base64
import urllib.parse
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass

from aiohttp import web
from loguru import logger

from hands.sessions.authority import Authority, Uninterceptable

# The one answer a CONNECT that is granted is given, before the tunnel carries anything.
ESTABLISHED = b"HTTP/1.1 200 Connection Established\r\n\r\n"
_CONNECT = b"CONNECT "
_HEAD_ENDS = b"\r\n\r\n"
# The longest CONNECT head read before it is refused: a request line and a few headers.
HEAD_LIMIT = 16 * 1024
# The port of the TLS the proxy ends for the API's host: HTTPS's, which a client reaching https://<host> opens.
HTTPS_PORT = 443
_PIPED = 1 << 16


@dataclass(frozen=True)
class Dialed:
    """A host the proxy reaches directly."""


@dataclass(frozen=True)
class Outer:
    """An outer proxy hands' environment names, which the proxy reaches a host through: where it listens, and the
    Proxy-Authorization its URL's credentials make, as a header the request to it carries, or none."""

    host: str
    port: int
    authorization: tuple[tuple[str, str], ...]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


Onward = Dialed | Outer


@dataclass(frozen=True)
class Outward:
    """How the proxy reaches a host it does not answer for, as hands' own environment says: through the proxy its
    HTTPS_PROXY or HTTP_PROXY names for the scheme, or directly, for a host its NO_PROXY names or a scheme with none."""

    https: Onward
    http: Onward
    exempt: str

    def onward(self, host: str, secure: bool) -> Onward:
        if exempted(host, self.exempt):
            return Dialed()
        return self.https if secure else self.http


def exempted(host: str, exempt: str) -> bool:
    """Whether NO_PROXY's `exempt` names `host`, as curl and Node's undici read it: `*` names every host, and an entry
    names its host and every host under it, written with a leading dot or `*.` or not. An entry's port is not weighed:
    it names its host on every port."""
    host = host.lower().strip("[]")
    return any(entry == "*" or (named := _host_of(entry)) != "" and (host == named or host.endswith(f".{named}")) for entry in (each.strip().lower() for each in exempt.split(",")))


def _host_of(entry: str) -> str:
    """The host a NO_PROXY entry names, without its port or the dot or `*.` it may open with."""
    named = entry.removeprefix("*").removeprefix(".")
    if named.startswith("["):
        return named[1:].split("]", 1)[0]
    # An IPv6 address unbracketed has colons of its own and no port.
    return named.rsplit(":", 1)[0] if named.count(":") == 1 else named


def outward(environ: Mapping[str, str]) -> Outward:
    """How the proxy reaches the world from `environ`; raises Uninterceptable for an outer proxy it cannot reach a host
    through."""
    return Outward(
        _outer(environ, "HTTPS_PROXY"),
        _outer(environ, "HTTP_PROXY"),
        environ.get("NO_PROXY") or environ.get("no_proxy") or "",
    )


def _outer(environ: Mapping[str, str], name: str) -> Onward:
    # Each spelling, the upper first, as curl and Node's undici read them.
    named = environ.get(name) or environ.get(name.lower()) or ""
    if not named:
        return Dialed()
    # A proxy named without a scheme is an HTTP proxy, as every client of these variables reads it.
    parsed = urllib.parse.urlsplit(named if "://" in named else f"http://{named}")
    if parsed.scheme != "http" or parsed.hostname is None:
        raise Uninterceptable(f"{name} names {named}: hands' proxy reaches an outer proxy at an http:// URL with a host alone")
    try:
        port = parsed.port or 80
    except ValueError as error:
        raise Uninterceptable(f"{name} names {named}, whose port is no port: {error}") from error
    credentials = () if parsed.username is None else ((
        "Proxy-Authorization",
        "Basic " + base64.b64encode(f"{urllib.parse.unquote(parsed.username)}:{urllib.parse.unquote(parsed.password or '')}".encode()).decode(),
    ),)
    return Outer(parsed.hostname, port, credentials)


class Unreached(Exception):
    """A host a tunnel could not be opened to, saying why."""


class Doors:
    """Every connection made to the proxy, from the moment it opens: what it carries is told apart here, each tunnel
    lives here, and closing the doors ends them all.

    [LAW:single-enforcer] the one place a connection to the proxy is told apart, as fritter's tap's door is the one place
    one to fritter is (fritter/intercept)."""

    def __init__(self, authority: Authority, outward: Outward, app: web.Server) -> None:
        self._authority = authority
        self._outward = outward
        self._app = app
        # [LAW:no-shared-mutable-globals] what is open at this door, added and taken away only here.
        self._open: set[asyncio.Transport] = set()
        self._working: set[asyncio.Task[None]] = set()

    def __call__(self) -> asyncio.Protocol:
        return _Door(self)

    def close(self) -> None:
        """End every connection not handed to the app, which its own server ends, and every tunnel."""
        for task in self._working:
            task.cancel()
        for transport in self._open:
            transport.close()

    def opened(self, transport: asyncio.Transport) -> None:
        self._open.add(transport)

    def closed(self, transport: asyncio.Transport) -> None:
        self._open.discard(transport)

    def served(self, transport: asyncio.Transport, head: bytes) -> None:
        """`transport` handed to the app, as a connection made to it from its first byte."""
        self._open.discard(transport)
        handler = self._app()
        transport.set_protocol(handler)
        handler.connection_made(transport)
        handler.data_received(head)

    def connect(self, transport: asyncio.Transport, head: bytes, early: bytes) -> None:
        """The CONNECT whose head is `head`, its client having sent `early` past it already."""
        transport.pause_reading()
        self._work(self._connected(transport, head, early))

    def _work(self, work: Coroutine[object, object, None]) -> None:
        task = asyncio.create_task(work)
        self._working.add(task)
        task.add_done_callback(self._worked)

    def _worked(self, task: asyncio.Task[None]) -> None:
        self._working.discard(task)
        # [LAW:no-silent-failure] what a connection's work raised that it did not expect is said, with its trace.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.opt(exception=error).error("hands' proxy failed on a CONNECT")

    async def _connected(self, transport: asyncio.Transport, head: bytes, early: bytes) -> None:
        match _target(head):
            case str() as refused:
                _refuse(transport, 400, refused)
            case (host, port) if (host, port) == (self._authority.host, HTTPS_PORT):
                await self._intercepted(transport, early)
            case (host, port):
                await self._tunnelled(transport, host, port, early)

    async def _intercepted(self, transport: asyncio.Transport, early: bytes) -> None:
        """The client's TLS to the API's host ended here, and the HTTP inside it served by the app."""
        if early:
            # TLS can be ended only from its first byte, and these were read before it began.
            _refuse(transport, 400, f"hands' proxy ends TLS for {self._authority.host} only once it has answered the CONNECT, and {len(early)} bytes came before")
            return
        transport.write(ESTABLISHED)
        handler = self._app()
        loop = asyncio.get_running_loop()
        try:
            tls = await loop.start_tls(transport, handler, self._authority.answer, server_side=True)
        except (OSError, EOFError) as error:
            # [LAW:no-silent-failure] a client refusing the proxy's TLS leaves its every request to the API unmade, and
            # nothing answers it that: said here, as fritter says its HandshakeRefused.
            logger.error(f"a client of hands' proxy did not finish TLS for {self._authority.host}, so none of its requests reach the API: {type(error).__name__}: {error}")
            transport.close()
            return
        finally:
            self._open.discard(transport)
        # start_tls's own type admits none, which it returns only for a loop that has no TLS.
        assert tls is not None, "the loop gave hands' proxy no TLS transport"
        # [LAW:no-ambient-temporal-coupling] started by start_tls without its connection_made, which is the app's from here.
        handler.connection_made(tls)

    async def _tunnelled(self, transport: asyncio.Transport, host: str, port: int, early: bytes) -> None:
        """Bytes piped both ways between the client and `host`, as each side sent them, until either ends."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        streamed = asyncio.StreamReaderProtocol(reader)
        transport.set_protocol(streamed)
        streamed.connection_made(transport)
        client = asyncio.StreamWriter(transport, streamed, reader, loop)
        try:
            try:
                far_reader, far_writer = await self._reach(host, port)
            except (Unreached, OSError) as error:
                logger.warning(f"hands' proxy could not open a tunnel to {host}:{port}: {error}")
                _refuse(transport, 502, f"hands' proxy could not reach {host}:{port}: {error}")
                return
            client.write(ESTABLISHED)
            far_writer.write(early)
            transport.resume_reading()
            try:
                await asyncio.gather(_piped(reader, far_writer), _piped(far_reader, client))
            finally:
                far_writer.close()
        finally:
            client.close()
            self._open.discard(transport)

    async def _reach(self, host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """A connection to `host`:`port`, directly or through the outer proxy hands' environment names for it."""
        match self._outward.onward(host, secure=True):
            case Dialed():
                return await asyncio.open_connection(host, port)
            case Outer() as outer:
                reader, writer = await asyncio.open_connection(outer.host, outer.port)
                authority = _authority(host, port)
                lines = [f"CONNECT {authority} HTTP/1.1", f"Host: {authority}", *(f"{name}: {value}" for name, value in outer.authorization)]
                writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
                try:
                    answer = await reader.readuntil(_HEAD_ENDS)
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as error:
                    writer.close()
                    raise Unreached(f"the outer proxy at {outer.url} gave no answer to its CONNECT: {error}") from error
                status = answer.split(b"\r\n", 1)[0]
                if status.split(b" ", 2)[1:2] != [b"200"]:
                    writer.close()
                    raise Unreached(f"the outer proxy at {outer.url} answered its CONNECT {status.decode(errors='replace')}")
                return reader, writer


class _Door(asyncio.Protocol):
    """One connection to the proxy, until what it carries is known from its first bytes: a CONNECT, whose head is read
    here, or HTTP for the app, handed to it whole."""

    def __init__(self, doors: Doors) -> None:
        self._doors = doors
        self._head = b""
        self._transport: asyncio.Transport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.Transport)
        self._transport = transport
        self._doors.opened(transport)

    def data_received(self, data: bytes) -> None:
        transport = self._transport
        assert transport is not None
        self._head += data
        if not _CONNECT.startswith(self._head[: len(_CONNECT)]):
            self._doors.served(transport, self._head)
        elif (ends := self._head.find(_HEAD_ENDS)) >= 0:
            self._doors.connect(transport, self._head[: ends + len(_HEAD_ENDS)], self._head[ends + len(_HEAD_ENDS):])
        elif len(self._head) > HEAD_LIMIT:
            _refuse(transport, 431, f"hands' proxy reads a CONNECT's head of {HEAD_LIMIT} bytes at most")

    def connection_lost(self, exc: Exception | None) -> None:
        if self._transport is not None:
            self._doors.closed(self._transport)


def _target(head: bytes) -> tuple[str, int] | str:
    """The host and port a CONNECT's head asks a tunnel to, or why it names none."""
    line = head.split(b"\r\n", 1)[0].decode("latin-1")
    match line.split(" "):
        case ["CONNECT", target, version] if version.startswith("HTTP/1."):
            pass
        case _:
            return f"hands' proxy reads a CONNECT as CONNECT <host>:<port> HTTP/1.1, not {line!r}"
    parsed = urllib.parse.urlsplit(f"//{target}")
    try:
        port = parsed.port
    except ValueError:
        port = None
    if parsed.hostname is None or port is None:
        return f"a CONNECT names a host and its port, not {target!r}"
    return parsed.hostname, port


def _authority(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _refuse(transport: asyncio.Transport, status: int, why: str) -> None:
    """The client told `why` its connection ends, as an HTTP answer it reads, and the connection ended."""
    said = why.encode()
    reason = {400: "Bad Request", 431: "Request Header Fields Too Large", 502: "Bad Gateway"}[status]
    transport.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(said)}\r\nConnection: close\r\n\r\n".encode() + said)
    transport.close()


async def _piped(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """What `reader` reads written on as it came, until it ends; then the other side's writing ends too."""
    try:
        while chunk := await reader.read(_PIPED):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except ConnectionError:
        # The other side hung up: the tunnel ends, which closing both sides says to each.
        writer.close()

