"""How hands' proxy answers a CONNECT, the method a client asks its HTTPS proxy for a tunnel to a host with, and how the
proxy reaches every host it does not answer for.

A CONNECT is a request aiohttp's parser reads like any other, answered, as every request is, only on this run's
credential (`admission`), and its handler here takes the connection it came on over from aiohttp's once it has answered
it (`hands.sessions.takeover`). To the API's host, the proxy ends the client's TLS itself, with its authority's
certificate for the host (`hands.sessions.authority`), and the HTTP inside it is served by the API's own server, every
request of it the API's. To any other host, the connection is a tunnel, its bytes piped both ways as they came. Each
CONNECT is one unit of work, its event emitted as its connection ends.

The world past the proxy is reached as hands' own environment says, read as Python's urllib reads it: through the outer
proxy its https_proxy or http_proxy names, for a host its no_proxy does not, or directly.
"""

import asyncio
import hmac
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import aiohttp
from aiohttp import web
from loguru import logger
from yarl import URL

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.authority import Authority, Uninterceptable
from hands.sessions.takeover import TlsRefused, streamed, tls_ended
from hands.sessions.wide import annotate, begun, count, fail, unit
from hands.tasks import cancel_and_wait

# How long reaching a host a tunnel is asked to may take, through an outer proxy's CONNECT or directly.
REACH_SECONDS = 30.0
# How long a client may take to finish its TLS to the API's host: one round trip on loopback, given a loaded machine.
HANDSHAKE_SECONDS = 10.0
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
    reach a host through: one that is no URL, any but http://, and one without a host or with port 0. One named without
    a scheme is http://, as urllib, curl and Claude Code read it."""
    try:
        proxy = URL(named if "://" in named else f"http://{named}")
        port = proxy.port
    except ValueError as error:
        # The URL itself is left out of what is said: it may carry the outer proxy's credentials.
        raise Uninterceptable(f"{scheme}_proxy or {scheme.upper()}_PROXY names an outer proxy by no URL hands' proxy can read ({error}): name it by an http:// URL with a host and a port") from error
    if proxy.scheme != "http" or not proxy.host or port == 0:
        said = f"its scheme {proxy.scheme!r}, its host {proxy.host!r} and its port {port}"
        raise Uninterceptable(f"{scheme}_proxy or {scheme.upper()}_PROXY names an outer proxy of {said}: hands' proxy reaches one at an http:// URL with a host and a port that is not 0")
    return proxy


class OuterRefused(Exception):
    """The outer proxy hands' environment names did not open a tunnel to a host, saying why."""


@dataclass(frozen=True)
class Admitted:
    """[LAW:parse-dont-validate] a request that carried this run's credential as its Proxy-Authorization: what every
    request on the proxy's own socket is answered on, so no other process on the machine reaches past the proxy with
    hands' way out, the API on hands' outer proxy among it."""


@dataclass(frozen=True)
class Unadmitted:
    """A request that came to the proxy without this run's credential, and why; never the credential."""

    why: str


def admission(told: str | None, credential: str) -> Admitted | Unadmitted:
    """The Proxy-Authorization a request was `told` with, an auth-scheme and its token68 (RFC 9110 §11.6.2), against
    `credential`, the run's, as `proxy_authorization` writes it."""
    match told:
        case None:
            return Unadmitted("it sent no Proxy-Authorization")
        case str():
            scheme, _, presented = told.partition(" ")
    # What a request presents is never said: it may be another of the user's credentials, sent to the wrong place.
    if scheme.lower() != "basic":
        return Unadmitted("its Proxy-Authorization is not Basic credentials")
    # Compared in a time that says nothing of how much of it matched. A header aiohttp read as other than UTF-8 holds its
    # bytes as surrogates, which are encoded back as they came: never this run's credential, and refused as any other.
    same = hmac.compare_digest(presented.strip().encode(errors="surrogateescape"), credential.partition(" ")[2].encode())
    return Admitted() if same else Unadmitted("its Proxy-Authorization is not this run's credential")


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

    async def answer(self, request: web.BaseRequest, _: Admitted) -> web.StreamResponse:
        """The CONNECT `request`, answered, and served until its connection ends."""
        # [LAW:nothing-unseen] one event for each CONNECT, held open as long as its connection is: where it asked for,
        # whether the proxy ended its TLS or tunnelled it, and how it ended, which is how its answering did.
        with unit("proxy.connect", self._record, begun(self._clock), ("tunnelled_to_host", "tunnelled_to_client")):
            answering = asyncio.ensure_future(self._answered(request))
            self._open.add(answering)
            try:
                return await answering
            finally:
                self._open.discard(answering)

    async def _answered(self, request: web.BaseRequest) -> web.StreamResponse:
        # aiohttp's reading of its authority-form target (RFC 9110 §9.3.6): a host and its port, which is not 0, as its
        # parser refuses any other before here.
        match request.url.host, request.url.port:
            case (str() as host, int() as port) if port > 0:
                pass
            case _:
                refused = f"a CONNECT names a host and a port that is not 0, not {request.raw_path!r}"
                fail(refused)
                return web.Response(status=400, text=refused)
        intercepted = self._authority.intercepts(host, port)
        annotate(host=host, port=port, intercepted=intercepted)
        return await (self._intercepted(request) if intercepted else self._tunnelled(request, host, port))

    async def _intercepted(self, request: web.BaseRequest) -> web.StreamResponse:
        """The client's TLS to the API's host ended here, and the HTTP inside it served by the API's server."""
        established = await _established(request)
        with tls_ended(request, self._inside(), self._authority.answer) as tls:
            try:
                # [LAW:no-ambient-temporal-coupling] a client that never finishes its handshake holds its connection no
                # longer than the deadline.
                async with self._clock.timeout(HANDSHAKE_SECONDS):
                    await tls.made()
            except (TlsRefused, TimeoutError) as error:
                # [LAW:no-silent-failure] a client refusing the proxy's TLS leaves its every request to the API unmade,
                # and nothing answers it that: said here, as fritter says its HandshakeRefused.
                how = f"its handshake took longer than {HANDSHAKE_SECONDS:g} seconds" if isinstance(error, TimeoutError) else str(error)
                said = f"a client of hands' proxy did not finish TLS for {self._authority.host}, so none of its requests reach the API: {how}"
                fail(said)
                logger.error(said)
                return established
            annotate(tls="made")
            # Shielded: a handler cancelled as the proxy shuts down leaves `ended` for its connection to set.
            await asyncio.shield(tls.ended)
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
            with streamed(request) as (reader, client, tail):
                far_writer.write(tail)
                count(tunnelled_to_host=len(tail))
                await asyncio.gather(
                    _piped(_Side("client", reader), _Side("host", far_writer), lambda n: count(tunnelled_to_host=n)),
                    _piped(_Side("host", far_reader), _Side("client", client), lambda n: count(tunnelled_to_client=n)),
                )
            return established
        finally:
            far_writer.close()


async def _established(request: web.BaseRequest) -> web.StreamResponse:
    """The CONNECT `request` granted: aiohttp's own 200, which it writes with no body and no framing (RFC 9110 §9.3.6)."""
    established = web.StreamResponse(status=200, reason="Connection Established")
    await established.prepare(request)
    await established.write_eof()
    return established


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


@dataclass(frozen=True)
class _Side[T]:
    """One end of a tunnel, named as its event names it: the client, or the host it asked for."""

    name: Literal["client", "host"]
    stream: T


class _Reset(Exception):
    """The connection of one side of a tunnel was reset, or failed as a socket fails."""

    def __init__(self, side: str, error: OSError) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.side = side


async def _piped(source: _Side[asyncio.StreamReader], sink: _Side[asyncio.StreamWriter], moved: Callable[[int], None]) -> None:
    """What `source` reads written on to `sink` as it came, each part's size told to `moved`, until it ends; then the
    sink's writing ends too. A side that resets ends the tunnel, and the CONNECT's event says which and how."""
    try:
        while chunk := await _of(source.name, source.stream.read(_PIPED)):
            moved(len(chunk))
            sink.stream.write(chunk)
            await _of(sink.name, sink.stream.drain())
        if sink.stream.can_write_eof():
            sink.stream.write_eof()
    except _Reset as reset:
        # [LAW:nothing-unseen] the tunnel ended by a reset, not a close: which side, and what it said. Closing both sides
        # says the end to each.
        annotate(reset_by=reset.side, reset=str(reset))
        sink.stream.close()


async def _of[T](side: str, awaited: Awaitable[T]) -> T:
    """What `awaited` gives, its connection's reset or failure told as `side`'s."""
    try:
        return await awaited
    except OSError as error:
        raise _Reset(side, error) from error
