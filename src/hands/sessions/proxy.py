"""The proxy: a Claude Code process reaches the API through it, and hands hears every request and every byte back.

    HTTPS_PROXY=<the proxy's url> NODE_EXTRA_CA_CERTS=<its trust file> claude ...

A client given it as its HTTPS proxy, and its trust file, reaches https://api.anthropic.com as it would Anthropic's own
API: the proxy ends that TLS itself and serves the requests inside it (`hands.sessions.connect`), so Claude Code behaves
as it does first-party. A request posted to the proxy's own origin, as hands' side questions are, is served the same way.
Every other host such a client reaches is tunnelled, untouched, and a request in absolute form is relayed; the proxy
reaches each as hands' own environment says, through its outer proxy or directly. Every request needs the run's
credential as its Proxy-Authorization, which a client given the proxy's url sends from its userinfo: the proxy's
loopback port is any local process's to reach.

It forwards each request upstream as it came, or with the changes its route gives it made to it, and
streams the reply back byte for byte, reading a copy of the same bytes into typed events as they pass. It never makes a
request of its own; it tells each request it takes, and what hands' side questions fork from the brain's comes back
through it as the brain's own Claude Code would send it (`hands.brain.fork`). The one request it does not forward is one
its route answers itself, with the words the route gives it, so the model is not asked at all.
"""

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiohttp
from aiohttp import hdrs, web
from loguru import logger
from multidict import CIMultiDict
from yarl import URL

from hands.clock import Clock
from hands.core import loopback
from hands.core.wire import (
    API_HOST,
    PROXY_USER,
    Answered,
    Answering,
    Body,
    Change,
    Ending,
    Exchanged,
    Garbled,
    Heard,
    Message,
    Observed,
    Reached,
    Received,
    Route,
    Seconds,
    Send,
    Sent,
    Synthesize,
    Synthetic,
    Text,
    ThroughProxy,
    Unfinished,
    Unreached,
    Written,
    edited,
    is_stream,
    proxy_authorization,
    read_request,
    refused_message,
    serialized,
    streams,
)
from hands.sessions.audit import Record
from hands.sessions.authority import Uninterceptable, also_trusted, mint, onward_trust, trusted
from hands.sessions.connect import Admitted, Connects, Unadmitted, admission, host_port, outward
from hands.sessions.replies import reply_reader, sent_of, shielded, spent
from hands.sessions.untap import untapped
from hands.sessions import wide
from hands.sessions.wide import annotate, begun, fail, root, unit

# Headers that describe one hop's connection rather than the request or reply, so each hop sets its own.
HOP_BY_HOP = frozenset({"connection", "content-length", "host", "keep-alive", "proxy-authorization", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade"})

# What the proxy answers a request it refuses on the API's behalf: one it could not reach, or one held final.
REFUSED = 502
# How many random bytes the password of the run's credential is.
_CREDENTIAL_BYTES = 16

Observe = Callable[[Observed], None]
Router = Callable[[Sent], Route]


class Listener(Protocol):
    """What speaks from the wire: it hears everything the proxy observes, and decides where each request goes.

    [LAW:no-ambient-temporal-coupling] the proxy tells a request before it routes it, in the one handler that does both
    (`serve_proxy`), so a listener routes only what it has heard."""

    def route(self, sent: Sent) -> Route: ...
    def hear(self, observed: Observed) -> None: ...


class _NoOne:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def route(self, sent: Sent) -> Route:
        return Send(span=root(self._clock))

    def hear(self, observed: Observed) -> None:
        pass


class Wire:
    """What the proxy tells and asks: the daemon's record of the wire always, and the one listener joined to it, while it is."""

    def __init__(self, clock: Clock, record: Observe) -> None:
        self._clock = clock
        self._record = record
        # [LAW:dataflow-not-control-flow] with nobody joined, a listener that forwards everything and hears nothing.
        self._listener: Listener = _NoOne(clock)

    def observe(self, observed: Observed) -> None:
        # [LAW:no-ambient-temporal-coupling] heard whether or not the record took it: what the listener routes by is never
        # the record's to lose. A record that fails still raises, once the listener has heard.
        try:
            self._record(observed)
        finally:
            self._listener.hear(observed)

    def route(self, sent: Sent) -> Route:
        return self._listener.route(sent)

    @contextmanager
    def joined(self, listener: Listener) -> Generator[None]:
        # [LAW:no-shared-mutable-globals] one listener at a time, joined and left only here.
        if not isinstance(self._listener, _NoOne):
            raise RuntimeError(f"{listener!r} joined the wire while {self._listener!r} still listens")
        self._listener = listener
        try:
            yield
        finally:
            self._listener = _NoOne(self._clock)


# How long a stopping proxy waits on each request still served before cancelling it, rather than on replies that stream
# for minutes. Not 0: aiohttp reads a timeout of 0 as none at all.
SHUTDOWN_SECONDS = 0.1


@dataclass(frozen=True)
class Proxy:
    """The proxy, serving: `through` is what a client is given to reach the API by it, hands' side questions among them."""

    through: ThroughProxy
    runner: web.ServerRunner
    inside: web.Server
    connects: Connects
    client: aiohttp.ClientSession

    async def close(self) -> None:
        # The requests inside each TLS the proxy ended, then each CONNECT's connection, then its own origin's.
        await self.inside.shutdown(SHUTDOWN_SECONDS)
        await self.connects.close()
        await self.runner.cleanup()
        await self.client.close()


async def serve_proxy(upstream: str, observe: Observe, route: Router, clock: Clock, environ: Mapping[str, str], trust: Path, record: Record) -> Proxy:
    """Listen on a free loopback port until the returned proxy is closed, ending the TLS of a CONNECT to the API's host
    with an authority of its own and forwarding the requests inside it, and those made to its own origin, to
    `upstream`. Its authority is written to `trust`, with the certificates `environ` trusts, and every other host is
    reached as `environ` says; raises Uninterceptable when either cannot be, or `upstream` is no URL of a host and a port.
    Every request is answered only on the credential the returned proxy's `through` carries."""
    # [LAW:one-source-of-truth] the world beyond the proxy as hands' own environment has it, wherever hands was started:
    # a hands started inside a tapped session reaches it as that session's own environment did.
    world = untapped(environ)
    reach = outward(world)
    also = also_trusted(world, trust)
    try:
        api = URL(upstream)
        at = api.host, api.port
    except ValueError as error:
        raise Uninterceptable(f"the API the proxy forwards to is no URL, {upstream!r}: {error}") from error
    match at:
        case str() as api_host, int() as api_port:
            through_api = reach.onward(api.scheme, api_host, api_port)
        case _:
            raise Uninterceptable(f"the API the proxy forwards to is reached at a host and a port, not {upstream}")
    # Every refusal is said before the trust file is written: nothing a refused start leaves behind is anyone's to trust.
    trusting = onward_trust(also)
    # [LAW:types-are-the-program] the host whose TLS the proxy ends is the API's, by construction.
    authority = mint(API_HOST, clock)
    written = trusted(authority, also, trust)
    # [LAW:single-enforcer] the run's credential, the one thing that has the proxy serve a request: any other process on
    # the machine reaches its loopback port too, and would go out the way hands does.
    password = clock.token(_CREDENTIAL_BYTES)
    credential = proxy_authorization(password)
    # The client's own timeouts govern a request: a long reply streams as long as the client keeps reading it.
    # Replies are read as sent, compressed or not, so the client gets the bytes the API wrote; and no header the client
    # left out is added, so the API is asked for no encoding the client did not ask for. [LAW:single-enforcer] every
    # connection it makes, to the API and every relayed origin, trusts what hands trusts.
    client = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None),
        auto_decompress=False,
        skip_auto_headers=("Accept-Encoding", "Content-Type", "User-Agent"),
        connector=aiohttp.TCPConnector(ssl=trusting),
    )

    tell = shielded(observe)

    async def forward(request: web.BaseRequest) -> web.StreamResponse:
        # Read whole whatever its size: the request is classified from its body before any of it goes upstream.
        body = await request.content.read()
        requested_at = clock.wall().timestamp()
        # Its path and query as the client wrote them, in origin form or absolute: aiohttp's path_qs rewrites a path that
        # begins with "//".
        path = request.rel_url.raw_path_qs
        sent = sent_of(clock, request.headers, path, body)
        exchange, session, kind, named, parsed = sent.exchange, sent.session, sent.kind, sent.named, sent.body
        tell(sent)
        try:
            routed = route(sent)
        except Exception:
            # [LAW:no-silent-failure] a route that raises is logged with its trace, and the request goes on as it came:
            # the conversation is not broken by hands failing to decide about it.
            logger.exception(f"the proxy's route failed on exchange {exchange}; forwarding it")
            routed = Send(span=root(clock))

        sent_at = clock.wall().timestamp()

        def exchanged(changes: tuple[Change, ...], reply: Ending, final: bool) -> Exchanged:
            return Exchanged(exchange, session, kind, named, request.method, path, len(body), changes, requested_at, sent_at, reply, final, routed.span)

        changes: tuple[Change, ...] = ()
        # [LAW:single-enforcer] from here until the API's answer has a head, whatever stops the exchange ends it here.
        try:
            headers = _end_to_end(request.headers)
            match routed:
                case Synthesize(said=said):
                    # [LAW:single-enforcer] exception: classify read this request and kept no Request, so its model is read
                    # again by the one reader; promptctl-brain-ny8p.mz4 has Sent carry what was read.
                    model = read_request(parsed).model
                    written = Written(Message(f"msg_hands_{clock.uuid()}", model, (Text(said),), "end_turn", {"input_tokens": 0, "output_tokens": 0}), streams(parsed))
                    content_type, answer = serialized(written)
                    # Its answer made before its ending is told: once told, nothing here can raise and tell another.
                    response = web.Response(status=200, body=answer, headers={"Content-Type": content_type})
                    # The request as the client sent it: the client holds it, answered, as its conversation's newest.
                    tell(Received(exchange, path, tuple(headers.items()), body, built=False))
                    tell(exchanged((), Synthetic(written, clock.wall().timestamp()), False))
                    return response
                case Send(changes=asked, refusal=refusal, built=built):
                    onward, changes = _edited(exchange, body, parsed, asked)
            tell(Received(exchange, path, tuple(headers.items()), onward, built))
            try:
                reached = await client.request(request.method, upstream + path, headers=headers, data=onward, proxy=through_api)
            except (aiohttp.ClientError, OSError) as error:
                # A client asks again after the proxy's 502 unless told the refusal is final.
                final = refusal == "final"
                refused = web.Response(status=REFUSED, text=f"hands' proxy could not reach {upstream}: {error}", headers=_told(final))
                tell(exchanged(changes, Unreached(f"{type(error).__name__}: {error}", clock.wall().timestamp()), final))
                return refused
        except BaseException as error:
            # [LAW:no-silent-failure] the error goes on as it came; the exchange is still recorded as ended, and how.
            tell(exchanged(changes, Unfinished(_unfinished(error), clock.wall().timestamp()), False))
            raise
        async with reached:
            first: Seconds | None = None
            ended: Seconds | None = None
            size = 0
            final = False
            # What a reply is recorded as when its handler is cancelled mid-way: the client hung up, or hands is stopping.
            reply: Body = Garbled("the proxy stopped reading the reply: its client hung up or hands stopped")

            async def heard() -> AsyncIterator[bytes]:
                nonlocal first, size
                async for chunk in reached.content.iter_any():
                    first = clock.wall().timestamp() if first is None else first
                    size += len(chunk)
                    # [LAW:no-ambient-temporal-coupling] read before it is written on: whatever the client does once it
                    # has these bytes, its result line on stdout included, comes after hands has heard them.
                    reader.feed(chunk)
                    yield chunk

            try:
                limit = spent(reached.status, reached.headers)
                tell(Answering(exchange, reached.status, limit))
                reader = reply_reader(kind, reached.headers, lambda event: tell(Heard(exchange, event)))
                if reached.status < 400 and is_stream(reached.headers.get("Content-Type", "")):
                    response = await _streamed(request, reached, heard())
                    ended = clock.wall().timestamp()
                    reply = reader.finish(None)
                else:
                    # A reply sent whole is read whole before it is answered: a message is heard only once it is all
                    # here, and a refusal is held final or not on what it says. Held final, it reaches the client as the
                    # proxy's own, which Claude Code ends on, since it asks again after an overload whatever the header
                    # says, and waits out a spent limit to continue the task at its reset (autoContinueAtUsageLimit,
                    # autoContinueAtUsageLimit); whoever speaks the turn's failure reads it off the wire.
                    parts = [chunk async for chunk in heard()]
                    ended = clock.wall().timestamp()
                    reply = reader.finish(None)
                    final = refusal == "final" and (limit is not None or _asked_again(reached.status, reached.headers, reply))
                    response = _refused(reached) if final else web.Response(status=reached.status, reason=reached.reason, headers=_end_to_end(reached.headers), body=b"".join(parts))
            except Exception as error:
                # Upstream dropped the reply, the client hung up on it, or the proxy failed on it: either way the client
                # has not had it whole, its connection ends, and the error says which.
                reply = Garbled(f"the reply was not relayed whole, {size} bytes in: {type(error).__name__}: {error}")
                raise
            finally:
                last = clock.wall().timestamp() if ended is None else ended
                tell(exchanged(changes, Reached(reached.status, last if first is None else first, last, size, reply), final))
                # [LAW:single-enforcer] every request hands built or changed passes here, so its 400 is told here alone,
                # after the exchange's end is told: a failure saying it cannot unsay that.
                if reached.status == 400 and (built or changes):
                    _hands_bug(exchange, reached.status, reached.headers.get("request-id"), parsed, built, changes, reply)
        return response

    async def relayed(request: web.BaseRequest, _: Admitted) -> web.StreamResponse:
        """A request in absolute form, a client asking its proxy for another origin's answer: relayed there and back as
        sent, through the outer proxy hands' environment names for it, and none of it the API's or recorded."""
        # [LAW:nothing-unseen] one event for each relay: the origin it asked, the way it went, and its answer's status or
        # why there was none.
        with unit("proxy.relay", record, begun(clock)):
            # aiohttp's reading of its absolute-form target (RFC 9112 §3.2.2).
            target = request.url
            match target.host, target.port:
                case str() as host, int() as port if port > 0:
                    pass
                case _:
                    refused = f"hands' proxy relays a request to a host and a port that is not 0, not {request.raw_path!r}"
                    fail(refused)
                    return web.Response(status=400, text=refused)
            onward = reach.onward(target.scheme, host, port)
            annotate(origin=str(target.origin()), through="directly" if onward is None else str(onward.origin()))
            try:
                reached = await client.request(request.method, target, headers=_end_to_end(request.headers), data=request.content, proxy=onward)
            except (aiohttp.ClientError, OSError) as error:
                unreached = f"hands' proxy could not reach {target.origin()}: {type(error).__name__}: {error}"
                fail(unreached)
                logger.warning(unreached)
                return web.Response(status=REFUSED, text=unreached)
            annotate(status=reached.status)
            async with reached:
                return await _streamed(request, reached, reached.content.iter_any())

    async def served(request: web.BaseRequest) -> web.StreamResponse:
        # [LAW:single-enforcer] the one check of the run's credential, ahead of everything the proxy does with a request on
        # its own socket. Inside the TLS a CONNECT ended, every request is the API's, on the connection its CONNECT was
        # admitted on: `inside` serves it.
        match admission(request.headers.get(hdrs.PROXY_AUTHORIZATION), credential):
            case Unadmitted(why=why):
                said = f"hands' proxy answers a request on this run's credential alone, and {why}"
                wide.ended("proxy.unadmitted", record, begun(clock), None, said, method=request.method, target=request.raw_path)
                return web.Response(status=407, text=said, headers={hdrs.PROXY_AUTHENTICATE: f'Basic realm="{PROXY_USER}"'})
            case Admitted() as admitted:
                pass
        # [LAW:dataflow-not-control-flow] an admitted request's form says where it goes (RFC 9112 §3.2): a CONNECT asks for
        # a tunnel; origin form, its target a path, is the API's, as hands' side questions are; and absolute form asks
        # for an origin's answer, the API's for the API's origin alone, as a CONNECT's is (`Authority.intercepts`).
        match request.method, request.raw_path.startswith("/"):
            case hdrs.METH_CONNECT, _:
                return await connects.answer(request, admitted)
            case _, True:
                return await forward(request)
            case _, False if authority.intercepts(request.url.host, request.url.port):
                return await forward(request)
            case _, False:
                return await relayed(request, admitted)

    inside = _server(forward)
    connects = Connects(authority, reach, inside, clock, record)
    runner = web.ServerRunner(_server(served), shutdown_timeout=SHUTDOWN_SECONDS)
    try:
        await runner.setup()
        await web.TCPSite(runner, loopback.HOST, 0).start()
    except BaseException:
        # A proxy that never listened closes what it opened for its requests before it says why.
        await runner.cleanup()
        await client.close()
        raise
    host, port = runner.addresses[0][:2]
    through = ThroughProxy(host_port(host, port), password, written)
    return Proxy(through=through, runner=runner, inside=inside, connects=connects, client=client)


def _server(handler: Callable[[web.BaseRequest], Awaitable[web.StreamResponse]]) -> web.Server:
    """[LAW:single-enforcer] a server of the proxy's, its own origin's and the API's inside each TLS it ends alike: a
    client that hangs up cancels its handler, which is how an exchange learns its client left, and a stopping daemon
    cancels every handler within SHUTDOWN_SECONDS."""
    return web.Server(handler, handler_cancellation=True, access_log=None)


def _edited(exchange: str, body: bytes, parsed: object, changes: tuple[Change, ...]) -> tuple[bytes, tuple[Change, ...]]:
    """The bytes that go upstream with `changes` made, and the changes they carry: none when they could not be made."""
    if not changes:
        # The bytes the client wrote, not a re-encoding of them.
        return body, ()
    try:
        # JSON.stringify's spacing. Non-ASCII is escaped: a string Claude Code cut mid-emoji holds a lone surrogate,
        # which UTF-8 cannot encode.
        return json.dumps(edited(parsed, changes), separators=(",", ":")).encode(), changes
    except ValueError:
        # [LAW:no-silent-failure] the request goes on as it came, and the log says it went unchanged.
        logger.exception(f"the proxy could not change exchange {exchange}; forwarding it as it came")
        return body, ()


def _hands_bug(exchange: str, status: int, request_id: str | None, request: object, built: bool, changes: tuple[Change, ...], reply: Body) -> None:
    """Say loudly that the API refused a request hands built or changed: hands must never send one it refuses."""
    said = str(reply)
    # hands' changes never add, drop, or re-role a message, so the request as it came names the same one.
    named = refused_message(request, said)
    made = "built" if built else f"changed ({len(changes)} changes)"
    where = "" if named is None else f" {named};"
    logger.error(f"hands' bug: hands {made} exchange {exchange} and the API refused it {status} (request-id {request_id}):{where} {said}")


def _unfinished(error: BaseException) -> str:
    """Why the proxy stopped waiting on the API's answer."""
    match error:
        # A client that hangs up cancels its handler, and so does a stopping daemon.
        case asyncio.CancelledError():
            return "cancelled before the API answered: its client hung up or hands stopped"
        case _:
            return f"{type(error).__name__}: {error}"


def _asked_again(status: int, headers: Mapping[str, str], reply: Body) -> bool:
    """Whether Claude Code asks again after this answer (`lMo`)."""
    told = headers.get("x-should-retry")
    match status, told, reply:
        case (int() as answered, _, _) if answered < 400:
            return False
        # Asking again after a 401 is how the client refreshes its login, and that stays its own.
        case (401, _, _):
            return False
        # Overloaded: asked again whatever the header says.
        case (529, _, _) | (_, _, Answered(body={"error": {"type": "overloaded_error"}})):
            return True
        case (_, "true" | "false", _):
            return told == "true"
        case _:
            return status in (408, 409, 429) or status >= 500


def _told(final: bool) -> dict[str, str]:
    """The header that tells the client a refusal is final, the SDK's own, read ahead of the status."""
    return {"x-should-retry": "false"} if final else {}


async def _streamed(request: web.BaseRequest, reached: aiohttp.ClientResponse, chunks: AsyncIterable[bytes]) -> web.StreamResponse:
    """[LAW:one-source-of-truth] the one tail of a reply streamed on: `reached`'s status and headers as they came, and
    its body byte for byte as `chunks` give it."""
    response = web.StreamResponse(status=reached.status, reason=reached.reason, headers=_end_to_end(reached.headers))
    if "Content-Length" in reached.headers:
        response.content_length = int(reached.headers["Content-Length"])
    await response.prepare(request)
    async for chunk in chunks:
        await response.write(chunk)
    await response.write_eof()
    return response


def _refused(reached: aiohttp.ClientResponse) -> web.Response:
    """The proxy's own refusal, told final, in place of the API's, saying what the API answered."""
    said = f"hands' proxy: the API answered {reached.status} {reached.reason} (request-id {reached.headers.get('request-id')}), and hands holds that final"
    return web.Response(status=REFUSED, text=said, headers=_told(True))


def _end_to_end(headers: Mapping[str, str]) -> CIMultiDict[str]:
    return CIMultiDict((name, value) for name, value in headers.items() if name.lower() not in HOP_BY_HOP)
