"""The phone's page: served over HTTPS to the LAN and the tailnet, where a phone's browser opens it and calls hands, and
sees what the brain is doing and stops it (`hands.voice.brainpage`); and the conversation page and the backlog page beside it
(`hands.voice.conversationpage`, `hands.voice.backlogpage`).

Where it is reached, and the certificate and key it is reached with, are `hands.voice.phoneaddress`'s.

The page is open to anyone who can reach the port; a call is not. The page sends the phone's key with its offer, and
an offer without it is refused.
"""

import json
import ssl
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Never

from aiohttp import web
from loguru import logger

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, begun, continuing, unit
from hands.voice.backlogpage import backlog_routes
from hands.voice.conversationpage import Conversation, conversation_routes
from hands.voice.gallerypage import Gallery, gallery_routes
from hands.voice.phone import Asked, CallDeclined, CallRefused, Offer, Phone, call_ended
from hands.voice.brainpage import Watched, brain_routes
from hands.voice.phoneaddress import PHONE_HOST, PHONE_PORT, Tailnet, Untailed, carries_key, own_certificate, tailnet
from hands.voice.summaries import SummaryStore

# How often a page served under the tailnet's name asks Tailscale for its certificate again: Tailscale renews one a
# month before its 90 days are out, so a day is far inside the time a renewed one has before the old one ends.
RENEW_SECONDS = 24 * 60 * 60.0


def parse_offer(body: object) -> Offer:
    """A page's offer as hands takes it; raises Rejected naming what is wrong with it."""
    match body:
        case {"sdp": str(sdp), "type": "offer", "rate": int(rate), "page": str(page), "claim": str(claim)} if rate > 0 and page and claim in ("take", "resume"):
            return Offer(sdp, "offer", rate, Asked(page, claim))
        case _:
            raise Rejected("an offer is {sdp, type: offer, rate, page, claim: take or resume}")


def phone_routes(clock: Clock, phone: Phone, key: str, record: Record) -> list[web.RouteDef]:
    """The page, the script every phone page asks hands through, and the one route that takes its calls."""
    page = resources.files("hands.voice").joinpath("phone.html").read_text()
    script = resources.files("hands.voice").joinpath("hands.js").read_text()

    async def shown(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    # Never kept by the browser, as no page is: a page and the script it asks hands through are always one run's.
    async def scripted(_request: web.Request) -> web.Response:
        return web.Response(text=script, content_type="text/javascript", headers={"Cache-Control": "no-store"})

    async def offered(request: web.Request) -> web.Response:
        remote = request.remote or "unknown"
        began = begun(clock)
        if not carries_key(request.headers.get("Authorization", ""), key):
            call_ended(record, began, remote, None, CallRefused("offered without the phone's key"))
            return web.Response(status=401, text="this page's address is missing the phone's key; open it from `hands phone`")
        try:
            offer = parse_offer(await request.json())
        except (Rejected, json.JSONDecodeError) as error:
            call_ended(record, began, remote, None, CallRefused(str(error)))
            return web.Response(status=400, text=str(error))
        except BaseException as error:
            # The body could not be read at all, or the request went as it was.
            call_ended(record, began, remote, None, error)
            raise
        match await phone.answer(offer, remote, began):
            case CallDeclined():
                return web.Response(status=409, text="another page has hands' phone; press Connect to take it")
            case answer:
                return web.json_response({"sdp": answer.sdp, "type": answer.type})

    return [web.get("/", shown), web.get("/hands.js", scripted), web.post("/offer", offered)]


def phone_pages(clock: Clock, phone: Phone, brain: Watched, typed: Callable[[str], Awaitable[None]], store: SummaryStore, audit: Path, key: str, record: Record, environment: Mapping[str, str], gallery: Gallery) -> web.Application:
    """Every page the phone opens, under the phone's `key`. The phone's page reads and stops `brain`. The conversation
    page reads the audit log at `audit` and hands `typed` what is typed into it; the backlog page reads its summaries
    from `store` and runs its writes in `environment`, the one hands was started in; the gallery page sets hands'
    voices, level, modality, and model through `gallery`.

    [LAW:one-source-of-truth] the one composition of the phone's routes: hands serves it, and every test of the routes
    asks it.
    """
    app = web.Application()
    app.add_routes(phone_routes(clock, phone, key, record))
    app.add_routes(brain_routes(clock, brain, key, record))
    app.add_routes(conversation_routes(clock, Conversation(audit), typed, key, record))
    app.add_routes(backlog_routes(clock, store, key, record, environment))
    app.add_routes(gallery_routes(clock, gallery, key, record))
    return app


async def serve_phone(clock: Clock, pages: web.Application, home: Home, record: Record) -> Never:
    """Serve `pages` on every address this machine has, until cancelled; under the tailnet's name, with the certificate
    Tailscale renews, asked for again every RENEW_SECONDS.

    [LAW:nothing-unseen] serving it is one unit of work, `phone.served`: the port, and the tailnet name it is served
    under, or, served on the LAN alone, why Tailscale gave it none.
    """
    runner = web.AppRunner(pages, access_log=None)
    await runner.setup()
    try:
        with unit("phone.served", record, begun(clock)):
            own = _context(*own_certificate(clock, home))
            kept: Callable[[], Coroutine[object, object, Never]]
            match await tailnet(clock, home):
                case Tailnet(name=name, cert=cert, key=key):
                    tailed = _context(cert, key)

                    def chosen(connection: ssl.SSLObject, server_name: str | None, _context: ssl.SSLContext) -> None:
                        # [LAW:single-enforcer] the one place a name picks its certificate: the name the phone asked for.
                        if server_name == name:
                            connection.context = tailed

                    own.sni_callback = chosen  # pyright: ignore[reportAttributeAccessIssue]  (typeshed types the callback for SSLSocket alone)
                    annotate(tailnet=name)
                    kept = partial(_keep_renewed, clock, tailed, home)
                case Untailed(reason=reason):
                    # [LAW:no-silent-failure] the page is still served on the LAN; why the tailnet name is not, is said.
                    logger.warning(f"the phone's page is served on the LAN alone: {reason}")
                    annotate(untailed=reason)
                    kept = partial(_keep_served, clock)
            # Every request the site takes is work for no unit open here: the server's tasks copy the context it is
            # started in, and `phone.served` ends as the page comes up.
            with continuing(None):
                await _site(runner, own)
            # The port the page was bound on, which a phone's address names.
            [(_host, port, *_)] = runner.addresses
            annotate(port=port)
        await kept()
    finally:
        await runner.cleanup()


async def _keep_renewed(clock: Clock, tailed: ssl.SSLContext, home: Home) -> Never:
    while True:
        await clock.sleep(RENEW_SECONDS)
        match await tailnet(clock, home):
            case Tailnet(cert=cert, key=key):
                # A context's chain, loaded again, is the one every handshake after it shows.
                tailed.load_cert_chain(cert, key)
            case Untailed(reason=reason):
                logger.warning(f"the tailnet's certificate for the phone's page was not renewed: {reason}")


async def _keep_served(clock: Clock) -> Never:
    while True:
        # Served until cancelled: there is no certificate of Tailscale's to renew.
        await clock.sleep(RENEW_SECONDS)


def _context(cert: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


async def _site(runner: web.AppRunner, context: ssl.SSLContext) -> None:
    try:
        await web.TCPSite(runner, PHONE_HOST, PHONE_PORT, ssl_context=context).start()
    except OSError as error:
        raise RuntimeError(f"cannot serve the phone's page on port {PHONE_PORT}: {error}") from error
