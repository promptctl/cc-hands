"""The phone's page: served over HTTPS to the LAN and the tailnet, where a phone's browser opens it and calls hands, and
sees what the brain is doing and stops it (`hands.voice.brainpage`); and the conversation page and the backlog page beside it
(`hands.voice.conversationpage`, `hands.voice.backlogpage`).

Where it is reached, and the certificate and key it is reached with, are `hands.voice.phoneaddress`'s.

The page is open to anyone who can reach the port; a call is not. The page sends the phone's key with its offer, and
an offer without it is refused.
"""

import datetime
import json
import ssl
from collections.abc import Awaitable, Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Never

from aiohttp import web

from hands.clock import Clock
from hands.core.bearer import bears
from hands.sessions.child import Children
from hands.sessions.audit import Record
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, begun, continuing, fail, unit
from hands.voice.backlogpage import backlog_routes
from hands.voice.conversationpage import Conversation, conversation_routes
from hands.voice.gallerypage import Gallery, gallery_routes
from hands.voice.pinnedfile import Release
from hands.voice.pagescript import script_route
from hands.voice.phone import PAGE_RATES, Asked, CallDeclined, CallRefused, Offer, Phone, call_ended
from hands.voice.brainpage import Watched, brain_routes
from hands.voice.phoneaddress import PHONE_HOST, PHONE_PORT, Tailnet, Untailed, own_certificate, server_context
from hands.voice.phonewake import wake_routes
from hands.voice.summaries import SummaryStore
from hands.voice.wakemodels import WakeModels

# How often the page asks Tailscale for the tailnet's certificate again: Tailscale renews one a month before its 90
# days are out, so a day is far inside the time a renewed one has before the old one ends.
RENEW_SECONDS = 24 * 60 * 60.0
# How soon Tailscale is asked again once it has given no certificate and none is served: it is still starting at login,
# and connects within a minute or so. Each refusal after doubles the wait, up to RENEW_SECONDS.
ASK_AGAIN_SECONDS = 60.0


def parse_offer(body: object) -> Offer:
    """A page's offer as hands takes it; raises Rejected naming what is wrong with it."""
    match body:
        case {"sdp": str(sdp), "type": "offer", "rate": int(rate), "page": str(page), "claim": str(claim)} if rate in PAGE_RATES and page and claim in ("take", "resume"):
            return Offer(sdp, "offer", rate, Asked(page, claim))
        case _:
            raise Rejected(f"an offer is {{sdp, type: offer, rate: a whole number from {PAGE_RATES.start} to {PAGE_RATES.stop - 1}, page, claim: take or resume}}")


def phone_routes(clock: Clock, phone: Phone, key: str, record: Record) -> list[web.RouteDef]:
    """The page, the script every phone page asks hands through, and the one route that takes its calls."""
    page = resources.files("hands.voice").joinpath("phone.html").read_text()

    async def shown(_request: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def offered(request: web.Request) -> web.Response:
        remote = request.remote or "unknown"
        began = begun(clock)
        if not bears(request.headers.get("Authorization", ""), key):
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

    return [web.get("/", shown), script_route("hands.js"), web.post("/offer", offered)]


def phone_pages(clock: Clock, phone: Phone, brain: Watched, typed: Callable[[str], Awaitable[None]], store: SummaryStore, audit: Path, key: str, record: Record, environment: Mapping[str, str], gallery: Gallery, models: WakeModels, runtime: Release) -> web.Application:
    """Every page the phone opens, under the phone's `key`. The phone's page reads and stops `brain`. The conversation
    page reads the audit log at `audit` and hands `typed` what is typed into it; the backlog page reads its summaries
    from `store` and runs its writes in `environment`, the one hands was started in; the gallery page sets hands'
    voices, level, modality, and model through `gallery`; the wake word page's detector runs the word `gallery` names, on the
    models `models` fetches and the runtime `runtime` serves (`hands.voice.phonewake`).

    [LAW:one-source-of-truth] the one composition of the phone's routes: hands serves it, and so do the tests of the phone's
    route and of the pages in a browser.
    """
    app = web.Application()
    app.add_routes(phone_routes(clock, phone, key, record))
    app.add_routes(brain_routes(clock, brain, key, record))
    app.add_routes(conversation_routes(clock, Conversation(audit), typed, key, record))
    # What runs lit for the page: on the phone's clock, each lit's event recorded with the page's.
    app.add_routes(backlog_routes(Children(clock, record), store, key, record, environment))
    app.add_routes(gallery_routes(clock, gallery, key, record))
    app.add_routes(wake_routes(clock, gallery.wake, models, runtime, key, record))
    return app


async def serve_phone(clock: Clock, pages: web.Application, home: Home, asked: Callable[[], Awaitable[Tailnet | Untailed]], record: Record) -> Never:
    """Serve `pages` on every address this machine has, until cancelled; under the tailnet's name, with the certificate
    Tailscale gives `asked`, asked for again when `_Served` says.

    [LAW:nothing-unseen] serving it is one unit of work, `phone.served`, and each renewal after it one of its own,
    `phone.renewed`, the root of a trace of its own: each says the port or what Tailscale answered, as `_Served` says it.
    """
    runner = web.AppRunner(pages, access_log=None)
    await runner.setup()
    try:
        with unit("phone.served", record, begun(clock)):
            served = _Served(own_certificate(clock, home))
            wait = served.answered(await asked())
            # Every request the site takes is work for no unit open here: the server's tasks copy the context it is
            # started in, and `phone.served` ends as the page comes up.
            with continuing(None):
                await _site(runner, served.context)
            # The port the page was bound on, which a phone's address names.
            [(_host, port, *_)] = runner.addresses
            annotate(port=port)
        # No renewal is part of whatever unit is open where the page is served.
        with continuing(None):
            while True:
                await clock.sleep(wait)
                with unit("phone.renewed", record, begun(clock)):
                    wait = served.answered(await asked())
    finally:
        await runner.cleanup()


class _Served:
    """The certificate a phone's handshake is answered with: hands' own, and the tailnet's under its name once Tailscale
    has issued one.

    [LAW:one-source-of-truth] the one holder of the tailnet chain served, and its one writer: the handshake reads it, and
    an answer of Tailscale's replaces it whole with a context that has already loaded, so the name and the chain served
    never part.
    """

    def __init__(self, own: tuple[Path, Path]) -> None:
        self.context = server_context(*own)
        self._tailnet: Tailnet | None = None
        # The wait after the next refusal with nothing served: exponential backoff, doubling up to a day.
        self._backoff = ASK_AGAIN_SECONDS
        self.context.sni_callback = self._chosen  # pyright: ignore[reportAttributeAccessIssue]  (typeshed types the callback for SSLSocket alone)

    def answered(self, answer: Tailnet | Untailed) -> float:
        """Serve what Tailscale answered: a chain it issued replaces the one served, and a refusal keeps it. The seconds
        until it is asked again: a day while a chain is served, and while none is, exponential backoff from ASK_AGAIN_SECONDS.

        [LAW:nothing-unseen] said on the unit open here: failed in Tailscale's words when it refused, the tailnet name
        served and when its certificate ends, if one is, and when Tailscale is asked again.
        """
        match answer:
            case Tailnet():
                self._tailnet = answer
            case Untailed(reason=reason):
                fail(reason)
        match self._tailnet:
            case Tailnet(name=name, not_after=ends):
                annotate(tailnet=name, tailnet_not_after=ends)
                wait, self._backoff = RENEW_SECONDS, ASK_AGAIN_SECONDS
            case None:
                wait, self._backoff = self._backoff, min(self._backoff * 2, RENEW_SECONDS)
        annotate(asked_again_in=datetime.timedelta(seconds=wait))
        return wait

    def _chosen(self, connection: ssl.SSLObject, server_name: str | None, _context: ssl.SSLContext) -> None:
        # [LAW:single-enforcer] the one place a name picks its certificate: the name the phone asked for.
        match self._tailnet:
            case Tailnet(name=name, context=context) if server_name == name:
                connection.context = context
            case _:
                connection.context = self.context


async def _site(runner: web.AppRunner, context: ssl.SSLContext) -> None:
    try:
        await web.TCPSite(runner, PHONE_HOST, PHONE_PORT, ssl_context=context).start()
    except OSError as error:
        raise RuntimeError(f"cannot serve the phone's page on port {PHONE_PORT}: {error}") from error
