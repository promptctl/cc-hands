"""The phone's app: served over HTTPS to the LAN and the tailnet, where a phone's browser opens it and calls hands, and
sees what the brain is doing and stops it (`hands.voice.brainpage`); and the conversation page and the backlog page beside it
(`hands.voice.conversationpage`, `hands.voice.backlogpage`), each served at its own path.

The app is one document, `app.html`, that holds the call and mounts the app's views under it, so no move between views is a
move of the call (design-docs/phone-sessions/design.md, "The one app"). `ROW` says which views there are.

Where it is reached, and the certificate and key it is reached with, are `hands.voice.phoneaddress`'s.

The page is open to anyone who can reach the port; a call is not. The page sends the phone's key with its offer, and
an offer without it is refused.
"""

import asyncio
import datetime
import html
import json
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Never

from aiohttp import web

from hands.clock import Clock
from hands.core.bearer import bears
from hands.sessions.backlog import Lit
from hands.sessions.child import Children
from hands.sessions.audit import Record
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, begun, continuing, fail, unit
from hands.voice.backlogpage import backlog_routes
from hands.voice.conversationpage import Conversation, conversation_routes
from hands.voice.gallerypage import Gallery, gallery_routes
from hands.voice.keyed import Handler
from hands.voice.pinnedfile import Release
from hands.voice.pagescript import script_route
from hands.voice.phone import PAGE_RATES, Asked, CallDeclined, CallRefused, Offer, Phone, call_ended
from hands.voice.brainpage import Watched, brain_routes
from hands.voice.phoneaddress import PHONE_HOST, PHONE_PORT, Tailnet, Untailed, own_certificate, server_context
from hands.voice.phonewake import wake_routes
from hands.voice.summaries import SummaryStore
from hands.voice.wakemodels import WakeModels


@dataclass(frozen=True)
class InApp:
    """A view of the app: shown in the app's document at `path`, mounted from `views/<view>.js`."""

    label: str
    path: str
    view: str


@dataclass(frozen=True)
class InTab:
    """A page served at its own `path`, opened from the app in a tab of its own."""

    label: str
    path: str


# [LAW:one-source-of-truth] the row of views, in the order it is drawn: the one place that says which views there are, what
# the row draws, which paths are answered with the app's document and which view scripts hands serves. A page's move into
# the app is its entry becoming an `InApp`.
ROW: tuple[InApp | InTab, ...] = (
    InApp("Call", "/", "call"),
    InTab("Conversation", "/conversation"),
    InApp("Sessions", "/sessions", "sessions"),
    InTab("Voices and options", "/gallery"),
)
APP_PATHS = tuple(entry.path for entry in ROW if isinstance(entry, InApp))
# The scripts hands serves for the app: the one every phone page asks hands through, the router, the call, and the views.
APP_SCRIPTS = ("hands.js", "app.js", "call.js", *(f"views/{entry.view}.js" for entry in ROW if isinstance(entry, InApp)))

_ROW_MARKER = "<!--row-->"

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


def app_page() -> str:
    """The app's one document, with the row of views drawn into it as `ROW` says."""
    document = resources.files("hands.voice").joinpath("app.html").read_text()
    before, marker, after = document.partition(_ROW_MARKER)
    if not marker:
        raise LookupError(f"app.html has no {_ROW_MARKER} to draw the row of views at")
    links = "".join(
        f'<a href="{entry.path}" data-view="{entry.view}">{html.escape(entry.label)}</a>' if isinstance(entry, InApp)
        else f'<a href="{entry.path}" target="_blank" rel="noopener">{html.escape(entry.label)}</a>'
        for entry in ROW
    )
    return f"{before}{links}{after}"


def phone_routes(clock: Clock, phone: Phone, key: str, record: Record) -> list[web.RouteDef]:
    """The app at every path of its views, the scripts it is made of, and the one route that takes its calls."""
    page = app_page()

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

    return [*(web.get(path, shown) for path in APP_PATHS), *(script_route(name) for name in APP_SCRIPTS), web.post("/offer", offered)]


# The methods HTTP defines as safe (RFC 9110, 9.2.1): a request that only reads, which asks nothing to be done.
_READS = frozenset({"GET", "HEAD", "OPTIONS"})


@web.middleware
async def deeds_run_to_their_end(request: web.Request, handler: Handler) -> web.StreamResponse:
    """A request that reads ends with its connection, as the server's runner ends it; one that does something, a choice
    taken, a write, a call answered, runs to its end whether or not the phone is still there to be told.

    [LAW:single-enforcer] the one place a deed is kept from the cancellation the runner delivers when a phone's connection
    drops: a deed cut partway leaves what it was changing half changed, and the wait for a page that has gone is all a
    cancellation is for. What the deed did is its own unit's event, as it was before; only the answer is lost.
    """
    if request.method in _READS:
        return await handler(request)
    deed = asyncio.ensure_future(handler(request))
    # A deed that fails after the phone has gone has no one to be raised to: its unit has already said how it failed.
    deed.add_done_callback(lambda done: done.cancelled() or done.exception())
    return await asyncio.shield(deed)


def phone_pages(clock: Clock, phone: Phone, brain: Watched, typed: Callable[[str], Awaitable[None]], store: SummaryStore, audit: Path, key: str, record: Record, environment: Mapping[str, str], gallery: Gallery, models: WakeModels, runtime: Release) -> web.Application:
    """Every page the phone opens, under the phone's `key`. The phone's page reads and stops `brain`. The conversation
    page reads the audit log at `audit` and hands `typed` what is typed into it; the backlog page reads its summaries
    from `store` and runs its writes in `environment`, the one hands was started in; the gallery page sets hands'
    voices, level, modality, and model through `gallery`; the wake word page's detector runs the word `gallery` names, on the
    models `models` fetches and the runtime `runtime` serves (`hands.voice.phonewake`).

    [LAW:one-source-of-truth] the one composition of the phone's routes: hands serves it, and so do the tests of the phone's
    route and of the pages in a browser.
    """
    app = web.Application(middlewares=[deeds_run_to_their_end])
    app.add_routes(phone_routes(clock, phone, key, record))
    app.add_routes(brain_routes(clock, brain, key, record))
    app.add_routes(conversation_routes(clock, Conversation(audit), typed, key, record))
    # What runs lit for the page: on the phone's clock, each lit's event recorded with the page's.
    app.add_routes(backlog_routes(Lit(Children(clock, record), environment), store, key, record))
    app.add_routes(gallery_routes(clock, gallery, key, record))
    app.add_routes(wake_routes(clock, gallery.wake, models, runtime, key, record))
    return app


async def serve_phone(clock: Clock, pages: web.Application, home: Home, asked: Callable[[], Awaitable[Tailnet | Untailed]], record: Record) -> Never:
    """Serve `pages` on every address this machine has, until cancelled; under the tailnet's name, with the certificate
    Tailscale gives `asked`, asked for again when `_Served` says.

    [LAW:nothing-unseen] serving it is one unit of work, `phone.served`, and each renewal after it one of its own,
    `phone.renewed`, the root of a trace of its own: each says the port or what Tailscale answered, as `_Served` says it.
    """
    # [LAW:nothing-unseen] a page that closes its connection ends the handler it was waiting on, so a read it left is a
    # `cancelled` event, and no long-poll goes on at hands for a view the phone has left. Only reads end with their
    # connection: `deeds_run_to_their_end` is what keeps a deed from ending with it.
    runner = web.AppRunner(pages, access_log=None, handler_cancellation=True)
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
