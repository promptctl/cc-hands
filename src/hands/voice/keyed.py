"""A page's route that answers only a request carrying the phone's key, each request one unit of work.

The pages beside the phone's (`hands.voice.conversationpage`, `hands.voice.backlogpage`) are open to anyone who can
reach the port; what they read and change is not.
"""

from collections.abc import Awaitable, Callable

from aiohttp import web

from hands.sessions.audit import Record
from hands.sessions.wide import fail, unit
from hands.voice.phoneaddress import carries_key

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

UNKEYED = "this page's address is missing the phone's key; open it from the address `hands phone` prints"


def keyed(event: str, record: Record, key: str, counts: tuple[str, ...] = ()) -> Callable[[Handler], Handler]:
    """`handler` run as the unit of work `event` for a request that carries the phone's key; one that does not is
    refused with a 401, and its event says so.

    [LAW:single-enforcer] the one place a page's route checks the key.
    """

    def wrapped(handler: Handler) -> Handler:
        async def handled(request: web.Request) -> web.StreamResponse:
            with unit(event, record, counts=counts):
                if not carries_key(request.headers.get("Authorization", ""), key):
                    fail("asked without the phone's key")
                    return web.Response(status=401, text=UNKEYED)
                return await handler(request)

        return handled

    return wrapped
