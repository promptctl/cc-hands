"""A page's read that waits for what the page shows to change before it answers: the conversation page's
(`hands.voice.conversationpage`), the brain on the phone's page (`hands.voice.brainpage`), and the gallery's
(`hands.voice.gallerypage`) alike.

[LAW:one-type-per-behavior] each page versions what it shows, and says with each read the version it was last given; the one
loop here waits on any version.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import TypedDict

from aiohttp import web

from hands.clock import Clock
from hands.sessions.wide import Fact, annotate

# How long a read waits for what its page shows to change before answering with it as it is, well inside the minute a
# browser or a proxy between gives a request; and how often it looks meanwhile.
WAIT_SECONDS = 25.0
POLL_SECONDS = 0.25


class Read[V, T](TypedDict):
    """A read's answer, every page's alike (hands.voice.hands.js reads it): the version of what it shows, which the page
    says it has seen with its next read; how long that read may wait at hands, so the page bounds it by hands' wait and
    not by a copy of it; the Mac's clock as hands answered, which the page times what hands shows by, never by its own;
    and what it shows, which the version is the version of."""

    seen: V
    waits: float
    now: float
    shown: T


def version_of(shown: object) -> str:
    """The version of what a page shows: the same for the same JSON, whatever order its keys were built in."""
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:16]


def page_left(request: web.Request) -> bool:
    """Whether the connection `request` came on has closed: the page it was for has gone, a view left or the phone away."""
    return request.transport is None or request.transport.is_closing()


async def waited[V: Fact, T](clock: Clock, look: Callable[[], Awaitable[tuple[V, T]]], seen: V | None, gone: Callable[[], bool]) -> Read[V, T]:
    """The read of what `look` finds once its version is not `seen`, or as it is once WAIT_SECONDS have passed, or as soon as
    `gone` says the page that asked has; a page that has seen nothing (None) is answered at once. The event of the unit open
    here says what the page had seen, how long it waited and how many looks it took, whether what it found had changed, and
    whether the page had gone.

    [LAW:single-enforcer] the one place a read ends with its connection: the wait is all a read does, so a page that left
    ends it and no long-poll goes on at hands for it, while every other handler runs to its end as it always has.
    """
    began = clock.now()
    looks = 0
    try:
        while True:
            looks += 1
            version, found = await look()
            left = gone()
            # [LAW:no-ambient-temporal-coupling] the page says what it has seen, so a change made between its reads is
            # never missed: it is answered at once with whatever has changed since.
            if version != seen or left or clock.now() - began >= WAIT_SECONDS:
                break
            await clock.sleep(POLL_SECONDS)
    finally:
        # [LAW:nothing-unseen] a look that fails partway through the wait still says how far the wait had come.
        annotate(seen=seen, waited_ms=round((clock.now() - began) * 1000, 1), looks=looks)
    annotate(changed=version != seen, page_left=left)
    return {"seen": version, "waits": WAIT_SECONDS, "now": clock.wall().timestamp(), "shown": found}
