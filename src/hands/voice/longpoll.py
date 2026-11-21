"""A page's read that waits for what the page shows to change before it answers: the conversation page's
(`hands.voice.conversationpage`), the brain on the phone's page (`hands.voice.brainpage`), and the gallery's
(`hands.voice.gallerypage`) alike.

[LAW:one-type-per-behavior] each page versions what it shows, and says with each read the version it was last given; the one
loop here waits on any version.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable

from hands.clock import Clock
from hands.sessions.wide import Fact, annotate

# How long a read waits for what its page shows to change before answering with it as it is, well inside the minute a
# browser or a proxy between gives a request; and how often it looks meanwhile.
WAIT_SECONDS = 25.0
POLL_SECONDS = 0.25


def version_of(shown: object) -> str:
    """The version of what a page shows: the same for the same JSON, whatever order its keys were built in."""
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:16]


async def waited[V: Fact, T](clock: Clock, look: Callable[[], Awaitable[tuple[V, T]]], seen: V | None) -> tuple[V, T]:
    """What `look` finds once its version is not `seen`, or as it is once WAIT_SECONDS have passed; a page that has seen
    nothing (None) is answered at once. The event of the unit open here says what the page had seen, how long it waited
    and how many looks it took, and whether what it found had changed.
    """
    began = clock.now()
    looks = 0
    try:
        while True:
            looks += 1
            version, found = await look()
            # [LAW:no-ambient-temporal-coupling] the page says what it has seen, so a change made between its reads is
            # never missed: it is answered at once with whatever has changed since.
            if version != seen or clock.now() - began >= WAIT_SECONDS:
                break
            await clock.sleep(POLL_SECONDS)
    finally:
        # [LAW:nothing-unseen] a look that fails partway through the wait still says how far the wait had come.
        annotate(seen=seen, waited_ms=round((clock.now() - began) * 1000, 1), looks=looks)
    annotate(changed=version != seen)
    return version, found
