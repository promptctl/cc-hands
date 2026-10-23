"""A page's read that waits for what the page shows to change before it answers: the conversation page's
(`hands.voice.conversationpage`) and the sessions on the phone's page (`hands.voice.sessionspage`) alike.

[LAW:one-type-per-behavior] each page marks what it shows, and says with each read the mark it was last given; the one
loop here waits on any mark.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from hands.sessions.wide import Fact, annotate

# How long a read waits for what its page shows to change before answering with it as it is, well inside the minute a
# browser or a proxy between gives a request; and how often it looks meanwhile.
WAIT_SECONDS = 25.0
POLL_SECONDS = 0.25


async def waited[M: Fact, T](look: Callable[[], Awaitable[tuple[M, T]]], seen: M | None) -> tuple[M, T]:
    """What `look` finds once its mark is not `seen`, or as it is once WAIT_SECONDS have passed; a page that has seen
    nothing (None) is answered at once. The event of the unit open here says what the page had seen, how long it waited
    and how many looks it took, and whether what it found had changed.
    """
    began = time.monotonic()
    mark, found = await look()
    looks = 1
    # [LAW:no-ambient-temporal-coupling] the page says what it has seen, so a change made between its reads is never
    # missed: it is answered at once with whatever has changed since.
    while mark == seen and time.monotonic() - began < WAIT_SECONDS:
        await asyncio.sleep(POLL_SECONDS)
        mark, found = await look()
        looks += 1
    annotate(seen=seen, waited_ms=round((time.monotonic() - began) * 1000, 1), looks=looks, changed=mark != seen)
    return mark, found
