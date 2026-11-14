"""A page's read that waits for what the page shows to change before it answers: the conversation page's
(`hands.voice.conversationpage`) and the brain on the phone's page (`hands.voice.brainpage`) alike.

[LAW:one-type-per-behavior] each page versions what it shows, and says with each read the version it was last given; the one
loop here waits on any version.
"""

from collections.abc import Awaitable, Callable

from hands.clock import Clock
from hands.sessions.wide import Fact, annotate

# How long a read waits for what its page shows to change before answering with it as it is, well inside the minute a
# browser or a proxy between gives a request; and how often it looks meanwhile.
WAIT_SECONDS = 25.0
POLL_SECONDS = 0.25


async def waited[V: Fact, T](clock: Clock, look: Callable[[], Awaitable[tuple[V, T]]], seen: V | None) -> tuple[V, T]:
    """What `look` finds once its version is not `seen`, or as it is once WAIT_SECONDS have passed; a page that has seen
    nothing (None) is answered at once. The event of the unit open here says what the page had seen, how long it waited
    and how many looks it took, and whether what it found had changed.
    """
    began = clock.now()
    version, found = await look()
    looks = 1
    # [LAW:no-ambient-temporal-coupling] the page says what it has seen, so a change made between its reads is never
    # missed: it is answered at once with whatever has changed since.
    while version == seen and clock.now() - began < WAIT_SECONDS:
        await clock.pause(POLL_SECONDS)
        version, found = await look()
        looks += 1
    annotate(seen=seen, waited_ms=round((clock.now() - began) * 1000, 1), looks=looks, changed=version != seen)
    return version, found
