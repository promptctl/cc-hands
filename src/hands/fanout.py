"""One stream told to every listener: each item published reaches each listener subscribed as it is published."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager


class Fanout[Item]:
    """Each item published, to every listener subscribed as it is, in the order published. Published on the loop; a thread
    publishes through `loop.call_soon_threadsafe`. What a listener has yet to read waits for it unbounded, so a listener
    that stops reading is to leave its subscription."""

    def __init__(self) -> None:
        self._listening: tuple[asyncio.Queue[Item], ...] = ()

    @asynccontextmanager
    async def subscribed(self) -> AsyncGenerator[AsyncIterator[Item]]:
        """Every item published from now until the subscription is left."""
        items: asyncio.Queue[Item] = asyncio.Queue()
        self._listening = (*self._listening, items)

        async def published() -> AsyncIterator[Item]:
            while True:
                yield await items.get()

        try:
            yield published()
        finally:
            self._listening = tuple(each for each in self._listening if each is not items)

    def publish(self, item: Item) -> None:
        for each in self._listening:
            each.put_nowait(item)
