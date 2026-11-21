"""Ending asyncio work hands started and no longer waits on."""

import asyncio
from collections.abc import Collection


async def cancelled[T](tasks: Collection[asyncio.Future[T]]) -> None:
    """Cancel `tasks`, and return once every one is done: what any raised as it ended is its done callbacks' to tell, or
    its caller's to read, before this returns, never after."""
    for task in tasks:
        task.cancel()
    await asyncio.wait(tasks)
