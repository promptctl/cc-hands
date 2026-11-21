"""Ending asyncio work hands started and no longer waits on."""

import asyncio
from typing import Any


async def cancelled(task: asyncio.Future[Any], *tasks: asyncio.Future[Any]) -> None:
    """Cancel `task` and `tasks`, and return once every one is done: what any raised as it ended is its done callbacks' to
    tell, or its caller's to read, before this returns, never after."""
    every = {task, *tasks}
    for each in every:
        each.cancel()
    await asyncio.wait(every)
