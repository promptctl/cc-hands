"""Cancelling asyncio work hands started, and waiting for it to end."""

import asyncio
from typing import Any


async def cancel_and_wait(task: asyncio.Future[Any], *tasks: asyncio.Future[Any]) -> None:
    """Cancel `task` and `tasks`, and return once every one is done: what any raised as it ended is its done callbacks' to
    tell, or its caller's to read, before this returns, unless this wait is itself cancelled. Each is cancelled in the
    order given; one already done is waited on as done."""
    every = list(dict.fromkeys((task, *tasks)))
    for each in every:
        each.cancel()
    await asyncio.wait(every)
