"""Cancelling asyncio work hands started, and waiting for it to end."""

import asyncio
from collections.abc import Awaitable
from typing import Any, overload


async def cancel_and_wait(task: asyncio.Future[Any], *tasks: asyncio.Future[Any]) -> None:
    """Cancel `task` and `tasks`, and return once every one is done: what any raised as it ended is its done callbacks' to
    tell, or its caller's to read, before this returns, unless this wait is itself cancelled. Each is cancelled in the
    order given; one already done is waited on as done."""
    every = list(dict.fromkeys((task, *tasks)))
    for each in every:
        each.cancel()
    await asyncio.wait(every)


@overload
async def settled[A, B](first: Awaitable[A], second: Awaitable[B], /) -> tuple[A, B]: ...
@overload
async def settled[A, B, C](first: Awaitable[A], second: Awaitable[B], third: Awaitable[C], /) -> tuple[A, B, C]: ...
@overload
async def settled[T](*sides: Awaitable[T]) -> tuple[T, ...]: ...
async def settled(*sides: Awaitable[object]) -> tuple[object, ...]:
    """Every side of a reading run together and waited out, then what each gave; where any raised, all they raised.

    [LAW:single-enforcer] the one way a reading runs its sides at once, each bounded by its own limit. Waited
    out, so a side that raises does not end the reading while another runs on and writes to its event after it has
    ended [LAW:no-ambient-temporal-coupling]. Not a TaskGroup, nor a gather that cancels the rest: a child told twice to
    stop leaves its git unreaped.
    """
    gave = await asyncio.gather(*sides, return_exceptions=True)
    raised = [side for side in gave if isinstance(side, BaseException)]
    # A side cancelled on its own, as the loop's shutdown cancels every task at once, is the reading cancelled: raised
    # as a cancellation, it is told as one and goes on up, where in a group it would be told as a failure.
    if any(isinstance(side, asyncio.CancelledError) for side in raised):
        raise asyncio.CancelledError
    if raised:
        raise BaseExceptionGroup("a side of the reading raised", raised)
    return tuple(gave)
