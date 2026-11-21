"""The sides of one reading, run together: how a reading asks several questions at once and waits out every answer."""

import asyncio
from collections.abc import Awaitable
from typing import overload


@overload
async def settled[A, B](first: Awaitable[A], second: Awaitable[B], /) -> tuple[A, B]: ...
@overload
async def settled[A, B, C](first: Awaitable[A], second: Awaitable[B], third: Awaitable[C], /) -> tuple[A, B, C]: ...
@overload
async def settled[T](*sides: Awaitable[T]) -> tuple[T, ...]: ...
async def settled(*sides: Awaitable[object]) -> tuple[object, ...]:
    """Every side of a reading run together and waited out, then what each gave; where any raised, all they raised.

    [LAW:single-enforcer] the one way a reading runs its sides at once, each bounded by the reading's deadline. Waited
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
