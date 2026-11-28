"""The one thread the process runs MLX on, and the token that proves a piece of work is running there.

MLX is not thread safe (ml-explore/mlx#2133): its scheduler, the Metal command encoders and kernel cache, and its
allocator's counters are the process's, unguarded, and 0.32 keeps a default stream and a compile cache for each thread
(#4377). A lock around evaluation is not enough either: an array made outside it allocates, and races the evaluation's
reads (#4835, filed against the 0.32.4 pyproject pins). Two threads in MLX at once killed the daemon five times, with a
null pointer read in `eval_impl`: as hands started, the desk's Smart Turn and the phone's each warmed up on an executor
thread of its own (promptctl-voice-oalb.8vn).

So MLX is confined to one thread. Every model's load, and every graph a model builds and evaluates, runs here, one
piece of work at a time in the order it was given, so each model's own order is kept by the one line. An array is
freed wherever its last reference goes, and an evaluated array's buffer goes back through MLX's allocator, which takes
its own lock.
"""

import asyncio
from collections.abc import Callable
from typing import final

from hands.threads import SerialThread


@final
class OnMLX:
    """Proof that the work holding it runs on the MLX thread: the thread hands one to each piece of work it runs, and
    [LAW:types-are-the-program] every function that runs MLX asks for one, so code that would run a model on a thread
    of its own does not type-check. A test whose models are fakes makes its own."""


# [LAW:no-shared-mutable-globals] the thread is the process's, as MLX's state is, so a restart in the same process keeps
# it, and this module is its one owner: `give`, `settles`, `run` and `call` hand it work, and nothing else does.
_THREAD = SerialThread("MLX")
_ON = OnMLX()


def give(work: Callable[[OnMLX], None]) -> None:
    """Queue work on the MLX thread behind everything given before it, and return without waiting for it to run."""
    _THREAD.give(lambda: work(_ON))


def settles[T](work: Callable[[OnMLX], T]) -> asyncio.Future[T]:
    """A future settled with work's outcome once the MLX thread has run it, queued now behind everything given before it."""
    return _THREAD.settles(lambda: work(_ON))


async def run[T](work: Callable[[OnMLX], T]) -> T:
    """The result of work, once the MLX thread has run it behind everything given before it."""
    return await settles(work)


def call[T](work: Callable[[OnMLX], T]) -> T:
    """The result of work, once the MLX thread has run it behind everything given before it, waited for by a thread
    that cannot await: one that builds the voice, or Pipecat's executor asking Smart Turn for a verdict."""
    return _THREAD.call(lambda: work(_ON))


def waiting() -> int:
    """How many pieces of MLX work wait behind the one running."""
    return _THREAD.waiting
