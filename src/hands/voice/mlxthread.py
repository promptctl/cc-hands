"""The one thread the process runs MLX on, and the token that proves a piece of work is running there.

MLX is not thread safe (ml-explore/mlx#2133): its scheduler, the Metal command encoders and the kernel cache are the
process's, unguarded, and 0.32 keeps a default stream and a compile cache for each thread (#4377). A lock around evaluation
is not enough either: an array made outside it allocates, and races the evaluation's reads (#4835, filed against the 0.32.4
pyproject pins). Two threads in MLX at once killed the daemon five times, with a null pointer read in `eval_impl`: as hands
started, the desk's Smart Turn and the phone's each warmed up on an executor thread of its own (promptctl-voice-oalb.8vn).

So MLX is confined to one thread. Every model's load, and every graph a model builds, evaluates and lets go of, runs here,
one piece of work at a time in the order it was given, and nothing else does: a model's weights are fetched before its load
is given, and a model's own order is kept by whoever gives its work. Work from different models runs in the order it comes,
and nothing depends on that order. A weight let go of anywhere else goes back through the allocator's `free`, which holds
the allocator's mutex (mlx/backend/metal/allocator.cpp at v0.32.4, the release pyproject.toml pins for it); its counters
are written under that mutex and read without it, so MLX's split of the memory it holds is read here too.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable
from typing import final

from hands.threads import SerialThread


@final
class OnMLX:
    """The token every function that runs MLX asks for, made only on the MLX thread. [LAW:single-enforcer] making one is the
    one check of the thread, and it raises anywhere else, so MLX asked for with a token of its own from another thread
    fails before it begins. The thread hands one to each piece of work it runs, for that piece alone: a token kept past its
    piece is checked nowhere it is used again. Code Pipecat calls there makes its own."""

    def __init__(self) -> None:
        if not _THREAD.here():
            raise RuntimeError(f"MLX runs only on the MLX thread, and this is {threading.current_thread().name}")


# [LAW:no-shared-mutable-globals] the thread is the process's, as MLX's state is: a second one anywhere in the process
# would be two threads in MLX again. This module is its one owner: `settles`, `call` and EXECUTOR hand it work, and
# nothing else does.
_THREAD = SerialThread("MLX")


def settles[T](work: Callable[[OnMLX], T]) -> asyncio.Future[T]:
    """A future settled with work's outcome once the MLX thread has run it, queued now behind everything given before it."""
    return _THREAD.settles(lambda: work(OnMLX()))


def call[T](work: Callable[[OnMLX], T]) -> T:
    """The result of work, once the MLX thread has run it behind everything given before it, waited for by a thread that
    cannot await: the one that builds the voice, or the line a model's own order is kept on."""
    return _THREAD.call(lambda: work(OnMLX()))


def waiting() -> int:
    """How many pieces of MLX work wait behind the one running."""
    return _THREAD.waiting


@final
class _Executor(concurrent.futures.Executor):
    """The MLX thread as an executor, for a library that runs its model on one it is given."""

    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> concurrent.futures.Future[T]:
        return _THREAD.submit(lambda: fn(*args, **kwargs))


# Pipecat's Smart Turn analyzer judges each segment on the executor it is given (hands.voice.engaged.SmartTurn).
EXECUTOR = _Executor()
