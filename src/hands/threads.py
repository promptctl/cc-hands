"""Blocking work run off the event loop on threads the process does not wait for.

asyncio.to_thread's executor threads are joined at exit, so work that never returns — a model still loading, a
write to an audio device that is gone — would hold a stopped daemon open for as long as it hangs.
"""

import asyncio
import concurrent.futures
import contextlib
import queue
import threading
from collections.abc import Callable

from loguru import logger


async def off_loop[T](work: Callable[[], T], name: str) -> T:
    """The result of work run on a daemon thread of its own."""
    return await settles_off_loop(work, name)


def settles_off_loop[T](work: Callable[[], T], name: str) -> asyncio.Future[T]:
    """A future settled with work's outcome once a daemon thread of its own has run it.

    Not a task, so the loop's shutdown does not cancel it: a caller cancelled while waiting can still wait for it.
    """
    done: concurrent.futures.Future[T] = concurrent.futures.Future()
    settled = _awaited(done)
    threading.Thread(target=_complete, args=(done, work), name=name, daemon=True).start()
    return settled


class SerialThread:
    """One daemon thread that runs the work it is given one piece at a time, in the order given.

    For work that must never overlap, such as writes to one audio stream: a caller cancelled while its work runs
    does not stop the work, and the next piece still waits for it to finish.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._queue: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while (work := self._queue.get()) is not None:
            try:
                work()
            except BaseException:
                # [LAW:single-enforcer] work given without a waiter fails here, said, and the thread outlives it, however
                # it failed: everything queued behind it, and everything given later, still runs.
                logger.exception(f"work on {self._name} failed")

    @property
    def closed(self) -> bool:
        """Whether this takes no more work: everything given before it was closed still runs."""
        return self._closed

    @property
    def waiting(self) -> int:
        """How many pieces of work given wait behind the one running."""
        return self._queue.qsize()

    def give(self, work: Callable[[], None]) -> None:
        """Queue work behind everything given before it, and return without waiting for it to run."""
        if self._closed:
            # [LAW:no-silent-failure] queued behind the end of the thread, it would never run, and a waiter never wake.
            raise RuntimeError(f"{self._name} is closed")
        self._queue.put(work)

    async def close(self) -> None:
        """Once everything given before this has run and the thread has ended, whichever close asked first."""
        self._closed = True
        self._queue.put(None)
        await off_loop(self._thread.join, f"ending {self._name}")

    async def run[T](self, work: Callable[[], T]) -> T:
        """The result of work, once everything given before it has finished and it has run."""
        return await self.settles(work)

    def here(self) -> bool:
        """Whether the thread asking is this one."""
        return threading.current_thread() is self._thread

    def submit[T](self, work: Callable[[], T]) -> concurrent.futures.Future[T]:
        """A future completed with work's outcome, the work queued now, behind everything given before it, and not run
        where the future is cancelled before its turn, as an executor's is."""
        done: concurrent.futures.Future[T] = concurrent.futures.Future()
        self.give(lambda: _complete(done, work))
        return done

    def settles[T](self, work: Callable[[], T]) -> asyncio.Future[T]:
        """A future settled with work's outcome, the work queued now, behind everything given before it: what is given
        after this returns runs after it, which a coroutine of `run` does not promise until it is first awaited. Cancelling
        it stops only the wait: the work still runs in its turn."""
        return _awaited(self.submit(work))

    def call[T](self, work: Callable[[], T]) -> T:
        """The result of work, once everything given before it has finished and it has run, waited for on the calling
        thread, which is not an event loop's: a thread that cannot await waits here."""
        if self.here():
            # [LAW:no-silent-failure] work this thread runs waiting on work queued behind it would wait forever.
            raise RuntimeError(f"work on {self._name} called for more work on {self._name}, which would wait for it forever")
        return self.submit(work).result()


def _complete[T](done: concurrent.futures.Future[T], work: Callable[[], T]) -> None:
    """[LAW:single-enforcer] the one place work's outcome is taken: run it on this thread, unless its future was cancelled
    first, as an executor's worker does, and complete the future with what it returned or raised."""
    if not done.set_running_or_notify_cancel():
        return
    try:
        result = work()
    except BaseException as error:
        done.set_exception(error)
    else:
        done.set_result(result)


def _awaited[T](done: concurrent.futures.Future[T]) -> asyncio.Future[T]:
    """A future of the running loop's, settled with `done`'s outcome once it has one: cancelling it lets go of the wait,
    never of the work."""
    loop = asyncio.get_running_loop()
    settled: asyncio.Future[T] = loop.create_future()
    done.add_done_callback(lambda done: tell(loop, _copied, done, settled))
    return settled


def _copied[T](done: concurrent.futures.Future[T], settled: asyncio.Future[T]) -> None:
    """Settle `settled` with `done`'s outcome, on the loop that awaits it, unless its waiter has gone."""
    if settled.cancelled():
        return
    match done.exception():
        case None:
            settled.set_result(done.result())
        case error:
            settled.set_exception(error)


def tell[*Args](loop: asyncio.AbstractEventLoop, callback: Callable[[*Args], object], *args: *Args) -> None:
    """Run `callback` on `loop`, from whichever thread. [LAW:single-enforcer] the one hop onto the loop that lets go of
    a loop already closed: that is only when the run has already ended, and there is nobody left to tell."""
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(callback, *args)
