"""The keyboard edges: Right Shift and Right Command, from any app, reach the desk (`hands.voice.desk`); `q` typed in
hands' own terminal quits."""

import asyncio
import sys
import termios
import tty
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager

from hands.voice import talkkey
from hands.voice.hold import KeyEvent, asked

QUIT = "q"


@asynccontextmanager
async def tapped(into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
    """Hand every event of the talk key to `into` on the event loop while open, each press's Ripe and Overlong too."""
    loop = asyncio.get_running_loop()

    def arrived(event: KeyEvent) -> None:
        into(event)
        # [LAW:no-ambient-temporal-coupling] what each event asks to be told later is told at its instant on the loop's own
        # clock, the monotonic one the press was stamped on; the hold is the one owner of what that means, and ignores the
        # ones a release has made stale.
        for at, later in asked(event):
            loop.call_at(at, into, later)

    def heard_on_the_tap(event: KeyEvent) -> None:
        loop.call_soon_threadsafe(arrived, event)

    stop = talkkey.tap(heard_on_the_tap)
    try:
        yield
    finally:
        stop()


@contextmanager
def raw_terminal(fd: int) -> Generator[None]:
    """Put the terminal in cbreak mode for the duration and always restore it."""
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def drive_quit(quit: Callable[[], None]) -> None:
    """Read hands' own terminal until `q`, which is told to `quit`."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    typed: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(fd, lambda: typed.put_nowait(sys.stdin.read(1)))
    try:
        with raw_terminal(fd):
            while await typed.get() != QUIT:
                pass
            quit()
    finally:
        loop.remove_reader(fd)
