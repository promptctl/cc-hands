"""Indicators as values: each shown one a pid, and none started (hands.daemon.indicatorprocess.Indicators)."""

from hands.clock import Clock
from hands.daemon.indicatorprocess import Retired
from hands.sessions.home import Home

# The pid the first indicator shown is given, and each after it the next.
FIRST = 42


class FakeIndicators:
    """Each shown indicator runs until it is retired, which ends it; one that is not running, as one the run before
    reaped, is found reaped, as the run after a restart finds it. Every home shown for is kept in `shown`, and every
    pid retired, with what became of it, in `retired`. Given `unshowable`, every show raises it, as a spawn the OS
    refuses does."""

    def __init__(self, unshowable: OSError | None = None) -> None:
        self.unshowable = unshowable
        self.shown: list[Home] = []
        self.retired: list[tuple[int, Retired]] = []

    def show(self, home: Home) -> int:
        if self.unshowable is not None:
            raise self.unshowable
        self.shown.append(home)
        return FIRST + len(self.shown) - 1

    def retire(self, indicator: int, clock: Clock) -> Retired:
        # [LAW:one-source-of-truth] running is what was shown and is not yet retired.
        running = FIRST <= indicator < FIRST + len(self.shown) and all(pid != indicator for pid, _ in self.retired)
        became: Retired = "ended" if running else "reaped"
        self.retired.append((indicator, became))
        return became
