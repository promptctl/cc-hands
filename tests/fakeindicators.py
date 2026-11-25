"""Indicators as values: each shown one a pid, and none started (hands.daemon.indicatorprocess.Indicators)."""

from itertools import count

from hands.clock import Clock
from hands.daemon.indicatorprocess import Retired
from hands.sessions.home import Home

# The pid the first indicator shown is given, and each after it the next.
FIRST = 42


class FakeIndicators:
    """Each shown indicator runs until it is retired, which ends it; one that is not running, as one the run before
    reaped, is found reaped, as the run after a restart finds it. Every home shown for is kept in `shown`, and every
    pid retired, with what became of it, in `retired`."""

    def __init__(self) -> None:
        self.running: set[int] = set()
        self.shown: list[Home] = []
        self.retired: list[tuple[int, Retired]] = []
        self._pids = count(FIRST)

    def show(self, home: Home) -> int:
        shown = next(self._pids)
        self.running.add(shown)
        self.shown.append(home)
        return shown

    def retire(self, indicator: int, clock: Clock) -> Retired:
        became: Retired = "ended" if indicator in self.running else "reaped"
        self.running.discard(indicator)
        self.retired.append((indicator, became))
        return became
