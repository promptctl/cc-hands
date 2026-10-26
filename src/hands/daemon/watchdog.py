"""The run's watchdog timer: a run whose heartbeat stops for HUNG ends itself, saying where every thread was, so whatever
supervises hands starts it again.

[LAW:one-source-of-truth] the heartbeat is the run's proof that its loop still turns, the one `heartbeat.judge` reads as
not responding; each beat kicks the watchdog, so the two cannot disagree about whether the run is making progress.
"""

import faulthandler
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TextIO

# How long a run may go without a heartbeat before it ends itself. Well past the stalls a working run recovers from,
# which the indicator shows as not responding after three missed beats; a voice app frozen this long is gone to its
# user, and a start again costs seconds.
HUNG = timedelta(seconds=60)


@dataclass(frozen=True)
class Watchdog:
    after: timedelta
    dump: TextIO  # where the tracebacks go as it ends the run

    def kick(self) -> None:
        """Push the deadline `after` from now."""
        # faulthandler's timer is a C thread that takes no GIL: a hang that holds it, as a regex backtracking in a worker
        # thread once did, runs no Python thread and no Python signal handler, SIGTERM's included, but this still fires.
        # It writes every thread's traceback to the dump and ends the process with _exit(1).
        faulthandler.dump_traceback_later(self.after.total_seconds(), exit=True, file=self.dump)


def dumped(path: Path) -> str | None:
    """What the watchdog of the run before wrote: its tracebacks where it ended that run, "" where it did not, and None
    where no run before kept one here."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


@contextmanager
def watching(path: Path, after: timedelta) -> Generator[Watchdog]:
    """The watchdog for one run, dumping into `path`, emptied for it; disarmed when the run ends.

    [LAW:no-ambient-temporal-coupling] the timer is the process's, so its life is bound to the run that owns it: armed
    by the run's first beat, cancelled as the run ends. A restart execs inside it, which ends the timer's thread and
    closes the dump with the process image it belonged to.
    """
    with path.open("w", encoding="utf-8") as dump:
        try:
            yield Watchdog(after, dump)
        finally:
            faulthandler.cancel_dump_traceback_later()
