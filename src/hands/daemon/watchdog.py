"""The run's watchdog timer: a run whose heartbeat stops for HUNG ends itself, saying where every thread was, rather than
stay up unable to hear, so whatever supervises hands, where something does, can start it again.

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

from loguru import logger

# How long a run may go without a heartbeat before it ends itself. Well past the stalls a working run recovers from,
# which the indicator shows as not responding after three missed beats, and past the steps of a run that no beater
# covers, the brain's launch and stop and the exporter's last flush among them, each bounded in seconds; a voice app
# frozen this long is gone to its user, and a start again costs seconds.
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


@contextmanager
def watching(path: Path, after: timedelta) -> Generator[Watchdog]:
    """The watchdog for one run, dumping into `path`, emptied for it; armed as the run begins, and disarmed as it ends.

    [LAW:nothing-unseen] a run its watchdog ended wrote nothing else as it went: what it dumped is logged as an error, on
    the terminal and in the audit log, before the file is emptied for this run, so a run that hangs before it is ready
    still leaves the hang before it on the log. [LAW:no-ambient-temporal-coupling] the timer is the process's, so its
    life is bound to the run that owns it: armed as the run begins, so a run stuck before its first beat ends too, and
    cancelled as it ends. A restart execs inside it, which ends the timer's thread and closes the dump with the process
    image it belonged to.
    """
    try:
        # The run before's tracebacks where its watchdog ended it, and nothing where it did not.
        tracebacks = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        tracebacks = ""
    if tracebacks:
        logger.error(f"the run before stopped beating, and its watchdog ended it; where every thread was:\n{tracebacks}")
    with path.open("w", encoding="utf-8") as dump:
        watchdog = Watchdog(after, dump)
        try:
            watchdog.kick()
            yield watchdog
        finally:
            faulthandler.cancel_dump_traceback_later()
