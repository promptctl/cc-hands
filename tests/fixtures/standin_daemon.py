"""A daemon with no voice: hands' own launch, restart, session sweep, and heartbeat, around a run that only lists sessions.

    python standin_daemon.py <home> <claude> [--restarted]

It stands in for `hands run`, which needs a microphone, the talk key's grant, and the speech models, so that a test
can restart a running daemon through the plugin, or by editing its settings, and see what the run after it knows. Its
brain's pinned claude is <claude>, a test's script, so the start's preflight an edit is weighed by starts no Claude Code.
"""

import asyncio
import hashlib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from hands.brain import pinned
from hands.clock import SYSTEM
from hands.daemon.cli import launch
from hands.daemon.config import edited, load
from hands.daemon.starting import Ended, Start, Stop, execv, keep_beating
from hands.sessions import heartbeat
from hands.sessions.audit import AuditLog
from hands.sessions.home import Home
from hands.sessions.liveness import sweep
from hands.sessions.registry import Sessions


def main() -> None:
    home, claude = Home(Path(sys.argv[1])), Path(sys.argv[2])
    # As tests/conftest.py pins it in a test's own process, whatever directory the brain is on.
    pinned.claude = lambda _config_dir: claude
    with claude.open("rb") as held:
        pinned.CHECKSUMS = dict.fromkeys(pinned.CHECKSUMS, hashlib.file_digest(held, "sha256").hexdigest())
    run_start = Start(restarted="--restarted" in sys.argv[3:], clock=SYSTEM)
    heart = heartbeat.Heart(home.status, os.getpid(), datetime.now(UTC), heartbeat.HEARTBEAT, kick_watchdog=lambda: None, clock=SYSTEM)
    heart.beat("starting", None, 0, listening=False, degraded=())
    audit = AuditLog(home.audit, clock=lambda: datetime.now(UTC))
    settings = load(home)

    async def run(stop: Stop) -> Ended:
        sessions = Sessions(clock=SYSTEM, record=audit.record)
        # As hands' run does before its models load: the sessions already running are listed from the home.
        await sweep(home, sessions, frozenset())
        beating = asyncio.create_task(keep_beating(lambda: heart.beat("running", None, sessions.live_count(), listening=False, degraded=()), heart.period.total_seconds(), SYSTEM))
        # As hands' run is once its pipeline has started.
        run_start.ended(audit.record, None)
        await stop.wait()
        beating.cancel()
        return Ended(None, sessions.live_count())

    match asyncio.run(launch(lambda: run, heart, lambda: (), lambda: edited(home, audit.record, settings, SYSTEM, period=0.05), audit.record, run_start, SYSTEM)):
        case "quit":
            return
        case "restart":
            # [LAW:one-source-of-truth] the run after is on the home and the claude this one is.
            execv([sys.executable, __file__, *sys.argv[1:3], "--restarted"])


if __name__ == "__main__":
    main()
