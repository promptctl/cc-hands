"""`hands run` under its watchdog, which ends whatever process it is in, so it is watched here and not in the test's.

    python watched_run.py <home> <deadline seconds> hangs|stalls|returns|beats

hangs: `hands run` beats starting, then is stuck as the daemon once was, its loop's thread waiting on the GIL a thread in
a backtracking regex holds. stalls: `hands run` is stuck the same way at its door, before its first beat. returns:
`hands run` beats its refusal and returns, then the process idles past the deadline, exiting 10 more than the run's code.
beats: a watchdog kicked well within its deadline, past two deadlines.
Only the door and the run's launch are stood in for, so the heartbeat and watchdog are the ones `hands run` makes.
"""

import re
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

from hands.daemon import cli, watchdog
from hands.daemon.starting import CannotStart
from hands.sessions.home import Home

# How often `beats` kicks: a twentieth of the deadline, so only a stall of nearly all of it ends the run.
KICKS_PER_DEADLINE = 20


def stuck_in_a_regex() -> None:
    # Nested quantifiers on a subject that fails to match at its end: 2**40 ways to try, never done in a test's time.
    re.search(r"(a+)+$", "a" * 40 + "b")


def stuck() -> None:
    threading.Thread(target=stuck_in_a_regex, name="worker").start()
    while True:
        time.sleep(0.05)


def hung(*arguments: object) -> None:
    heart = arguments[4]
    assert isinstance(heart, cli.heartbeat.Heart)
    heart.beat("starting", None, 0, listening=False, degraded=())
    stuck()


def stalled_at_the_door(*_arguments: object) -> None:
    stuck()


def opened(*_arguments: object) -> None:
    """The door, opened onto settings the stood-in run never reads."""


def refused(*_arguments: object) -> None:
    raise CannotStart("refused for the test")


def main() -> None:
    home, deadline, mode = Home(Path(sys.argv[1])), timedelta(seconds=float(sys.argv[2])), sys.argv[3]
    watchdog.HUNG = deadline
    match mode:
        case "hangs":
            setattr(cli, "door", opened)
            setattr(cli, "run_here", hung)
            sys.exit(cli.main(["--home", str(home.root), "run"]))
        case "stalls":
            setattr(cli, "door", stalled_at_the_door)
            sys.exit(cli.main(["--home", str(home.root), "run"]))
        case "returns":
            # A restart's run holds the heartbeat from the outset, so its refusal beats, and kicks the watchdog.
            setattr(cli, "door", refused)
            code = cli.main(["--home", str(home.root), "run", "--restarted", "0"])
            time.sleep(deadline.total_seconds() * 2)
            sys.exit(10 + code)
        case "beats":
            with watchdog.watching(home.hang, deadline) as watched:
                for _ in range(2 * KICKS_PER_DEADLINE):
                    watched.kick()
                    time.sleep(deadline.total_seconds() / KICKS_PER_DEADLINE)
        case other:
            raise SystemExit(f"no such mode: {other}")


main()
