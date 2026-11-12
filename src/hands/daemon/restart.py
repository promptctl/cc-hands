"""`hands restart`: start the running daemon again, so it runs the code, prompt, and brain setup on disk now.

The plugin's /hands:restart skill runs it as the installed hands' own command.

A change to hands is never taken up live: the brain's prompt is hands' source, given to the brain when it launches; the
brain's setup in its config directory is read by the brain's Claude Code when it launches; and a skill in the plugin is
loaded by the session that uses it (on /reload-plugins), never by the daemon. So asking always restarts. The daemon
stops as `q` stops it and starts again in the same process (hands.daemon.starting.again), and this waits until the
heartbeat says the new run's pipeline is running.
"""

import os
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from hands.clock import Clock
from hands.sessions import heartbeat
from hands.sessions.home import Home

# [LAW:one-source-of-truth] the one signal both ends mean "restart" by: the daemon's handler and this sender.
RESTART_SIGNAL = signal.SIGUSR1
# How long the new run has to say its pipeline is running: its start loads the speech models. Well under the two
# minutes a Claude Code command is given by default, so the deadline is said rather than cut off.
BACK_WITHIN = timedelta(seconds=90)
# How often the heartbeat is looked at while waiting.
LOOK_SECONDS = 0.2
# How often a command gone on in a process of its own looks whether the one it left has exited.
PARENT_LOOK_SECONDS = 0.01


@dataclass(frozen=True)
class Restarted:
    status: heartbeat.Status
    took: timedelta


@dataclass(frozen=True)
class NotRunning:
    """Nothing was asked: hands is not running with its pipeline up, which is the only run whose restart is heard."""

    verdict: heartbeat.Verdict


@dataclass(frozen=True)
class NotBack:
    """hands was asked, and the heartbeat has not said a new run is running: it failed, or is still starting."""

    verdict: heartbeat.Verdict
    waited: timedelta


Outcome = Restarted | NotRunning | NotBack


def restart(home: Home, now: Callable[[], datetime], wait: Callable[[], None], within: timedelta = BACK_WITHIN) -> Outcome:
    """Ask the daemon to restart and wait for the run it starts again to be running."""
    asked = now()
    match heartbeat.look(home.status, asked):
        # [LAW:no-ambient-temporal-coupling] only a running pipeline is asked: by then the run has long since put in
        # its handler, which a run in its first moments has not, and the signal's default would end it.
        case heartbeat.Up(status=before) if before.pipeline == "running":
            pass
        case verdict:
            return NotRunning(verdict)
    try:
        os.kill(before.pid, RESTART_SIGNAL)
    except ProcessLookupError:
        # It ended between the look and the ask: what its heartbeat says now is why it was not restarted.
        return NotRunning(heartbeat.look(home.status, now()))
    while True:
        at = now()
        match heartbeat.look(home.status, at):
            # The new run's heartbeat is told from the old one's by when it started: the pid is the same.
            case heartbeat.Up(status=status) if status.started_at > before.started_at and status.pipeline == "running":
                return Restarted(status, at - asked)
            # Gone, or stopped, or unreadable: nothing more is coming, so it is said now rather than at the deadline.
            case heartbeat.Down() | heartbeat.Stopped() | heartbeat.Refused() | heartbeat.Unreadable() | heartbeat.NeverRan() as verdict:
                return NotBack(verdict, at - asked)
            # The old run stopping, the new one starting, or a moment unheard between the two.
            case verdict if at - asked >= within:
                return NotBack(verdict, at - asked)
            case _:
                wait()


def said(outcome: Outcome, now: datetime) -> str:
    match outcome:
        case Restarted(status=status, took=took):
            return f"hands restarted: pid {status.pid} is running again after {took.total_seconds():.0f}s, with {heartbeat.live_sessions(status)}."
        case NotRunning(verdict=verdict):
            return f"hands was not restarted: {heartbeat.describe(verdict, now)}."
        case NotBack(verdict=verdict, waited=waited):
            return f"hands was asked to restart {waited.total_seconds():.0f}s ago and is not running again: {heartbeat.describe(verdict, now)}."


def detach(clock: Clock) -> Callable[[int], bool]:
    """Go on in a process of its own, outside the caller's process tree, session, and process group; and what answers
    the caller, still waiting in the process it started, with the exit to end with.

    The brain runs `hands restart` and `hands update` from its shell, and the restart stops the brain: Claude Code ends
    the commands it ran, the brain's terminal hangs up on its session, and hands kills the brain's process group. The
    command outlives all three only once it is in a session of its own and launchd has taken it in, so it asks for the
    restart, waits for it, and writes its event only after that. What it prints after this still goes where its
    output went, and the process the caller started waits for its answer and exits with it: a caller the restart does
    not end, a terminal, has the outcome before its prompt, and its exit.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    read, write = os.pipe()
    middle = os.fork()
    if middle != 0:
        os.close(write)
        # The middle process exits at once, leaving the command to launchd.
        os.waitpid(middle, 0)
        with os.fdopen(read, "rb") as answers:
            answer = answers.read()
        if len(answer) != 1:
            # [LAW:no-silent-failure] the command ended, a crash or a kill, before it said how its restart went.
            print("hands: the command ended before it said how hands' restart went; `hands status` says whether hands is running", file=sys.stderr, flush=True)
        # The process the caller started ends here, with nothing of Python's exit: the unit of work open in it is the
        # detached command's to end, and its event its to write.
        os._exit(answer[0] if len(answer) == 1 else 1)
    os.close(read)
    os.setsid()
    os.dup2(os.open(os.devnull, os.O_RDONLY), sys.stdin.fileno())
    middle = os.getpid()
    if os.fork() != 0:
        os._exit(0)
    # [LAW:no-ambient-temporal-coupling] out of the caller's tree once its parent has exited, not after a guess at how long that takes.
    while os.getppid() == middle:
        clock.rest(PARENT_LOOK_SECONDS)
    return lambda code: reply(write, code)


def reply(write: int, code: int) -> bool:
    """Give the caller waiting at `write` the exit `code`; False when it is gone, ended by the restart."""
    try:
        os.write(write, bytes([code]))
    except BrokenPipeError:
        return False
    finally:
        os.close(write)
    return True
