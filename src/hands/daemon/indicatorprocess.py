"""The menu-bar indicator's process: started for each run, waited on while it runs, and ended by the run after a restart,
so the one shown is always the running run's own."""

import os
import queue
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Literal, Protocol

from loguru import logger

from hands.clock import Clock
from hands.daemon.starting import invocation
from hands.sessions import child
from hands.sessions.home import Home
from hands.sessions.processes import Pid, parse_pid

# What became of the indicator the run before a restart showed: ended as this run asked, killed by it once it had not
# within the grace, found exited on its own and said, or reaped already by the run before, which said so as it did.
type Retired = Literal["ended", "killed", "exited", "reaped", "unreaped"]

# How long the indicator has to end once asked before its process group is killed.
RETIRE_GRACE_SECONDS = 1.0


class Indicators(Protocol):
    """[LAW:effects-at-boundaries] what starts and ends the indicator each run shows: SpawnedIndicators, and a fake of it
    in tests (tests/fakeindicators.py)."""

    def show(self, home: Home) -> Pid:
        """The indicator for this run, started and waited on, so one that exits early is reaped and said: its pid, which
        a restart hands to the run after it."""
        ...

    def retire(self, indicator: Pid, clock: Clock) -> Retired:
        """End the indicator the run before a restart showed, so the one this run starts is the only one in the menu bar."""
        ...


@dataclass(frozen=True)
class SpawnedIndicators:
    """The Indicators hands runs on: each a process of its own, as AppKit wants a main thread and this process's is the
    daemon's. Each is started apart from Children, whose children are supervised by a thread of the process that starts
    them, which a restart's exec does not keep. The indicator outlives that exec: a restart keeps the pid, so it stays a
    child of the run after it, which ends it by that pid."""

    grace: float = RETIRE_GRACE_SECONDS

    def show(self, home: Home) -> Pid:
        argv = invocation(home, "indicator", "--parent", str(os.getpid()))
        # [LAW:single-enforcer] exception: started, waited on and ended apart from hands.sessions.child, which cannot yet
        # hold a child across the restart's exec, until promptctl-testing-c3u.3pn. The indicator runs as long as the run
        # does and ends itself once the run that started it is gone (menubar.show), however the run ended; a session of
        # its own keeps the terminal's Ctrl-C and hangup from ending it first, before it has said so. Its output shares this terminal, so an indicator that fails is seen where the daemon's own are.
        shown = os.posix_spawn(sys.executable, argv, os.environ, file_actions=[(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0)], setsid=True)
        threading.Thread(target=_reap, args=(shown,), name="indicator", daemon=True).start()
        return parse_pid(shown)

    def retire(self, indicator: Pid, clock: Clock) -> Retired:
        try:
            exited, status = os.waitpid(indicator, os.WNOHANG)
        except ChildProcessError:
            return "reaped"
        if exited:
            _said_exited(status)
            return "exited"
        # Still this process's child and unreaped, so the pid is the indicator's and no other process's. Its group goes
        # with it, as hands.sessions.child kills the group of a child it started.
        _signalled(indicator, signal.SIGTERM)
        statuses: queue.SimpleQueue[int] = queue.SimpleQueue()
        threading.Thread(target=lambda: statuses.put(os.waitpid(indicator, 0)[1]), name="retire", daemon=True).start()
        try:
            ended = clock.take(statuses, self.grace)
        except queue.Empty:
            _signalled(indicator, signal.SIGKILL)
            try:
                ended = clock.take(statuses, child.KILL_GRACE_SECONDS)
            except queue.Empty:
                # [LAW:no-silent-failure] killed, and not dead: the start goes on without it, and its event says so.
                logger.error(f"killed the menu-bar indicator ({indicator}), and it was still not reaped {child.KILL_GRACE_SECONDS:.1f}s later")
                return "unreaped"
        # [LAW:no-silent-failure] the status says how it ended, not the asking: one that crashed as it was asked is said.
        code = os.waitstatus_to_exitcode(ended)
        if code in (0, -signal.SIGTERM):
            return "ended"
        if code == -signal.SIGKILL:
            logger.info(f"killed the menu-bar indicator ({indicator}) and everything it started, because it had not ended {self.grace:.1f}s after it was asked to")
            return "killed"
        _said_exited(ended)
        return "exited"


def _signalled(group: int, signum: signal.Signals) -> None:
    """Send `signum` to the process group `group` leads, where any of it has yet to exit."""
    try:
        os.killpg(group, signum)
    except (ProcessLookupError, PermissionError):
        # [LAW:no-silent-failure] exception: until promptctl-testing-c3u.3pn, a refusal is taken for a group that had
        # exited: gone, ProcessLookupError; exited and not yet reaped, macOS says PermissionError. The wait that follows
        # reaps the leader all the same.
        pass


def _reap(shown: int) -> None:
    """Wait on the indicator, so one that exits early is reaped and said, not left a zombie under the run."""
    # It exits of its own accord only once the run is gone, so an exit this process lives to see is a failure.
    _, status = os.waitpid(shown, 0)
    _said_exited(status)


def _said_exited(status: int) -> None:
    logger.error(f"the menu-bar indicator exited ({os.waitstatus_to_exitcode(status)}) while hands runs; hands is not shown in the menu bar")
