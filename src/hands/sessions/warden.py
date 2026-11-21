"""The process that ends hands' children once hands has ended, however it ended.

hands starts this once, on its first child, in a session of its own, reading the end of a pipe whose other end hands
alone holds. hands writes `+<pgid>` once a child's process group exists and `-<pgid>` once it has killed what was left
of it. The kernel closes hands' end when hands exits, is killed, crashes, or execs in its own place, since the end is
close-on-exec; this reads that close as the end of its input, kills every group still recorded, and writes one event
saying how many it killed, zero among them. An exec that fails closes nothing, so nothing is killed.

[LAW:no-ambient-temporal-coupling] the end of hands is the kernel's to report, so no Python running in hands as it ends
is needed for its children to end with it: SIGKILL runs none, a signal handler can be taken away, and an exec can fail
after a sweep that cannot be undone.
"""

import os
import signal
import sys
from pathlib import Path


def kill(groups: frozenset[int]) -> int:
    """SIGKILL each of `groups`; how many still had a process in them."""
    killed = 0
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError):
            # Gone, or nothing left in it but processes that have exited, which macOS refuses to signal.
            pass
    return killed


def after(groups: frozenset[int], line: bytes) -> frozenset[int]:
    """The groups recorded once `line` of the pipe is read: `+<pgid>` adds one, `-<pgid>` takes it away."""
    match line[:1], int(line[1:]):
        case b"+", group:
            return groups | {group}
        case b"-", group:
            return groups - {group}
        case _:
            raise ValueError(f"the warden was written {line!r}, which is neither +<pgid> nor -<pgid>")


def main(audit: Path, hands: int) -> None:
    from hands.clock import SYSTEM
    from hands.sessions import wide
    from hands.sessions.audit import AuditLog

    # [LAW:effects-at-boundaries] this process's one binding of the machine's clock.
    clock = SYSTEM
    # Its event is timed from its start, so its duration is how long the hands it watched ran with children.
    began = wide.begun(clock)
    groups: frozenset[int] = frozenset()
    # One line at a time, as hands writes them: a write under PIPE_BUF bytes arrives whole.
    for line in sys.stdin.buffer:
        groups = after(groups, line.strip())
    killed = kill(groups)
    wide.ended("hands.warden", AuditLog(audit, clock=clock.wall).record, began, None, None, hands=hands, recorded=len(groups), killed=killed)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
