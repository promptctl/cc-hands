"""The process that ends hands' children once hands has ended, however it ended.

hands starts this on its first child, in a session of its own, reading the end of a pipe whose other end hands alone
holds. hands writes `added(pgid)` once a child's process group exists and `removed(pgid)` once it has killed what was
left of it. The kernel closes hands' end when hands exits, is killed, crashes, or execs in its own place, since the end is
close-on-exec; this reads that close as the end of its input, kills every group still recorded, and writes one event
saying how many it killed, zero among them. An exec that fails closes nothing, so nothing is killed.

[LAW:no-ambient-temporal-coupling] the end of hands is the kernel's to report, so no Python running in hands as it ends
is needed for its children to end with it: SIGKILL runs none, a signal handler can be taken away, and an exec can fail
after a sweep that cannot be undone.

[LAW:one-source-of-truth] this module is both ends of the pipe's protocol, and the one kill of a group, hands' and its own.
"""

import os
import signal
import sys
from pathlib import Path


def added(group: int) -> bytes:
    """The line that records `group` with the warden."""
    return b"+%d\n" % group


def removed(group: int) -> bytes:
    """The line that takes `group`'s record back."""
    return b"-%d\n" % group


def after(groups: frozenset[int], line: bytes) -> frozenset[int]:
    """The groups recorded once `line` of the pipe is read: an `added` line adds one, a `removed` line takes it away."""
    refused = ValueError(f"the warden was written {line!r}, which is neither +<pgid> nor -<pgid>")
    try:
        group = int(line[1:])
    except ValueError:
        raise refused from None
    match line[:1]:
        case b"+":
            return groups | {group}
        case b"-":
            return groups - {group}
        case _:
            raise refused


def killed(group: int) -> bool:
    """SIGKILL the process group `group`; whether anything in it was still running.

    [LAW:no-silent-failure] exception: macOS answers a group of nothing but exited, unreaped processes with EPERM, as it
    answers one with a live process of another user's, and no errno tells the two apart. Every child's own group is the
    first at the kill after its exit, its leader unreaped until then, so EPERM is read as nothing left; hands' children
    run as hands' user.
    """
    try:
        os.killpg(group, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main(audit: Path, hands: int) -> None:
    from hands.clock import SYSTEM
    from hands.sessions import wide
    from hands.sessions.audit import AuditLog

    # [LAW:effects-at-boundaries] this process's one binding of the machine's clock.
    clock = SYSTEM
    # Its event is timed from its start, so its duration is how long the hands it watched ran with children.
    began = wide.begun(clock)
    groups: frozenset[int] = frozenset()
    unread: list[str] = []
    try:
        # One line at a time, as hands writes them: a write under PIPE_BUF bytes arrives whole.
        for line in sys.stdin.buffer:
            try:
                groups = after(groups, line.strip())
            except ValueError as error:
                # [LAW:no-silent-failure] a line hands never writes is kept for the event, and the groups read before
                # and after it are ended all the same.
                unread.append(str(error))
    finally:
        # However the reading ended, every group still recorded is ended, and said.
        ended = sum(map(killed, groups))
        wide.ended("hands.warden", AuditLog(audit, clock=clock.wall).record, began, None, "; ".join(unread) or None, hands=hands, recorded=len(groups), killed=ended)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
