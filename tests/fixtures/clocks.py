"""Clocks a test and a child process it starts alike run a command on."""

from hands.virtualclock import VirtualClock


class Interrupting(VirtualClock):
    """A clock whose first blocking sleep is a Ctrl-C, as `hands log` is stopped while it follows: run on it, a command
    that waits is stopped where it would wait, and one that never waits is not touched."""

    def sleep_blocking(self, seconds: float) -> None:
        raise KeyboardInterrupt
