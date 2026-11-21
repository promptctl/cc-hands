"""Which session the brain's Claude Code is in now."""

from collections.abc import Callable
from typing import Protocol

from hands.core.session import SessionId


class SessionView(Protocol):
    """[LAW:single-enforcer] BrainSession as a reader holds it: read and watched, never moved."""

    def id(self) -> SessionId: ...
    def watch(self, moved: Callable[[], None]) -> None: ...


class BrainSession:
    """[LAW:one-source-of-truth] which session the brain's Claude Code is in now: the conversation it launches in, until
    a /clear or a /resume at its keyboard moves it. The brain's start step is its one writer (`Brain._entered`). A reader
    reads `id` as it is when it reads, and one that waits on the conversation the brain is in is told of each move.

    [LAW:no-ambient-temporal-coupling] a move is told to every watcher in the step it is made, so a wait on the session
    the brain is in wakes as it moves, whatever else it waits on."""

    def __init__(self, session: SessionId) -> None:
        self._id = session
        self._watching: list[Callable[[], None]] = []

    def id(self) -> SessionId:
        return self._id

    def watch(self, moved: Callable[[], None]) -> None:
        """Has `moved` called as the brain enters each session from now on."""
        self._watching.append(moved)

    def enter(self, session: SessionId) -> SessionId:
        """Moves the brain to `session`, and returns the one it was in."""
        left, self._id = self._id, session
        for moved in self._watching:
            moved()
        return left
