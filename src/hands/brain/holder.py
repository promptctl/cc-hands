"""The brain as a reader of its wire needs it: which session's traffic is the brain's, now."""

from typing import Protocol

from typing_extensions import TypeIs

from hands.core.session import SessionId


class SessionHolder(Protocol):
    """A Claude Code process as a reader of its traffic needs it: the session it holds now, which a /clear or a /resume
    moves, or None before it has said which."""

    @property
    def session(self) -> SessionId | None: ...


def is_held(session: SessionId | None, holder: SessionHolder) -> TypeIs[SessionId]:
    """[LAW:single-enforcer] whether traffic of `session` is the holder's: of the session it holds now, never of none."""
    return session is not None and session == holder.session
