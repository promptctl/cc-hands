"""The session the brain's Claude Code holds now: the one thing that says which traffic on the wire, and which hook at its
listener, is the brain's."""

from dataclasses import dataclass

from hands.core.session import SessionId


@dataclass
class BrainSession:
    """[LAW:one-source-of-truth] the brain's session, from before its spawn, as hands chose it, to its end. Its one writer
    is the brain's SessionStart (`Brain.move`): a /clear or /resume moves the process to another session, and everything
    that reads the brain's traffic reads this, at the moment it reads, and keeps no copy."""

    # [LAW:types-are-the-program] never None: hands names the session it launches in (--session-id or --resume), so the
    # brain holds one from the first request on.
    session: SessionId
