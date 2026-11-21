"""The brain's conversation: the session its Claude Code holds, kept across hands' restarts in its config directory, and
moved when a /clear or /resume in it begins another."""

import glob
from dataclasses import dataclass
from pathlib import Path

from hands.core.session import SessionId


@dataclass(frozen=True)
class Fresh:
    """A conversation begun new, under a session chosen by hands; `untranscribed` is the one held before it, when there
    was one, which Claude Code has no transcript of: it never had a turn (Claude Code writes none until one), or its
    transcript was cleaned away."""

    # [LAW:one-source-of-truth] chosen by hands, so the brain's requests are known as its own from the first one on the
    # wire, and its hooks from the first one posted.
    session: SessionId
    untranscribed: SessionId | None


@dataclass(frozen=True)
class Resumed:
    """The conversation the brain held when hands last ran, taken up again under its own session: a restart, an upgrade,
    or a change of model goes on from what was said, compacted as Claude Code compacts any conversation."""

    session: SessionId


Conversation = Fresh | Resumed


def held(config_dir: Path) -> Path:
    """The session of the brain's conversation, written by hands alone: as each brain on `config_dir` starts, and as
    the brain moves to another (`CurrentSession.move`)."""
    return config_dir / "conversation"


def conversation(config_dir: Path, new: SessionId) -> Conversation:
    """The conversation a brain on `config_dir` starts in: the one held, while Claude Code still has its transcript, or
    else one begun under `new`."""
    session = _holding(config_dir)
    if session is None:
        return Fresh(new, None)
    # [LAW:one-source-of-truth] Claude Code's own transcript says whether there is a conversation to resume: named by
    # its session, under any project directory, as Claude Code finds the session it resumes.
    if glob.glob(glob.escape(str(config_dir / "projects")) + f"/*/{glob.escape(session)}.jsonl"):
        return Resumed(session)
    return Fresh(new, session)


def hold(config_dir: Path, session: SessionId) -> None:
    """Remember `session` as the brain's conversation, replacing whatever was held whole."""
    path = held(config_dir)
    written = path.with_name(f"{path.name}.writing")
    written.write_text(f"{session}\n")
    written.replace(path)


def let_go(config_dir: Path) -> SessionId | None:
    """Forget the brain's conversation, so the next brain on `config_dir` begins a new one: the session that was held,
    when one was."""
    session = _holding(config_dir)
    held(config_dir).unlink(missing_ok=True)
    return session


def _holding(config_dir: Path) -> SessionId | None:
    try:
        return SessionId(held(config_dir).read_text().strip())
    except FileNotFoundError:
        return None


class CurrentSession:
    """The session the brain's Claude Code holds now. [LAW:one-source-of-truth] the one copy every reader of the brain's
    hooks and requests asks as it reads: none keeps its own, so none goes on reading a conversation the brain has left."""

    def __init__(self, config_dir: Path, session: SessionId) -> None:
        self._config_dir = config_dir
        self._session = session

    @property
    def session(self) -> SessionId:
        return self._session

    def move(self, to: SessionId) -> None:
        """The brain's Claude Code holds `to` now, as its SessionStart said: held for the next brain too, so a restart
        takes up the conversation the brain moved to."""
        hold(self._config_dir, to)
        self._session = to
