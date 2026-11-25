"""How a live session stands: at its prompt, at a dialog, working, and the rest; and the one sentence that says it.

[LAW:one-source-of-truth] `standing` is the one place a session's registry state, its dialogs and its turn are read into
a standing. The brain's listing says it as `spoken_state`, `read_session` and `close_session` ask it whether the session is
over, and the phone's card folds and orders by it, so none of them re-derives what the others read.
"""

from dataclasses import dataclass

from hands.core import status
from hands.core.progress import Doing, said
from hands.core.session import Blocker, Held, Idle, Opened, Permission, Plan, Question, Running, Session, Unreported, delegating
from hands.core.spoken import counted
from hands.core.status import Busy, Going, Reason, Shell, Unknown, UnknownReason, Waiting


@dataclass(frozen=True)
class AtDialog:
    """At a dialog whose hook hands holds, said by what it asks, though the status saying it waits may not be read yet."""

    held: Held


@dataclass(frozen=True)
class AtUnheldDialog:
    """Claude Code says it waits at a dialog, and hands holds no hook for it: nothing hands can read says what it asks."""

    reason: Reason | UnknownReason


@dataclass(frozen=True)
class Working:
    """A turn runs: the latest thing it set out to do, once it has made a call."""

    latest: Doing | None


@dataclass(frozen=True)
class Delegating:
    """At its prompt, with only subagents it started in the background working (`delegating`)."""

    agents: int


@dataclass(frozen=True)
class ShellRunning:
    """At its prompt, with a shell command it started in the background still running."""


@dataclass(frozen=True)
class AtPrompt:
    """At its prompt with nothing out: no dialog, no turn open, no shell or subagent in the background."""


@dataclass(frozen=True)
class NotReported:
    """Heard of, with no status read for it yet."""


@dataclass(frozen=True)
class UnknownStatus:
    """Claude Code said a status this version of hands does not know, kept by its name."""

    name: str


Standing = AtDialog | AtUnheldDialog | Working | Delegating | ShellRunning | AtPrompt | NotReported | UnknownStatus


def standing(session: Session) -> Standing:
    # The dialog on its screen is the first it asked; the rest are queued beneath it.
    match next(iter(session.dialogs), None):
        case Held() as held:
            return AtDialog(held)
        case None:
            return _undialoged(session)


def _undialoged(session: Session) -> Standing:
    match session.state:
        case Unreported():
            return NotReported()
        case Idle(status=status.Idle()) if isinstance(session.turn, Opened):
            # A turn opened at its prompt, before the status that says it is busy is read.
            return Working(None)
        case Idle(status=Shell()):
            # No turn runs: it does nothing until that task ends and its notification opens one.
            return ShellRunning()
        case Idle():
            return AtPrompt()
        case Running(status=going):
            return _going(session, going)


def _going(session: Session, going: Going) -> Standing:
    match going:
        case Busy():
            match session.turn:
                case Opened(latest=Doing() as latest):
                    # [LAW:one-source-of-truth] the progress a session that is not the focus is noted with, for the brain,
                    # whose notes are this listing at the tail of its every request.
                    return Working(latest)
                case _ if delegating(session):
                    # No turn runs: a prompt typed at it runs at once, and each subagent's report opens a turn of its own.
                    return Delegating(len(session.background))
                case _:
                    return Working(None)
        case Waiting(reason=reason):
            return AtUnheldDialog(reason)
        case Unknown(name=name):
            return UnknownStatus(name)


def spoken(of: Standing) -> str:
    match of:
        case AtDialog(held=Held(on=on)):
            return _waiting_on(on)
        case AtUnheldDialog(reason=UnknownReason(name=name)):
            return f"waiting at a dialog: {name}"
        case AtUnheldDialog(reason=reason):
            return f"waiting at a dialog: {reason}"
        case Working(latest=Doing() as latest):
            return f"working; the last thing it set out to do: {said((latest,))}"
        case Working():
            return "working"
        case Delegating(agents=agents):
            return f"idle, with {counted(agents, 'subagent')} it started in the background still working"
        case ShellRunning():
            return "idle, with a shell command it started in the background still running"
        case AtPrompt():
            return "idle"
        case NotReported():
            return "not reported yet"
        case UnknownStatus(name=name):
            return f"in a state hands does not know: {name}"


def spoken_state(session: Session) -> str:
    """How a live session stands, in the one sentence the brain's listing says of it."""
    return spoken(standing(session))


def _waiting_on(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return f"waiting for permission to use {tool}"
        case Question():
            return "waiting for the user to answer its question"
        case Plan():
            return "waiting for the user to approve its plan"
