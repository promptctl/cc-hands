"""How a live session stands: at its prompt, at a dialog, working, and the rest; and the one sentence that says it.

[LAW:one-source-of-truth] `standing_of` is the one place a session's registry state, its dialogs and its turn are read into
a standing, and `phase` the one place a standing says where the session's turn is. The brain's listing says it as
`spoken_state`; `read_session`, `close_session`, the keyboard and the draft send read it; none re-derives what the others
read.

What the hooks tell hands outruns the status it reads: Claude Code sets the status before a prompt's hooks run, but hands may
read it after them. So a dialog hands holds, and a turn its prompt opened, decide the standing ahead of the status, whatever
the status still says; the status decides only where neither is known.
"""

from dataclasses import dataclass
from typing import Literal

from hands.core.progress import Doing, said
from hands.core.session import Blocker, Held, Idle, Opened, Permission, Plan, Question, Running, Session, Unreported, delegating
from hands.core.spoken import counted
from hands.core.status import Busy, Going, Reason, Shell, Unknown, UnknownReason, Waiting


@dataclass(frozen=True)
class AtDialog:
    """At a dialog: one whose hook hands holds, said by what it asks, though the status saying it waits may not be read yet;
    or one Claude Code says it waits at and hands holds no hook for, which nothing hands can read says the content of."""

    by: Held | Reason | UnknownReason


@dataclass(frozen=True)
class Working:
    """A turn runs: the latest thing it set out to do, once it has made a call."""

    latest: Doing | None


@dataclass(frozen=True)
class Delegating:
    """At its prompt, with only subagents it started in the background working."""

    agents: int


@dataclass(frozen=True)
class ShellRunning:
    """At its prompt, with a shell command it started in the background still running."""


@dataclass(frozen=True)
class AtRest:
    """At its prompt with nothing out: no dialog, no turn open, no words queued behind its last turn, no shell or subagent
    in the background. Not `status.AtPrompt`, which a background shell is at too."""


@dataclass(frozen=True)
class NotReported:
    """Heard of, with no status read for it yet, and no turn opened."""


@dataclass(frozen=True)
class UnknownStatus:
    """Claude Code said a status this version of hands does not know, kept by its name."""

    name: str


Standing = AtDialog | Working | Delegating | ShellRunning | AtRest | NotReported | UnknownStatus


def standing_of(session: Session) -> Standing:
    # The dialog on its screen is the first it asked; the rest are queued beneath it.
    match next(iter(session.dialogs), None):
        case Held() as held:
            return AtDialog(held)
        case None:
            return _undialoged(session)


def _undialoged(session: Session) -> Standing:
    match session.state:
        case Unreported() | Idle() if isinstance(session.turn, Opened):
            return Working(None)
        case Unreported():
            return NotReported()
        case Idle() if session.background:
            # A subagent's launch written before the idle that came after it: still out, whatever the status says.
            return Delegating(len(session.background))
        case Idle() if session.turn.waiting:
            # Words queued behind its last turn run as a turn of their own once its Stop returns.
            return Working(None)
        case Idle(status=Shell()):
            # No turn runs: it does nothing until that task ends and its notification opens one.
            return ShellRunning()
        case Idle():
            return AtRest()
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
            return AtDialog(reason)
        case Unknown(name=name):
            return UnknownStatus(name)


# Where a session's turn is: "ended" for a session the registry no longer has, "unknown" while no status says.
Phase = Literal["ended", "over", "mid-turn", "unknown"]


def phase(of: Standing | None) -> Phase:
    """Where the session's last turn is, which only a session at its prompt, or one whose only work is in the background,
    proves over. None is a session no longer live."""
    match of:
        case None:
            return "ended"
        case AtRest() | ShellRunning() | Delegating():
            return "over"
        case AtDialog() | Working() | UnknownStatus():
            return "mid-turn"
        case NotReported():
            return "unknown"


def spoken(of: Standing) -> str:
    match of:
        case AtDialog(by=Held(on=on)):
            return _waiting_on(on)
        case AtDialog(by=UnknownReason(name=name)):
            return f"waiting at a dialog: {name}"
        case AtDialog(by=reason):
            return f"waiting at a dialog: {reason}"
        case Working(latest=Doing() as latest):
            return f"working; the last thing it set out to do: {said((latest,))}"
        case Working():
            return "working"
        case Delegating(agents=agents):
            return f"idle, with {counted(agents, 'subagent')} it started in the background still working"
        case ShellRunning():
            return "idle, with a shell command it started in the background still running"
        case AtRest():
            return "idle"
        case NotReported():
            return "not reported yet"
        case UnknownStatus(name=name):
            return f"in a state hands does not know: {name}"


def spoken_state(session: Session) -> str:
    """How a live session stands, in the one sentence the brain's listing says of it."""
    return spoken(standing_of(session))


def _waiting_on(on: Blocker) -> str:
    match on:
        case Permission(tool=tool):
            return f"waiting for permission to use {tool}"
        case Question():
            return "waiting for the user to answer its question"
        case Plan():
            return "waiting for the user to approve its plan"
