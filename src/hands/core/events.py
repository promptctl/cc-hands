"""Everything that can happen to the registry: parsed at the edges, reduced here."""

from dataclasses import dataclass
from typing import Literal

from hands.core.occurrences import Occurrence
from hands.core.progress import Doing
from hands.core.session import Blocker, Instant, Membership, FinishedCall, Mode, PromptId, RequestId, SessionId
from hands.core.status import Report, Stamp
from hands.core.stopfailure import StopFailed
from hands.core.turn import AgentId, AgentTask


# `fork` starts a new session forked from another, under its own id.
StartSource = Literal["startup", "resume", "clear", "compact", "fork"]
# Claude Code reports why a session ended; a closed terminal reports `other`.
EndReason = Literal["clear", "resume", "logout", "prompt_input_exit", "bypass_permissions_disabled", "other"]


@dataclass(frozen=True)
class Joined:
    membership: Membership
    source: StartSource


@dataclass(frozen=True)
class Attached:
    """A membership file names a running process: a session from before the daemon started, or one whose start hook never arrived."""

    membership: Membership


@dataclass(frozen=True)
class Located:
    """A hook's session as the file its shim wrote just before posting names it: where the session works now.

    [LAW:types-are-the-program] a sweep's Attached may have read the file before a hook landed, so it never moves a
    session the registry holds; this is never older than the hook it came with, so it does: Claude Code moves a
    session's transcript when it enters a worktree."""

    membership: Membership


@dataclass(frozen=True)
class Died:
    """A session's process is no longer running, and no hook said the session ended."""

    membership: Membership


@dataclass(frozen=True)
class MovedOn:
    """A session's process runs, but holds another session now: a /clear or a resume in it, whose end hook never arrived."""

    membership: Membership


@dataclass(frozen=True)
class Prompted:
    session: SessionId
    at: Instant
    # The permission_mode the hook carried; None only when it carried none.
    mode: Mode | None
    # [LAW:no-ambient-temporal-coupling] the prompt_id that names the turn this prompt opens. Only the opening names the
    # turn: every hook of a background subagent carries the prompt_id of the turn that started it, even after that
    # turn stopped and another opened, so a turn read off any later hook could be one already over.
    prompt: PromptId
    # The words submitted, as Claude Code's queue records carry them: how words filed under a running turn are matched
    # to the record that takes them out of its queue.
    text: str


@dataclass(frozen=True)
class Stopped:
    session: SessionId
    closing: str | None  # the reply the turn closed with, as the Stop hook carries it; None when there was none
    # The permission_mode the hook carried; None only when it carried none.
    mode: Mode | None
    # The prompt_id of the turn that stopped: the id it last went on under, and a turn a background task's notification
    # opened has its own, which its UserPromptSubmit carried too.
    prompt: PromptId
    # Whether this is the turn stopping again: another Stop hook blocked its last Stop, so Claude went on in it under the
    # same id (stop_hook_active). The one Stop that ends a turn already told.
    again: bool
    # When hands heard it, on the clock Claude Code stamps a status and a record with: every record of the turn it ends
    # was written before it fired, so a transcript read through a moment past this holds them.
    heard: Stamp
    # The hook's connection, which Claude Code waits on until the reducer has decided whose Stop it is (see Reply).
    request: RequestId
    # What failed the turn, when the API did: Claude Code fires StopFailure in place of Stop, and the turn is over as at
    # a Stop. None for a Stop.
    failure: StopFailed | None = None


@dataclass(frozen=True)
class Closed:
    """The model ended a turn with a reply of text (stop_reason end_turn), as it came back on the wire: the turn's closing
    reply, heard before Claude Code fires the Stop hook that carries it too.

    Never made of a reply with no text, to which Claude Code answers by asking again: that turn goes on."""

    session: SessionId
    prompt: PromptId
    closing: str


@dataclass(frozen=True)
class Interrupted:
    """The user stopped the turn at the keyboard, with Escape or Ctrl-C: no Stop hook fires for that, so it is read
    from the record Claude Code writes in the transcript instead."""

    session: SessionId
    # The turn the record says it stopped: only the untold turn it names waits for it to be told.
    prompt: PromptId
    at: Instant  # when the record was read


@dataclass(frozen=True)
class CarriedOut:
    """A slash command Claude Code carries out itself, such as /model, printed what it had to say, and its turn is over.
    Read from the transcript, since nothing else says so: no Stop fires for it, and while a subagent works in the
    background Claude Code sets no idle either."""

    session: SessionId
    # The turn the command opened, by the id its own records carry.
    prompt: PromptId
    at: Instant  # when the record was read


@dataclass(frozen=True)
class Taken:
    """Claude Code took the prompt: the transcript holds a record of its turn, which it writes only once the prompt's
    hooks are done and it was not cancelled.

    At the prompt it is a turn no hook opened, running as Claude Code says: a message queued while a turn ran, taken
    once that turn's Stop lands; a `!` command, which Claude answers once it has run; a command such as /compact.
    What happened in it is what is told, which for a command is nothing."""

    session: SessionId
    prompt: PromptId
    # When Claude Code wrote the record, on the clock it stamps a status with, so an idle it set after it can be told
    # from one it set before. None when the record's time could not be read: either way it opens nothing.
    written: Stamp | None
    at: Instant  # when the record was read


@dataclass(frozen=True)
class Continued:
    """Claude went on answering under another prompt's id without the turn ending: a queued message taken in mid-turn,
    or flushed by an Escape, carries its own new id from then on, and no hook ever names it. Read from the
    transcript, where the user's side of every record Claude answers carries the id it answers under."""

    session: SessionId
    # The id it was answering under, so a record read after the next prompt opened moves nothing.
    was: PromptId
    now: PromptId


@dataclass(frozen=True)
class Dequeued:
    """Claude Code took the oldest message out of its queue, to run: into the turn running, or as a turn of its own."""


@dataclass(frozen=True)
class Removed:
    """Claude Code took one message out of its queue by itself (remove): delivered into the running turn, to an agent,
    or for no reason it gives; or pulled it back out of the queue (popAll, one record for each message)."""

    content: str


# [LAW:types-are-the-program] what a queue-operation record says left Claude Code's queue; an enqueue says nothing the
# prompt's hook did not say first, and is no value here.
Unqueuing = Dequeued | Removed


@dataclass(frozen=True)
class Unqueued:
    """Claude Code took a message out of its queue, as its transcript's queue-operation record says."""

    session: SessionId
    taken: Unqueuing


@dataclass(frozen=True)
class Launched:
    """Claude started a subagent in the background, as the result of the call that started it says: an async agent, or
    a skill forked to run in the background. Claude Code says busy from here until the turn that reports it back ends."""

    session: SessionId
    agent: AgentId
    # When Claude Code wrote the result, on its status clock: a launch written before the idle it set since was over by
    # then, since it sets no idle while one works. None, as for Taken, when there was no time to read.
    written: Stamp | None


@dataclass(frozen=True)
class ReportedBack:
    """A background task's notification was read, at the prompt or mid-turn: the task it names, completed, failed, or
    stopped, works no more. A background command's or a monitor's names a task that is no subagent."""

    session: SessionId
    task: AgentId


@dataclass(frozen=True)
class Read:
    """The session's transcript has given everything it will of what Claude Code had written by `through`, on the clock
    it stamps a status with: read through it, or found missing or unreadable. A record of how a turn ended that was not
    in it, Claude Code had not written by then, or hands cannot read."""

    session: SessionId
    through: Stamp


@dataclass(frozen=True)
class Progressed:
    """The session made calls since its transcripts were last read: what each sets out to do, read as its record landed."""

    session: SessionId
    # Whose calls they are. The turn's own go by every id the turn they were read into goes by, sorted, so calls read
    # late, of a turn since over, move nothing. A subagent's are its own, whatever turn its parent is in: run in the
    # background, it works on after the turn that started it ends.
    of: tuple[PromptId, ...] | AgentTask
    doings: tuple[Doing, ...]
    at: Instant  # when they were read


@dataclass(frozen=True)
class Displayed:
    """Claude Code displayed lines of Claude's text as it streamed them: its MessageDisplay hook, one batch of finished
    lines, or every line since the last batch when the one before was slow to be taken."""

    session: SessionId
    # The id of the turn it was written in, as the hook names it; a tuple, as a Progressed's is, though a hook names one.
    turn: tuple[PromptId, ...]
    text: str
    at: Instant  # when it was heard


@dataclass(frozen=True)
class StatusReported:
    """Claude Code set the session's status: read from the file it keeps for the session each time its stamp moves, so a
    status set again to what it was is heard, and so is a busy that came and went between two reads."""

    session: SessionId
    report: Report
    at: Instant  # when it was read


@dataclass(frozen=True)
class PermissionRequested:
    session: SessionId
    at: Instant
    request: RequestId
    on: Blocker
    # The permission_mode the hook carried; None only when it carried none.
    mode: Mode | None


@dataclass(frozen=True)
class ToolFinished:
    """A tool call ran to its end, or failed. The call is named as the request to run it was, so the two can be matched."""

    session: SessionId
    at: Instant
    call: FinishedCall
    # The permission_mode the hook carried; None only when it carried none.
    mode: Mode | None


@dataclass(frozen=True)
class Occurred:
    """A hook said something happened in the session that moves nothing in it, for the user to hear of as they set."""

    session: SessionId
    occurrence: Occurrence


@dataclass(frozen=True)
class Ended:
    session: SessionId
    reason: EndReason


@dataclass(frozen=True)
class Abandoned:
    """The hook waiting on a permission reply went away before one was decided, so nothing can be answered."""

    session: SessionId
    request: RequestId
    at: Instant


@dataclass(frozen=True)
class Tick:
    """The one clock the reducer hears: when a burst of progress has settled is read against it, never against a timer."""

    at: Instant


# Events about a session the registry must already know; a join is how it comes to.
# What moves a live session on its axes; its end is the one session event that moves none of them.
Moving = Prompted | Stopped | Closed | Interrupted | CarriedOut | Taken | Continued | Unqueued | Launched | ReportedBack | Read | StatusReported | PermissionRequested | ToolFinished
SessionEvent = Moving | Progressed | Displayed | Occurred | Ended
# What a session's transcript says that none of its hooks do: of its turn, of its subagents in the background, and how
# far it has been read.
Transcribed = Taken | Interrupted | CarriedOut | Continued | Unqueued | Launched | ReportedBack | Progressed | Read
# What the liveness sweep saw in one membership file.
Observed = Attached | Died | MovedOn
Event = Joined | Located | Observed | SessionEvent | Abandoned | Tick
