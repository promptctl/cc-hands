"""The session lifecycle as one pure function.

[LAW:no-ambient-temporal-coupling] hooks, records, and statuses reach it on three timings, so which of two things Claude
Code did first is read from what Claude Code put on them, never from the order hands heard them in: the prompt id it
files a hook or a record under, and the clock it stamps a status, a record, and a reading of the transcript with.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace

from hands.core.effects import (
    AfterEnd,
    Audit,
    AuditRecord,
    Effect,
    Narrate,
    Progress,
    Asking,
    Holding,
    HoldingBehind,
    NoneBehind,
    Overtaken,
    QueuedBehind,
    Resumed,
    Unfiled,
    Unmarked,
    Reply,
    SessionGone,
    Compare,
    Snapshot,
    Submitted,
    Summarise,
    Tell,
    Unmatched,
    Unclosed,
    Unregistered,
    Unsettled,
    Withdraw,
)
from hands.core.events import (
    Abandoned,
    Attached,
    Died,
    Ended,
    Ending,
    EndReason,
    MovedOn,
    Moving,
    Occurred,
    Event,
    Interrupted,
    CarriedOut,
    Continued,
    Closed,
    Taken,
    Joined,
    Located,
    Launched,
    Read,
    ReportedBack,
    Displayed,
    PermissionRequested,
    Prompted,
    Progressed,
    SessionEvent,
    StartSource,
    StatusReported,
    Stopped,
    Tick,
    ToolFinished,
    Dequeued,
    Unqueued,
    Unqueuing,
    Removed,
)
from hands.core.occurrences import Cleared
from hands.core.progress import Doing, Gathering
from hands.core.stopfailure import StopFailed
from hands.core.session import followed, ids, Blocker, DenyAndInterrupt, Gone, Held, Idle, Instant, Known, Membership, Mode, Opened, Plan, PlanApproved, PromptId, Question, FinishedCall, Registry, RequestId, Running, Session, SessionId, SessionState, Told, Turn, Behind, HeldStop, Unnamed, Unreported, Untold, holds, status_stamp
from hands.core.status import AtPrompt, Going, Report, Stamp
from hands.core.turn import AgentId, AgentTask

# How long past the idle Claude Code set a turn waits to be told for the record of how it ended, and past a Stop hands
# heard it waits for a record naming its id, in milliseconds on Claude Code's own clock: an interrupt's is written 37 ms
# after the idle, a turn's reply up to 40 ms after its Stop fires, and some ways of stopping a turn
# write none. The transcript read through this point holds it if it was ever coming.
UNTOLD = 1000


def reduce(registry: Registry, event: Event) -> tuple[Registry, list[Effect]]:
    """One event in; the next registry and the effects it calls for out. No I/O."""
    # [LAW:effects-at-boundaries] time arrives inside the event, so when a burst settles is arithmetic on values,
    # never a clock read.
    match event:
        case Joined(membership=membership, source=source):
            previous = registry.sessions.get(membership.id)
            started = _started(membership, source, previous)
            joined, effects = _join(registry, started)
            # A Stop held in the process before is let go: its transcript is read afresh from here, as history.
            held = previous.stops if isinstance(previous, Session) else ()
            waiting = previous.turn.waiting if isinstance(previous, Session) else ()
            return joined, [*_unsettled(membership.id, [stop for stop in held if stop not in started.stops], waiting), *effects, *_said_at_start(membership.id, source)]
        case Attached(membership=membership) | Located(membership=membership) if membership.id not in registry.sessions:
            # A membership file says nothing of the mode; the session's next hook will.
            return registry.put(Session(membership, Unreported(), mode=None)), []
        case Located(membership=membership):
            return _located(registry, membership), []
        case Attached():
            # [LAW:no-ambient-temporal-coupling] a session the registry already knows was heard from its hooks, which
            # know more than its file: a sweep that read the file before a hook landed never overwrites it.
            return registry, []
        case Died(membership=membership):
            return _ended_unheard(registry, membership, [SessionGone(membership.id)])
        case MovedOn(membership=membership):
            # The user moved the process on at the keyboard, so there is nothing to tell them.
            return _ended_unheard(registry, membership, [])
        case Abandoned(session=session, request=request):
            return _abandoned(registry, session, request), []
        case Ending(session=session, refusal=refusal):
            return _refusing(registry, session, refusal)
        case Tick(at=at):
            return _ticked(registry, at)
        case _:
            return _enter(registry, event)


def _located(registry: Registry, membership: Membership) -> Registry:
    """A session the registry holds, with the transcript its own hook's file names now; one that ended stays ended.

    Only the transcript: the process and the project are the start's, and a late hook of a process the session has since
    left, which writes its own whole file, does not take the session back to it.
    """
    match registry.sessions[membership.id]:
        case Session() as session:
            return registry.put(replace(session, membership=replace(session.membership, transcript=membership.transcript)))
        case Gone():
            return registry


def _started(membership: Membership, source: StartSource, previous: Known | None) -> Session:
    # Compaction starts a session again in the middle of what it was doing, in the same process, so all of it carries
    # over. Every other start is a new process, or a new session in one, whose status is read afresh: a session resumed
    # after a crash was never told it stopped. SessionStart carries no permission_mode, and a session started or resumed
    # in a new process may have been given any mode.
    match (source, previous):
        case ("compact", Session() as previous):
            return replace(previous, membership=membership)
        case (_, Session(turn=Untold() as untold, earlier=earlier)):
            # A turn ended before the restart is still told. One that ended was told as it ended. Nothing waits behind it:
            # Claude Code's queue does not outlive its process.
            return Session(membership, Unreported(), mode=None, turn=replace(untold, waiting=()), earlier=earlier)
        case (_, Session(turn=Told() as last, earlier=earlier)):
            # Told before the restart, so a late Stop of it ends nothing after it.
            return Session(membership, Unreported(), mode=None, earlier=earlier | ids(last))
        case (_, Session(earlier=earlier)):
            # A turn open when its process went is not told, so a Stop of it can still tell it.
            return Session(membership, Unreported(), mode=None, earlier=earlier)
        case _:
            return Session(membership, Unreported(), mode=None)


def _join(registry: Registry, session: Session) -> tuple[Registry, list[Effect]]:
    previous = registry.sessions.get(session.membership.id)
    return registry.put(session), _transition(previous, session)


def _ended_unheard(registry: Registry, membership: Membership, said: list[Effect]) -> tuple[Registry, list[Effect]]:
    """A session the sweep found over, though no end hook said so; `said` is what the user hears about it."""
    match registry.sessions.get(membership.id):
        case None | Gone():
            # A session this run never listed ended while the daemon was down, or before a reboot: the user was not told
            # of it here, so there is nothing to take back.
            return registry, []
        case Session(membership=held) if held.pid != membership.pid:
            # Started again in a new process since the sweep looked; what it saw ending is not this session.
            return registry, []
        case Session() as was:
            return _end(registry, was, membership, said)


def _end(registry: Registry, was: Session, membership: Membership, said: list[Effect]) -> tuple[Registry, list[Effect]]:
    """The session is over, however that was heard: the hook it was held on is let go, and a turn ended and not told yet
    is told before the session is said to be gone, since it happened first."""
    _, telling = _told(membership.id, was.turn, None)
    return registry.put(Gone(membership)), [*_dialogs(membership.id, was.dialogs, ()), *telling, *_unsettled(membership.id, was.stops, was.turn.waiting), *said]


def _unsettled(session: SessionId, held: Iterable[HeldStop], waiting: tuple[str, ...]) -> list[Effect]:
    """Stops held for a record that will not be read now: each is a line, and its hook is let go."""
    # [LAW:nothing-unseen] said as what it is, never as a Stop the transcript was read through without naming.
    return [effect for stop in held for effect in (Audit(_undecided(session, stop, waiting)), *_let_go(session, stop))]


def _undecided(session: SessionId, stop: HeldStop, waiting: tuple[str, ...]) -> AuditRecord:
    match stop:
        case Unnamed(prompt=prompt):
            # It ended nothing: no record named its id.
            return Unsettled(session, prompt)
        case Behind():
            # It told its turn; the words behind it run, if they run, with no mark taken for them.
            return Unmarked(session, len(waiting))


def _let_go(session: SessionId, stop: HeldStop) -> list[Effect]:
    """The Stop is decided, after whatever its deciding called for, so Claude Code goes on only once that is done: its
    hook is answered, unless it stopped waiting."""
    match stop.hook:
        case None:
            return []
        case hook:
            return [Reply(session, hook, Withdraw())]


def _enter(registry: Registry, event: SessionEvent) -> tuple[Registry, list[Effect]]:
    """An event of one session: it moves or ends a live one, and is recorded of any other."""
    match registry.sessions.get(event.session):
        case None:
            # [LAW:no-silent-failure] an event for a session that never joined is a record, not a drop.
            return registry, _unheard(event, Unregistered(event))
        case Gone():
            # Ended is final until the session starts again; a hook that lands late cannot revive it.
            return registry, _unheard(event, AfterEnd(event))
        case Session() as was:
            match event:
                case Ended(reason=reason):
                    return _end(registry, was, was.membership, _said_at_end(was.membership.id, reason))
                case Progressed(of=AgentTask() as agent, doings=doings, at=at):
                    return registry.put(replace(was, subagents=_helped(was.subagents, agent, doings, at))), []
                case Progressed() | Displayed():
                    return registry.put(replace(was, turn=_gathered(was.turn, event))), []
                case Occurred(occurrence=occurrence):
                    return registry, [Tell(was.membership.id, occurrence)]
                case _:
                    return _moved(registry, was, event)


def _said_at_start(session: SessionId, source: StartSource) -> list[Effect]:
    match source:
        case "clear":
            # The session the /clear started, in the process the cleared one ran in; the cleared one ended saying nothing.
            return [Tell(session, Cleared())]
        case "startup" | "resume" | "compact" | "fork":
            return []


def _said_at_end(session: SessionId, reason: EndReason) -> list[Effect]:
    match reason:
        case "other":
            # Nobody ended it at the keyboard: its terminal closed. Spoken, as a process found dead is.
            return [SessionGone(session)]
        case "clear" | "resume" | "logout" | "prompt_input_exit" | "bypass_permissions_disabled":
            return []


def _moved(registry: Registry, was: Session, event: Moving) -> tuple[Registry, list[Effect]]:
    """The event moves a live session on each of its axes: what Claude Code says it is doing, the dialog, and the turn."""
    held = was.mode
    turn, told = _turned(event, was)
    reported = _reported(event)
    # [LAW:dataflow-not-control-flow] every hook that carries a mode sets it, so a mode changed at the keyboard
    # is heard at the session's next hook, whatever that hook moves the session to.
    mode = held if reported is None else reported
    background, overtaken = _backgrounded(event, was)
    moved = replace(was, state=_stated(event, was), mode=mode, turn=turn, dialogs=_dialog(event, was), ending=_still_ending(event, was), background=background)
    moved, stopped = _stopped(moved, event)
    after, settled = _settled(moved, event)
    # [LAW:single-enforcer] a turn another replaces was told as it was replaced, so its ids are earlier from here on.
    after = replace(after, earlier=was.earlier | (ids(was.turn) - ids(after.turn)))
    # A turn left untold is told before what the event calls for, so before a prompt marks the next, and before the
    # prompt's words are said to be taken: a telling the prompt decides is never theirs.
    return registry.put(after), [*_refused(event, was.ending), *_transition(was, after), *told, *_submitted(was, event), *stopped, *settled, *overtaken]


def _submitted(was: Session, event: Moving) -> list[Effect]:
    """Where a prompt hook's words went: filed under a turn the session already had, or opening one of their own."""
    match event:
        case Prompted(prompt=prompt, text=text):
            return [Submitted(was.membership.id, prompt, text, queued=_names(was.turn, prompt))]
        case _:
            return []


def _stopped(session: Session, event: Moving) -> tuple[Session, list[Effect]]:
    """What a Stop does as it is heard: ends the turn it names, or is held until the transcript says whose it is; then,
    its turn told, decides what waits behind it, or is held until the transcript read through its firing says."""
    match event:
        case Stopped(prompt=prompt, closing=closing, again=again, heard=fired, request=request, failure=failure):
            stop = Unnamed(prompt, closing, again, failure, Stamp(fired), request)
            match _stopping(session, stop):
                case None:
                    # Its hook is held with it, so what its deciding calls for is done before Claude Code goes on.
                    return replace(session, stops=(*session.stops, stop)), [Audit(Holding(session.membership.id, prompt))]
                case (turn, effects):
                    ended, behind = _told_behind(replace(session, turn=turn), Behind(prompt, stop.fired, request), event)
                    return ended, [*effects, *behind]
        case _:
            return session, []


def _settled(session: Session, event: Moving) -> tuple[Session, list[Effect]]:
    """Every Stop held on the session, tried against its turn as the event left it, in the order they were heard: one
    whose id a record read since names ends that turn; one the transcript was read through without naming is a line;
    one whose turn is told decides what waits behind it once the transcript is read through its firing.

    Never told as a turn hands never had open (see _stopping): that a record of it may still come is why it is held."""
    id, effects = session.membership.id, list[Effect]()
    settled = replace(session, stops=())
    for stop in session.stops:
        match (stop, event):
            case (Behind(), _):
                settled, behind = _behind(settled, stop, event)
                effects += behind
            case (Unnamed(), _) if (ended := _ends(settled, stop)) is not None:
                settled, behind = _told_behind(replace(settled, turn=ended[0]), Behind(stop.prompt, stop.fired, stop.hook), event)
                effects += [*ended[1], *behind]
            case (Unnamed(), Read(through=through)) if through >= stop.fired + UNTOLD:
                effects += [Audit(Unmatched(id, stop.prompt)), *_let_go(id, stop)]
            case (Unnamed(), _):
                settled = replace(settled, stops=(*settled.stops, stop))
    return settled, effects


def _told_behind(session: Session, stop: Behind, event: Moving) -> tuple[Session, list[Effect]]:
    """A Stop that has just told its turn: let go at once with nothing waiting behind that turn; else decided as one held
    behind it is, and held, which is a line, until it can be."""
    match session.turn:
        case Told(waiting=()):
            return session, _let_go(session.membership.id, stop)
        case _:
            after, effects = _behind(session, stop, event)
            held = [Audit(HoldingBehind(session.membership.id, stop.prompt, len(session.turn.waiting)))] if stop in after.stops else []
            return after, [*effects, *held]


def _behind(session: Session, stop: Behind, event: Moving) -> tuple[Session, list[Effect]]:
    """A Stop whose turn is told and words waited behind, with what it calls for of those still waiting, which Claude Code
    runs once the Stop's hook returns as a turn no prompt hook opens: marked while the hook holds Claude Code, so before
    that turn can have changed anything [LAW:no-ambient-temporal-coupling]. Held until the transcript is read through the
    Stop's firing, which holds the queue records Claude Code wrote before it that deliver them into the turn.

    [LAW:single-enforcer] the one place a turn is marked for words that run with no hook of their own."""
    id = session.membership.id
    match session.turn:
        case Told(waiting=()) as told if _names(told, stop.prompt):
            # [LAW:nothing-unseen] queue records took them all out before it fired: delivered into the turn, or pulled back.
            return session, [Audit(NoneBehind(id, stop.prompt)), *_let_go(id, stop)]
        case Told(waiting=waiting) as told if _names(told, stop.prompt) and _read_through(session, event, stop.fired):
            return session, [Audit(QueuedBehind(id, stop.prompt, len(waiting))), Snapshot(id, session.membership.cwd), *_let_go(id, stop)]
        case Told() as told if _names(told, stop.prompt):
            return replace(session, stops=(*session.stops, stop)), []
        case _:
            # The turn it ended is not the last any more: a prompt opened one, which said the words run unmarked (Unmarked).
            return session, _let_go(id, stop)


def _read_through(session: Session, event: Moving, fired: Stamp) -> bool:
    """Whether everything Claude Code wrote by `fired` has been read, or none of it will be: no session's transcript is read
    before its status is (see Tails.catch_up), and holding for a read that never comes would hold Claude Code."""
    return (isinstance(event, Read) and event.through >= fired) or status_stamp(session.state) is None


def _stated(event: Moving, was: Session) -> SessionState:
    """What Claude Code says the session is doing after the event.

    [LAW:one-source-of-truth] only a status read moves it between idle and running: a hook or a record says which turn
    it is, what it did, and which request it waits on, never whether the session runs.
    """
    match (event, was.state):
        # [LAW:one-source-of-truth] which statuses are at the prompt is status.AtPrompt's to say, never these patterns'.
        case (StatusReported(report=Report(status=at, stamp=stamp)), Idle(after=after) as idle) if isinstance(at, AtPrompt) and was.turn.turn == after:
            # Set idle again with no turn heard since: the same idle period, though a background shell began or ended in
            # it. One with a turn heard since, as a prompt cancelled during its hooks or a turn over between two reads,
            # is a new period, below.
            return replace(idle, status=at, stamp=stamp)
        case (StatusReported(report=Report(status=at, stamp=stamp)), _) if isinstance(at, AtPrompt):
            return Idle(at, stamp, after=was.turn.turn)
        case (StatusReported(report=Report(status=going, stamp=stamp)), Running() as running) if isinstance(going, Going):
            return replace(running, status=going, stamp=stamp)
        case (StatusReported(report=Report(status=going, stamp=stamp)), Idle(stamp=idled)) if isinstance(going, Going):
            return Running(going, stamp, idled=idled)
        case (StatusReported(report=Report(status=going, stamp=stamp)), _) if isinstance(going, Going):
            # First read running, as when hands attaches mid-turn: no idle before it was read.
            return Running(going, stamp, idled=None)
        case (_, state):
            # No hook or record moves it.
            return state


def _backgrounded(event: Moving, was: Session) -> tuple[frozenset[AgentId], list[Effect]]:
    """The subagents the session started in the background that have not reported back, after the event."""
    match event:
        case Launched(agent=agent, written=written) if _since_idle(was.state, written):
            return was.background | {agent}, []
        case Launched(agent=agent):
            # From before Claude Code's last idle, which it sets none of while a subagent works: over by then, and its
            # report read already or in what is read next. [LAW:nothing-unseen] the decision is a line.
            return was.background, [Audit(Overtaken(was.membership.id, frozenset({agent})))]
        case ReportedBack(task=task):
            return was.background - {task}, []
        case StatusReported(report=Report(status=at)) if isinstance(at, AtPrompt) and was.background:
            # [LAW:one-source-of-truth] Claude Code sets no idle while a subagent works, so its idle says none does,
            # whether or not each report has been read yet.
            return frozenset(), [Audit(Overtaken(was.membership.id, was.background))]
        case _:
            return was.background, []


def _since_idle(state: SessionState, written: Stamp | None) -> bool:
    """Whether a record was written since Claude Code last set the session idle, on its own clock; with no idle read, or
    no time on the record, it is not known to be older."""
    match state:
        case Idle(stamp=idled) | Running(idled=idled):
            return idled is None or written is None or written >= idled
        case Unreported():
            return True


def _dialog(event: Moving, was: Session) -> tuple[Held, ...]:
    """The dialogs the session is at after the event, as its hooks and Claude Code's idle tell it, the one on its screen
    first."""
    dialogs = was.dialogs
    match event:
        case StatusReported(report=Report(status=at)) if isinstance(at, AtPrompt):
            # At its prompt: no dialog is up, whether it was answered at the keyboard or escaped.
            return ()
        case PermissionRequested() if was.ending is not None:
            # Asked while hands ends what the session runs: refused as it arrives (_refused), and never held.
            return dialogs
        case PermissionRequested(request=request) if holds(was, request):
            # A request it already waits on, heard again: it is the same dialog, by request id.
            return dialogs
        case PermissionRequested(request=request, on=on, at=at):
            # Drawn beneath the dialogs already up, as Claude Code queues one asked while another is on its screen.
            return (*dialogs, Held(on=on, request=request, asked=at))
        case ToolFinished(call=call):
            # The tool a dialog was waiting to run has run, so that dialog was answered at the keyboard.
            ran = next((held for held in dialogs if _same_call(held.on, call)), None)
            return tuple(held for held in dialogs if held is not ran)
        case Prompted(at=at):
            # Typed at the session: a dialog it was typed past was answered, and one escaped is waited on no more, for no
            # prompt is typed while a dialog holds the keyboard. A dialog asked after the prompt reached hands was not up
            # as it was typed: its hook is held as it arrives (Reader.asked), and the prompt's is applied after a read.
            return tuple(held for held in dialogs if held.asked > at)
        case _:
            return dialogs


def _refused(event: Moving, ending: DenyAndInterrupt | None) -> list[Effect]:
    """A dialog asked while hands ends what the session runs is answered at once with the ending's refusal."""
    match (event, ending):
        case (PermissionRequested(session=session, request=request), DenyAndInterrupt() as refusal):
            return [Reply(session, request, refusal)]
        case _:
            return []


def _still_ending(event: Moving, was: Session) -> DenyAndInterrupt | None:
    """[LAW:single-enforcer] what hands is ending lasts until a prompt starts new work: one filed into a turn still
    running starts nothing, and the work the ending was for still runs."""
    match event:
        case Prompted(prompt=prompt) if not _filed(was.turn, prompt):
            return None
        case _:
            return was.ending


def _refusing(registry: Registry, session: SessionId, refusal: DenyAndInterrupt) -> tuple[Registry, list[Effect]]:
    """hands is ending what the session runs: every dialog it holds is refused now, and each one asked until a prompt
    starts new work is refused as it arrives. The first ending stays."""
    match registry.sessions.get(session):
        case Session(dialogs=dialogs, ending=ending) as was:
            # [LAW:one-source-of-truth] the hooks are sent the refusal in the step the model lets their dialogs go.
            ended = registry.put(replace(was, dialogs=(), ending=refusal if ending is None else ending))
            return ended, [Reply(session, held.request, refusal) for held in dialogs]
        case _:
            # A session not yet heard, or over, runs nothing a dialog could be asked of.
            return registry, []


def _turned(event: Moving, was: Session) -> tuple[Turn, list[Effect]]:
    """The turn after the event, and what the event calls for of it: told, compared, marked."""
    id, turn = was.membership.id, was.turn
    match (event, turn):
        case (StatusReported(report=Report(status=at, stamp=stamp)), Opened() as opened) if isinstance(at, AtPrompt):
            # [LAW:one-source-of-truth] Claude Code says the turn is over, however it was stopped, so it is. A prompt
            # cancelled by an Escape during its hooks ends here too, and is told as itself if it ran. It is told once the
            # transcript says how it ended: see Untold.
            return Untold(opened.turn, opened.others, Stamp(stamp + UNTOLD), opened.waiting), []
        case (StatusReported(report=Report(status=going)), Untold(turn=prompt, others=others, waiting=waiting)) if isinstance(going, Going):
            # Claude Code set it going again after the idle that made the turn wait, with no Stop of it heard, as when send
            # now moves a running call to the background and it sets shell for a few ms: open again, so what it does is
            # gathered and its deadline gone. Whether this turn goes on or one after it runs, only what ends it says, its
            # Stop, its interrupt record, or a record written after that idle (see Taken), never how long hands has read
            # the transcript for [LAW:no-ambient-temporal-coupling].
            return Opened(prompt, others, waiting, resumed=True), [Audit(Resumed(id, prompt))]
        case (Prompted(prompt=prompt, text=text), _) if _filed(turn, prompt):
            # Claude Code files what is submitted while a turn runs, the user's words or a task's notification as it is
            # delivered, under the running turn's id, and leaves its status as it was: filed, until a queue record says
            # Claude Code took it out of its queue. It marks nothing, or the mark would move into the middle of the work
            # it is there to measure: words still waiting when the Stop before them fires are marked for there (_behind).
            # Behind words already waiting, after the turn is told too: they run first, and the Stop of their turn marks
            # for these (see Taken). A turn's own prompt hook, heard after the shim gave up on a slow daemon, is read the
            # same way.
            return replace(turn, waiting=(*turn.waiting, text)), []
        case (Prompted(prompt=prompt), Told() as told) if _names(told, prompt):
            # Submitted after the wire told the turn, or after its Stop, with nothing waiting before it: filed under the
            # turn's id and run as its own turn once that Stop's hook returns. Marked here, while its own prompt hook holds
            # Claude Code, whether or not the Stop is still to come, so it waits on no Stop for its mark; its turn opens
            # when it is taken (see Taken).
            return turn, [Snapshot(id, was.membership.cwd)]
        case (Prompted(prompt=prompt), Opened() as opened):
            # A prompt under an id of its own is one Claude Code took at its prompt, the only place it gives one: the
            # open turn is over, though its idle may never have been read. An Escape and the next prompt set idle and
            # busy again 93 ms apart, inside one status read. Told as it stands, and compared now, while this
            # prompt's hook holds Claude Code, so before the next turn has changed anything; then that turn is marked.
            return Opened(prompt), [*_over(id, opened)[1], *_unmarked(id, turn), Snapshot(id, was.membership.cwd)]
        case (Prompted(prompt=prompt), Untold() | Told()):
            # A turn opens at the prompt, and is marked, so what it changes is read against a repository it has not
            # touched yet: after the one before is told, so that one is compared against its own mark.
            return Opened(prompt), [*_told(id, turn, None)[1], *_unmarked(id, turn), Snapshot(id, was.membership.cwd)]
        case (Taken(prompt=prompt), Opened() as opened) if _names(opened, prompt):
            return turn, []
        case (Taken(prompt=prompt, written=written), Opened(resumed=True) as opened) if _opens(was, prompt, written):
            # Gone on past an idle, so read as a turn waiting to be told is: a record written since that idle is of a turn
            # after it, which opens, and this one is over.
            return Opened(prompt, waiting=opened.waiting), _over(id, opened)[1]
        case (Taken(), Opened(resumed=True)):
            # Written before that idle: of this turn going on under the words' id or of the turn after it, which only
            # their ends tell apart (see Continued). Nothing moves.
            return turn, []
        case (Taken(prompt=prompt), Opened() as opened):
            # A flush's id: an Escape sends a message queued behind the turn on in the same turn, under the message's own
            # id, which the interrupt record carries first; or send now takes it into the turn (see Unqueued).
            return replace(opened, others=opened.others | {prompt}), []
        case (Taken(prompt=prompt, written=written), Untold() | Told()) if _opens(was, prompt, written):
            # A turn no hook opened: a message queued behind a turn, a `!` command, a command such as /compact
            # Named, so its Stop ends it, or what it printed for a command Claude Code carries out itself (see
            # CarriedOut). Not marked here, where a mark could land after Claude has begun changing the repository: a
            # queued message was marked while the Stop hook before it held Claude Code (see _behind), and one nothing
            # was queued for, as a `!` command's answer, is told by its steps. What still waits behind the turn before
            # runs after this one.
            return Opened(prompt, waiting=turn.waiting), _told(id, turn, None)[1]
        case (Continued(was=going, now=now), Opened() | Untold() as going_on) if _names(going_on, going):
            # Still working, now on the queued message: the turn goes by its id, so the Stop that ends it names it, and
            # still by the one it went on from, which a hook sent before Claude Code moved on carries. Read after the idle
            # that ended it, the turn waiting to be told is the same turn, and waits for the Stop under its new id.
            return replace(going_on, turn=now, others=going_on.others | {going_on.turn}), []
        case (Unqueued(taken=taken), _):
            match _unqueued(turn.waiting, taken):
                case None:
                    # [LAW:nothing-unseen] none of the words filed under this turn: a notification's, or of a turn before.
                    return turn, [Audit(Unfiled(id, taken))]
                case left:
                    # Delivered into the running turn, run after a told one (the turn its Stop marked for), or taken out
                    # of the queue for another reason: either way it no longer waits.
                    return replace(turn, waiting=left), []
        case (Closed(prompt=prompt, closing=closing), _):
            # The reply the turn closed with, from the wire, where it exists first: told as its Stop would tell it, and
            # before that Stop fires, which then ends nothing, as any Stop of a turn told does. Only a first ending: that
            # the turn went on after another Stop hook blocked its Stop is the Stop's to say (its again), not the wire's.
            ended = _ending(was, prompt, closing, None, again=False)
            # [LAW:nothing-unseen] a reply that ends nothing, as a Stop that ends nothing is, is a line.
            return (turn, [Audit(Unclosed(id, prompt))]) if ended is None else ended
        case (Interrupted(prompt=prompt), Untold() as untold) if _names(untold, prompt):
            # The record of how the turn Claude Code said is over ended: what its telling waits for.
            return _told(id, turn, None)
        case (Interrupted(prompt=prompt), Opened(turn=answering) as opened) if prompt == answering:
            # The user stopped the turn Claude was answering, and Claude Code goes on in no turn it interrupted (on every
            # interrupt record in this machine's transcripts): over, whether or not the idle it set was read, and told
            # as it stands. An interrupt that flushes a queued message names that message's id instead (see Taken).
            return _over(id, opened)
        case (CarriedOut(prompt=prompt), Opened(turn=running) as opened) if prompt == running:
            # The command's turn is over, though a subagent working in the background keeps Claude Code from setting the
            # idle that would say so: told as it stands. Only the turn it opened: a command run while another turn runs
            # joins that turn's others under its own id (see Taken), and ends nothing of it. One written under the
            # running turn's own id follows that turn's interrupt record (every such command in this machine's
            # transcripts), so the turn is over before it is read.
            return _over(id, opened)
        case (CarriedOut(prompt=prompt), Untold(turn=waiting)) if prompt == waiting:
            # Its idle was read first: what it printed is the record of how it ended.
            return _told(id, turn, None)
        case (Read(through=through), Untold(by=by)) if through >= by:
            # Read through the point where the record of how it ended would be, and it was not there: told with what was read.
            return _told(id, turn, None)
        case _:
            # A Stop, which _stopped decides. A record of a turn that is neither the one open nor the one waiting to be
            # told: read after that turn ended, or of one this registry never heard open. Ending it would end another turn
            # and spend its mark: nothing moves.
            return turn, []


def _ending(was: Session, prompt: PromptId, closing: str | None, failure: StopFailed | None, again: bool) -> tuple[Turn, list[Effect]] | None:
    """The turn the id names, ended with its reply, whether its Stop or the wire heard that; None when it names no turn
    it can end."""
    id = was.membership.id
    match was.turn:
        case Opened() as opened if _names(opened, prompt):
            # Compared before the turn is handed over to be summarised, never after: see Compare. A turn queued behind it
            # is marked by its Stop (see _behind), however the turn was told. Read against its prompt's mark even when the
            # Stop is one again: a first Stop hands never heard told nothing, so no reading left off anywhere for this turn.
            return _telling(id, Told(opened.turn, opened.others, opened.waiting), prompt, closing, failure, again=False)
        case Untold(turn=turn, others=others, waiting=waiting) as untold if _names(untold, prompt):
            # Its Stop fired after Claude Code set idle, as an Escape's can: told with the reply it carries, under the id
            # it names.
            return _telling(id, Told(turn, others, waiting), prompt, closing, failure, again=False)
        case Told() as told if again and _names(told, prompt):
            # The last turn stopping again after another Stop hook blocked its Stop: its telling holds only what was not
            # told before. Any other ending of a turn told ends nothing.
            return _alone(was, prompt, closing, failure, again)
        case _:
            return None


def _alone(was: Session, prompt: PromptId, closing: str | None, failure: StopFailed | None, again: bool) -> tuple[Turn, list[Effect]]:
    """The ending told as a turn that goes by its id alone. The words waiting behind the session's last turn still wait,
    behind this one: no queue record has taken them out."""
    return _telling(was.membership.id, Told(prompt, frozenset(), was.turn.waiting), prompt, closing, failure, again)


def _telling(session: SessionId, told: Told, prompt: PromptId | None, closing: str | None, failure: StopFailed | None, again: bool) -> tuple[Turn, list[Effect]]:
    """The turn told: compared before it is handed over to be summarised, never after (see Compare), and said with what
    still waits behind it as it is told. [LAW:single-enforcer] every telling is made here."""
    return told, [Compare(session, again), Summarise(session, prompt, closing, told.waiting, failure)]


def _stopping(was: Session, stop: Unnamed) -> tuple[Turn, list[Effect]] | None:
    """What the Stop does to the session's turn as it is heard; None while no record read so far says whose it is."""
    id, turn, prompt = was.membership.id, was.turn, stop.prompt
    match (_ends(was, stop), turn):
        case (None, Told()) if not followed(was, prompt):
            # A turn hands never had open, such as the one a session was in when it was attached. Told here for the first
            # time whatever its Stop says: where the last reading left off is another turn's end [LAW:one-source-of-truth].
            return _alone(was, prompt, stop.closing, stop.failure, again=False)
        case (None, Opened() | Untold()) if not followed(was, prompt) and status_stamp(was.state) is not None:
            # Whether it is the open or waiting turn's, gone on under a queued message whose record is unread, or a turn
            # after it whose record is unread, only the transcript says, and a Stop told as the wrong turn is heard as
            # that turn's answer [LAW:one-source-of-truth]: never guessed from how hands sampled the status. Held only
            # where the transcript is read, which is once a status is (see Tails.catch_up): else no record would come.
            return None
        case (None, _):
            # [LAW:nothing-unseen] a Stop is a hook Claude Code fired: one that ends nothing is still a line.
            return turn, [Audit(Unmatched(id, prompt))]
        case (ended, _):
            return ended


def _ends(was: Session, stop: Unnamed) -> tuple[Turn, list[Effect]] | None:
    """What the Stop ends of the session's turn: the turn it names, as any ending does; or, of a turn the wire told
    already with words waiting behind it, nothing, leaving it what waits behind (see _behind). None when it names no
    turn it can end."""
    match (_ending(was, stop.prompt, stop.closing, stop.failure, stop.again), was.turn):
        case (None, Told(waiting=waiting) as told) if waiting and _names(told, stop.prompt):
            return told, []
        case (ended, _):
            return ended


def _unqueued(waiting: tuple[str, ...], taken: Unqueuing) -> tuple[str, ...] | None:
    """The words still waiting once the queue record took its message out; None when it took none of them."""
    match taken:
        case Removed(content=content) if content in waiting:
            # [LAW:one-source-of-truth] by the text the record names, so a message removed that the user never filed
            # under this turn never takes theirs out.
            at = waiting.index(content)
            return (*waiting[:at], *waiting[at + 1 :])
        case Dequeued() if waiting:
            # Claude Code dequeues the oldest message; a notification is dequeued only into a turn its own prompt opened,
            # which files nothing, so what waits here is the oldest of what it held.
            return waiting[1:]
        case _:
            return None


def _unmarked(session: SessionId, turn: Turn) -> list[Effect]:
    """A turn of a prompt of its own opening while words waited behind the last: they run after it, unmarked."""
    return [Audit(Unmarked(session, len(turn.waiting)))] if turn.waiting else []


def _reported(event: Moving) -> Mode | None:
    """The mode the event's hook reported; None from the hooks that carry none."""
    match event:
        case Prompted(mode=mode) | Stopped(mode=mode) | PermissionRequested(mode=mode) | ToolFinished(mode=mode):
            return mode
        case Closed() | Taken() | Interrupted() | CarriedOut() | Continued() | Unqueued() | Launched() | ReportedBack() | Read() | StatusReported():
            return None


def _told(session: SessionId, turn: Turn, closing: str | None) -> tuple[Turn, list[Effect]]:
    """The turn once one Claude Code ended and hands has not told yet is told, with the reply its Stop carried."""
    match turn:
        case Untold(turn=prompt, others=others, waiting=waiting):
            return _telling(session, Told(prompt, others, waiting), prompt, closing, None, again=False)
        case Opened() | Told():
            return turn, []


def _over(session: SessionId, opened: Opened) -> tuple[Turn, list[Effect]]:
    """The open turn over before Claude Code's idle was read, told now as it stands."""
    return _telling(session, Told(opened.turn, opened.others, opened.waiting), opened.turn, None, None, again=False)


def _names(turn: Turn, prompt: PromptId) -> bool:
    """Whether the id is one the turn goes by."""
    return prompt in ids(turn)


def _filed(turn: Turn, prompt: PromptId) -> bool:
    """Whether a prompt submitted under this id is filed into the work the turn still runs, behind or into it, rather
    than starting work of its own: the turn runs, or words wait behind it to run."""
    match turn:
        case Opened() | Untold() | Told(waiting=(_, *_)):
            return _names(turn, prompt)
        case Told():
            return False


def _opens(session: Session, prompt: PromptId, written: Stamp | None) -> bool:
    """Whether a prompt taken with no turn open opens one no hook opened: under an id no turn told went by, written
    since Claude Code last set the session idle."""
    # [LAW:one-source-of-truth] Claude Code's own two clocks, never the order hands read them in: a record written before
    # the idle it set is of a turn over by then. It sets idle for ~3 ms between a turn and the message queued behind it
    # which a read can land on, and which came first; and it stamps a `!` command's record before the busy it
    # sets for it. Whether a record is from before hands followed the session is the tail's to say (see Tails._read).
    # [LAW:single-enforcer] a turn told is over, which `earlier` says of every one: a record of it read late opens
    # nothing, though no idle came between, as none does while a subagent works in the background.
    match session.state:
        case Idle(stamp=idled) | Running(idled=idled):
            # With no idle read, as for a session running since hands first read it, the turn its transcript is in is
            # the one running.
            return not followed(session, prompt) and (idled is None or (written is not None and written >= idled))
        case Unreported():
            # No status says whether any turn runs. The catch-up reads no session before its status (see Tails.catch_up); a
            # Stop's telling may, of the turn that Stop ends.
            return False


def _same_call(asked: Blocker, call: FinishedCall) -> bool:
    match (asked, call):
        case (Question(asked=questions), Question(asked=answered)):
            # A question answered at the keyboard comes back with the answers added to its input: it is the same
            # call when it asks the same questions.
            return questions == answered
        case (Plan(), PlanApproved()):
            # A session has one plan up at a time, and ExitPlanMode running is that plan approved.
            return True
        case _:
            return asked == call


def _unheard(event: SessionEvent, record: AuditRecord) -> list[Effect]:
    """What an event for a session the registry does not hold live calls for."""
    match event:
        case Taken() | Interrupted() | CarriedOut() | Continued() | Unqueued() | Launched() | ReportedBack() | Progressed() | Read() | Displayed():
            # Read from a transcript the tail goes on reading a moment after its session ends, or displayed after it:
            # behind, not wrong.
            return []
        case _:
            return [Audit(record), *_unwaited(event)]


def _unwaited(event: SessionEvent) -> list[Effect]:
    # A permission or Stop hook from a session this registry cannot block, such as one that started before the
    # daemon did, is let go at once rather than left to hang until Claude Code kills it.
    match event:
        case PermissionRequested(session=session, request=request) | Stopped(session=session, request=request):
            return [Reply(session, request, Withdraw())]
        case _:
            return []


def _abandoned(registry: Registry, session: SessionId, request: RequestId) -> Registry:
    match registry.sessions.get(session):
        case Session(dialogs=dialogs) as was if holds(was, request):
            # No hook waits for a reply, so there is nothing to withdraw, answer, or deny; the dialogs beneath it stay.
            return registry.put(replace(was, dialogs=tuple(held for held in dialogs if held.request != request)))
        case Session(stops=stops) as was if any(stop.hook == request for stop in stops):
            # A held Stop's hook stopped waiting: the Stop still tells its turn once decided, and answers no hook.
            return registry.put(replace(was, stops=tuple(replace(stop, hook=None) if stop.hook == request else stop for stop in stops)))
        case _:
            return registry


def _transition(before: Known | None, after: Session) -> list[Effect]:
    # [LAW:dataflow-not-control-flow] every change of a live session passes through here, so no event
    # can leave a hook waiting or a request unspoken by taking a path that forgot to.
    id = after.membership.id
    match before:
        case Session(dialogs=dialogs):
            return _dialogs(id, dialogs, after.dialogs)
        case None | Gone():
            return _dialogs(id, (), after.dialogs)


def _dialogs(session: SessionId, before: tuple[Held, ...], after: tuple[Held, ...]) -> list[Effect]:
    """Each dialog the session moved on from without a voice answer is let go, deciding nothing: the user answered it at
    the keyboard and the tool ran, or the turn went on or ended. Each dialog newly up is put to the user."""
    kept = {held.request for held in after}
    asked = {held.request for held in before}
    return [
        *(Reply(session, held.request, Withdraw()) for held in before if held.request not in kept),
        *(Narrate(Asking(session, held.request, held.on)) for held in after if held.request not in asked),
    ]


def _gathered(turn: Turn, event: Progressed | Displayed) -> Turn:
    """The turn with the calls it made and the text it wrote gathered, until they settle; either of a turn since over
    moves nothing, since that turn's result is told instead. Text is displayed for seconds after its turn's Stop."""
    match turn, event:
        case Opened(gathering=None), Displayed(text=text) if not text.strip():
            # Blank lines alone are nothing said, so they begin no burst: one would be told as the session's name and
            # nothing after it. Within a burst they part its paragraphs.
            return turn
        case Opened(gathering=gathering), Progressed(of=tuple() as of) | Displayed(turn=of) if not ids(turn).isdisjoint(of):
            began = Gathering((), "", event.at, event.at) if gathering is None else gathering
            match event:
                case Progressed(doings=doings):
                    return replace(turn, gathering=began.joined(doings, event.at), latest=doings[-1])
                case Displayed(text=text):
                    return replace(turn, gathering=began.wrote(text, event.at))
        case _:
            return turn


def _helped(subagents: Mapping[AgentTask, Gathering], agent: AgentTask, doings: tuple[Doing, ...], at: Instant) -> Mapping[AgentTask, Gathering]:
    """The subagent's calls gathered with what it did before them that nobody has been told of yet."""
    began = subagents.get(agent, Gathering((), "", at, at))
    return {**subagents, agent: began.joined(doings, at)}


def _ticked(registry: Registry, at: Instant) -> tuple[Registry, list[Effect]]:
    after, effects = registry, list[Effect]()
    for session in registry.live():
        id = session.membership.id
        turn, settled = _burst(id, session.turn, at)
        # [LAW:dataflow-not-control-flow] each subagent's burst settles on the same clock as its parent's own.
        due = {agent: gathering for agent, gathering in session.subagents.items() if at >= gathering.due()}
        subagents = {agent: gathering for agent, gathering in session.subagents.items() if agent not in due}
        after = after.put(replace(session, turn=turn, subagents=subagents))
        effects += [*settled, *(Progress(id, agent, gathering.doings, gathering.written) for agent, gathering in due.items())]
    return after, effects


def _burst(session: SessionId, turn: Turn, at: Instant) -> tuple[Turn, list[Effect]]:
    """[LAW:no-ambient-temporal-coupling] gathered calls are told at the tick that finds them settled, so how often the
    transcript is read moves when a burst is heard, never what it holds."""
    match turn:
        case Opened(gathering=Gathering() as gathering) if at >= gathering.due():
            return replace(turn, gathering=None), [Progress(session, ids(turn), gathering.doings, gathering.written)]
        case _:
            return turn, []
