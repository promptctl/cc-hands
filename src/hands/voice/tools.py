"""hands' own tools: the acts, the same to any caller.

Each is a `hands.voice.tool.Tool`: a plain async body, called with the model's arguments and returning what the model
is handed back. [LAW:decomposition] what each tool does knows nothing of who calls it: hands' MCP server is the adapter
over the bodies, and what a call is to a voice turn is `hands.voice.braindoor`'s.
"""

import asyncio
import functools
import json
import re
from datetime import timedelta
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal, TypedDict

from loguru import logger

from hands.clock import Clock
from hands.voice.modelswitch import OwnModel, chosen
from hands.voice.tool import Answer, Result, Tool, read_back, tool
from hands.brain.usage import Usage
from hands.core.drafts import AmendDraft, DiscardDraft, DraftAmended, DraftOutcome, DraftStaged, SendDraft, StageDraft
from hands.core.effects import Allow, Answers, Approve, Command, Decision, Deny, KeepPlanning, ModeAfterPlan
from hands.core.keyboard import Interrupt, SendCommand
from hands.core.session import Membership, CommandName, KEYSTROKES, PromptText, RequestId, Resolution, Session, SessionId, Staged
from hands.core.delta import Delta
from hands.core.attention import Attention, Kind, Overlay, Spoken, Withheld
from hands.core.drilldown import drill
from hands.core.summaries import Due, cut, turn_digest
from hands.core.tmux import InPane, NotInTmux, Pane, PaneUnread, Unanswered
from hands.core.standing import phase, spoken_state, standing_of
from hands.core.turn import Budget, Happening, Opening, body, describe, turns
from hands.sessions.child import Children
from hands.sessions.processes import Processes
from hands.sessions.backfill import Reading, read_transcript
from hands.sessions.backlog import BACKLOG, Backlog, Lit, LitFailed, Untracked
from hands.sessions import catchup, closesession, panes, startsession
from hands.sessions.terminals import Terminals
from hands.sessions.audit import Record
from hands.sessions.focus import Unreadable, focused
from hands.sessions.payload import Rejected
from hands.sessions.home import Home
from hands.sessions.overlays import Overlays
from hands.sessions.registry import Listing, Sessions
from hands.sessions import attention as settings
from hands.core.place import Modality
from hands.voice.listening import Level, Listening
from hands.voice.trigger import PhoneLevel, described, described_at_phone
from hands.voice.wakeword import Word
from hands.voice.narrator import Recount, Recounts, delivery, set_to
from hands.voice.summaries import Sighted, SummaryStore, sighted
from hands.voice.readback import identifier, keyboard_readback, readback, spoken_mode, spoken_name
from hands.voice.speech import answer_readback, told
from hands.voice.voices import Voices
from hands.voice.voicechoice import VOICES, parse_voice, spoken
from hands import spotify

# How much of a session one reading hands over. A session that has run for an hour has hundreds of steps, and
# all of them at once is a context spent on history; the rest are read on from `more_since`, in order.
READBACK_COUNT = 40

# What the agent reads when the user says no and gives no reason.
DENIED_BY_VOICE = "The user denied this by voice."

# What the agent reads when the user sends a plan back without saying what to change.
SENT_BACK_BY_VOICE = "The user sent the plan back by voice without saying what to change. Ask them what they want different."

# answer_plan's approvals, by the mode each leaves plan mode for.
_APPROVALS: Mapping[str, ModeAfterPlan] = {"approve": "resume", "auto-accept edits": "acceptEdits", "manually approve edits": "default"}


# A slash command's name, with the slash the model may have kept from what the user said.
_COMMAND_NAME = re.compile(r"/?([A-Za-z0-9][A-Za-z0-9_:-]*)")


def acts(
    sessions: Sessions, terminals: Terminals, processes: Processes, children: Children, store: SummaryStore, home: Home, recounts: Recounts, switch: Callable[[Modality], None], levels: Listening, at_phone: Callable[[], PhoneLevel | None], wake: Word, own: OwnModel, voices: Voices, catalogue: spotify.Catalogue, environment: Mapping[str, str],
    record: Record,
) -> list[Tool[Answer]]:
    """hands' own tools, in the order its schema lists them, each the act alone: [LAW:types-are-the-program] built with
    no player, no refocus and no cue, so none can speak a voice turn's line, wait on a heard line, or move the focus. A
    room act still sounds in the room, for any caller: hear_voices says voices, use_voice changes the voice hands speaks
    in, and Spotify plays. Every tool reads time on `sessions.clock`, the run's one clock, and runs a child through
    `children`. `environment` is the run's, which says where tmux keeps its sockets.

    [LAW:one-source-of-truth] the daemon serves the brain these through its `BrainDoor`, and the intermediary's eval
    (tests/test_intermediary.py) judges the prompt against them served the same way, so a tool added here is one the
    eval's model is offered too.
    """
    overlays = Overlays(home)
    return [
        list_sessions_tool(sessions, terminals, processes, overlays, home, environment),
        *session_tools(sessions, store),
        tell_turn_tool(sessions, recounts),
        expand_tool(sessions, recounts),
        *backlog_tools(sessions, store, Lit(children, environment)),
        *draft_tools(sessions),
        *keyboard_tools(sessions),
        read_screen_tool(sessions, terminals, processes, environment),
        set_overlay_tool(sessions, overlays),
        *permission_tools(sessions),
        catch_up_tool(sessions, home),
        attention_tool(home),
        modality_tool(switch),
        *listening_tools(levels, at_phone, wake),
        *voice_tools(voices),
        *model_tools(own),
        *spotify_tools(spotify.Player(functools.partial(spotify.osascript, children)), catalogue),
        # A session is started by the daemon, in the environment hands was started in, which is the user's, and is started
        # once the registry holds it, so the brain can stage for it at once; it is closed once the registry no longer does.
        start_session_tool(sessions.clock, terminals, processes, home, record, environment, sessions.live_members),
        close_session_tool(processes, home, record, sessions),
    ]


def list_sessions_tool(sessions: Sessions, terminals: Terminals, processes: Processes, overlays: Overlays, home: Home, environment: Mapping[str, str]) -> Tool[Result]:
    async def list_sessions() -> Result:
        """List the running Claude Code sessions by name, what each is doing, and the permission mode each is in.

        A session's name is its project, then a short name for its work: "cc-hands, naming fix". That is how the user
        speaks of a session, and how you speak of one to them.

        Call this when the user asks what is running, what sessions exist, what
        Claude is working on, or what mode a session is in. A session's mode is the
        one it reported when it last did something: a mode changed at its keyboard
        while it sits at its prompt is seen when it is next prompted, and one changed
        in the middle of a turn at its next tool call. `overlay` says how the user hears the turns it finishes:
        watched, normal, or muted (set_overlay). `focus` is the id of the session the user is talking to when they name
        none (focus_session), null when none is focused, or says why it cannot be read. `tmux` is the tmux pane a
        session runs in, as it is now: its server's `socket`, the `pane`, and the tmux `session` and `window` it is in,
        so keys reach it by `tmux -S <socket> send-keys -t <pane>`; or "not in tmux", or says why it cannot be read.
        """
        listings = sessions.live()
        in_panes, focus = await asyncio.gather(panes.panes(terminals, processes, [listing.session.membership.pid for listing in listings], environment), _focus(home))
        return {
            "sessions": [{**describe_listing(listing), "overlay": await _overlay(overlays, listing.session.membership.id), "tmux": _in_pane(pane)} for listing, pane in zip(listings, in_panes, strict=True)],
            "focus": focus,
        }

    return tool(list_sessions)



def read_screen_tool(sessions: Sessions, terminals: Terminals, processes: Processes, environment: Mapping[str, str]) -> Tool[Result]:
    async def read_screen(session: str) -> Result:
        """What a session's terminal shows now, as text, read off the tmux pane it runs in.

        Call this before you send keys to a session at a dialog hands let go of, to see the dialog and its options, and
        when the user asks what such a dialog asks: say what it asks and its options in a sentence or two, never the
        screen whole. `screen` is the pane's text, down to its last line drawn on; `tmux` is the pane it was read
        from, as list_sessions names it. A session in no tmux pane has no screen hands can read, and the error says so.

        Args:
            session: The session's id, from list_sessions.
        """
        id = SessionId(session)
        live = sessions.live_session(id)
        if live is None:
            return {"error": f"no session {session} is running: list_sessions names the ones that are"}
        [pane] = await panes.panes(terminals, processes, [live.membership.pid], environment)
        match pane:
            case Pane(socket=socket, id=pane_id):
                match await terminals.shown(socket, pane_id):
                    case str(screen):
                        return {"screen": screen, "tmux": _in_pane(pane)}
                    case Unanswered(reason=reason):
                        return {"error": f"the screen of {spoken_name(sessions, id)} could not be read: {reason}", "tmux": _in_pane(pane)}
            case NotInTmux():
                return {"error": f"{spoken_name(sessions, id)} runs in no tmux pane, and hands reads a screen only off one"}
            case PaneUnread(reason=reason):
                return {"error": f"which tmux pane {spoken_name(sessions, id)} runs in could not be read: {reason}"}

    return tool(read_screen)


def _in_pane(pane: InPane) -> str | Mapping[str, str | int]:
    match pane:
        case Pane(socket=socket, id=id, session=session, window=window):
            return {"socket": str(socket), "pane": id, "session": session, "window": window}
        case NotInTmux():
            return "not in tmux"
        case PaneUnread(reason=reason):
            return {"cannot_read": reason}


async def _focus(home: Home) -> SessionId | None | Mapping[str, str]:
    match await asyncio.to_thread(focused, home):
        case Unreadable(reason):
            return {"cannot_read": reason}
        case focus:
            return focus


async def _overlay(overlays: Overlays, session: SessionId) -> str:
    try:
        return await asyncio.to_thread(overlays.of, session)
    except (Rejected, OSError) as error:
        logger.error(f"cannot read the overlay of session {session} to list it: {error}")
        return f"unknown, its setting cannot be read: {error}"


# How much of one step the intermediary is shown when it reads a session back: enough to say what happened,
# and short enough that a screenful of them still leaves room for the conversation they are read into.
# A reading is of the transcript alone, so it shows no repository delta and budgets none: what a turn changed
# in git is told when the turn stops, to the summariser, and is not a session's to be read back out of.
READBACK_BUDGET = Budget(opening=200, said=400, input=120, result=200, steps=READBACK_COUNT, files=0, commits=0, changes=0)


# How much of a finished turn the summariser is shown to say it in one sentence: its request, and how it started and
# ended. What is longer than the summariser is shown of one thing loses its middle, never its end.
TURN_SUMMARY_BUDGET = Budget(opening=400, said=250, input=100, result=120, steps=16, files=0, commits=0, changes=0)

# How many turns one reading of a session hands over, newest last: a day of work is hundreds of turns, and what it did
# lately is what is asked about.
TURNS_PAGE = 40


def session_tools(sessions: Sessions, store: SummaryStore) -> list[Tool[Result]]:
    """read_session, read_turn: what a session has done, a summary per finished turn first and one turn's steps on request."""

    async def read_session(session: str, before: int = 0) -> Result:
        """What a session has done, one turn at a time, in order: a summary for each turn it has finished, and the request of the one it is on.

        Call this when the user asks what a session has been doing, or to catch up on one that was already
        running before hands attached. Answer from the summaries; call read_turn to hear more of one turn, or of the
        one it is on. A finished turn with no summary yet has only what opened it (a request, or a command the user ran), and its summary is being written.
        When `working` comes back true the session is still on its last turn. You are given its newest turns; when
        `earlier` is above zero there are that many before them, and calling again with `before` set to the first
        turn you were given reads those.

        Args:
            session: The session's id, from list_sessions.
            before: A turn number from an earlier call, to read the turns before it. 0 reads the newest.
        """
        found = await _session_reading(sessions, session)
        if isinstance(found, str):
            return {"error": found}
        member, reading = found
        spans = turns(reading.happenings)
        if not (before == 0 or 1 <= before <= len(spans)):
            return {"error": f"session {session} has turns 1 to {len(spans)}, and no turn {before}"}
        end = len(spans) if before == 0 else before - 1
        start = max(0, end - TURNS_PAGE)
        live = sessions.live_session(member.id)
        # [LAW:one-source-of-truth] the file cannot tell a turn that ended from one waiting on a long call, so whether
        # the last turn is over is the standing's to say: a summary of half a turn would be kept for good.
        stands = None if live is None else standing_of(live)
        over = phase(stands) in ("ended", "over")
        entries: list[dict[str, object]] = []
        due: list[Due] = []
        for number in range(start + 1, end + 1):
            happened = reading.happenings[spans[number - 1].start : spans[number - 1].stop]
            key = turn_digest(happened) if number < len(spans) or over else None
            summary = None if key is None else store.known(key)
            if key is not None and summary is None:
                due.append(Due(f"turn-{number}", key, body(happened, Delta(), TURN_SUMMARY_BUDGET), ()))
            entries.append({"turn": number, **({"summary": summary} if summary is not None else _opened(happened[0]))})
        # Every read is a sighting: the turns without a summary are said in the background, never while this call waits.
        store.want_turns(member.id, due)
        return {
            "turns": entries,
            "earlier": start,
            "unsummarised": len(due),
            "working": phase(stands) == "mid-turn",
        }

    async def read_turn(session: str, turn: int, since: str = "") -> Result:
        """The steps of one turn of a session, in the order it took them, from the point you last read to.

        Call this when the user wants more of a turn than its summary, or wants to know what a session is doing
        in the turn it is on. When `more` comes back true there is more of the turn after what you were given:
        ask the user whether to hear it, and call again with `since` set to `more_since` if they want it. When
        `working` comes back true the session is in the middle of a call that has not come back; say what it is
        in the middle of, and read on later rather than now, when what it did will be there.

        Args:
            session: The session's id, from list_sessions.
            turn: The turn's number, from read_session.
            since: The record id you last read to, from an earlier call's `more_since`. Empty reads from the turn's start.
        """
        found = await _session_reading(sessions, session)
        if isinstance(found, str):
            return {"error": found}
        member, reading = found
        spans = turns(reading.happenings)
        if not 1 <= turn <= len(spans):
            return {"error": f"session {session} has turns 1 to {len(spans)}, and no turn {turn}"}
        span = spans[turn - 1]
        # A mark names a record and a reading goes on from after the whole of it.
        marked = [index for index in span if since and reading.happenings[index].ref == since]
        if since and not marked:
            # [LAW:no-silent-failure] a mark from another turn, or from a transcript since rewritten, is said rather
            # than read as "from the start", which would tell the whole turn over again unasked.
            return {"error": f"turn {turn} of session {session} has no record {since}; call again with since empty to read from its start"}
        start = marked[-1] + 1 if marked else span.start
        rest = reading.happenings[start : span.stop]
        # The earliest of what it has not had, not the newest: read on from `more_since` and a turn is
        # caught up on in order, which is the only order any of it makes sense in.
        shown = _page(rest)
        # A call the session is still waiting on is shown but never marked as read, so its result is told once
        # it lands rather than falling into the gap between one reading and the next.
        # [LAW:one-source-of-truth] whether a session can still answer a call is the registry's to say, not the
        # file's. One that has ended will never write the result of the call it was killed inside, and a mark
        # held behind that call would leave the intermediary saying a dead session is still running something.
        ended = sessions.live_session(member.id) is None
        settled = shown if ended else shown[: max(0, min(reading.settled - start, len(shown)))]
        # [LAW:one-source-of-truth] the mark names a record, and one record can carry both a settled happening
        # and the call the session is still inside — the text and the call it introduces are written together.
        # Marking that record would go on from after the whole of it, losing the very result the mark is held
        # back for, so the mark is the last record every happening of which is settled.
        waiting = {happening.ref for happening in shown[len(settled) :]}
        return {
            "happened": [{"record": happening.ref, "what": describe(happening, READBACK_BUDGET)} for happening in shown],
            # Two different facts, so two answers: more of the turn this reading did not reach, and a call that has
            # not come back. Told as one, the model cannot tell "read on" from "wait and ask again".
            "more": len(rest) > len(shown),
            "working": len(settled) < len(shown),
            # A record carries no uuid only rarely, and naming nothing reads as "from the start" next time.
            "more_since": next(
                (happening.ref for happening in reversed(settled) if happening.ref is not None and happening.ref not in waiting),
                since,
            ),
        }

    return [tool(read_session), tool(read_turn)]


def _opened(first: Happening) -> dict[str, object]:
    """How a turn with no summary is told: by what opened it, or, in a transcript that starts part way through one, by the
    step it was first read at, which opened nothing."""
    return {"opened" if isinstance(first, Opening) else "began": describe(first, READBACK_BUDGET)}


async def _session_reading(sessions: Sessions, session: str) -> tuple[Membership, Reading] | str:
    """All of what a session has done, read fresh from its transcript; or why it could not be read."""
    member = sessions.membership(SessionId(session))
    if member is None:
        return f"there is no session {session}"
    try:
        return member, await asyncio.to_thread(read_transcript, member.transcript)
    except OSError as error:
        # [LAW:no-silent-failure] the model is told why it got nothing, rather than being handed nothing.
        logger.error(f"cannot read what session {session} did from {member.transcript}: {error}")
        return f"the transcript of session {session} could not be read"


def _page(happenings: list[Happening]) -> list[Happening]:
    """As much of a reading as one call hands over, ending where a record does.

    The mark the reader comes back with names a record, and a reading goes on from after that record, so a page
    that ended inside one would lose the rest of it [LAW:one-source-of-truth]. Nearly every record carries one
    happening and cannot be split: 8 of the 628,822 on this machine carry more than one — a text and the call
    it introduces, and one record with two calls in it. Rare enough to be left to chance is exactly what this
    is not, because the loss is silent: the reading simply never mentions what the skipped block did.
    """
    end = min(READBACK_COUNT, len(happenings))
    ref = happenings[end - 1].ref if end else None
    if end == len(happenings) or ref is None or happenings[end].ref != ref:
        return happenings[:end]
    start = end
    while start > 0 and happenings[start - 1].ref == ref:
        start -= 1
    if start > 0:
        return happenings[:start]
    # One record carrying a whole page of them: handed over long rather than short, because a page cut to fit
    # would name that record as the mark and drop everything after the cut [LAW:no-silent-failure].
    while end < len(happenings) and happenings[end].ref == ref:
        end += 1
    return happenings[:end]


def backlog_tools(sessions: Sessions, store: SummaryStore, lit: Lit) -> list[Tool[Result]]:
    """read_backlog, read_ticket: a session's project backlog, a summary per ticket first and the ticket's own words on request."""

    async def read_backlog(session: str) -> Result:
        """The backlog of the project a session works in: one summary for the whole of it, and each epic and loose ticket in the order they are to be worked, each with its own summary.

        Call this when the user asks what is in the backlog, what is left to do, or what comes next. Answer from the
        summaries; call read_ticket to hear more of one. A ticket with no summary yet has only its title, and its
        summary is being written. `directory` is where this backlog was read; lit run there works its tracker.
        `tracked` false means the project has no backlog at all, and `why` says why: lit has no workspace there, or
        the directory no longer exists.

        Args:
            session: The id, from list_sessions, of a session working in the project.
        """
        match await _backlog(sessions, store, session, lit):
            case str() as error:
                return {"error": error}
            case Untracked(project) as untracked:
                return {"directory": str(project), "tracked": False, "why": untracked.why, "items": []}
            case (project, Sighted(backlog=backlog, said=said)):
                roots = backlog.roots()
                return {
                    "directory": str(project),
                    **({"summary": said[BACKLOG]} if BACKLOG in said else {}),
                    "items": [_ticket_line(backlog, said, id) for id in roots],
                    "unsummarised": sum(id not in said for id in roots),
                }

    async def read_ticket(session: str, ticket: str, full: bool = False) -> Result:
        """One ticket or epic from a session's project backlog: its summary, where it sits, and what is still open under it.

        Call this when the user asks about a ticket or an epic, such as how an epic is going. Set `full` only when
        the user wants the detail the summary leaves out: it hands over the ticket's whole description and every
        comment on it, which is long.

        Args:
            session: The id, from list_sessions, of a session working in the project.
            ticket: The ticket's id, from read_backlog or an earlier read_ticket.
            full: True to be handed the ticket's own words: its description and its comments.
        """
        match await _backlog(sessions, store, session, lit):
            case str() as error:
                return {"error": error}
            case Untracked(project) as untracked:
                return {"error": f"{project} has no backlog, so no ticket {ticket}: {untracked.why}"}
            case (_, Sighted(backlog=backlog)) if ticket not in backlog.tickets:
                return {"error": f"the backlog has no ticket {ticket}"}
            case (_, Sighted(backlog=backlog, said=said)):
                parent = backlog.parent.get(ticket)
                found = backlog.tickets[ticket]
                return {
                    **_ticket_line(backlog, said, ticket),
                    **({} if parent is None else {"parent": _ticket_line(backlog, said, parent)}),
                    "open_children": [_ticket_line(backlog, said, child) for child in backlog.open_children(ticket)],
                    **({"description": found.description, "comments": [{"by": comment.by, "at": comment.at, "body": comment.body} for comment in backlog.comments.get(ticket, ())]} if full else {}),
                }

    return [tool(read_backlog), tool(read_ticket)]


async def _backlog(sessions: Sessions, store: SummaryStore, session: str, lit: Lit) -> tuple[Path, Sighted] | Untracked | str:
    """The session's project and its backlog sighted there by `lit`; that lit has no workspace there, so it has no backlog; or
    why it could not be read."""
    member = sessions.membership(SessionId(session))
    if member is None:
        return f"there is no session {session}"
    try:
        read = await sighted(lit, store, member.cwd)
    except (LitFailed, Rejected) as error:
        # [LAW:no-silent-failure] the model is told why it got nothing, and the log keeps it.
        logger.error(f"cannot read the backlog of session {session} in {member.cwd}: {error}")
        return f"the backlog in {member.cwd} could not be read: {error}"
    match read:
        case Untracked():
            return read
        case Sighted():
            return member.cwd, read


def _ticket_line(backlog: Backlog, said: Mapping[str, str], id: str) -> dict[str, object]:
    """A ticket as the backlog tools hand it over: its summary where one is said, its state, and for a parent, how far its children are."""
    ticket = backlog.tickets[id]
    children = backlog.children.get(id, ())
    open_children = backlog.open_children(id)
    return {
        "id": id,
        "title": ticket.title,
        **({"summary": said[id]} if id in said else {}),
        # An epic has no status of its own in lit; how far it is lies in its children's counts.
        **({} if ticket.status is None else {"status": ticket.status}),
        **({"children_open": len(open_children), "children_done": len(children) - len(open_children)} if children else {}),
    }


def standing(sessions: Sessions) -> list[dict[str, str]]:
    """Every live session by its id, name, state and mode, as list_sessions describes each before what it adds."""
    return [describe_listing(listing) for listing in sessions.live()]


def describe_listing(listing: Listing[Session]) -> dict[str, str]:
    return {
        "id": listing.session.membership.id,
        "name": identifier(listing),
        "state": spoken_state(listing.session),
        "mode": "not reported yet" if listing.session.mode is None else spoken_mode(listing.session.mode),
    }


def tell_turn_tool(sessions: Sessions, recounts: Recounts) -> Tool[Result]:
    async def tell_turn(session: str) -> Result:
        """What a session's last finished turn did, as hands tells a turn when it finishes, and how it stands now.

        Call this when the user asks what a session just did or how its last turn went. Tell them as the returned turn
        says to. `now` is how the session stands at this moment, as list_sessions says it: a question the turn ended
        on may have been answered at the keyboard since, and a session working again is no longer waiting on it.

        Args:
            session: The session's id, from list_sessions.
        """
        id = SessionId(session)
        live = sessions.live_session(id)
        if live is None:
            return {"error": f"no running session has the id {id!r}; take one from list_sessions"}
        name = spoken_name(sessions, id)
        now = spoken_state(live)
        match recounts.of(id):
            case None:
                # [LAW:no-silent-failure] said as what it is, never as a turn that did nothing.
                return {"error": f"no turn of {name} has finished since hands started; read_session reads what it did before"}
            case Recount(tellings=(), unread=False):
                turn = f"[hands] The Claude Code session {name} finished a turn with nothing in it hands could tell. Tell the user so."
            case Recount(tellings=tellings, unread=unread):
                failed = (f"[hands] hands could not read {'the rest of ' if tellings else ''}the turn the Claude Code session {name} finished. Tell the user so.",)
                turn = "\n\n".join((*(told(id, name, (telling,), "full") for telling in tellings), *(failed if unread else ())))
        return {"turn": turn, "now": now}

    return tool(tell_turn)


def expand_tool(sessions: Sessions, recounts: Recounts) -> Tool[Result]:
    async def expand(session: str, part: str = "") -> Result:
        """More of the last turn a session finished: with no part, the parts it opens into, such as the change, the
        tests, or the commit, each in a line; with a part, that part's records, at more length each time it is asked for.

        When `deeper` comes back false, asking for that part again tells no more.

        Args:
            session: The id of the session whose turn you just told, from the [hands] message that told it or from list_sessions.
            part: The part the user wants more of, by the name a call with it empty gave, such as "the tests". Empty for the list of parts.
        """
        id = SessionId(session)
        name = spoken_name(sessions, id)
        held = recounts.of(id)
        if held is None or not held.parts:
            # [LAW:no-silent-failure] said as what it is, never as a turn with nothing in it.
            return {"error": f"hands holds no finished turn of {name} to open; read_session and read_turn read what it did"}
        topic = part.strip().lower()
        chosen = held.parts if not topic else tuple(segment for segment in held.parts if segment.topic.name == topic)
        if not chosen:
            named = ", ".join(dict.fromkeys(segment.topic.name for segment in held.parts))
            return {"error": f"the last turn of {name} has no part {part!r}; its parts are {named}"}
        # The turn as a whole is told only as its parts' lines, the first rung, however often it is asked for: told
        # deeper, every part at once would be more than one answer can carry. A part asked for by name opens at the
        # rung below its line, and a rung further each time it is asked for: [LAW:one-source-of-truth] its depth is
        # counted by the part, so the list asked for between two askings neither moves nor repeats it.
        depth = recounts.open(id, topic) + 1 if topic else 0
        drilled = drill(chosen, depth)
        # [LAW:nothing-unseen] the depth rides on the result, so the call's event says how far down this asking went.
        return {"parts": [{"part": topic, "told": told} for topic, told in drilled.told], "deeper": drilled.deeper, "depth": depth}

    return tool(expand)


# How much of the sessions' closing words one catch-up carries, shared among the sessions that finished: a morning away
# can be a hundred sessions, and every one of them is still named. Each gets its share, and never less than enough to
# say what came of its work.
CATCH_UP_CLOSINGS = 12000
CATCH_UP_LEAST = 150


def _shown(value: object) -> object:
    """A field of an occurrence as catch_up hands it on: text, or a call's input, bounded as a closing's least share is."""
    match value:
        case str():
            return cut(value, CATCH_UP_LEAST)
        case dict():
            return cut(json.dumps(value, ensure_ascii=False), CATCH_UP_LEAST)
        case _:
            return value


def catch_up_tool(sessions: Sessions, home: Home) -> Tool[Result]:
    async def catch_up(minutes: int = 0) -> Result:
        """What the user missed: every session that finished work while they were away and the newest words each closed
        a turn with, the sessions that ended, what their hooks said happened, and what hands announced.

        Call this when the user asks what they missed, what happened while they were away, or what went on in the last
        while. Sum it up the way a colleague would after a break: each session that finished in a sentence, by name, then
        any that ended. Leave nothing out of `finished`: the user is asking because they heard none of it. `turns` is
        how many turns a session finished; read_session reads what each did, when they want more of one. `occurred` is
        what sessions' hooks said happened, each kind once a session with how many `times` and the newest one's
        fields, by its type: AutoDenied, auto mode refusing a call; SubagentStarted and SubagentStopped; TaskCompleted;
        ConfigChanged, its settings or skills changing; Compacting; Cleared, a /clear.

        Args:
            minutes: How far back to look, when the user says, such as 60 for "the last hour". 0 for since the user last spoke to hands before this.
        """
        if minutes < 0:
            return {"error": f"minutes is how far back to look, so it cannot be {minutes}"}
        at = sessions.clock.wall()
        opening = catchup.LastSpoke() if minutes == 0 else at - timedelta(minutes=minutes)
        try:
            missed = await asyncio.to_thread(catchup.missed, home.audit, opening)
        except OSError as error:
            return {"error": f"hands could not read its log: {error}"}
        share = max(CATCH_UP_LEAST, CATCH_UP_CLOSINGS // max(1, len(missed.finished)))
        return {
            # [LAW:nothing-unseen] where the window opened and what of it could not be read ride on the result, so the
            # call's event says what was read.
            "since_minutes_ago": None if missed.since is None else round((at - missed.since).total_seconds() / 60),
            "unreadable_lines": missed.unreadable,
            "finished": [
                {"session": spoken_name(sessions, done.session), "turns": done.turns, "closing": None if done.closing is None else cut(done.closing, share)}
                for done in missed.finished
            ],
            "ended": [spoken_name(sessions, session) for session in missed.ended],
            "occurred": [
                {"session": spoken_name(sessions, happened.session), "times": happened.times, **{key: _shown(value) for key, value in happened.newest.items()}}
                for happened in missed.occurred
            ],
            "announced": [{"text": said.text, "times": said.times} for said in missed.announced],
        }

    return tool(catch_up)


class Change(TypedDict):
    kind: Kind
    level: Literal["off", "brief", "full", "on"]


def attention_tool(home: Home) -> Tool[Result]:
    async def attention(changes: list[Change]) -> Result:
        """Change what hands says to the user without being asked, or hear what is set: with no changes, it only says.

        Kinds, and their levels:
        - finished: each turn a session finishes. off holds it until they ask (tell_turn, catch_up), except a watched
          session's; brief tells it in a few words; full tells what it did.
        - progress: the focused session's steps as it works. off says none; brief says what it says it is doing; full
          says each step.
        - ended: a session ending, on or off.
        - Claude Code's hooks, each off, brief (what happened), or full (and what the hook says of it), and off until
          set: permission_denied, auto mode refusing a call; subagent_start and subagent_stop, a subagent the session
          started starting and finishing; task_completed, a task on its list marked done; config_change, its settings
          or skills changing; pre_compact, its context about to be compacted; clear, a /clear.
        - quiet: on holds everything above, whatever its level, until it is off again, and leaves the levels as they
          were. "Be quiet for a while" is quiet on; "you can talk again" is quiet off.
        What a session asks, a permission, a question, or a plan, is said whatever is set: it needs an answer.

        Call this as soon as the user says what they want to hear more or less of, mapping their words to the nearest
        kind and level yourself, several changes at once if they said several; never ask which they meant. "Stop telling
        me when sessions finish" is finished off; "just the headlines" is finished brief; "less detail while it works" is
        progress brief. Call it with no changes when they ask what hands tells them. It lasts until they change it,
        across restarts, from the next thing hands would say. Say the returned readback to the user.

        Args:
            changes: each kind to set and its level, in the order said; none to hear what is set.
        """
        try:
            said = [(change["kind"], change["level"]) for change in changes]
            to = await asyncio.to_thread(settings.asked, home, said)
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        return {"readback": settings.described(to)}

    return tool(attention)


def modality_tool(switch: Callable[[Modality], None]) -> Tool[Result]:
    async def set_modality(modality: Modality) -> Result:
        """Take the user as able to see a screen, or as audio-only, from now on.

        Each of their turns says which they are: talking at the Mac starts as screen, and from the phone as audio-only.
        This switches it until hands next moves between the Mac and the phone, as a call comes or goes. It is a hint for choosing what to do, never a limit on what
        hands can do. Call this when the user asks to go audio-only, or back to a screen. Say the returned readback.

        Args:
            modality: screen, or audio-only.
        """
        switch(modality)
        return {"modality": modality, "readback": _modality_readback(modality)}

    return tool(set_modality)


def _modality_readback(modality: Modality) -> str:
    match modality:
        case "screen":
            return "Okay, you can see a screen."
        case "audio-only":
            return "Okay, audio only."


def listening_tools(levels: Listening, at_phone: Callable[[], PhoneLevel | None], wake: Word) -> list[Tool[Result]]:
    """How the place hands is at listens between the user's turns. At the Mac: saying the level the desk is at, and
    choosing another while hands runs, as a triple tap does; `wake` is the wake word. At the phone, `at_phone` says the
    phone's level, which only the page's own button sets, so choosing says so and changes neither level."""

    async def listening_level() -> Result:
        """Say how hands listens between turns where the user is: at the desk, off, for the wake word, or engaged, hearing
        speech alone; at the phone, off or for the wake word.

        Call this when the user asks how to talk to hands, or whether the microphone is on. Say the returned readback.
        """
        match at_phone():
            case None:
                return {"level": levels.level, "readback": described(levels.level, wake)}
            case phone:
                return {"level": phone, "readback": described_at_phone(phone)}

    async def set_listening_level(level: Level) -> Result:
        """Choose how the desk listens between the user's turns; holding Right Shift talks at every level. At the phone this
        changes nothing, and the readback says what sets the phone's level.

        Call this when the user asks for the microphone off, for the wake word, or for engaged conversation. Say the
        returned readback.

        Args:
            level: "off", the microphone closed between turns but for a few seconds after each reply at the desk, for
                the user's answer; "wake word", listening for the wake word, and for an answer as "off" does; or
                "engaged", where speech alone opens each turn.
        """
        match at_phone():
            case None:
                pass
            case phone:
                # [LAW:one-source-of-truth] the phone's level is its page's: a choice made here would be a second setting for it.
                return {"level": phone, "readback": described_at_phone(phone)}
        was = await levels.choose(level, "tool")
        # [LAW:nothing-unseen] the level asked from lands on the call's event, so a change and a no-op read apart.
        return {"level": level, "was": was, "readback": f"{'Already' if was == level else 'Okay'}. {described(level, wake)}"}

    return [tool(listening_level), tool(set_listening_level)]


def voice_tools(voices: Voices) -> list[Tool[Result]]:
    """Choosing the voice hands speaks in, by ear: the user hears the voices said by hands, and keeps one.

    [LAW:nothing-unseen] each result names the voice hands speaks in after the call, and a hearing the voices it said,
    so the call's event holds what was heard and what was kept.
    """

    async def voices_on_offer() -> Result:
        """The voices hands can speak in, and the one it speaks in now.

        Call this when the user asks what voice hands speaks in, or which voices there are. Say a voice's name as a person
        would: bill_boerst is Bill Boerst. Voices in Kyutai's voice library are not listed here; find_voices searches them.
        """
        return {"speaking_in": str(voices.speaking_in()), "voices": VOICES}

    async def hear_voices(names: list[str]) -> Result:
        """Hands says a line in each voice named, one after another, each saying the line the user set for it or, where
        they set none, its own name, then speaks on in the voice it was using.

        Call this when the user wants to hear what voices sound like: a few at a time, at most five unless they ask for
        more, since each takes a few seconds. Hearing a voice does not choose it; use_voice does. To hear voices again,
        call this again.

        Args:
            names: the voices to hear, as voices_on_offer lists them.
        """
        try:
            heard = tuple(parse_voice(name) for name in names)
            return {"heard": [str(voice) for voice in heard], "speaking_in": str(await voices.hear(heard))}
        except (Rejected, OSError) as error:
            return {"error": str(error)}

    async def use_voice(name: str) -> Result:
        """Hands speaks in this voice from now on. It lasts until the user chooses another, across restarts.

        Call this when the user picks a voice. Say the returned readback.

        Args:
            name: the voice, as voices_on_offer lists it.
        """
        try:
            voice = parse_voice(name)
            await voices.use(voice)
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        return {"speaking_in": str(voice), "readback": f"This is {spoken(voice)}, and I'll speak in this voice from now on."}

    async def find_voices(query: str, limit: int = 10) -> Result:
        """Search Kyutai's voice library for voices whose paths contain the words given. Each name it returns is a voice
        hear_voices and use_voice take, by its path there.

        Call this when the user asks for a voice that is not one of the built-in names, or asks what voices the library has.
        Say the names the search found, and if the listing was made for this search, that it was just made.

        Args:
            query: the words to look for in a voice's path, such as vctk or p225.
            limit: the most names to return.
        """
        try:
            found = await voices.search(query, limit)
        except OSError as error:
            return {"error": str(error)}
        return {"voices": list(found.names), "listed_at": found.listed_at, "from_cache": found.from_cache}

    return [tool(voices_on_offer), tool(hear_voices), tool(use_voice), tool(find_voices)]


def model_tools(own: OwnModel) -> list[Tool[Answer]]:
    """Choosing the model hands runs on, which is a setting of hands' own and never a session's.

    [LAW:nothing-unseen] each result names the model hands runs on and, for a choice, the one it switches to, so the
    call's event holds both.
    """

    async def model_in_use() -> Result:
        """The model hands runs on now. Call this when the user asks what model hands runs on."""
        return {"running_on": own.running()}

    async def use_model(model: str) -> Answer:
        """Run hands on another model: hands restarts on it, back in a few seconds, and stays on it across restarts.

        Call this when the user asks to change hands' own model. It never touches a session; a session's model is changed
        by send_command running model in it.

        Args:
            model: the model's id, such as claude-opus-5-5.
        """
        try:
            return (await chosen(own, model)).answer
        except (Rejected, OSError) as error:
            return {"error": str(error)}

    return [tool(model_in_use), tool(use_model)]


def usage_tool(usage: Usage) -> Tool[Result]:
    """[LAW:nothing-unseen] the result is the reading, so the call's event holds it."""

    async def context_usage() -> Result:
        """How many tokens hands' main conversation holds now, and how many hands has spent since it started, as the API
        counted them on its replies.

        Call this when the user asks how much context or how many tokens hands has used. These are the real figures.
        """
        reading = usage.reading()
        if reading.spent.replies == 0:
            return {"error": "No reply has been counted yet."}
        # A conversation the brain has moved to and had no reply in yet holds nothing counted: its model and size are null.
        held = reading.in_context
        return {
            "model": None if held is None else held.model,
            "in_context_tokens": None if held is None else held.tokens,
            "spent": {"input_tokens": reading.spent.input_tokens, "output_tokens": reading.spent.output_tokens, "replies": reading.spent.replies},
        }

    return tool(context_usage)


def spotify_tools(player: spotify.Player, catalogue: spotify.Catalogue) -> list[Tool[Result]]:
    """Spotify on this Mac: what it plays, played, paused, skipped, and set, and its catalogue searched for what to play.

    [LAW:nothing-unseen] each result is what Spotify said or did, or why it would not, so the call's event holds it.
    """

    async def spotify_now_playing() -> Result:
        """What Spotify is playing on this Mac: whether it is playing, paused, stopped, or not running, and the track."""
        return await _refusable(_now(player))

    async def spotify_play(link: str = "") -> Result:
        """Play music in Spotify on this Mac, starting Spotify if it is not running.

        To play something the user names, find it with spotify_search and pass the link of the one they meant.

        Args:
            link: The spotify: link of the track, album, artist, or playlist to play from its start, or an
                open.spotify.com link the user gave. Leave it out to go on with what was playing.
        """
        uri = spotify.link(link) if link else None
        if link and uri is None:
            return {"error": f"{link!r} is no Spotify link; find what to play with spotify_search"}
        return await _refusable(_done(player.play(uri), {"playing": uri or "what was playing"}))

    async def spotify_pause() -> Result:
        """Pause Spotify on this Mac."""
        return await _refusable(_done(player.pause(), {"paused": True}))

    async def spotify_skip(to: Literal["next", "previous"]) -> Result:
        """Skip Spotify on this Mac to the next track, or back to the previous one.

        Args:
            to: Which track to skip to.
        """
        return await _refusable(_done(player.skip(to), {"skipped_to": to}))

    async def spotify_volume(level: int) -> Result:
        """Set Spotify's own volume on this Mac, which is not the Mac's.

        Args:
            level: From 0, silent, to 100, its loudest.
        """
        if not 0 <= level <= 100:
            return {"error": f"Spotify's volume runs from 0 to 100, not {level}"}
        return await _refusable(_done(player.volume(level), {"volume": level}))

    async def spotify_shuffle(on: bool) -> Result:
        """Turn Spotify's shuffle on or off on this Mac.

        Args:
            on: True to shuffle, false to play in order.
        """
        return await _refusable(_done(player.shuffle(on), {"shuffle": on}))

    async def spotify_repeat(on: bool) -> Result:
        """Turn Spotify's repeat on or off on this Mac.

        Args:
            on: True to repeat, false not to.
        """
        return await _refusable(_done(player.repeat(on), {"repeat": on}))

    async def spotify_search(query: str, kind: spotify.Kind = "track") -> Result:
        """Search Spotify's catalogue for something to play: up to five, best match first, each with the link
        spotify_play takes.

        Args:
            query: What the user named, in their words: a song, an album, an artist, or a playlist, and who made it
                where they said.
            kind: What they asked for.
        """
        return await _refusable(_searched(catalogue, query, kind))

    return [
        tool(spotify_now_playing),
        tool(spotify_play),
        tool(spotify_pause),
        tool(spotify_skip),
        tool(spotify_volume),
        tool(spotify_shuffle),
        tool(spotify_repeat),
        tool(spotify_search),
    ]


async def _refusable(act: Awaitable[Result]) -> Result:
    """What `act` handed back, or, where Spotify refused it, why: a refusal the model is asked to answer."""
    try:
        return await act
    except spotify.Refused as refused:
        return {"error": str(refused)}


async def _now(player: spotify.Player) -> Result:
    return spotify.said(await player.now_playing())


async def _done(act: Awaitable[None], result: Result) -> Result:
    await act
    return result


async def _searched(catalogue: spotify.Catalogue, query: str, kind: spotify.Kind) -> Result:
    return {"found": [spotify.said(found) for found in await catalogue.search(query, kind)]}


def start_session_tool(clock: Clock, terminals: Terminals, processes: Processes, home: Home, record: Record, environment: Mapping[str, str], members: Callable[[], Iterable[Membership]]) -> Tool[Result]:
    """[LAW:nothing-unseen] each start is its own session.start event, inside the call's."""

    async def start_session(folder: str, model: str = "") -> Result:
        """Start a new Claude Code session for the user: `claude` in the folder, in a new window of the tmux session named
        for the folder, as the user would start it at a terminal. Returns once the session has joined hands, with its id
        and the tmux pane it runs in, or says why it did not. A session started has been told nothing.

        Args:
            folder: The folder the session works in, its whole path or one from ~.
            model: The model it starts on, as the user named it (opus, sonnet, haiku, or a model id). Empty for Claude Code's own default.
        """
        try:
            started = await startsession.start(home, clock, terminals, processes, record, Path(folder), model or None, environment, members)
        except startsession.NotStarted as why:
            return {"error": str(why)}
        return {"session": started.session, "tmux_session": started.tmux_session, "pane": started.pane}

    return tool(start_session)


def close_session_tool(processes: Processes, home: Home, record: Record, sessions: Sessions) -> Tool[Result]:
    """[LAW:nothing-unseen] each close is its own session.close event, inside the call's."""

    async def closed(id: SessionId, asked: closesession.Asked) -> Result:
        # Its name as the user knows it, read while it is still listed.
        name = spoken_name(sessions, id)
        try:
            match await closesession.close(processes, home, sessions.clock, record, sessions.live_session, id, asked):
                case closesession.Closed():
                    return {"closed": name}
                case closesession.LeftRunning(session=found):
                    # [LAW:one-source-of-truth] what it was found doing, as list_sessions says it.
                    return {"left_running": name, "state": spoken_state(found)}
        except closesession.NotClosed as why:
            return {"error": str(why)}

    async def close_session(sessions: list[str], asked: closesession.Asked) -> Result:
        """End Claude Code sessions the user is finished with: each one's claude exits, as at a closed terminal, and it
        leaves your session listing. Returns once they have, with each one's name, or says why one did not. A tmux window
        hands opened for one closes with it; a terminal the user started one from stays.

        Args:
            sessions: Each session's id, from list_sessions. They are closed together, so one call closes them all.
            asked: `named` when the user named these sessions: each ends whatever it is doing. `done` when they asked for
                the sessions that are done: each ends only at its prompt, with no dialog up and no shell or subagent
                running in the background, and is otherwise left running, which its result says.
        """
        return {"sessions": list(await asyncio.gather(*(closed(SessionId(session), asked) for session in sessions)))}

    return tool(close_session)


def set_overlay_tool(sessions: Sessions, overlays: Overlays) -> Tool[Result]:
    async def set_overlay(session: str, overlay: Overlay) -> Result:
        """Set how the user hears a session's finished turns: watched, normal, or muted.

        `watched` tells them each turn it finishes, as it finishes, even with finished turns off. `normal` tells its
        turns as finished turns are set to be told (attention), and otherwise when they ask (tell_turn); every session is
        normal until they change it. `muted` tells its turns only when they ask. Whatever its overlay, a session's
        permission requests, questions, and plans are said: they need an answer. Call this with watched when the user
        asks to be told when a session finishes, with muted when they ask to stop hearing about it or to mute it, and
        with normal when they unmute or unwatch it. It lasts until they change it, across restarts. Say the returned
        readback to the user: it says how the session's turns now reach them, what is set considered.

        Args:
            session: The session's id, from list_sessions.
            overlay: watched, normal, or muted.
        """
        try:
            id = SessionId(session)
            # [LAW:single-enforcer] the registry is the one judge of which sessions are running.
            live = sessions.live_session(id)
            if live is None:
                raise Rejected(f"no running session has the id {id!r}; take one from list_sessions")
            await asyncio.to_thread(overlays.set, id, overlay)
        except (Rejected, OSError) as error:
            return {"error": str(error)}
        # [LAW:one-source-of-truth] the delivery the narrator computes, from what is set as it reads it, so the readback
        # says what will happen to the session's next turn.
        return {"readback": _overlay_readback(spoken_name(sessions, id), await set_to(lambda: settings.attention(overlays.home)), overlay)}

    return tool(set_overlay)


def _overlay_readback(name: str, attention: Attention, overlay: Overlay) -> str:
    match delivery(attention, overlay):
        case Spoken(why=why):
            return {
                "watched": f"I'll tell you each turn {name} finishes.",
                "finished": f"I'll tell you each turn {name} finishes, as I tell every session's.",
            }[why]
        case Withheld(why="quiet"):
            # What quiet holds is told as it is set once hands talks again, so that is said too.
            return f"For now I'm keeping quiet and holding {name}'s turns. After that, {_overlay_readback(name, replace(attention, quiet='off'), overlay)}"
        case Withheld(why=why):
            return {
                "off": f"I'll hold {name}'s turns until you ask for one.",
                "muted": f"{name} is muted: I'll hold its turns until you ask. It still speaks when it needs your answer.",
            }[why]


class Resolved(TypedDict):
    heard: str
    meant: str


def draft_tools(sessions: Sessions) -> list[Tool[Result]]:
    """stage_draft, amend_draft, discard_draft, send_draft: a prompt dictated for a session, read back until it is right, then sent."""

    def aloud(outcome: DraftOutcome, name: str) -> Result:
        """A staged or amended draft is read back by hands, as written: what the user checks it by is spelled for the
        ear, and a model asked to say it would say it in its own words. Any other outcome changed nothing, so it is the
        model's to answer, as a refusal: to retry, as with the session's id, or to say what went wrong."""
        match outcome:
            case DraftStaged():
                return read_back(readback(outcome, name), f"the draft for {name} read back")
            case DraftAmended():
                return read_back(readback(outcome, name), f"the amended draft for {name} read back")
            case _:
                return {"error": readback(outcome, name)}

    async def stage_draft(session: str, text: str, resolutions: list[Resolved]) -> Result:
        """Stage a prompt the user dictated for a session. It is not sent until the user says to send it.

        Args:
            session: The session's id, from list_sessions.
            text: The prompt.
            resolutions: Each spoken phrase you turned into something exact, such as a file name, with what you made of it. Empty when you resolved nothing.
        """
        return await _answer(sessions, session, lambda id: StageDraft(id, parse_draft(text, resolutions)), sessions.draft, aloud)

    async def amend_draft(session: str, text: str, resolutions: list[Resolved]) -> Result:
        """Replace a session's staged draft with a corrected one when the user changes it.

        Args:
            session: The session's id, from list_sessions.
            text: The whole corrected prompt, not only the changed words.
            resolutions: Every resolution the corrected prompt relies on.
        """
        return await _answer(sessions, session, lambda id: AmendDraft(id, parse_draft(text, resolutions)), sessions.draft, aloud)

    async def discard_draft(session: str) -> Result:
        """Throw away a session's staged draft without sending it.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer(sessions, session, DiscardDraft, sessions.draft, _for_the_model(readback))

    async def send_draft(session: str) -> Result:
        """Type a session's staged draft into it and press Return. Call it only once the user has said to send it.

        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer(sessions, session, SendDraft, sessions.draft, _for_the_model(readback))

    return [tool(stage_draft), tool(amend_draft), tool(discard_draft), tool(send_draft)]


def _for_the_model[O](say: Callable[[O, str], str]) -> Callable[[O, str], Result]:
    """An outcome's readback handed to the model, which says it."""
    return lambda outcome, name: {"readback": say(outcome, name)}


async def _answer[R, O](
    sessions: Sessions,
    session: str,
    request: Callable[[SessionId], R],
    apply: Callable[[R], Awaitable[O]],
    say: Callable[[O, str], Result],
) -> Result:
    # [LAW:no-silent-failure] the model hears each failure and says it; the tool's event keeps it.
    id = SessionId(session)
    try:
        outcome = await apply(request(id))
    except Rejected as error:
        return {"error": str(error)}
    return say(outcome, spoken_name(sessions, id))


def keyboard_tools(sessions: Sessions) -> list[Tool[Result]]:
    """send_command, interrupt_session: what the user would otherwise do at a session's keyboard besides send it a prompt."""

    async def send_command(session: str, command: str, args: str = "") -> Result:
        """Run a slash command in a session, such as compact, clear, or model. Call it only when the user asks for a command by name.

        What the user dictates as a prompt is a draft, even when it begins with a slash; this is only for commands.
        Hands itself is not a session: its own model is changed by use_model.
        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
            command: The command's name, such as "compact" or "model".
            args: What follows the name, such as "opus" for model. Empty when the user gave nothing.
        """
        return await _answer(sessions, session, lambda id: SendCommand(id, parse_command(command, args)), sessions.keyboard, _for_the_model(keyboard_readback))

    async def interrupt_session(session: str) -> Result:
        """Stop what a session is doing, as pressing Escape at its keyboard does. At a permission dialog that is the dialog's no.

        Say the returned readback to the user.

        Args:
            session: The session's id, from list_sessions.
        """
        return await _answer(sessions, session, Interrupt, sessions.keyboard, _for_the_model(keyboard_readback))

    return [tool(body) for body in (send_command, interrupt_session)]


def permission_tools(sessions: Sessions) -> list[Tool[Result]]:
    """answer_permission, answer_question, answer_plan: the only ways a voice answer reaches a session waiting on its dialog."""

    async def answer_permission(request: str, decision: str, message: str = "") -> Result:
        """Answer a session's permission request with what the user decided. Call it only after the user has said to allow or deny.

        Say the returned readback to the user.

        Args:
            request: The request id given with the permission request.
            decision: "allow" to let the tool run, or "deny" to refuse it.
            message: Only when denying: what the user wants the session to know or do instead, in their words.
        """
        return await _decide(sessions, request, lambda: parse_decision(decision, message))

    async def answer_question(request: str, answers: list[str]) -> Result:
        """Answer the questions a session asked with what the user chose. Call it only once the user has answered every one.

        Say the returned readback to the user.

        Args:
            request: The request id given with the questions.
            answers: One answer per question, in the order they were asked: the label of the option the user chose, or their own words when no option fits. Where more than one may be chosen, join the labels with ", ". An empty answer when the user chose none.
        """
        return await _decide(sessions, request, lambda: Answers(tuple(answers)))

    async def answer_plan(request: str, decision: str, message: str = "") -> Result:
        """Answer a session's plan with what the user decided. Call it only after the user has approved the plan or asked for changes.

        Say the returned readback to the user.

        Args:
            request: The request id given with the plan.
            decision: "approve" to approve the plan and go on in the mode the session had before it planned; "auto-accept edits" or "manually approve edits" only when the user says how edits should go; or "keep planning" to send it back.
            message: Only when it keeps planning: what the user wants changed, in their words.
        """
        return await _decide(sessions, request, lambda: parse_plan_decision(decision, message))

    return [tool(body) for body in (answer_permission, answer_question, answer_plan)]


async def _decide(sessions: Sessions, request: str, decision: Callable[[], Decision]) -> Result:
    # [LAW:no-silent-failure] the model hears a refused answer and says it; the tool's event keeps it.
    try:
        outcome = sessions.answer(_request_id(request), decision())
    except Rejected as error:
        return {"error": str(error)}
    return {"readback": answer_readback(outcome, lambda id: spoken_name(sessions, id))}


def parse_decision(decision: str, message: str) -> Decision:
    """The model's answer, parsed once into the only two things a person can decide."""
    # [LAW:parse-dont-validate] a Decision is made here and nowhere else, so nothing but "allow" runs a tool.
    match (decision, message):
        case ("allow", ""):
            return Allow()
        case ("allow", _):
            raise Rejected("a message goes only with deny; an allow carries none, so nothing was answered")
        case ("deny", ""):
            return Deny(DENIED_BY_VOICE)
        case ("deny", _):
            return Deny(message)
        case (other, _):
            raise Rejected(f"decision should be 'allow' or 'deny', got {other!r}")


def parse_plan_decision(decision: str, message: str) -> Approve | KeepPlanning:
    """The model's answer to a plan, parsed once into an approval for a mode or a plan sent back."""
    match (decision, message):
        case ("keep planning", ""):
            return KeepPlanning(SENT_BACK_BY_VOICE)
        case ("keep planning", _):
            return KeepPlanning(message)
        case (choice, "") if choice in _APPROVALS:
            return Approve(_APPROVALS[choice])
        case (choice, _) if choice in _APPROVALS:
            raise Rejected("a message goes only with keep planning; an approval carries none, so nothing was answered")
        case (other, _):
            raise Rejected(f"decision should be 'approve', 'auto-accept edits', 'manually approve edits', or 'keep planning', got {other!r}")


def _request_id(request: str) -> RequestId:
    if not request:
        raise Rejected(f"request should be the request id string, got {request!r}")
    return RequestId(request)


def parse_draft(text: str, resolutions: list[Resolved]) -> Staged:
    """The model's arguments, parsed once into a draft whose text is safe to type."""
    # [LAW:parse-dont-validate] PromptText is made here and nowhere else.
    return Staged(_prompt_text(text, "the draft text"), tuple(Resolution(heard=item["heard"], meant=item["meant"]) for item in resolutions))


def parse_command(name: str, args: str) -> Command:
    """The model's arguments, parsed once into a command whose name and arguments are safe to type after a slash."""
    return Command(_command_name(name), _command_args(args))


def _command_name(name: str) -> CommandName:
    # [LAW:parse-dont-validate] CommandName is made here and nowhere else. A slash the model kept from what the user
    # said is the one the command is typed with, not a second.
    named = _COMMAND_NAME.fullmatch(name)
    if named is None:
        raise Rejected(f"command should be a slash command's name, such as compact, got {name!r}")
    return CommandName(named.group(1))


def _command_args(args: str) -> PromptText | None:
    if not args.strip():
        return None
    if "\n" in args:
        # A command is one line: what a line break does inside one, pasted, is unmeasured.
        raise Rejected("a command's arguments are one line, and these hold a line break")
    return _prompt_text(args, "the argument string")


def _prompt_text(text: str, what: str) -> PromptText:
    if not text.strip():
        raise Rejected(f"{what} is empty")
    if KEYSTROKES.search(text):
        # A tab is one: typed into a session it cycles the mode, and fritter refuses it by name at the socket.
        # Refusing it here instead means the model is told while it still has the words to fix.
        raise Rejected(f"{what} holds a control character, which would press a key when it is typed")
    if text.endswith("\\"):
        raise Rejected(f"{what} ends with a backslash, which turns the Return that sends it into a newline")
    return PromptText(text)
