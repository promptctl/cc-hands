"""Progress while a session works: its calls read as they are made, gathered until they settle, and heard by the focus."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path, PurePath

import pytest
from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame

from hands.clock import SYSTEM
from hands.core.attention import Amount, Attention, Level, Overlay, Route, Switch, progress_route
from hands.core.effects import Progress
from hands.core.events import Displayed, Progressed, Read, StatusReported, Tick
from hands.core.pending import Finished, News, Pending, Unread, Working, coalesce
from hands.core.progress import LINE, EDITING, READING, WRITING, LONGEST, RUNNING, SETTLE, Doing, Gathering, Words, doing, explained, intent, said
from hands.core.progress import Path as Filed
from hands.core.reducer import reduce
from hands.core.session import Gone, Idle, Membership, Opened, PromptId, Registry, Running, Session, SessionId, Told, Turn, Untold
from hands.core import status
from hands.core.status import Busy, Report, Stamp
from hands.core.turn import AgentId, AgentTask
from hands.sessions.audit import Entry, jsonable
from hands.sessions.wide import WideEvent, begun, unit
from hands.sessions.tail import Tails
from hands.voice.speech import Aloud, Unprompted, frames, relay, sent
from hands.voice.utterance import Fate, Utterance, Utterances
from hands.voice.summary import SummaryFailed
from hands.voice.working import Playing, keep_playing
from hands.voice.tools import describe_listing
from hands.sessions.registry import Listing

from test_narrator import heard, rendered

SID = SessionId("bf411065-dc5c-4ec9-8302-61b84bdb5c53")
OTHER = SessionId("other")
MUTED = SessionId("muted")
TURN = PromptId("p1")
IN_TURN = frozenset({TURN})
IN_NEXT = frozenset({PromptId("p2")})
MEMBER = Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl"))
TESTS = Doing(RUNNING, "run the test suite", Words("run the test suite"))


def burst(*calls: tuple[str, Mapping[str, object]]) -> tuple[Doing, ...]:
    return tuple(made for tool, input in calls if (made := doing(tool, input)) is not None)


def edit(name: str) -> Doing:
    return Doing(EDITING, f"edit {name}", Filed(f"/a/{name}"))


@pytest.mark.parametrize("tool", ["AskUserQuestion", "ExitPlanMode"])
def test_a_call_that_asks_the_user_is_no_progress(tool: str) -> None:
    # Its PermissionRequest hook speaks it as it is asked.
    assert doing(tool, {}) is None


CWD = PurePath("/code/a")


@pytest.mark.parametrize(
    ("tool", "input", "heard", "line"),
    [
        pytest.param("Bash", {"command": "uv run pytest -q\nmake check", "description": "Run the tests"}, "run the tests", "run the tests", id="a command by its description"),
        pytest.param("Bash", {"command": "gh pr view", "description": "PR status"}, "PR status", "PR status", id="an initialism kept as written"),
        pytest.param("Bash", {"command": "uv run pytest -q\nmake check"}, "run a command", "uv run pytest -q", id="a command with none by its first line, which the voice does not read"),
        pytest.param("Edit", {"file_path": "/code/a/src/tail.py"}, "edit tail.py", "edit src/tail.py", id="an edit by the file and its path from the cwd"),
        pytest.param("MultiEdit", {"file_path": "/code/a/src/tail.py"}, "edit tail.py", "edit src/tail.py", id="a multi edit"),
        pytest.param("NotebookEdit", {"notebook_path": "/code/a/n.ipynb"}, "edit n.ipynb", "edit n.ipynb", id="a notebook edit"),
        pytest.param("Write", {"file_path": "/code/a/new.md", "content": "x"}, "write new.md", "write new.md", id="a write"),
        pytest.param("Read", {"file_path": "/elsewhere/README.md"}, "read README.md", "read /elsewhere/README.md", id="a path outside the cwd is shown whole"),
        pytest.param("Grep", {"pattern": "Progressed", "path": "/code/a/src"}, "search for Progressed", "search for Progressed in src", id="a search of words, with where"),
        pytest.param("Grep", {"pattern": r"def\s+_\w+\("}, "search the code", r"search for def\s+_\w+\(", id="a regular expression, which the voice does not read"),
        pytest.param("Glob", {"pattern": "**/*.py"}, "search the code", "search for **/*.py", id="a glob"),
        pytest.param("WebFetch", {"url": "https://docs.python.org/3/library/re.html"}, "read a page on docs.python.org", "read a page on docs.python.org", id="a fetch by its site"),
        pytest.param("WebSearch", {"query": "xterm.js mobile"}, "search the web for xterm.js mobile", "search the web for xterm.js mobile", id="a web search"),
        pytest.param("Agent", {"description": "Find hook callers", "subagent_type": "Explore", "prompt": "..."}, "start a subagent to find hook callers", "start a subagent to find hook callers (Explore)", id="a subagent and its type"),
        pytest.param("Task", {"description": "Find hook callers"}, "start a subagent to find hook callers", "start a subagent to find hook callers", id="a subagent by its older name"),
        pytest.param("Skill", {"skill": "laws:law-review", "args": "557 1"}, "use the law review skill", "/laws:law-review 557 1", id="a skill and its arguments"),
        pytest.param("SendMessage", {"to": "a471895fc050c643e", "summary": "Plan revised", "message": "The plan\nis revised"}, "send a message saying Plan revised", "send a message to a471895fc050c643e: Plan revised", id="a message by its summary, to an id the voice does not say"),
        pytest.param("SendMessage", {"to": "main", "message": "Done.\nSee the PR."}, "send a message to main saying Done.", "send a message to main: Done.", id="a message with no summary by its first line"),
        pytest.param("TaskCreate", {"subject": "Implement decode_octal", "description": "..."}, "add task Implement decode_octal", "add task Implement decode_octal", id="a task by its subject"),
        pytest.param("TaskUpdate", {"taskId": "3", "status": "completed"}, "update task 3: completed", "update task 3: completed", id="a task's status change beside its id"),
        pytest.param("TaskUpdate", {"taskId": "3", "status": "in_progress"}, "update task 3: in progress", "update task 3: in progress", id="a task's status said as words, not as Claude Code's key"),
        pytest.param("TaskUpdate", {"taskId": "3", "status": "blocked"}, "update task 3: blocked", "update task 3: blocked", id="a status hands has not met is said as Claude Code names it"),
        pytest.param("WebFetch", {"url": "https://example.com?next=user@corp.com"}, "read a page on example.com", "read a page on example.com", id="a site is the URL's host, whatever its query holds"),
        pytest.param("Skill", {"skill": "review", "args": "  557 1\nsecond line "}, "use the review skill", "/review 557 1", id="a skill's arguments are trimmed and the line is the first of them"),
        pytest.param("Skill", {"skill": "review"}, "use the review skill", "/review", id="a skill with no arguments"),
        pytest.param("TodoWrite", {"todos": []}, "update its plan", "update its plan", id="a todo list"),
        pytest.param("mcp__claude_ai_Firecrawl__firecrawl_search", {"query": "x"}, "use firecrawl search", 'use firecrawl search {"query": "x"}', id="an MCP tool by its own name and its input"),
    ],
)
def test_a_call_is_one_intent_the_voice_and_the_page_each_tell(tool: str, input: Mapping[str, object], heard: str, line: str) -> None:
    made = doing(tool, input)
    assert made is not None and said((made,)) == heard
    assert intent(tool, input).line(CWD) == line


@pytest.mark.parametrize(
    ("tool", "input", "line"),
    [
        pytest.param("AskUserQuestion", {"questions": [{"question": "Which way?", "options": ["a", "b"]}, {"question": "And then?", "options": ["c"]}]}, "Which way?", id="the first question"),
        pytest.param("ExitPlanMode", {"plan": "1. do it"}, "asked to approve its plan", id="a plan to approve"),
    ],
)
def test_a_call_that_asks_the_user_has_a_line_though_the_voice_leaves_it_to_its_hook(tool: str, input: Mapping[str, object], line: str) -> None:
    assert intent(tool, input).line(CWD) == line


def test_a_tool_the_page_has_no_row_for_is_cut_to_a_line() -> None:
    assert len(intent("mcp__s__t", {"text": "x" * 500}).line(CWD)) == LINE


@pytest.mark.parametrize(
    ("tool", "act"),
    [
        ("Edit", "edit a file"),
        ("Write", "write a file"),
        ("Read", "read a file"),
        ("Grep", "search the code"),
        ("WebFetch", "read a page"),
        ("WebSearch", "look something up on the web"),
        ("Agent", "start a subagent"),
        ("Skill", "use a skill"),
        ("SendMessage", "send a message"),
        ("TaskCreate", "add a task"),
        ("TaskUpdate", "update a task"),
        ("AskUserQuestion", "ask the user"),
        ("ExitPlanMode", "asked to approve its plan"),
    ],
)
def test_a_call_whose_input_says_nothing_of_what_it_is_done_to_is_its_act_alone(tool: str, act: str) -> None:
    assert intent(tool, {}).line(CWD) == act


@pytest.mark.parametrize(
    ("doings", "sentence"),
    [
        pytest.param((TESTS,), "run the test suite", id="one call"),
        pytest.param(tuple(edit(f"f{n}.py") for n in range(10)), "edit ten files", id="ten edits are one clause"),
        pytest.param((edit("tail.py"),) * 10, "edit tail.py", id="ten edits of one file are that file"),
        pytest.param((Doing(RUNNING, None),) * 8, "run eight commands", id="calls that say nothing of themselves are each their own"),
        pytest.param((edit("a.py"), TESTS, edit("b.py"), TESTS), "edit two files, then run the test suite", id="each kind once, in the order first done"),
        pytest.param(burst(("Write", {"file_path": "/c/a/x.py"}), ("Edit", {"file_path": "/c/a/x.py"})), "write x.py, then edit x.py", id="a file written and then edited is said as each"),
        pytest.param(burst(("Write", {"file_path": "/c/a/x.py"}), ("Write", {"file_path": "/c/a/y.py"})), "write two files", id="two writes are two files written"),
        pytest.param(burst(("Edit", {"file_path": "/c/a/__init__.py"}), ("Edit", {"file_path": "/c/b/__init__.py"})), "edit two files", id="two files of one name are two files"),
        pytest.param(burst(("Edit", {"file_path": "/x/" + "y" * 130 + "/a.py"}), ("Edit", {"file_path": "/x/" + "y" * 130 + "/b.py"})), "edit two files", id="files that differ only past what the page shows are two files"),
        pytest.param(burst(("SendMessage", {"to": "a0123456789abcdef"}), ("SendMessage", {"to": "afedcba9876543210"})), "send two messages", id="messages to two agents are two messages"),
        pytest.param(burst(("Grep", {"pattern": "a.*b"}), ("Grep", {"pattern": "a.*b"})), "search the code", id="one regular expression twice is one search, though the voice does not read it"),
        pytest.param(burst(("Bash", {"command": "ls"}), ("Bash", {"command": "ls"}), ("Bash", {"command": "pwd"})), "run two commands", id="a command twice is one command"),
        pytest.param(burst(("WebSearch", {"query": "example.com"}), ("WebFetch", {"url": "https://example.com"})), "search the web for example.com, then read a page on example.com", id="a search and a fetch of one text are two things"),
        pytest.param(burst(("mcp__s__firecrawl_search", {"query": "a"}), ("mcp__s__firecrawl_search", {"query": "b"})), "use firecrawl search", id="one tool twice is that tool"),
        pytest.param(burst(("Edit", {"file_path": "/c/a/a.py"}), ("Read", {"file_path": "/c/a/b.py"}), ("Write", {"file_path": "/c/a/a.py"})), "edit a.py, then read b.py, then write a.py", id="each kind says its own act whichever call comes last"),
    ],
)
def test_a_burst_is_one_clause(doings: tuple[Doing, ...], sentence: str) -> None:
    assert said(doings) == sentence


def test_a_burst_is_due_once_it_settles_and_no_later_than_its_longest_wait() -> None:
    assert Gathering((TESTS,), "", 10.0, 11.0).due() == 11.0 + SETTLE
    assert Gathering((TESTS,), "", 10.0, 10.0 + LONGEST).due() == 10.0 + LONGEST


@pytest.mark.parametrize(
    ("focused", "overlay", "route"),
    [
        (True, "normal", "full"),
        (True, "watched", "full"),
        (True, "muted", "note"),
        (False, "normal", "note"),
        (False, "watched", "note"),
        (False, "muted", "note"),
    ],
)
def test_only_the_focus_is_heard_working_and_never_a_muted_one(focused: bool, overlay: Overlay, route: Route) -> None:
    assert progress_route(Attention(progress="full"), focused, overlay) == route


def test_no_session_is_heard_working_until_the_user_sets_progress_to_be() -> None:
    assert progress_route(Attention(), True, "normal") == "note"


@pytest.mark.parametrize(
    ("progress", "quiet", "route"),
    [("full", "off", "full"), ("brief", "off", "brief"), ("off", "off", "note"), ("full", "on", "note"), ("brief", "on", "note")],
)
def test_the_focus_is_heard_working_as_much_as_progress_is_set_and_not_at_all_while_quiet(progress: Level, quiet: Switch, route: Route) -> None:
    assert progress_route(Attention(progress=progress, quiet=quiet), True, "normal") == route


def working(session: SessionId, turn: Turn) -> Session:
    return Session(Membership(session, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl")), Running(Busy(), Stamp(1000), None), mode=None, turn=turn)


# Each session as the floor reads it letting go: still in the turn its progress came from, and the one after it.
LIVE = {session: working(session, Opened(TURN, others=IN_NEXT)) for session in (SID, OTHER)}


def finished(session: SessionId) -> Finished:
    return Finished(session, (News(TURN, "Done.", "", "", (), frozenset()),), "full")


@pytest.mark.parametrize(
    ("pending", "told"),
    [
        pytest.param(
            (Working(SID, IN_TURN, (edit("a.py"),)), Working(OTHER, IN_TURN, (TESTS,)), Working(SID, IN_TURN, (edit("b.py"),))),
            (Working(SID, IN_TURN, (edit("a.py"), edit("b.py"))), Working(OTHER, IN_TURN, (TESTS,))),
            id="a session's progress folds into one telling where its first stood",
        ),
        pytest.param(
            (Working(SID, IN_TURN, (TESTS,)), finished(SID), Working(OTHER, IN_TURN, (TESTS,))),
            (finished(SID), Working(OTHER, IN_TURN, (TESTS,))),
            id="progress the turn's result follows is said by the result",
        ),
        pytest.param(
            (finished(SID), Working(SID, IN_NEXT, (TESTS,))),
            (finished(SID), Working(SID, IN_NEXT, (TESTS,))),
            id="progress of the turn after a result is still heard",
        ),
        pytest.param(
            (Working(SID, IN_NEXT, (TESTS,)), finished(SID)),
            (Working(SID, IN_NEXT, (TESTS,)), finished(SID)),
            id="progress of the next turn is heard though the turn before's result is told after it",
        ),
        pytest.param(
            (Working(SID, IN_NEXT, (TESTS,)), Unread(SID)),
            (Unread(SID),),
            id="a turn that could not be read goes by no id, so the progress it follows is said by it",
        ),
    ],
)
def test_progress_folds_and_gives_way_to_the_result(pending: tuple[Pending, ...], told: tuple[Pending, ...]) -> None:
    assert tuple(each.pending for each in coalesce(pending, LIVE)) == told


def test_progress_heard_is_said_as_written_in_hands_lane() -> None:
    [said] = sent(frames(Working(SID, IN_TURN, (TESTS,)), lambda _: "cc-hands"), ())
    assert isinstance(said, Aloud) and said.spoken.text == "cc-hands: run the test suite."


def running(turn: Turn) -> Registry:
    session = Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=turn)
    return Registry(sessions={SID: session}, drafts={})


def turn_of(registry: Registry) -> Turn:
    session = registry.sessions[SID]
    assert isinstance(session, Session)
    return session.turn


def test_calls_are_gathered_on_the_turn_and_told_once_they_settle() -> None:
    gathered, effects = reduce(running(Opened(TURN)), Progressed(SID, (TURN,), (edit("a.py"),), at=10.0))
    gathered, more = reduce(gathered, Progressed(SID, (TURN,), (TESTS,), at=11.0))
    assert effects == more == []
    assert turn_of(gathered) == Opened(TURN, gathering=Gathering((edit("a.py"), TESTS), "", 10.0, 11.0), latest=TESTS)
    waiting, early = reduce(gathered, Tick(11.0 + SETTLE - 0.5))
    assert early == [] and waiting == gathered
    told, effects = reduce(gathered, Tick(11.0 + SETTLE))
    assert effects == [Progress(SID, frozenset({TURN}), (edit("a.py"), TESTS), "")]
    # Told once: what the session last set out to do is kept for anyone who asks.
    assert turn_of(told) == Opened(TURN, latest=TESTS)
    assert reduce(told, Tick(30.0)) == (told, [])


def test_a_session_that_never_pauses_is_heard_at_its_longest_wait() -> None:
    registry = running(Opened(TURN))
    for at in range(0, int(LONGEST) + 1):
        registry, _ = reduce(registry, Progressed(SID, (TURN,), (edit(f"f{at}.py"),), at=float(at)))
    _, effects = reduce(registry, Tick(LONGEST))
    assert [type(effect) for effect in effects] == [Progress]


@pytest.mark.parametrize(
    "turn",
    [
        pytest.param(Opened(PromptId("p2")), id="a turn by another id"),
        pytest.param(Untold(TURN, frozenset(), Stamp(2000)), id="a turn since over, whose result is told instead"),
        pytest.param(Told(TURN), id="a turn told"),
    ],
)
def test_calls_read_and_text_displayed_late_move_nothing(turn: Turn) -> None:
    before = running(turn)
    assert reduce(before, Progressed(SID, (TURN,), (TESTS,), at=10.0)) == (before, [])
    # Claude Code goes on displaying a reply for seconds after its turn's Stop.
    assert reduce(before, Displayed(SID, (TURN,), "1. A line of it.\n", at=10.0)) == (before, [])


def test_what_a_turn_gathered_and_did_not_tell_goes_with_it_as_it_ends() -> None:
    gathered, _ = reduce(running(Opened(TURN)), Progressed(SID, (TURN,), (TESTS,), at=10.0))
    ended, _ = reduce(gathered, StatusReported(SID, Report(status.Idle(), Stamp(3000)), at=11.0))
    assert isinstance(turn_of(ended), Untold)
    assert reduce(ended, Tick(30.0))[1] == []


def lines(*records: str) -> str:
    return "".join(f"{record}\n" for record in records)


ASKED = '{"type":"user","promptId":"p1","message":{"role":"user","content":"Run the tests."}}'
CALL = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"uv run pytest","description":"Run the test suite"}}]}}'
RESULT = '{"type":"user","promptId":"p1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"3 passed"}]}}'
EDITS = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Edit","input":{"file_path":"/a/x.py"}},{"type":"tool_use","id":"t3","name":"AskUserQuestion","input":{}}]}}'


class Known:
    """As much of the registry as the tail asks about."""

    def __init__(self, transcript: Path) -> None:
        self.member = Membership(SID, pid=4242, cwd=transcript.parent, transcript=transcript)

    def live_members(self) -> list[Membership]:
        return [self.member]

    def status_read(self, session: SessionId) -> bool:
        return True

    def now(self) -> float:
        return 7.0

    def stamp(self) -> Stamp:
        return Stamp(5000)

    def membership(self, session: SessionId) -> Membership | None:
        return self.member

    def followed(self, session: SessionId) -> Callable[[PromptId], bool]:
        # No hook of the session is heard: the tail decides alone what of the file is history.
        return lambda _prompt: False


async def test_each_call_a_running_turn_makes_is_read_once_as_progress(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    with transcript.open("a") as file:
        file.write(lines(CALL))
    assert [event for event in await tails.catch_up() if isinstance(event, Progressed)] == [Progressed(SID, (TURN,), (TESTS,), 7.0)]
    with transcript.open("a") as file:
        file.write(lines(RESULT, EDITS))
    # The result is the turn's to tell; a question is spoken as it is asked.
    assert [event for event in await tails.catch_up() if isinstance(event, Progressed)] == [Progressed(SID, (TURN,), (edit("x.py"),), 7.0)]
    assert [event for event in await tails.catch_up() if isinstance(event, Progressed)] == []


async def test_the_calls_a_turn_made_before_hands_followed_it_are_history(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED, CALL))
    assert [event for event in await Tails(Known(transcript)).catch_up() if isinstance(event, Progressed)] == []


async def test_the_relay_hands_the_focus_on_to_be_played_and_cued_leaves_any_other_to_the_listing_and_says_why() -> None:
    queued: list[Frame] = []
    played: list[tuple[Progress, Amount, Utterance]] = []
    recorded: list[Entry] = []
    working: list[None] = []

    class Heard:
        def __init__(self) -> None:
            self.waiting: asyncio.Queue[Progress] = asyncio.Queue()

        async def heard(self) -> Progress:
            return await self.waiting.get()

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    async def attending(session: SessionId) -> tuple[Attention, bool, Overlay]:
        return Attention(progress="brief"), session in (SID, MUTED), "muted" if session == MUTED else "normal"

    sessions = Heard()
    for session in (SID, OTHER, MUTED):
        sessions.waiting.put_nowait(Progress(session, frozenset({TURN}), (TESTS,), ""))
    utterances = Utterances(SYSTEM, recorded.append)
    keeping = asyncio.create_task(utterances.keep())
    relaying = asyncio.create_task(relay(sessions, utterances, queue_frame, attending, lambda progress, amount, utterance: played.append((progress, amount, utterance)), lambda: working.append(None)))  # pyright: ignore[reportArgumentType]  (only heard() is asked)
    while len([entry for entry in recorded if isinstance(entry, WideEvent)]) < 2:
        await asyncio.sleep(0.01)
    relaying.cancel()
    keeping.cancel()
    assert queued == []
    [(progress, amount, utterance)] = played
    assert (progress, amount) == (Progress(SID, frozenset({TURN}), (TESTS,), ""), "brief")
    # Only progress that is heard sounds: another session's burst made none, nor a muted focus's.
    assert working == [None]
    routed = {key: utterance.facts[key] for key in ("session", "attention", "focused", "overlay", "route")}
    assert routed == {"session": SID, "attention": Attention(progress="brief"), "focused": True, "overlay": "normal", "route": "brief"}
    noted = [{key: entry.facts[key] for key in ("session", "focused", "overlay", "route", "fate")} for entry in recorded if isinstance(entry, WideEvent) and entry.event == "utterance"]
    assert noted == [
        {"session": OTHER, "focused": False, "overlay": "normal", "route": "note", "fate": "noted"},
        {"session": MUTED, "focused": True, "overlay": "muted", "route": "note", "fate": "noted"},
    ]


def test_what_a_session_last_set_out_to_do_is_in_its_listing() -> None:
    session = Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=Opened(TURN, latest=TESTS))
    assert describe_listing(Listing(session, "cc-hands"))["state"] == "working; the last thing it set out to do: run the test suite"
    assert describe_listing(Listing(Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=Opened(TURN)), "cc-hands"))["state"] == "working"


def test_calls_read_are_a_line_the_audit_log_can_write() -> None:
    # Found live: a set of prompt ids is no line, and the event that carried one went unrecorded.
    assert jsonable(Progressed(SID, (TURN,), (TESTS,), at=7.0)) == {
        "type": "Progressed",
        "session": SID,
        "of": ["p1"],
        "doings": [{"type": "Doing", "work": {"type": "Work", "several": "run {count} commands", "one": "run a command", "act": "", "hooked": False}, "alone": "run the test suite", "of": {"type": "Words", "text": "run the test suite", "aside": None}}],
        "at": 7.0,
    }


def test_progress_heard_is_a_fact_the_audit_log_can_write() -> None:
    # Found live: the log could not write the set progress carries its turn as, and every burst's lines went unrecorded.
    written = jsonable(Progress(SID, frozenset({PromptId("p2"), TURN}), (TESTS,), "a line\n"))
    assert isinstance(written, dict) and written["of"] == ["p1", "p2"] and written["written"] == "a line\n"


def test_calls_read_after_their_session_ended_are_behind_not_wrong() -> None:
    # The tail reads a transcript a moment past its session's end: no audit line says a hook came late.
    gone = Registry(sessions={SID: Gone(MEMBER)}, drafts={})
    assert reduce(gone, Progressed(SID, (TURN,), (TESTS,), at=10.0)) == (gone, [])
    assert reduce(gone, Displayed(SID, (TURN,), "a line\n", at=10.0)) == (gone, [])


def test_text_is_gathered_with_the_calls_and_told_with_them_once_both_settle() -> None:
    registry, _ = reduce(running(Opened(TURN)), Displayed(SID, (TURN,), "First, how DNS works.\n", at=10.0))
    registry, _ = reduce(registry, Displayed(SID, (TURN,), "1. The OS asks its resolver.\n2. The resolver asks the root.\n", at=11.0))
    registry, _ = reduce(registry, Progressed(SID, (TURN,), (TESTS,), at=12.0))
    # A line displayed holds the burst open as a call does: a long explanation is told once it has settled or waited longest.
    assert reduce(registry, Tick(11.0 + SETTLE))[1] == []
    told, effects = reduce(registry, Tick(12.0 + SETTLE))
    assert effects == [Progress(SID, frozenset({TURN}), (TESTS,), "First, how DNS works.\n1. The OS asks its resolver.\n2. The resolver asks the root.\n")]
    # Text is not a call: what the session last set out to do is still the call it made.
    assert turn_of(told) == Opened(TURN, latest=TESTS)


def test_blank_lines_alone_begin_no_burst_but_part_the_paragraphs_of_one() -> None:
    # A burst of nothing would be told as the session's name and nothing after it.
    idle = running(Opened(TURN))
    assert reduce(idle, Displayed(SID, (TURN,), "\n", at=10.0)) == (idle, [])
    registry, _ = reduce(idle, Displayed(SID, (TURN,), "First.\n", at=10.0))
    registry, _ = reduce(registry, Displayed(SID, (TURN,), "\n", at=10.5))
    registry, _ = reduce(registry, Displayed(SID, (TURN,), "Then.\n", at=11.0))
    _, effects = reduce(registry, Tick(11.0 + SETTLE))
    assert effects == [Progress(SID, frozenset({TURN}), (), "First.\n\nThen.\n")]


def test_an_explanation_still_being_written_is_told_at_its_longest_wait() -> None:
    registry = running(Opened(TURN))
    for at in range(0, int(LONGEST) * 4):
        registry, _ = reduce(registry, Displayed(SID, (TURN,), f"{at}. A line.\n", at=at / 4))
    _, effects = reduce(registry, Tick(LONGEST))
    assert [(type(effect), effect.doings) for effect in effects if isinstance(effect, Progress)] == [(Progress, ())]


def test_text_is_said_by_its_summary_ahead_of_the_calls() -> None:
    assert said((explained("Explain how DNS resolution works."), TESTS)) == "explain how DNS resolution works, then run the test suite"
    assert said((Doing(WRITING, None), TESTS)) == "write something, then run the test suite"
    # Two bursts whose text could not be summarised, folded before they are said: what either says is not known.
    assert said((Doing(WRITING, None), Doing(WRITING, None))) == "write two things"


async def played(progress: Progress, turn: Callable[[], Turn], explain: Callable[[str], Awaitable[str]], amount: Amount = "full") -> tuple[list[Pending], Utterance]:
    """What the player hands the floor of one progress, and its utterance, with the session's turn as `turn` says when
    it is asked."""
    queued: list[Frame] = []
    utterance = heard()

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    def live_session(_session: SessionId) -> Session:
        return Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=turn())

    playing = Playing()
    playing.put_nowait((progress, amount, utterance))
    player = asyncio.create_task(keep_playing(playing, live_session, queue_frame, explain))
    while not (queued or utterance.settled.done()):
        await asyncio.sleep(0.01)
    player.cancel()
    return [frame.pending for frame in queued if isinstance(frame, Unprompted)], utterance


def handed_on(utterance: Utterance, explained: str | None, failure: str | None = None, fate: Fate | None = None) -> bool:
    """Whether the player found what `utterance`'s text came to, failed it as `failure`, and settled it as `fate`, None
    for one it handed on to be said."""
    settled = utterance.settled.result() if utterance.settled.done() else None
    return (utterance.facts, utterance.failure, settled) == ({"explained": explained}, failure, fate)


async def unasked(text: str) -> str:
    raise AssertionError(f"a burst of calls alone is said as written, not summarised: {text!r}")


async def test_a_burst_of_calls_alone_is_played_as_written() -> None:
    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), ""), lambda: Opened(TURN), unasked)
    assert queued == [Working(SID, IN_TURN, (TESTS,))]
    assert handed_on(told, None)


WRITTEN = "First, how DNS works.\n1. The OS asks its resolver.\n"


async def test_briefly_the_focus_is_heard_saying_what_it_is_doing_without_its_calls() -> None:
    asked: list[WideEvent] = []

    async def explain(_text: str) -> str:
        # As the side question it is asked as opens its unit of work.
        with unit("brain.aside", asked.append, begun(SYSTEM)):
            return "Explain how DNS resolution works."

    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), WRITTEN), lambda: Opened(TURN), explain, "brief")
    assert queued == [Working(SID, IN_TURN, (explained("explain how DNS resolution works"),))]
    assert handed_on(told, "explain how DNS resolution works")
    # The explanation is a part of the utterance it delays.
    [aside] = asked
    assert (aside.trace_id, aside.parent_id) == (told.begun.span.trace_id, told.begun.span.span_id)


async def test_briefly_a_burst_of_calls_alone_is_not_said_and_its_utterance_says_so() -> None:
    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), ""), lambda: Opened(TURN), unasked, "brief")
    assert queued == []
    assert handed_on(told, None, fate="noted")


async def test_the_focus_is_heard_explaining_by_a_summary_of_what_it_wrote() -> None:
    asked: list[str] = []

    async def explain(text: str) -> str:
        asked.append(text)
        return "Explain how DNS resolution works."

    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), WRITTEN), lambda: Opened(TURN), explain)
    assert asked == [WRITTEN.strip()]
    assert queued == [Working(SID, IN_TURN, (explained("explain how DNS resolution works"), TESTS))]
    assert handed_on(told, "explain how DNS resolution works")
    [spoken] = rendered(queued[0], names=lambda _: "cc-hands")
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands: explain how DNS resolution works, then run the test suite."


async def test_text_that_cannot_be_summarised_is_said_to_have_been_written_and_never_read_out() -> None:
    async def explain(_text: str) -> str:
        raise SummaryFailed("nothing came back")

    queued, told = await played(Progress(SID, frozenset({TURN}), (), WRITTEN), lambda: Opened(TURN), explain)
    assert queued == [Working(SID, IN_TURN, (Doing(WRITING, None),))]
    assert handed_on(told, None, "the text could not be summarised: SummaryFailed: nothing came back")


async def test_progress_whose_turn_ended_while_it_was_summarised_is_not_played() -> None:
    turn: list[Turn] = [Opened(TURN)]

    async def explain(_text: str) -> str:
        # The turn ends in the seconds the summary takes; its result is told instead.
        turn[0] = Untold(TURN, frozenset(), Stamp(2000))
        return "explain how DNS resolution works"

    queued, told = await played(Progress(SID, frozenset({TURN}), (TESTS,), WRITTEN), lambda: turn[0], explain)
    assert queued == []
    assert handed_on(told, "explain how DNS resolution works", fate="dropped")


async def test_a_burst_is_not_kept_waiting_on_the_summary_of_the_one_ahead_of_it() -> None:
    queued: list[Frame] = []
    asked: list[str] = []
    second_asked = asyncio.Event()

    async def explain(text: str) -> str:
        asked.append(text)
        if text == "first":
            # The first summary is still out when the second burst is routed; it comes back only once that one began.
            await second_asked.wait()
        else:
            second_asked.set()
        return f"explain the {text} thing"

    async def queue_frame(frame: Frame) -> None:
        queued.append(frame)

    playing = Playing()
    player = asyncio.create_task(keep_playing(playing, lambda _: Session(MEMBER, Running(Busy(), Stamp(1000), None), mode=None, turn=Opened(TURN)), queue_frame, explain))
    playing.put_nowait((Progress(SID, frozenset({TURN}), (), "first"), "full", heard()))
    playing.put_nowait((Progress(SID, frozenset({TURN}), (), "second"), "full", heard()))
    await asyncio.wait_for(second_asked.wait(), timeout=1.0)
    while len(queued) < 2:
        await asyncio.sleep(0.01)
    player.cancel()
    assert asked == ["first", "second"]
    # Played in the order they settled, whichever summary came back first.
    assert [frame.pending for frame in queued if isinstance(frame, Unprompted)] == [
        Working(SID, IN_TURN, (explained("explain the first thing"),)),
        Working(SID, IN_TURN, (explained("explain the second thing"),)),
    ]


# A focused session's subagent, heard step by step while it works, each step said as the work of the call that started it.

AGENT = AgentId("a0eea659e39132f84")
REVIEW = AgentTask(AGENT, "Review the parser change")
READ_TAIL = Doing(READING, "read tail.py", Filed("/a/tail.py"))

# The job a subagent is given is the record its transcript starts from, and no call of its own.
JOB = '{"type":"user","parentUuid":null,"uuid":"u0","message":{"role":"user","content":"Review the diff."}}'
# A fork's transcript starts from the parent's call that launched it, copied in.
LAUNCH = '{"type":"assistant","parentUuid":null,"uuid":"u0","message":{"content":[{"type":"tool_use","id":"p9","name":"Agent","input":{"description":"Review the parser change","prompt":"..."}}]}}'
READS = '{"type":"assistant","parentUuid":"u0","uuid":"u1","message":{"content":[{"type":"tool_use","id":"s1","name":"Read","input":{"file_path":"/a/tail.py"}}]}}'
RUNS = '{"type":"assistant","parentUuid":"u1","uuid":"u2","message":{"content":[{"type":"tool_use","id":"s2","name":"Bash","input":{"command":"uv run pytest","description":"Run the test suite"}}]}}'
SKILL = '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Skill","input":{"skill":"code-review","args":"high 152"}}]}}'


def subagent(transcript: Path, meta: dict[str, object], *records: str, id: AgentId = AGENT) -> Path:
    """A subagent of the session `transcript` is the transcript of, as Claude Code starts it: the file naming its job, then its own transcript."""
    folder = transcript.with_suffix("") / "subagents"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"agent-{id}.meta.json").write_text(json.dumps(meta))
    own = folder / f"agent-{id}.jsonl"
    own.write_text(lines(*records))
    return own


async def progressed(tails: Tails) -> list[Progressed]:
    return [event for event in await tails.catch_up() if isinstance(event, Progressed)]


async def logged[T](level: str, heard: Awaitable[T]) -> tuple[T, list[str]]:
    """What `heard` comes to, and what the log says at `level` while it runs."""
    said: list[str] = []
    sink = logger.add(lambda message: said.append(str(message)), level=level)
    try:
        return await heard, said
    finally:
        logger.remove(sink)


@pytest.mark.parametrize("job", [JOB, LAUNCH], ids=["a prompt", "a fork's launching call"])
async def test_a_subagent_is_heard_call_by_call_as_the_work_of_the_call_that_started_it(tmp_path: Path, job: str) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    own = subagent(transcript, {"agentType": "general-purpose", "description": "Review the parser change"}, job, READS)
    heard, infos = await logged("INFO", progressed(tails))
    assert heard == [Progressed(SID, REVIEW, (READ_TAIL,), 7.0)]
    # [LAW:nothing-unseen] which said who started it.
    assert any(f"subagent {AGENT} is heard as the work of the job agent-{AGENT}.meta.json names: 'Review the parser change'" in line for line in infos)
    with own.open("a") as file:
        file.write(lines(RUNS))
    assert await progressed(tails) == [Progressed(SID, REVIEW, (TESTS,), 7.0)]
    assert await progressed(tails) == []


async def test_a_subagent_already_working_as_hands_follows_its_session_is_heard_from_then_on(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    own = subagent(transcript, {"description": "Review the parser change"}, JOB, READS)
    tails = Tails(Known(transcript))
    # What it did before hands followed its session is history, and the log says how much.
    heard, infos = await logged("INFO", progressed(tails))
    assert heard == []
    assert any(f"subagent {AGENT} of session {SID} was working before hands followed the session" in line and f"{own.stat().st_size} bytes of it are history" in line for line in infos)
    with own.open("a") as file:
        file.write(lines(RUNS))
    assert await progressed(tails) == [Progressed(SID, REVIEW, (TESTS,), 7.0)]


async def test_a_skill_run_in_a_subagent_of_its_own_is_the_work_of_the_skill_its_parent_invoked(tmp_path: Path) -> None:
    # Claude Code names no job for it: measured, 124 foreground skill forks with no description beside them.
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    with transcript.open("a") as file:
        file.write(lines(SKILL))
    subagent(transcript, {"agentType": "general-purpose", "requestShape": "foreground"}, JOB, READS)
    heard, infos = await logged("INFO", progressed(tails))
    assert [event for event in heard if isinstance(event.of, AgentTask)] == [Progressed(SID, AgentTask(AGENT, "/code-review high 152"), (READ_TAIL,), 7.0)]
    assert any(f"subagent {AGENT} names no job, so it is heard as the work of the one skill its parent is running: '/code-review high 152'" in line for line in infos)


async def test_a_skill_forked_into_the_background_is_the_work_of_the_call_whose_result_names_it(tmp_path: Path) -> None:
    # Run in the background, the skill's call has its result at once, so it is running no longer as its subagent works.
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    forked = {"success": True, "commandName": "code-review", "status": "forked", "background": True, "agentId": AGENT, "result": "Running in the background as @code-review"}
    launched = json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "launched"}]}, "toolUseResult": forked}, separators=(",", ":"))
    with transcript.open("a") as file:
        file.write(lines(SKILL, launched))
    subagent(transcript, {"agentType": "general-purpose", "requestShape": "background"}, JOB, READS)
    heard, infos = await logged("INFO", progressed(tails))
    assert [event for event in heard if isinstance(event.of, AgentTask)] == [Progressed(SID, AgentTask(AGENT, "/code-review high 152"), (READ_TAIL,), 7.0)]
    assert any(f"subagent {AGENT} names no job, so it is heard as the work of the call whose result names it" in line for line in infos)


async def test_a_subagent_already_working_is_followed_from_its_last_whole_record(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    own = subagent(transcript, {"description": "Review the parser change"}, JOB)
    # Claude Code is part way through writing a record as hands follows the session.
    with own.open("a") as file:
        file.write(READS[:40])
    tails = Tails(Known(transcript))
    _, errors = await logged("ERROR", tails.catch_up())
    with own.open("a") as file:
        file.write(READS[40:] + "\n")
    heard, more = await logged("ERROR", progressed(tails))
    assert heard == [Progressed(SID, REVIEW, (READ_TAIL,), 7.0)] and errors == more == []


async def test_a_subagent_s_calls_wait_for_the_file_naming_its_job_to_be_readable(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    subagent(transcript, {}, JOB, READS)
    meta = transcript.with_suffix("") / "subagents" / f"agent-{AGENT}.meta.json"
    # Caught part way through being written.
    meta.write_text('{"description": "Review')
    assert await progressed(tails) == []
    meta.write_text(json.dumps({"description": "Review the parser change"}))
    assert await progressed(tails) == [Progressed(SID, REVIEW, (READ_TAIL,), 7.0)]


async def test_a_subagent_a_subagent_started_is_heard_as_the_work_of_the_call_in_the_session_that_started_the_first(tmp_path: Path) -> None:
    # Measured: Claude Code keeps a subagent's own subagents beside the session's, each naming the one that started it.
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    subagent(transcript, {"description": "Review the parser change", "spawnDepth": 1}, JOB, READS)
    nested = AgentId("a0b72288f3a9809d1")
    subagent(transcript, {"description": "Check the fold", "parentAgentId": AGENT, "spawnDepth": 2}, JOB, RUNS, id=nested)
    heard = await progressed(tails)
    assert sorted(heard, key=str) == sorted([Progressed(SID, REVIEW, (READ_TAIL,), 7.0), Progressed(SID, REVIEW, (TESTS,), 7.0)], key=str)


async def test_a_subagent_s_transcript_written_again_shorter_is_read_again_from_its_start(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    own = subagent(transcript, {"description": "Review the parser change"}, JOB, READS, RUNS)
    assert await progressed(tails) == [Progressed(SID, REVIEW, (READ_TAIL, TESTS), 7.0)]
    own.write_text(lines(JOB, READS))
    _, warnings = await logged("WARNING", tails.catch_up())
    assert any(f"the transcript of subagent {AGENT} of session {SID} is shorter than what was read of it" in line for line in warnings)
    with own.open("a") as file:
        file.write(lines(RUNS))
    assert await progressed(tails) == [Progressed(SID, REVIEW, (TESTS,), 7.0)]


async def test_a_session_moved_into_a_worktree_is_read_on_from_where_it_was_subagents_and_all(tmp_path: Path) -> None:
    """Claude Code renames a session's transcript, its subagents' folder with it, into the worktree's project as
    the session enters one, and its hooks name the new path from then on: what was read of it still is, so its prompt
    is not taken again and the calls made after the move are heard, its subagent's among them."""
    transcript = tmp_path / "repo" / "t.jsonl"
    transcript.parent.mkdir()
    transcript.write_text(lines(ASKED))
    known = Known(transcript)
    tails = Tails(known)
    await tails.catch_up()
    own = subagent(transcript, {"description": "Review the parser change"}, JOB, READS)
    assert await progressed(tails) == [Progressed(SID, REVIEW, (READ_TAIL,), 7.0)]
    moved = tmp_path / "repo--claude-worktrees-lane" / "t.jsonl"
    moved.parent.mkdir()
    transcript.rename(moved)
    transcript.with_suffix("").rename(moved.with_suffix(""))
    known.member = replace(known.member, transcript=moved)
    with moved.open("a") as file:
        file.write(lines(CALL))
    with (moved.with_suffix("") / own.relative_to(transcript.with_suffix(""))).open("a") as file:
        file.write(lines(RUNS))
    heard = [event for event in await tails.catch_up() if not isinstance(event, Read)]
    assert sorted(heard, key=str) == sorted([Progressed(SID, (TURN,), (TESTS,), 7.0), Progressed(SID, REVIEW, (TESTS,), 7.0)], key=str)


async def test_a_subagent_nothing_says_the_start_of_is_said_in_the_log_and_never_told_as_another_s(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(lines(ASKED))
    tails = Tails(Known(transcript))
    await tails.catch_up()
    own = subagent(transcript, {"agentType": "general-purpose"}, JOB, READS)
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        assert await progressed(tails) == []
        # Said once: every call it makes after is told as nobody's, and the log does not say so again for each.
        with own.open("a") as file:
            file.write(lines(RUNS))
        assert await progressed(tails) == []
    finally:
        logger.remove(sink)
    assert len(errors) == 1 and f"subagent {AGENT} of session {SID} sets out to do is never told: Unstarted" in errors[0]


def idle() -> Registry:
    # The parent's turn is over: a subagent run in the background works on after it.
    session = Session(Membership(SID, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl")), Idle(status.Idle(), Stamp(1000), TURN), mode=None, turn=Told(TURN))
    return Registry(sessions={SID: session}, drafts={})


def test_a_subagent_s_calls_are_gathered_whatever_its_parent_is_doing_and_told_once_they_settle() -> None:
    registry, effects = reduce(idle(), Progressed(SID, REVIEW, (READ_TAIL,), at=10.0))
    registry, more = reduce(registry, Progressed(SID, REVIEW, (TESTS,), at=11.0))
    assert effects == more == []
    assert reduce(registry, Tick(10.0 + SETTLE))[1] == []
    told, effects = reduce(registry, Tick(11.0 + SETTLE))
    assert effects == [Progress(SID, REVIEW, (READ_TAIL, TESTS), "")]
    assert reduce(told, Tick(30.0))[1] == []


def test_a_subagent_s_burst_and_its_parent_s_are_told_apart() -> None:
    session = idle().sessions[SID]
    assert isinstance(session, Session)
    registry = Registry(sessions={SID: Session(session.membership, session.state, None, turn=Opened(TURN))}, drafts={})
    registry, _ = reduce(registry, Progressed(SID, (TURN,), (Doing(EDITING, "edit a.py"),), at=10.0))
    registry, _ = reduce(registry, Progressed(SID, REVIEW, (TESTS,), at=10.0))
    assert reduce(registry, Tick(10.0 + SETTLE))[1] == [Progress(SID, frozenset({TURN}), (Doing(EDITING, "edit a.py"),), ""), Progress(SID, REVIEW, (TESTS,), "")]


def test_a_subagent_s_work_folds_only_with_its_own_and_gives_way_to_the_result_it_reports_back_to() -> None:
    own = Working(SID, frozenset({TURN}), (Doing(EDITING, "edit a.py"),))
    reported = Finished(SID, (News(PromptId("p2"), "Done.", "", "", (), frozenset({AGENT})),), "full")
    assert tuple(each.pending for each in coalesce((Working(SID, REVIEW, (READ_TAIL,)), own, Working(SID, REVIEW, (TESTS,))), LIVE)) == (Working(SID, REVIEW, (READ_TAIL, TESTS)), own)
    # The turn it reports back to tells its work better, wherever its last burst settled.
    assert tuple(each.pending for each in coalesce((Working(SID, REVIEW, (TESTS,)), reported), LIVE)) == (reported,)
    assert tuple(each.pending for each in coalesce((reported, Working(SID, REVIEW, (TESTS,))), LIVE)) == (reported,)


def test_a_subagent_working_on_in_the_background_is_still_news_after_a_result_that_does_not_report_it() -> None:
    unrelated = Finished(SID, (News(PromptId("p2"), "Done.", "", "", (), frozenset()),), "full")
    assert tuple(each.pending for each in coalesce((Working(SID, REVIEW, (TESTS,)), unrelated), LIVE)) == (Working(SID, REVIEW, (TESTS,)), unrelated)


@pytest.mark.parametrize(
    ("live", "told"),
    [
        pytest.param({OTHER: LIVE[OTHER]}, (Working(OTHER, IN_TURN, (TESTS,)),), id="a session gone meanwhile, its ending said or not"),
        pytest.param({**LIVE, SID: working(SID, Told(TURN))}, (Working(SID, REVIEW, (TESTS,)), Working(OTHER, IN_TURN, (TESTS,))), id="a turn finished meanwhile, its result told or held, its subagent still news"),
        pytest.param({**LIVE, SID: working(SID, Opened(PromptId("p3")))}, (Working(SID, REVIEW, (TESTS,)), Working(OTHER, IN_TURN, (TESTS,))), id="a turn the next one replaced, its subagent still news"),
    ],
)
def test_progress_of_a_turn_that_ended_while_it_was_held_is_dropped_as_it_is_let_go(live: Mapping[SessionId, Session], told: tuple[Pending, ...]) -> None:
    """Whatever was queued of the ending: with ended or finished turns off, or quiet, nothing of it reaches the floor."""
    assert tuple(each.pending for each in coalesce((Working(SID, IN_TURN, (edit("a.py"),)), Working(SID, REVIEW, (TESTS,)), Working(OTHER, IN_TURN, (TESTS,))), live)) == told


def test_a_subagent_s_work_is_said_as_the_job_its_call_gave_it() -> None:
    [spoken] = rendered(Working(SID, REVIEW, (READ_TAIL, TESTS)), names=lambda _: "cc-hands")
    assert isinstance(spoken, TTSSpeakFrame) and spoken.text == "cc-hands, its subagent to review the parser change: read tail.py, then run the test suite."


async def test_a_subagent_s_work_is_played_though_its_parent_s_turn_is_over() -> None:
    queued, _ = await played(Progress(SID, REVIEW, (TESTS,), ""), lambda: Told(TURN), unasked)
    assert queued == [Working(SID, REVIEW, (TESTS,))]
