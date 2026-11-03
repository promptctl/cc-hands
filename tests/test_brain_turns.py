"""The brain built bare, in the process and on a VirtualClock the test moves: how hands types a turn, matches what Claude
Code takes and tells to the turns it asked, and answers the hooks it posts, with no Claude Code running."""

from types import SimpleNamespace
import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest
from hands.virtualclock import VirtualClock
from loguru import logger

from hands.brain import process
from hands.sessions.reader import Reader
from hands.sessions.tail import Tails
from hands.brain.process import ClaudeTurns, BROKEN, LET_GO, NOBODY, STOPPED, SUBMIT, LOOK_SECONDS, SETTLE_SECONDS, TAKE_SECONDS, UNREAD, UNVOICED, Asked, Brain, BrainAnswered, BrainStopped, Untaken, _listen, _Turn, model  # pyright: ignore[reportPrivateUsage]
from hands.core.effects import Allow, Deny, HookReply, Submitted, Summarise
from hands.sessions.hooks import hook_output
from hands.core.session import DenyAndInterrupt, Held, Permission, RequestId
from hands.sessions.audit import Entry
from conftest import events, patiently, settled
from brains import dialogs, failures, permissions, unasked

from hands.core.events import Abandoned, Dequeued, Ending, Located, Prompted, Removed, StatusReported, Stopped, Taken, Unqueued
from hands.core.status import Busy, Idle, Report, Stamp
from hands.core.session import Membership, PromptId, PromptText, SessionId
from hands.core.wire import MainTurn, Sent
from hands.sessions.wide import WideEvent, begun, here, root, unit


async def until(check: Callable[[], bool]) -> None:
    """Until `check` holds: what a bare brain does next it does in the loop's turns, so a wait that stalls is the failure."""
    await patiently(check, lambda: None)


async def test_a_turn_whose_screen_never_shows_the_prompt_is_untaken_saying_what_it_showed_with_nothing_typed() -> None:
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, [])
    keys.shown = DIALOG_SHOWN
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Each look that sees the dialog waits a look's pause, so four pauses waited out are four looks.
    for _ in range(4):
        await passed(clock, LOOK_SECONDS)
    clock.advance_to(TAKE_SECONDS)
    with pytest.raises(Untaken, match="showed no empty prompt to type the turn at.*`hands login` answers them") as untaken:
        await asking
    assert DIALOG_SHOWN in str(untaken.value)
    assert keys.pressed == []


async def test_words_left_in_the_input_while_claude_code_runs_its_own_turn_are_deleted_before_a_turn_is_pasted_and_that_turn_runs_on() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, recorded)
    # Claude Code runs a turn it opened itself, and words are left in its input, drawn plain.
    _took(brain, "n1", NOTICE)
    keys.shown = DRAFTED_SHOWN
    pressed = keys.press

    def press(key: str) -> None:
        pressed(key)
        if key == "ctrl_h":
            keys.shown = PROMPT_SHOWN

    keys.press = press

    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Seen drafted, emptied, then seen empty and settled: the turn is pasted, and the take runs out on it.
    await passed(clock, SETTLE_SECONDS)
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: len(keys.pressed) == 5)
    clock.advance_to(TAKE_SECONDS)
    with pytest.raises(Untaken):
        await asking
    assert [key.strip() for key in keys.pressed] == ["ctrl_e", "ctrl_u", "ctrl_k", "ctrl_h", "and now?"]
    assert len(brain._claude_turns.opened_itself) == 1  # pyright: ignore[reportPrivateUsage]
    await until(lambda: len(events(recorded, "brain.turn")) == 1)
    [turn] = events(recorded, "brain.turn")
    assert "a person's words" in str(turn.facts["emptied"])


async def test_a_turn_whose_input_keeps_its_words_is_untaken_saying_what_it_showed_and_never_pasted() -> None:
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, [])
    keys.shown = DRAFTED_SHOWN
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Each look sees the words, settles, and empties them; the screen still shows them, so the next look sees them again.
    for _ in range(3):
        await passed(clock, SETTLE_SECONDS)
    clock.advance_to(TAKE_SECONDS)
    with pytest.raises(Untaken, match="showed no empty prompt") as untaken:
        await asking
    assert "a person's words" in str(untaken.value) and "\x1b" not in str(untaken.value)
    assert not any("and now?" in key for key in keys.pressed) and "escape" not in keys.pressed and "ctrl_c" not in keys.pressed


def _bare() -> Brain:
    """A Brain no launch built, holding what every turn's take reads from `__init__`: the tests set the rest."""
    brain = object.__new__(Brain)
    brain._untaken = None  # pyright: ignore[reportPrivateUsage]
    return brain


async def test_a_dialog_between_turns_or_with_a_body_that_does_not_parse_is_answered_no_and_said() -> None:
    recorded: list[Entry] = []
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    clock = VirtualClock()
    with unit("brain.launch", recorded.append, begun(clock)):
        launched = here()
    brain._launched = launched  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    _modelled(brain, clock)
    listener, url = await _serving(brain)

    async def answered(event: str, body: bytes) -> dict[str, object]:
        async with aiohttp.ClientSession() as client, client.post(f"{url}/{event}", data=body) as reply:
            assert reply.status == 200
            return (await reply.json())["hookSpecificOutput"]

    try:
        assert await answered("Elicitation", b"not json") == {"hookEventName": "Elicitation", "action": "decline"}
        assert await answered("PermissionRequest", b"not json") == {"hookEventName": "PermissionRequest", "decision": {"behavior": "deny", "message": UNREAD}}
        # An MCP server asking while it connects, and a permission with no turn in flight to ask the user in: no prompt id.
        elicited = {"hook_event_name": "Elicitation", "session_id": "b1", "mcp_server_name": "probe", "message": "Which?"}
        assert (await answered("Elicitation", json.dumps(elicited).encode()))["action"] == "decline"
        permission = {"hook_event_name": "PermissionRequest", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
        assert (await answered("PermissionRequest", json.dumps(permission).encode()))["decision"] == {"behavior": "deny", "message": NOBODY}
        # A dialog of questions is never put to the user by voice, in a turn or not: the brain is told to ask in words.
        questions: dict[str, object] = {**permission, "tool_name": "AskUserQuestion", "tool_input": {"questions": []}}
        assert (await answered("PermissionRequest", json.dumps(questions).encode()))["decision"] == {"behavior": "deny", "message": UNVOICED}
    finally:
        await listener.cleanup()
    # Each is one event, a part of the brain's launch, the unreadable ones failed saying so.
    unread, elicited_event = events(recorded, "brain.elicitation")
    assert unread.outcome == "failed" and unread.error is not None and unread.error.startswith("hands could not read the Elicitation hook")
    assert (elicited_event.outcome, elicited_event.facts) == ("ok", {"prompt": None, "server": "probe"})
    unreadable, *read = events(recorded, "brain.permission")
    assert unreadable.outcome == "failed" and unreadable.error is not None and unreadable.error.startswith("hands could not read the permission request")
    assert permissions(read) == [(None, "Write", Deny(NOBODY)), (None, "AskUserQuestion", Deny(UNVOICED))]
    assert {event.parent_id for event in events(recorded, "brain.permission") + events(recorded, "brain.elicitation")} == {launched.span_id}


async def test_a_hook_with_a_field_that_does_not_parse_is_passed_over_and_hooks_are_still_heard() -> None:
    brain = _bare()
    brain.session = SessionId("b1")
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    _modelled(brain, VirtualClock())
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(message.record["message"]), level="WARNING")
    try:
        # A prompt id that is not text names no turn: the hook is logged and passed over, never raised out of the loop that hears hooks.
        failed = {"hook_event_name": "StopFailure", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": {"id": "p1"}, "error": "unknown"}
        await brain.posted("StopFailure", json.dumps(failed).encode())
    finally:
        logger.remove(sink)
    assert [line.split(":")[0] for line in warned] == ["the brain posted a hook that does not parse"]


def test_a_brain_turn_without_hands_tools_is_an_error() -> None:
    recorded: list[Entry] = []
    brain = _bare()
    brain.session = SessionId("b1")
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    errors, sink = failures()
    try:
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Read"}]}
        with_hands = {**body, "tools": [{"name": "Read"}, {"name": "mcp__hands__read_session"}]}
        brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), None, with_hands))
        brain.hear(Sent("x2", SessionId("b1"), MainTurn(None), None, with_hands))
        brain.hear(Sent("x3", SessionId("elsewhere"), MainTurn(None), None, body))
        assert errors == []
        brain.hear(Sent("x4", SessionId("b1"), MainTurn(None), None, body))
        # Deferred, as Claude Code offers MCP tools under tool search: the server may be connected, and is not blamed.
        deferred = {**body, "tools": [{"name": "Read"}, {"name": "ToolSearch"}, {"name": "DeferredToolPlaceholder", "defer_loading": True}]}
        brain.hear(Sent("x5", SessionId("b1"), MainTurn(None), None, deferred))
    finally:
        logger.remove(sink)
    assert errors == [
        "the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server (('Read',))",
        "the brain's turn went to the model without hands' tools and with ToolSearch: tool search is on though hands' --settings"
        " turn it off (('Read', 'ToolSearch', 'DeferredToolPlaceholder'))",
    ]
    # What a turn's requests offered is on its brain turn's event.
    assert recorded == []


def _modelled(brain: Brain, clock: VirtualClock) -> None:
    """A brain built bare on `clock`, given the model it reads its turns with, and what it waits on that model with, as
    `Brain` builds it; a part of a launch of its own where the test gave it none."""
    brain._clock = clock  # pyright: ignore[reportPrivateUsage]
    brain._sessions = model(brain._clock, lambda _entry: None, brain._told, brain._took)  # pyright: ignore[reportPrivateUsage]
    brain._reader = Reader(brain._sessions, Tails(brain._sessions), brain._apply)  # pyright: ignore[reportPrivateUsage]
    brain._moved = asyncio.Event()  # pyright: ignore[reportPrivateUsage]
    brain._claude_turns = ClaudeTurns(brain._settle)  # pyright: ignore[reportPrivateUsage]
    brain.__dict__.setdefault("_launched", root(clock))


async def _serving(brain: Brain) -> tuple[web.AppRunner, str]:
    """The listener a bare brain's hooks are posted to, answered by its `posted` as `start` wires it."""
    posting: asyncio.Future[Callable[[str, bytes], Awaitable[Mapping[str, object]]]] = asyncio.get_running_loop().create_future()
    posting.set_result(brain.posted)
    return await _listen(posting)  # pyright: ignore[reportPrivateUsage]


def _hooked(event: str, id: str, **fields: object) -> bytes:
    """A hook of the brain's session under prompt `id`, as Claude Code fills it."""
    return json.dumps({"hook_event_name": event, "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": id, **fields}).encode()


async def read_through[T](clock: VirtualClock, hook: Awaitable[T]) -> T:
    """`hook`, a Stop with words waiting, answered once it is held for the transcript (its hold's deadline scheduled) and
    the reading on a 0.01s period has read again since."""
    held = asyncio.ensure_future(hook)
    await clock.scheduled(clock.at + TAKE_SECONDS)
    await passed(clock, 0.01)
    return await held


@pytest.mark.parametrize(("delivered", "answer", "matched"), [(True, "p1", "id"), (False, "p2", "behind")], ids=["taken into the turn running", "run after it"])
async def test_words_filed_under_a_running_turn_are_answered_by_the_turn_that_answers_them(delivered: bool, answer: str, matched: str) -> None:
    # [LAW:one-source-of-truth] which turn answers them is the model's to say, from Claude Code's queue records: the turn
    # they were typed into, when a remove took them out of the queue before its Stop, or the turn they ran as after it.
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    clock = VirtualClock()
    _modelled(brain, clock)
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(clock), TAKE_SECONDS, PromptText(" and add 3?"), pasted=True)  # pyright: ignore[reportPrivateUsage]
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    # Read on its period, as a running brain's model is: a Stop with words waiting is held until the transcript is read
    # through it, and let go unread, the order of the brain's turns is lost (`Brain._ordered`).
    reading = asyncio.ensure_future(brain._reader.keep_reading(0.01))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="count to ten"))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    # Typed while p1 runs: Claude Code files the words under its id.
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="and add 3?"))  # pyright: ignore[reportPrivateUsage]
    assert (turn.taken.done(), turn.taken_under, turn.queued) == (True, "p1", True)
    if delivered:
        await brain._apply(Unqueued(SessionId("b1"), Removed("and add 3?")))  # pyright: ignore[reportPrivateUsage]
    await read_through(clock, brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Ten.", stop_hook_active=False)))  # pyright: ignore[reportPrivateUsage]
    if not delivered:
        assert not turn.answered.done()
        await brain._apply(Unqueued(SessionId("b1"), Dequeued()))  # pyright: ignore[reportPrivateUsage]
        await brain._apply(Taken(SessionId("b1"), PromptId("p2"), Stamp(3100), 3.1))  # pyright: ignore[reportPrivateUsage]
        await read_through(clock, brain.posted("Stop", _hooked("Stop", "p2", last_assistant_message="Thirteen.", stop_hook_active=False)))  # pyright: ignore[reportPrivateUsage]
    reading.cancel()
    await asyncio.wait({reading})
    assert turn.running is not None
    assert (turn.answered.result(), turn.running.ids, turn.running.matched) == (BrainAnswered(answer, None), frozenset({PromptId(answer)}), matched)


async def test_a_permission_of_a_turn_dequeued_under_a_new_id_is_asked_in_it_though_only_the_transcript_says_it_opened(tmp_path: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] the permission is owned once Claude Code is read through now, not when a period
    # next reads the record that opened the turn it asks in.
    transcript = tmp_path / "projects" / "-brain-cwd" / "b1.jsonl"
    transcript.parent.mkdir(parents=True)
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    recorded: list[Entry] = []
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    clock = VirtualClock()
    with unit("brain.launch", lambda _entry: None, begun(clock)):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    _modelled(brain, clock)
    asked: list[Asked] = []

    def allowed(held: Asked) -> None:
        asked.append(held)
        held.settle(Allow())

    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), allowed, begun(clock), TAKE_SECONDS, PromptText(" and add 3?"), pasted=True)  # pyright: ignore[reportPrivateUsage]
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    path = str(transcript)
    # Read on its period, as a running brain's model is, so the Stop with words waiting is decided and not let go.
    reading = asyncio.ensure_future(brain._reader.keep_reading(0.01))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="count to ten", transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="and add 3?", transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await read_through(clock, brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Ten.", stop_hook_active=False, transcript_path=path)))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Unqueued(SessionId("b1"), Dequeued()))  # pyright: ignore[reportPrivateUsage]
    # Claude Code runs the words as p2, which no hook has named, and the model asks for a tool before any read since.
    transcript.write_text('{"type":"user","promptId":"p2","timestamp":"1970-01-01T00:00:03.100Z","message":{"role":"user","content":"and add 3?"}}\n')
    permission = {**json.loads(_hooked("PermissionRequest", "p2", transcript_path=path)), "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    reply = await brain.posted("PermissionRequest", json.dumps(permission).encode())
    reading.cancel()
    await asyncio.wait({reading})
    assert (len(asked), reply) == (1, hook_output(Allow()))


async def test_a_brain_whose_reading_failed_still_lets_go_of_what_it_held_as_it_stops(tmp_path: Path) -> None:
    # What stopped the reading is `exited`'s to raise, to whoever waits on the brain: the stop ends it and lets go anyway.
    ended = asyncio.Event()

    class Ended:
        async def stop(self) -> None:
            ended.set()

    class Listener:
        cleaned = False

        async def cleanup(self) -> None:
            self.cleaned = True

    async def exited() -> int:
        # Its end is heard a moment after its process ends, as `_run_out` hears it.
        await ended.wait()
        await settled()
        return 0

    async def reading() -> None:
        raise OSError("the transcript went")

    brain = _bare()

    brain._claude = Ended()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    brain._exit = asyncio.ensure_future(exited())  # pyright: ignore[reportPrivateUsage]
    brain._reading = asyncio.ensure_future(reading())  # pyright: ignore[reportPrivateUsage]
    await asyncio.wait({brain._reading})  # pyright: ignore[reportPrivateUsage]
    listener = brain._listener = Listener()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    brain._sockets = tmp_path / "sockets"  # pyright: ignore[reportPrivateUsage]
    brain._sockets.mkdir()  # pyright: ignore[reportPrivateUsage]
    await brain.stop()
    assert (listener.cleaned, brain._sockets.exists()) == (True, False)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(OSError, match="the transcript went"):
        brain._reading.result()  # pyright: ignore[reportPrivateUsage]


async def test_a_hook_whose_hearing_fails_unexpectedly_is_said_once_and_the_hooks_after_it_are_still_heard() -> None:
    recorded: list[Entry] = []

    def record(entry: Entry) -> None:
        # The first event the log is handed cannot be written.
        if not recorded:
            recorded.append(entry)
            raise ValueError("the log is full")
        recorded.append(entry)

    brain = _bare()

    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = record  # pyright: ignore[reportPrivateUsage]
    clock = VirtualClock()
    with unit("brain.launch", lambda _entry: None, begun(clock)):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    # The elicitation that breaks is a part of the turn in flight: it was declined before it was heard, so the turn runs on.
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(clock), TAKE_SECONDS, PromptText(" probe"), pasted=True)  # pyright: ignore[reportPrivateUsage]
    turn.taken.set_result(None)
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    # Claude Code's turn p1 in flight, as the model holds it: the turn taken.
    _modelled(brain, clock)
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path("/brain/projects/-brain-cwd/b1.jsonl"))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "probe"))  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    listener, url = await _serving(brain)
    said, sink = failures()

    async def answered(event: str, body: Mapping[str, object]) -> dict[str, object]:
        async with aiohttp.ClientSession() as client, client.post(f"{url}/{event}", data=json.dumps(body).encode()) as reply:
            assert reply.status == 200
            return (await reply.json())["hookSpecificOutput"]

    try:
        elicited = {"hook_event_name": "Elicitation", "session_id": "b1", "prompt_id": "p1", "mcp_server_name": "probe", "message": "Which?"}
        assert (await answered("Elicitation", elicited))["action"] == "decline"
        permission = {"hook_event_name": "PermissionRequest", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
        assert (await answered("PermissionRequest", permission))["decision"] == {"behavior": "deny", "message": NOBODY}
    finally:
        logger.remove(sink)
        await listener.cleanup()
    assert said == ["the brain's own work failed: ValueError('the log is full')"]
    assert (turn.broken, turn.answered.done(), brain._turn) == (None, False, turn)  # pyright: ignore[reportPrivateUsage]
    assert [event.event for event in recorded if isinstance(event, WideEvent)] == ["brain.elicitation", "brain.permission"]


async def test_a_break_of_a_turn_already_over_ends_nothing_the_turn_in_flight_runs() -> None:
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    clock = VirtualClock()
    _modelled(brain, clock)
    brain._turn, brain._typing = None, set()  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path("/brain/projects/-brain-cwd/b1.jsonl"))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p2"), "next"))  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    over, current = (_Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(clock), TAKE_SECONDS, PromptText(f" {typed}"), pasted=True) for typed in ("first", "next"))  # pyright: ignore[reportPrivateUsage]
    brain._turn = current  # pyright: ignore[reportPrivateUsage]
    said, sink = failures()
    try:
        # What the turn before it left running breaks late, once the next turn's prompt has started new work.
        brain._broke(over, RuntimeError("late"))  # pyright: ignore[reportPrivateUsage]
    finally:
        logger.remove(sink)
    held = brain._modelled()  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    assert (said, held.ending, current.broken, brain._typing) == (["the brain's own work failed: RuntimeError('late')"], None, None, set())  # pyright: ignore[reportPrivateUsage]


async def test_a_permission_another_session_posts_to_the_brain_is_refused_and_is_one_event_naming_that_session() -> None:
    recorded: list[Entry] = []
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    _modelled(brain, VirtualClock())
    permission = {"hook_event_name": "PermissionRequest", "session_id": "b2", "transcript_path": "/brain/projects/-brain-cwd/b2.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": "p9", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    assert await brain.posted("PermissionRequest", json.dumps(permission).encode()) == hook_output(Deny(NOBODY))
    assert [(event.facts["session"], event.facts["prompt"], event.facts["tool"], event.facts["decision"]) for event in events(recorded, "brain.permission")] == [("b2", "p9", "Write", Deny(NOBODY))]


TRANSCRIPT = "/brain/projects/-brain-cwd/b1.jsonl"


def _permission_hook(prompt: str, path: str = "/tmp/x") -> bytes:
    """A PermissionRequest of the brain's session under prompt `prompt`, as Claude Code fills it."""
    return _hooked("PermissionRequest", prompt, tool_name="Write", tool_input={"file_path": path})


async def _in_turn(recorded: list[Entry], asks: Callable[[Asked], None]) -> tuple[Brain, _Turn]:  # pyright: ignore[reportPrivateUsage]
    """A brain built bare, with a turn in flight that Claude Code took as p1, which the model holds running."""
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._record = recorded.append  # pyright: ignore[reportPrivateUsage]
    clock = VirtualClock()
    with unit("brain.launch", lambda _entry: None, begun(clock)):
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    brain._typing = set()  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), asks, begun(clock), TAKE_SECONDS, PromptText(" write"), pasted=True)  # pyright: ignore[reportPrivateUsage]
    turn.taken.set_result(None)
    brain._turn = turn  # pyright: ignore[reportPrivateUsage]
    _modelled(brain, clock)
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "write"))  # pyright: ignore[reportPrivateUsage]
    return brain, turn


async def test_a_permission_request_is_held_in_the_model_before_anything_is_awaited_while_a_stop_posted_before_it_is_still_open() -> None:
    """[LAW:no-ambient-temporal-coupling] each hook is handled as it arrives: a dialog posted behind a Stop the model holds
    for its record is held in the model while the Stop's post is still open, before its own read of Claude Code."""
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _ = await _in_turn(recorded, asked.append)
    seen: list[tuple[tuple[Held, ...], bool]] = []
    read = brain._reader.read  # pyright: ignore[reportPrivateUsage]
    # A Stop under an id no record names yet, with the turn read busy: held until the transcript says whose it is.
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    stopping = asyncio.ensure_future(brain.posted("Stop", _hooked("Stop", "p9", last_assistant_message="", stop_hook_active=False)))

    async def reading() -> None:
        # What the model holds as the permission's own read of Claude Code begins, and whether the Stop is still open.
        seen.append((dialogs(brain), stopping.done()))
        await read()

    await until(lambda: any(stop.prompt == "p9" for stop in brain._modelled().stops))  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    brain._reader.read = reading  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    asking = asyncio.ensure_future(brain.posted("PermissionRequest", _permission_hook("p1")))
    await until(lambda: len(asked) == 1)
    [(held, stopped)] = seen
    assert [dialog.on for dialog in held] == [Permission("Write", {"file_path": "/tmp/x"})] and not stopped
    asked[0].settle(Allow())
    assert await asking == hook_output(Allow())
    stopping.cancel()
    await asyncio.wait({stopping})


async def test_a_dialog_asked_after_hands_stopped_its_turn_is_refused_with_interrupt_and_never_put_to_the_user_whatever_the_brain_still_holds() -> None:
    """The model, not the brain's turn in flight, says what hands is ending: once the stop's Escape went, a dialog Claude
    Code asks before a prompt starts new work ends its turn, though the brain let go of its turn already (a stop
    StopUnrecorded ended), where it would otherwise be nobody's and refused only as an answer."""
    recorded: list[Entry] = []
    brain, turn = await _in_turn(recorded, unasked)
    brain._sessions.hear(Ending(SessionId("b1"), DenyAndInterrupt(STOPPED)))  # pyright: ignore[reportPrivateUsage]
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(DenyAndInterrupt(STOPPED))
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    assert await brain.posted("PermissionRequest", _permission_hook("p1", "/tmp/y")) == hook_output(DenyAndInterrupt(STOPPED))
    assert dialogs(brain) == ()
    assert permissions(recorded) == [("p1", "Write", DenyAndInterrupt(STOPPED)), ("p1", "Write", DenyAndInterrupt(STOPPED))]
    owned, unowned = events(recorded, "brain.permission")
    assert (owned.parent_id, unowned.parent_id) == (turn.began.span.span_id, brain._launched.span_id)  # pyright: ignore[reportPrivateUsage]


async def test_a_dialog_claude_code_went_on_past_unanswered_is_refused_and_its_asker_and_event_say_so() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _ = await _in_turn(recorded, asked.append)
    asking = asyncio.ensure_future(brain.posted("PermissionRequest", _permission_hook("p1")))
    await until(lambda: len(asked) == 1)
    # Claude Code at its prompt, the dialog never answered: the model lets it go, and the brain refuses it.
    await brain._apply(StatusReported(SessionId("b1"), Report(Idle(), Stamp(9000)), 9.0))  # pyright: ignore[reportPrivateUsage]
    assert await asking == hook_output(Deny(LET_GO))
    assert await asked[0].decided() == Deny(LET_GO)
    assert permissions(recorded) == [("p1", "Write", Deny(LET_GO))]


async def test_a_permission_whose_read_breaks_is_no_turns_and_is_refused_said_once_and_left_held_by_nothing() -> None:
    recorded: list[Entry] = []
    brain, turn = await _in_turn(recorded, unasked)

    async def broken() -> None:
        raise OSError("the transcript went")

    brain._reader.read = broken  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    said, sink = failures()
    try:
        assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    finally:
        logger.remove(sink)
    assert said == ["the brain's own work failed: OSError('the transcript went')"]
    assert (dialogs(brain), turn.broken) == ((), None)
    [permission] = events(recorded, "brain.permission")
    assert (permission.outcome, permission.error, permission.facts["decision"]) == ("failed", "OSError: the transcript went", Deny(BROKEN))
    assert permission.trace


async def test_a_permission_the_model_could_not_hold_is_refused_at_its_hook_and_said_once() -> None:
    recorded: list[Entry] = []
    brain, _ = await _in_turn(recorded, unasked)

    def unheld(_hook: object) -> "asyncio.Future[HookReply]":
        raise RuntimeError("the model is gone")

    brain._reader.asked = unheld  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    said, sink = failures()
    try:
        assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    finally:
        logger.remove(sink)
    assert said == ["the brain's own work failed: RuntimeError('the model is gone')"]
    assert permissions(recorded) == [("p1", "Write", Deny(BROKEN))]


async def test_a_dialog_of_a_turn_hands_broke_is_refused_and_never_put_to_the_user() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, turn = await _in_turn(recorded, asked.append)
    # Broken once Claude Code took it, and its prompt cleared any ending the model held: its stop is still to come.
    turn.broken = RuntimeError("no voice")
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    assert (asked, permissions(recorded)) == ([], [("p1", "Write", Deny(BROKEN))])


async def test_a_permission_whose_event_cannot_be_written_is_refused_at_its_hook_and_said_once() -> None:
    """Never left unanswered, which would leave it to Claude Code's own dialog on a screen nobody sees."""
    recorded: list[Entry] = []
    brain, _ = await _in_turn(recorded, unasked)

    def full(entry: Entry) -> None:
        if isinstance(entry, WideEvent) and entry.event == "brain.permission":
            raise ValueError("the log is full")
        recorded.append(entry)

    brain._record = full  # pyright: ignore[reportPrivateUsage]
    other = {"hook_event_name": "PermissionRequest", "session_id": "b2", "transcript_path": "/brain/projects/-brain-cwd/b2.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": "p9", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    said, sink = failures()
    try:
        assert await brain.posted("PermissionRequest", json.dumps(other).encode()) == hook_output(Deny(BROKEN))
    finally:
        logger.remove(sink)
    assert said == ["the brain's own work failed: ValueError('the log is full')"]


# A turn Claude Code opens for a background task's result, as its prompt hook names it (backgrounded_then_notified.jsonl, p2).
NOTICE = "<task-notification>\n<task-id>b74jw7b5h</task-id>\n<status>completed</status>\n</task-notification>"


def _owning(clock: VirtualClock) -> Brain:
    """A brain built bare on `clock`, its model and its `ClaudeTurns` as `Brain` builds them, heard only through what the
    model hands over: the words each prompt hook filed (`_took`), and each telling (`_told`)."""
    brain = _bare()
    brain.session = SessionId("b1")
    brain._pid, brain._cwd = 4242, Path("/brain/cwd")  # pyright: ignore[reportPrivateUsage]
    brain._turn = None  # pyright: ignore[reportPrivateUsage]
    _modelled(brain, clock)
    return brain


def _typed_turn(brain: Brain, words: str) -> _Turn:  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    turn = brain._turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), unasked, begun(brain._clock), TAKE_SECONDS, PromptText(f" {words}"), pasted=True)  # pyright: ignore[reportPrivateUsage]
    return turn


def _took(brain: Brain, prompt: str, words: str, queued: bool = False) -> None:
    brain._took(Submitted(SessionId("b1"), PromptId(prompt), words, queued))  # pyright: ignore[reportPrivateUsage]


def _told(brain: Brain, prompt: str | None, ids: Sequence[str], waiting: tuple[str, ...] = ()) -> None:
    brain._told(Summarise(SessionId("b1"), None if prompt is None else PromptId(prompt), "Done.", waiting, ids=frozenset(map(PromptId, ids))))  # pyright: ignore[reportPrivateUsage]


async def test_a_turn_told_under_an_id_its_hook_did_not_name_ends_its_own_turn_by_the_ids_its_telling_carries() -> None:
    # A message queued into the turn flushed it onto p2: its telling goes by p2, and carries p1 too (`Summarise.ids`).
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    assert turn.running is not None and turn.running.ids == {"p1"}
    _told(brain, "p2", ("p1", "p2"))
    assert (turn.answered.result(), turn.running.matched, brain._claude_turns.open) == (BrainAnswered(PromptId("p2"), None), "id", ())  # pyright: ignore[reportPrivateUsage]


async def test_words_claude_code_takes_before_a_turn_pasted_its_own_are_never_that_turns() -> None:
    # An untaken turn's words, taken late as its clear goes, while the same words asked again wait for the input.
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    turn.pasted = False
    _took(brain, "p1", "hello")
    assert (turn.taken.done(), [running.opener for running in brain._claude_turns.open]) == (False, ["other"])  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("queued", [False, True])
async def test_words_of_an_untaken_turn_that_claude_code_takes_next_are_stopped(queued: bool) -> None:
    brain = _owning(VirtualClock())
    keys = _keyed(brain, [])
    brain._over(_typed_turn(brain, "hello"), Untaken("the brain did not take the turn typed into it"))  # pyright: ignore[reportPrivateUsage]
    _took(brain, "p1", "hello", queued=queued)
    (late,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await until(lambda: late.ended.done())
    assert (type(late.ended.exception()), keys.pressed, brain._claude_turns.open) == (process.Unran, ["escape", "ctrl_c"], ())  # pyright: ignore[reportPrivateUsage]


async def test_words_of_an_untaken_turn_taken_after_a_task_s_result_are_still_stopped() -> None:
    # A task's result posts its hook as it is delivered, not read from the input: the untaken words are still next in it.
    brain = _owning(VirtualClock())
    keys = _keyed(brain, [])
    brain._over(_typed_turn(brain, "hello"), Untaken("the brain did not take the turn typed into it"))  # pyright: ignore[reportPrivateUsage]
    _took(brain, "p1", "<task-notification>done</task-notification>")
    _took(brain, "p1", "hello", queued=True)
    (notified,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await until(lambda: notified.ended.done())
    assert (keys.pressed, brain._claude_turns.open) == (["escape", "ctrl_c"], ())  # pyright: ignore[reportPrivateUsage]


async def test_words_of_an_untaken_turn_taken_after_a_stop_cleared_the_input_are_a_person_s_and_run_on() -> None:
    # A stop's Escape and Ctrl-C clear Claude Code's queue and input, so the untaken words cannot come late after one.
    brain = _owning(VirtualClock())
    keys = _keyed(brain, [])
    brain._over(_typed_turn(brain, "yes"), Untaken("the brain did not take the turn typed into it"))  # pyright: ignore[reportPrivateUsage]
    await brain._clear(None)  # pyright: ignore[reportPrivateUsage]
    _took(brain, "p1", "yes")
    assert (len(brain._claude_turns.open), keys.pressed, brain._typing) == (1, ["escape", "ctrl_c"], set())  # pyright: ignore[reportPrivateUsage]


async def test_words_of_an_untaken_turn_a_turn_asked_since_was_filed_into_run_on_to_answer_it() -> None:
    # The next turn's words, read after the late ones, run in the same turn of Claude Code's: stopping it would end a turn
    # its asker asked.
    brain = _owning(VirtualClock())
    keys = _keyed(brain, [])
    brain._over(_typed_turn(brain, "hello"), Untaken("the brain did not take the turn typed into it"))  # pyright: ignore[reportPrivateUsage]
    _took(brain, "p1", "hello")
    asked = _typed_turn(brain, "what's the weather")
    _took(brain, "p1", "what's the weather", queued=True)
    await until(lambda: not brain._typing)  # pyright: ignore[reportPrivateUsage]
    (late,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert (asked.running, asked.answered.done(), keys.pressed) == (late, False, [])


async def test_words_of_an_untaken_turn_taken_after_other_words_are_a_person_s_and_run_on() -> None:
    # Claude Code reads its input in order: once it took other words, the untaken ones cannot come late, and the same words
    # typed at the pane later are a person's.
    brain = _owning(VirtualClock())
    keys = _keyed(brain, [])
    brain._over(_typed_turn(brain, "yes"), Untaken("the brain did not take the turn typed into it"))  # pyright: ignore[reportPrivateUsage]
    _took(brain, "p1", "something else")
    _told(brain, "p1", ["p1"])
    _took(brain, "p2", "yes")
    assert (len(brain._claude_turns.open), keys.pressed, brain._typing) == (1, [], set())  # pyright: ignore[reportPrivateUsage]


async def test_a_telling_that_names_no_prompt_ends_the_turn_its_ids_are() -> None:
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, None, ("p1",))
    assert (turn.answered.result(), brain._claude_turns.open) == (BrainAnswered(None, None), ())  # pyright: ignore[reportPrivateUsage]


async def test_a_notification_opened_before_the_last_turns_stop_is_applied_is_a_turn_of_its_own_each_ending_on_its_own_telling() -> None:
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    # Claude Code ran p1 to its end and opened p2 for a background task's result before hands applied p1's Stop.
    _took(brain, "p2", NOTICE)
    typed, notified = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert (brain.current(), typed.opener, notified.opener, notified.awaited) == (notified, "typed", "notified", False)
    _told(brain, "p1", ("p1",))
    assert (turn.answered.result(), notified.ended.done(), brain._claude_turns.opened_itself) == (BrainAnswered(PromptId("p1"), None), False, (notified,))  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p2", ("p2",))
    assert (notified.ended.result(), notified.matched, brain.current(), brain._claude_turns.open) == (BrainAnswered(PromptId("p2"), None), "id", None, ())  # pyright: ignore[reportPrivateUsage]


async def test_hands_words_filed_into_a_notification_turn_are_answered_by_that_turn() -> None:
    brain = _owning(VirtualClock())
    _took(brain, "p1", NOTICE)
    (notified,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    turn = _typed_turn(brain, "and the tests?")
    # Typed while the notification's turn runs: Claude Code files the words under its id, and delivers them into it.
    _took(brain, "p1", "and the tests?", queued=True)
    assert (turn.running, notified.awaited) == (notified, True)
    _told(brain, "p1", ("p1",))
    assert (turn.answered.result(), notified.ended.result()) == (BrainAnswered(PromptId("p1"), None), BrainAnswered(PromptId("p1"), None))


async def test_a_notification_queued_into_a_turn_nobody_waits_on_ends_nothing_and_the_turn_after_ends_on_its_own_telling() -> None:
    # The v8 review's case: a second notification's hook comes under the running turn's id, which binds nothing new.
    brain = _owning(VirtualClock())
    _took(brain, "p1", NOTICE)
    _took(brain, "p1", NOTICE.replace("b74jw7b5h", "c11"), queued=True)
    (first,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p1", ("p1",))
    _took(brain, "p2", NOTICE.replace("b74jw7b5h", "d22"))
    (second,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p2", ("p2",))
    assert (first.ended.result(), second.ended.result(), brain._claude_turns.open) == (BrainAnswered(PromptId("p1"), None), BrainAnswered(PromptId("p2"), None), ())  # pyright: ignore[reportPrivateUsage]


async def test_a_turn_of_a_kind_added_to_the_openers_opens_and_ends_with_no_other_change(monkeypatch: pytest.MonkeyPatch) -> None:
    # [LAW:one-type-per-behavior] a scheduled prompt is one more row: nothing that matches a turn to its end reads the kind.
    monkeypatch.setattr(process, "OPENERS", [*process.OPENERS, ("scheduled", re.compile(r"<scheduled-prompt>"))])
    brain = _owning(VirtualClock())
    _took(brain, "p1", "<scheduled-prompt>check the build</scheduled-prompt>")
    (scheduled,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    _told(brain, "p1", ("p1",))
    assert (scheduled.opener, scheduled.ended.result(), scheduled.matched) == ("scheduled", BrainAnswered(PromptId("p1"), None), "id")


# The brain's screen with its prompt taking keys, and with a permission dialog up in its place.
PROMPT_SHOWN = f"{'─' * 40}\n❯ \n{'─' * 40}"


DRAFTED_SHOWN = f"{'─' * 40}\n❯\u00a0a person's words\x1b[7m \x1b[0m\n{'─' * 40}"


DIALOG_SHOWN = " Write file\n Do you want to create notes.txt?\n ❯ 1. Yes\n   2. No"


class _Keys:
    """The brain's keyboard, as fritter's typist takes keys: each typing and press, in order; and its screen, as tmux
    shows it, which shows the prompt unless the test says else: every capture it was read for is counted."""

    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.shown = PROMPT_SHOWN
        self.looked = 0

    def type(self, text: str, submit: str) -> None:
        # Every turn is sent by the brain's own chord, never by Return.
        assert submit == SUBMIT
        self.pressed.append(text)

    def press(self, key: str) -> None:
        self.pressed.append(key)

    async def screen(self) -> str:
        self.looked += 1
        return self.shown


def _keyed(brain: Brain, recorded: list[Entry]) -> _Keys:
    """A bare brain a turn can be asked of: typed into a keyboard of the test's, read off a screen of the test's, its take
    run out only by the test moving its clock."""
    loop = asyncio.get_running_loop()
    keys = _Keys()
    brain._typist, brain._claude, brain._record, brain._typing, brain._input = keys, keys, recorded.append, set(), asyncio.Lock()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    brain._exit, brain._cleared = loop.create_future(), float("-inf")  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    with unit("brain.launch", lambda _entry: None, begun(brain._clock)):  # pyright: ignore[reportPrivateUsage]
        brain._launched = here()  # pyright: ignore[reportPrivateUsage]
    return keys


async def passed(clock: VirtualClock, seconds: float) -> None:
    """Once a pause of `seconds` begun now is waited on, `clock` moved to its end: the brain's look loop waits each out
    in turn, so a test spends exactly the pauses it names."""
    end = clock.at + seconds
    await clock.paused(end)
    clock.advance_to(end)


async def test_a_turn_asked_while_a_notification_turn_runs_is_typed_at_once_into_its_queue_and_never_escapes_it() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, recorded)
    reading = asyncio.ensure_future(brain._reader.keep_reading(0.01))  # pyright: ignore[reportPrivateUsage]
    # Claude Code opens a turn of its own for a background task's result, as its prompt hook says.
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt=NOTICE))
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Typed with nothing held behind the notification's turn, and nothing pressed to stop it: no Escape, no Ctrl-C.
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: keys.pressed == [" and now?"])
    # Claude Code queues the words under the turn it runs: they are filed into it, which now answers them.
    _took(brain, "p1", "and now?", queued=True)
    assert (brain.answering(), notified.awaited, notified.ended.done()) == (notified, True, False)
    asking.cancel()
    reading.cancel()
    await asyncio.wait({reading})


async def test_a_brain_stop_let_go_undecided_fails_every_turn_open_and_the_turn_it_answers_saying_the_order_was_lost() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, recorded)
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(1000)), 1.0))  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask("hello", unasked))
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: keys.pressed == [" hello"])
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="hello"))
    # A notification Claude Code opened, then a Stop under an id no record names, which no read decides in its hold.
    _took(brain, "p2", NOTICE)
    stopping = asyncio.ensure_future(brain.posted("Stop", _hooked("Stop", "q", last_assistant_message="Hi.", stop_hook_active=False)))
    await clock.scheduled(clock.at + TAKE_SECONDS)
    clock.advance(TAKE_SECONDS)
    await stopping
    with pytest.raises(process.OrderLost):
        await asking
    assert brain._claude_turns.open == ()  # pyright: ignore[reportPrivateUsage]
    await until(lambda: len(events(recorded, "brain.turn")) == 1)
    [turn] = events(recorded, "brain.turn")
    assert (turn.outcome, turn.facts["order_lost"], turn.facts["ids"]) == ("failed", "the Stop of q was let go before it was decided", ("p1",))


def _heard_late(brain: Brain, clock: VirtualClock) -> None:
    """What the bare brain hears is heard a second past the bound hands has to answer a hook posted now in: its every
    hearing, however many a hook's answer waits on, at that one time."""
    heard, posted = brain._heard, clock.at  # pyright: ignore[reportPrivateUsage]

    async def late() -> None:
        clock.advance_to(max(clock.at, posted + TAKE_SECONDS + 1))
        await heard()

    brain._heard = late  # pyright: ignore[reportPrivateUsage]


async def test_a_brain_prompt_hook_answered_past_its_timeout_fails_every_turn_open() -> None:
    # Answered later than Claude Code waits on it, the turn's first request may have left before hands heard its id.
    clock = VirtualClock()
    brain = _owning(clock)
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _heard_late(brain, clock)
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p2", prompt=NOTICE))
    with pytest.raises(process.OrderLost):
        turn.answered.result()
    assert (turn.order_lost, brain._claude_turns.open) == (f"the UserPromptSubmit hook of p2 was answered in {TAKE_SECONDS + 1:.1f}s, past the {TAKE_SECONDS:g}s hands has to answer it in", ())  # pyright: ignore[reportPrivateUsage]


async def test_a_permission_hook_claude_code_gave_up_on_fails_no_turn() -> None:
    # Abandoned is a permission's, not a Stop's: the order of the brain's turns does not rest on it.
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Abandoned(SessionId("b1"), RequestId("r1"), 1.0))  # pyright: ignore[reportPrivateUsage]
    assert (turn.answered.done(), len(brain._claude_turns.open)) == (False, 1)  # pyright: ignore[reportPrivateUsage]


async def test_a_task_result_queued_behind_a_turn_runs_as_its_own_turn_behind_it_never_as_the_typed_turns() -> None:
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    # Claude Code queued the notification under the running turn's id, and it still waits as that turn is told.
    _took(brain, "p1", NOTICE, queued=True)
    _told(brain, "p1", ["p1"], waiting=(NOTICE,))
    assert turn.answered.result() == BrainAnswered(PromptId("p1"), None)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert (notified.opener, notified.awaited, notified.ids) == ("notified", False, frozenset())
    # Its telling is its own: no turn of hands' is ended by it.
    _told(brain, "p9", ["p9"])
    assert (notified.ended.result(), notified.matched, brain._claude_turns.open) == (BrainAnswered(PromptId("p9"), None), "behind", ())  # pyright: ignore[reportPrivateUsage]


async def test_a_task_result_queued_behind_a_turn_already_told_runs_as_the_turn_behind_it() -> None:
    brain = _owning(VirtualClock())
    _took(brain, "p1", NOTICE, queued=True)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert (notified.opener, notified.ids) == ("notified", frozenset())


async def test_the_answering_turn_follows_hands_words_from_the_turn_they_were_filed_into_to_the_turn_behind_it() -> None:
    brain = _owning(VirtualClock())
    _took(brain, "n1", NOTICE)
    turn = _typed_turn(brain, "and the tests?")
    _took(brain, "n1", "and the tests?", queued=True)
    (notified,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert brain.answering() is notified
    _told(brain, "n1", ["n1"], waiting=(" and the tests?",))
    behind = brain.answering()
    assert behind is not None and behind is not notified and not turn.answered.done()
    _told(brain, "p2", ["p2"])
    assert turn.answered.result() == BrainAnswered(PromptId("p2"), None)


async def test_a_turn_claude_code_opened_itself_is_stopped_when_its_work_is_the_work_shown() -> None:
    recorded: list[Entry] = []
    brain = _owning(VirtualClock())
    keys = _keyed(brain, recorded)
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    # No turn of hands' is in flight: the notification's is stopped all the same, so the user is never locked out by it.
    assert brain.interrupt(notified) is True
    await until(lambda: "escape" in keys.pressed)
    # A turn of no kind Claude Code opens itself, and none at all, is not the brain's to stop on its own.
    assert brain.interrupt(None) is False
    for task in brain._typing:  # pyright: ignore[reportPrivateUsage]
        task.cancel()


async def test_a_task_result_ctrl_c_cleared_from_claude_codes_queue_ends_its_turn_unran_and_holds_no_turn_after_it() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, recorded)
    _took(brain, "n1", NOTICE)
    # A second result waits as the first is told: it runs as the turn behind, which no hook opens.
    _told(brain, "n1", ["n1"], waiting=(NOTICE,))
    (behind,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert brain.interrupt(behind) is True
    await until(lambda: behind.ended.done())
    # Ctrl-C cleared it before it ran: no telling will end it, so the clear does, and nothing is left open to wait on.
    assert ("ctrl_c" in keys.pressed, type(behind.ended.exception()), brain._claude_turns.open) == (True, process.Unran, ())  # pyright: ignore[reportPrivateUsage]
    turn = asyncio.ensure_future(brain.ask("and now?", unasked))
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: " and now?" in keys.pressed)
    turn.cancel()


async def test_a_telling_of_no_turn_names_no_turn_behind() -> None:
    brain = _owning(VirtualClock())
    _took(brain, "n1", NOTICE)
    turn = _typed_turn(brain, "hello")
    _took(brain, "n1", "hello", queued=True)
    _told(brain, "n1", ["n1"], waiting=("hello",))
    (behind,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert brain.answering() is behind
    # A telling that goes by no id ends nothing: the typed words still wait to be answered by the turn they run as.
    _told(brain, None, [])
    assert (behind.ended.done(), behind.ids, turn.answered.done()) == (False, frozenset(), False)


async def test_a_brain_prompt_hook_that_does_not_parse_fails_every_turn_open_as_the_order_lost() -> None:
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    await brain.posted("UserPromptSubmit", b"{not json")
    with pytest.raises(process.OrderLost):
        turn.answered.result()
    assert brain._claude_turns.open == ()  # pyright: ignore[reportPrivateUsage]


async def test_a_brain_stop_answered_past_its_bound_fails_every_turn_open() -> None:
    # Decided in its hold, but answered later than hands has to: Claude Code may have gone on, so its order is not trusted.
    clock = VirtualClock()
    brain = _owning(clock)
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    _heard_late(brain, clock)
    await brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Hi.", stop_hook_active=False))
    # The turn it told is answered; the one still open is not trusted to be the one Claude Code runs next.
    assert turn.answered.result() == BrainAnswered(PromptId("p1"), None)
    with pytest.raises(process.OrderLost, match="the Stop hook of p1 was answered in"):
        notified.ended.result()


async def test_a_notification_that_opens_while_a_turn_waits_for_the_input_is_never_escaped_for_it() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, recorded)
    await brain._input.acquire()  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    await until(lambda: brain._turn is not None)  # pyright: ignore[reportPrivateUsage]
    # Past the ask, waiting on the input: Claude Code opens a turn of its own meanwhile.
    _took(brain, "n1", NOTICE)
    brain._input.release()  # pyright: ignore[reportPrivateUsage]
    # Typed into Claude Code's queue behind it, with nothing pressed to stop it.
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: keys.pressed == [" and now?"])
    _told(brain, "n1", ["n1"])
    await until(lambda: keys.pressed == [" and now?"])
    asking.cancel()


async def test_a_brain_stop_released_by_shutdown_loses_no_order() -> None:
    brain = _owning(VirtualClock())
    _took(brain, "n1", NOTICE)
    stop = Stopped(SessionId("b1"), "done", mode=None, prompt=PromptId("n1"), again=False, heard=Stamp(0), request=RequestId("r"))
    brain._ordered(SimpleNamespace(happened=stop), "released", 0.0)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    assert len(brain._claude_turns.open) == 1  # pyright: ignore[reportPrivateUsage]


async def test_stopping_a_notification_turn_stops_the_turn_typed_behind_it_before_it_was_taken() -> None:
    recorded: list[Entry] = []
    brain = _owning(VirtualClock())
    keys = _keyed(brain, recorded)
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    # Typed behind the notification, and not yet taken, as the user stops the notification's work shown.
    await brain._input.acquire()  # pyright: ignore[reportPrivateUsage]
    turn = _typed_turn(brain, "and now?")
    assert brain.interrupt(notified) is True
    _took(brain, "n1", "and now?", queued=True)
    brain._input.release()  # pyright: ignore[reportPrivateUsage]
    # Its words waited behind the turn stopped, and Ctrl-C cleared them: the turn ends stopped, never left waiting.
    await until(lambda: turn.answered.done())
    assert (turn.answered.result(), "escape" in keys.pressed, brain._claude_turns.open) == (BrainStopped(), True, ())  # pyright: ignore[reportPrivateUsage]


async def test_a_telling_of_a_turn_already_over_ends_no_turn_behind() -> None:
    brain = _owning(VirtualClock())
    turn = _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, "p1", ["p1"], waiting=(NOTICE,))
    (behind,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    # Told again, as another Stop hook's block re-tells it: the turn is over, and the result waiting is not named by it.
    _told(brain, "p1", ["p1"])
    assert (turn.answered.result(), behind.ended.done(), behind.ids) == (BrainAnswered(PromptId("p1"), None), False, frozenset())


async def test_words_queued_while_which_turn_runs_is_unknown_take_no_turn_for_the_one_running() -> None:
    brain = _owning(VirtualClock())
    _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    brain._lost("a test")  # pyright: ignore[reportPrivateUsage]
    # Claude Code may still run p1: a task result queued now runs behind it, never as the turn its requests are of.
    _took(brain, "p1", NOTICE, queued=True)
    assert brain.current() is None
    _took(brain, "p2", "next")
    assert brain.current() is not None and brain.current().ids == {"p2"}  # pyright: ignore[reportOptionalMemberAccess]


async def test_a_turn_is_typed_only_once_the_brains_screen_shows_its_prompt_in_place_of_a_dialog() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain = _owning(clock)
    keys = _keyed(brain, recorded)
    keys.shown = DIALOG_SHOWN
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Looked at four times over, the screen still shows the dialog: a dialog would drop the paste, so nothing goes in.
    for _ in range(3):
        await passed(clock, LOOK_SECONDS)
    await clock.paused(clock.at + LOOK_SECONDS)
    assert (keys.looked, keys.pressed) == (4, [])
    keys.shown = PROMPT_SHOWN
    clock.advance(LOOK_SECONDS)
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: keys.pressed == [" and now?"])
    asking.cancel()
