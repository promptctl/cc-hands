"""The brain built of fakes, in the process and on a VirtualClock the test moves: how hands types a turn, matches what Claude
Code takes and tells to the turns it asked, and answers the hooks it posts, with no Claude Code running."""

from types import SimpleNamespace
import asyncio
import os
import json
import threading
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import aiohttp
from aiohttp import web
import pytest
from hands.virtualclock import VirtualClock

from hands.brain import process
from hands.sessions.reader import Reader
from hands.brain.session import BrainSession
from hands.brain.process import BROKEN, EXIT_SECONDS, LET_GO, NOBODY, STOPPED, SUBMIT, LOOK_SECONDS, SETTLE_SECONDS, TAKE_SECONDS, UNREAD, UNVOICED, Asked, Brain, BrainAnswered, BrainGone, BrainStopped, Untaken, _listen, _Turn  # pyright: ignore[reportPrivateUsage]
from hands.core.effects import Allow, Deny, Effect, HookReply, Submitted, Summarise, ToldByRecord, ToldByStop, TurnEnding
from hands.core.stopfailure import LoginExpired, StopFailed
from hands.sessions.hooks import hook_output
from hands.core.session import DenyAndInterrupt, Held, Keystroke, Permission, RequestId
from hands.sessions.audit import Entry, LoginReported, Record
from conftest import Unstarting, events, laid_out, patiently, settled, taken
from fixtures.pins import LAID_OUT
from hands.core.session import Runnable
from hands.daemon import readiness, run
from hands.sessions import heartbeat
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from brains import dialogs, permissions, unasked

from hands.core.events import Abandoned, Dequeued, Ending, Interrupted, Located, Prompted, Removed, StatusReported, Stopped, Taken, UnknownSource, Unqueued
from hands.core.status import Busy, Idle, Report, Stamp
from hands.core.session import Membership, PromptId, SessionId
from hands.core.wire import MainTurn, Sent
from hands.sessions.wide import WideEvent, begun, here, unit
from hands.core.trace import Span
from claudescreen import ClaudeScreen
from hands.sessions.tmux import KEYS
from hands.sessions.typing import Untyped
from terminalcontract import CHORD

# [LAW:no-silent-failure] an error hands logs that a test does not take fails it (`errors`).
pytestmark = pytest.mark.usefixtures("errors")


async def until(check: Callable[[], bool]) -> None:
    """Until `check` holds: what a bare brain does next it does in the loop's turns, so a wait that stalls is the failure."""
    await patiently(check, lambda: None)


async def test_a_turn_whose_screen_never_shows_the_prompt_is_untaken_saying_what_it_showed_with_nothing_typed() -> None:
    clock = VirtualClock()
    brain, keys = _built(clock)
    keys.claude_code.at_dialog()
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Each look that sees the dialog waits a look's pause, so four pauses waited out are four looks.
    for _ in range(4):
        await passed(clock, LOOK_SECONDS)
    clock.advance_to(TAKE_SECONDS)
    with pytest.raises(Untaken, match="showed no empty prompt to type the turn at.*`hands login` answers them") as untaken:
        await asking
    assert keys.claude_code.shown() in str(untaken.value)
    assert keys.pressed == []


async def test_words_left_in_the_input_while_claude_code_runs_its_own_turn_are_deleted_before_a_turn_is_pasted_and_that_turn_runs_on() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append)
    # Claude Code runs a turn it opened itself, and words are left in its input, drawn plain.
    _took(brain, "n1", NOTICE)
    keys.claude_code.type("a person's words")
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
    brain, keys = _built(clock)
    keys.claude_code.type("a person's words")
    # An input the brain's keys do not reach: they are pressed, and the words stay.
    keys.press = lambda key: keys.pressed.append(key)  # pyright: ignore[reportAttributeAccessIssue]
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Each look sees the words, settles, and empties them; the screen still shows them, so the next look sees them again.
    for _ in range(3):
        await passed(clock, SETTLE_SECONDS)
    clock.advance_to(TAKE_SECONDS)
    with pytest.raises(Untaken, match="showed no empty prompt") as untaken:
        await asking
    assert "a person's words" in str(untaken.value) and "\x1b" not in str(untaken.value)
    assert not any("and now?" in key for key in keys.pressed) and "escape" not in keys.pressed and "ctrl_c" not in keys.pressed


def _launch(clock: VirtualClock) -> Span:
    """A launch of the brain's, which what the brain does that no turn asked for is a part of."""
    with unit("brain.launch", lambda _entry: None, begun(clock)):
        return here()


def _read_every_hundredth(reader: Reader) -> Awaitable[None]:
    """A brain's Claude Code read on a period of its own, as a running brain's is (`_read_on_its_period`)."""
    return reader.keep_reading(0.01)


async def _unread(_reader: Reader) -> None:
    """A brain's Claude Code read only as its test reads it: its own reading reads nothing until the brain is stopped."""
    await asyncio.Event().wait()


def _built(
    clock: VirtualClock,
    record: Record = lambda _entry: None,
    *,
    listener: "_Listener | None" = None,
    sockets: Path = Path("/brain/sockets"),
    config_dir: Path = Path("/brain"),
    launched: Span | None = None,
    reading: Callable[[Reader], Awaitable[None]] = _unread,
) -> tuple[Brain, "_Keys"]:
    """A brain built as `start` builds it, of fakes: its terminal and keyboard the test's, fritter 4241 wrapping Claude
    Code 4242 in session b1, working in /brain/cwd; read as `reading` reads it, on `clock`, which only the test moves."""
    keys = _Keys()
    brain = Brain(keys, 4241, 4240, BrainSession(SessionId("b1")), keys, _Listener() if listener is None else listener, sockets, config_dir, Path("/brain/cwd"), record, _launch(clock) if launched is None else launched, clock, reading)
    return brain, keys


@pytest.mark.outside
async def test_a_dialog_between_turns_or_with_a_body_that_does_not_parse_is_answered_no_and_said() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    with unit("brain.launch", recorded.append, begun(clock)):
        launched = here()
    brain = _owning(clock, recorded.append, launched)
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


async def test_a_hook_with_a_field_that_does_not_parse_is_passed_over_and_hooks_are_still_heard(errors: list[str]) -> None:
    recorded: list[Entry] = []
    brain = _owning(VirtualClock(), recorded.append)
    # A prompt id that is not text names no turn: the hook is one event and passed over, never raised out of the loop that hears hooks.
    failed = {"hook_event_name": "StopFailure", "session_id": "b1", "transcript_path": "/brain/projects/-brain-cwd/b1.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": {"id": "p1"}, "error": "unknown"}
    assert await brain.posted("StopFailure", json.dumps(failed).encode()) == {}
    [unread] = events(recorded, "brain.unread")
    assert unread.facts == {"hook": "StopFailure"} and unread.outcome == "failed" and "the brain's StopFailure hook does not parse" in str(unread.error)
    assert taken(errors) == ["the brain lost the order of its turns: the brain's StopFailure hook does not parse: field 'prompt_id' should be a string, got dict"]


async def test_a_brain_turn_without_hands_tools_is_an_error(errors: list[str]) -> None:
    recorded: list[Entry] = []
    brain = _owning(VirtualClock(), recorded.append)
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "Read"}]}
    with_hands = {**body, "tools": [{"name": "Read"}, {"name": "mcp__hands__read_session"}]}
    brain.hear(Sent("x1", SessionId("b1"), MainTurn(None), None, with_hands))
    brain.hear(Sent("x2", SessionId("b1"), MainTurn(None), None, with_hands))
    assert errors == []
    brain.hear(Sent("x4", SessionId("b1"), MainTurn(None), None, body))
    # Deferred, as Claude Code offers MCP tools under tool search: the server may be connected, and is not blamed.
    deferred = {**body, "tools": [{"name": "Read"}, {"name": "ToolSearch"}, {"name": "DeferredToolPlaceholder", "defer_loading": True}]}
    brain.hear(Sent("x5", SessionId("b1"), MainTurn(None), None, deferred))
    assert taken(errors) == [
        "the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server (('Read',))",
        "the brain's turn went to the model without hands' tools and with ToolSearch: tool search is on though hands' --settings"
        " turn it off (('Read', 'ToolSearch', 'DeferredToolPlaceholder'))",
    ]
    # What a turn's requests offered is on its brain turn's event.
    assert recorded == []


async def _serving(brain: Brain) -> tuple[web.AppRunner, str]:
    """The listener a brain built of fakes has its hooks posted to, answered by its `posted` as `start` wires it."""
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
    clock = VirtualClock()
    # Read on its period, as a running brain's model is: a Stop with words waiting is held until the transcript is read
    # through it, and let go unread, the order of the brain's turns is lost (`Brain._ordered`).
    brain, _ = _built(clock, reading=_read_every_hundredth)
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt=NOTICE))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    # Typed while p1, a turn Claude Code opened itself for a task's result, runs: Claude Code files the words under its id.
    turn = await _typed_turn(brain, "and add 3?")
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
    assert turn.running is not None
    assert (turn.answered.result(), turn.running.ids, turn.running.matched) == (BrainAnswered(answer, None), frozenset({PromptId(answer)}), matched)


async def test_a_permission_of_a_turn_dequeued_under_a_new_id_is_asked_in_it_though_only_the_transcript_says_it_opened(tmp_path: Path) -> None:
    # [LAW:no-ambient-temporal-coupling] the permission is owned once Claude Code is read through now, not when a period
    # next reads the record that opened the turn it asks in.
    transcript = tmp_path / "projects" / "-brain-cwd" / "b1.jsonl"
    transcript.parent.mkdir(parents=True)
    recorded: list[Entry] = []
    clock = VirtualClock()
    # Read on its period, as a running brain's model is, so the Stop with words waiting is decided and not let go.
    brain, _ = _built(clock, recorded.append, reading=_read_every_hundredth)
    asked: list[Asked] = []

    def allowed(held: Asked) -> None:
        asked.append(held)
        held.settle(Allow())

    path = str(transcript)
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt=NOTICE, transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Busy(), Stamp(2000)), 2.0))  # pyright: ignore[reportPrivateUsage]
    await _typed_turn(brain, "and add 3?", allowed)
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p1", prompt="and add 3?", transcript_path=path))  # pyright: ignore[reportPrivateUsage]
    await read_through(clock, brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Ten.", stop_hook_active=False, transcript_path=path)))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Unqueued(SessionId("b1"), Dequeued()))  # pyright: ignore[reportPrivateUsage]
    # Claude Code runs the words as p2, which no hook has named, and the model asks for a tool before any read since.
    transcript.write_text('{"type":"user","promptId":"p2","timestamp":"1970-01-01T00:00:03.100Z","message":{"role":"user","content":"and add 3?"}}\n')
    permission = {**json.loads(_hooked("PermissionRequest", "p2", transcript_path=path)), "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    reply = await brain.posted("PermissionRequest", json.dumps(permission).encode())
    assert (len(asked), reply) == (1, hook_output(Allow()))


async def test_a_brain_whose_reading_failed_still_lets_go_of_what_it_held_as_it_stops(tmp_path: Path) -> None:
    # What stopped the reading is `exited`'s to raise, to whoever waits on the brain: the stop ends it and lets go anyway.
    recorded: list[Entry] = []

    async def reading(_reader: Reader) -> None:
        raise OSError("the transcript went")

    listener = _Listener()
    (tmp_path / "sockets").mkdir()
    brain, _ = _built(VirtualClock(), recorded.append, listener=listener, sockets=tmp_path / "sockets", reading=reading)
    await asyncio.wait({brain._reading})  # pyright: ignore[reportPrivateUsage]
    await brain.stop()
    assert (listener.cleaned, (tmp_path / "sockets").exists()) == (True, False)
    with pytest.raises(OSError, match="the transcript went"):
        brain._reading.result()  # pyright: ignore[reportPrivateUsage]
    [shutdown] = events(recorded, "brain.shutdown")
    assert (shutdown.outcome, shutdown.facts) == ("ok", {"interrupted": False, "waited_on": 0, "waited_after_exit": 0})


@pytest.mark.outside
async def test_a_hook_whose_hearing_fails_unexpectedly_is_said_once_and_the_hooks_after_it_are_still_heard(errors: list[str]) -> None:
    recorded: list[Entry] = []

    def record(entry: Entry) -> None:
        # The first hook's event the log is handed cannot be written.
        recorded.append(entry)
        if [each for each in recorded if isinstance(each, WideEvent) and each.event.startswith(("brain.elicitation", "brain.permission"))] == [entry]:
            raise ValueError("the log is full")

    brain, _ = _built(VirtualClock(), record)
    # The elicitation that breaks is a part of the turn in flight: it was declined before it was heard, so the turn runs on.
    turn = await _typed_turn(brain, "probe")
    # Claude Code's turn p1 in flight, as the model holds it: the turn taken.
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path("/brain/projects/-brain-cwd/b1.jsonl"))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "probe"))  # pyright: ignore[reportPrivateUsage]
    listener, url = await _serving(brain)

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
        await listener.cleanup()
    assert taken(errors) == ["the brain's own work failed: ValueError('the log is full')"]
    assert (turn.broken, turn.answered.done(), brain._turn) == (None, False, turn)  # pyright: ignore[reportPrivateUsage]
    assert [event.event for event in recorded if isinstance(event, WideEvent) and event.event != "applied"] == ["brain.elicitation", "brain.permission"]


async def test_a_break_of_a_turn_already_over_ends_nothing_the_turn_in_flight_runs(errors: list[str]) -> None:
    brain, _ = _built(VirtualClock())
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path("/brain/projects/-brain-cwd/b1.jsonl"))))  # pyright: ignore[reportPrivateUsage]
    over = await _untaken(brain, "first")
    current = await _typed_turn(brain, "next")
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p2"), "next"))  # pyright: ignore[reportPrivateUsage]
    # What the turn before it left running breaks late, once the next turn's prompt has started new work.
    brain._broke(over, RuntimeError("late"))  # pyright: ignore[reportPrivateUsage]
    held = brain._modelled()  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    await settled()
    assert (taken(errors), held.ending, current.broken, current.answered.done(), _keys(brain).pressed) == (["the brain's own work failed: RuntimeError('late')"], None, None, False, [" first", " next"])


async def test_a_permission_any_session_posts_to_the_brain_is_the_brains_refused_with_no_turn_to_ask_in_and_is_one_event_naming_that_session() -> None:
    recorded: list[Entry] = []
    brain = _owning(VirtualClock(), recorded.append)
    permission = {"hook_event_name": "PermissionRequest", "session_id": "b2", "transcript_path": "/brain/projects/-brain-cwd/b2.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": "p9", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    assert await brain.posted("PermissionRequest", json.dumps(permission).encode()) == hook_output(Deny(NOBODY))
    assert [(event.facts["session"], event.facts["prompt"], event.facts["tool"], event.facts["decision"]) for event in events(recorded, "brain.permission")] == [("b2", "p9", "Write", Deny(NOBODY))]


TRANSCRIPT = "/brain/projects/-brain-cwd/b1.jsonl"


def _permission_hook(prompt: str, path: str = "/tmp/x") -> bytes:
    """A PermissionRequest of the brain's session under prompt `prompt`, as Claude Code fills it."""
    return _hooked("PermissionRequest", prompt, tool_name="Write", tool_input={"file_path": path})


async def _in_turn(record: Record, asks: Callable[[Asked], None], config_dir: Path = Path("/brain")) -> tuple[Brain, _Turn]:  # pyright: ignore[reportPrivateUsage]
    """A brain (`_built`) with a turn in flight that Claude Code took as p1, which the model holds running."""
    brain, _ = _built(VirtualClock(), record, config_dir=config_dir)
    turn = await _typed_turn(brain, "write", asks)
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "write"))  # pyright: ignore[reportPrivateUsage]
    assert turn.taken.done()
    return brain, turn


async def test_a_permission_request_is_held_in_the_model_before_anything_is_awaited_while_a_stop_posted_before_it_is_still_open() -> None:
    """[LAW:no-ambient-temporal-coupling] each hook is handled as it arrives: a dialog posted behind a Stop the model holds
    for its record is held in the model while the Stop's post is still open, before its own read of Claude Code."""
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _ = await _in_turn(recorded.append, asked.append)
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
    brain, turn = await _in_turn(recorded.append, unasked)
    brain._sessions.hear(Ending(SessionId("b1"), DenyAndInterrupt(STOPPED)))  # pyright: ignore[reportPrivateUsage]
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(DenyAndInterrupt(STOPPED))
    brain._over(turn, process.StopUnrecorded("the brain's Claude Code did not stop what hands pressed Escape for"))  # pyright: ignore[reportPrivateUsage]
    assert await brain.posted("PermissionRequest", _permission_hook("p1", "/tmp/y")) == hook_output(DenyAndInterrupt(STOPPED))
    assert dialogs(brain) == ()
    assert permissions(recorded) == [("p1", "Write", DenyAndInterrupt(STOPPED)), ("p1", "Write", DenyAndInterrupt(STOPPED))]
    owned, unowned = events(recorded, "brain.permission")
    assert (owned.parent_id, unowned.parent_id) == (turn.began.span.span_id, brain._launched.span_id)  # pyright: ignore[reportPrivateUsage]


async def test_a_dialog_claude_code_went_on_past_unanswered_is_refused_and_its_asker_and_event_say_so() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _ = await _in_turn(recorded.append, asked.append)
    asking = asyncio.ensure_future(brain.posted("PermissionRequest", _permission_hook("p1")))
    await until(lambda: len(asked) == 1)
    # Claude Code at its prompt, the dialog never answered: the model lets it go, and the brain refuses it.
    await brain._apply(StatusReported(SessionId("b1"), Report(Idle(), Stamp(9000)), 9.0))  # pyright: ignore[reportPrivateUsage]
    assert await asking == hook_output(Deny(LET_GO))
    assert await asked[0].decided() == Deny(LET_GO)
    assert permissions(recorded) == [("p1", "Write", Deny(LET_GO))]


async def test_a_permission_whose_read_breaks_is_no_turns_and_is_refused_said_once_and_left_held_by_nothing(errors: list[str]) -> None:
    recorded: list[Entry] = []
    brain, turn = await _in_turn(recorded.append, unasked)

    async def broken() -> None:
        raise OSError("the transcript went")

    brain._reader.read = broken  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    assert taken(errors) == ["the brain's own work failed: OSError('the transcript went')"]
    assert (dialogs(brain), turn.broken) == ((), None)
    [permission] = events(recorded, "brain.permission")
    assert (permission.outcome, permission.error, permission.facts["decision"]) == ("failed", "OSError: the transcript went", Deny(BROKEN))
    assert permission.stacktrace


async def test_a_permission_the_model_could_not_hold_is_refused_at_its_hook_and_said_once(errors: list[str]) -> None:
    recorded: list[Entry] = []
    brain, _ = await _in_turn(recorded.append, unasked)

    def unheld(_hook: object) -> "asyncio.Future[HookReply]":
        raise RuntimeError("the model is gone")

    brain._reader.asked = unheld  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    assert taken(errors) == ["the brain's own work failed: RuntimeError('the model is gone')"]
    assert permissions(recorded) == [("p1", "Write", Deny(BROKEN))]


async def test_a_dialog_of_a_turn_hands_broke_is_refused_and_never_put_to_the_user() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, turn = await _in_turn(recorded.append, asked.append)
    # Broken once Claude Code took it, and its prompt cleared any ending the model held: its stop is still to come.
    turn.broken = RuntimeError("no voice")
    assert await brain.posted("PermissionRequest", _permission_hook("p1")) == hook_output(Deny(BROKEN))
    assert (asked, permissions(recorded)) == ([], [("p1", "Write", Deny(BROKEN))])


async def _interrupted(brain: Brain, prompt: str) -> None:
    """Claude Code's answer to an Escape on turn `prompt`: the record of the interrupt in its transcript, then its idle."""
    await brain._apply(Interrupted(SessionId("b1"), PromptId(prompt), 9.0))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(StatusReported(SessionId("b1"), Report(Idle(), Stamp(9000)), 9.0))  # pyright: ignore[reportPrivateUsage]


async def _held_at(count: int, recorded: list[Entry], asks: Callable[[Asked], None]) -> tuple[Brain, "_Keys", list["asyncio.Task[Mapping[str, object]]"]]:
    """A brain in a turn Claude Code took as p1, at a keyboard of the test's, holding `count` dialogs Claude Code stacked:
    each posted as its hook, the later beneath the earlier, and each put to `asks`."""
    brain, _ = await _in_turn(recorded.append, asks)
    keys = _keys(brain)
    posts = [asyncio.ensure_future(brain.posted("PermissionRequest", _permission_hook("p1", f"/tmp/{each}"))) for each in range(count)]
    await until(lambda: len(dialogs(brain)) == count)
    return brain, keys, posts


async def test_two_dialogs_claude_code_stacks_are_held_in_the_order_asked_and_each_refusal_reaches_the_one_it_answers() -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, _, posts = await _held_at(2, recorded, asked.append)
    # The model holds them as Claude Code stacked them, and hands puts each to the user in that order.
    assert [dialog.on for dialog in dialogs(brain)] == [Permission("Write", {"file_path": "/tmp/0"}), Permission("Write", {"file_path": "/tmp/1"})]
    assert [each.permission for each in asked] == [Permission("Write", {"file_path": "/tmp/0"}), Permission("Write", {"file_path": "/tmp/1"})]
    # The second is refused first: its refusal goes to it, and the first stays held.
    asked[1].settle(Deny("no"))
    assert await posts[1] == hook_output(Deny("no"))
    assert (posts[0].done(), [dialog.on for dialog in dialogs(brain)]) == (False, [Permission("Write", {"file_path": "/tmp/0"})])
    asked[0].settle(Deny("not that either"))
    assert await posts[0] == hook_output(Deny("not that either"))
    assert dialogs(brain) == ()


@pytest.mark.parametrize("count", [1, 2], ids=["one dialog held", "two dialogs stacked"])
async def test_a_turn_stopped_while_its_dialogs_are_held_refuses_each_with_interrupt_ends_stopped_and_holds_none(count: int) -> None:
    recorded: list[Entry] = []
    asked: list[Asked] = []
    brain, keys, posts = await _held_at(count, recorded, asked.append)
    turn = brain._turn  # pyright: ignore[reportPrivateUsage]
    assert turn is not None
    assert brain.interrupt(brain.answering()) is True
    # Each dialog is refused in the step the Escape goes, whether or not Claude Code has read the key, or it is pressed yet.
    await settled()
    assert [post.result() for post in posts] == [hook_output(DenyAndInterrupt(STOPPED))] * count
    await until(lambda: "escape" in keys.pressed)
    # Claude Code stops what the Escape stopped, writes the record of it, and is at its prompt.
    await _interrupted(brain, "p1")
    await brain.drained()
    assert turn.answered.done()
    assert (turn.answered.result(), dialogs(brain)) == (BrainStopped(), ())
    assert [await each.decided() for each in asked] == [DenyAndInterrupt(STOPPED)] * count
    # A yes after the stop answers nothing.
    asked[0].settle(Allow())
    assert await asked[0].decided() == DenyAndInterrupt(STOPPED)
    assert permissions(recorded) == [("p1", "Write", DenyAndInterrupt(STOPPED))] * count


async def test_a_turn_whose_asker_breaks_while_its_dialog_is_held_is_refused_with_interrupt_and_ends_broken(errors: list[str]) -> None:
    recorded: list[Entry] = []
    asks = 0

    def broken(_: Asked) -> None:
        nonlocal asks
        asks += 1
        raise RuntimeError("no voice")

    brain, _ = await _in_turn(recorded.append, broken)
    keys = _keys(brain)
    turn = brain._turn  # pyright: ignore[reportPrivateUsage]
    assert turn is not None
    reply = await brain.posted("PermissionRequest", _permission_hook("p1"))
    # The dialog is refused as the turn is broken, in the step it is learned, and the turn is stopped as an Escape stops it.
    assert (reply, asks, taken(errors)) == (hook_output(DenyAndInterrupt(BROKEN)), 1, ["the brain's own work failed: RuntimeError('no voice')"])
    await until(lambda: "escape" in keys.pressed)
    await _interrupted(brain, "p1")
    await brain.drained()
    assert turn.answered.done() and not turn.answered.cancelled()
    assert (type(turn.answered.exception()), dialogs(brain)) == (RuntimeError, ())
    assert permissions(recorded) == [("p1", "Write", DenyAndInterrupt(BROKEN))]


async def test_a_permission_whose_event_cannot_be_written_is_refused_at_its_hook_and_said_once(errors: list[str]) -> None:
    """Never left unanswered, which would leave it to Claude Code's own dialog on a screen nobody sees."""
    recorded: list[Entry] = []

    def full(entry: Entry) -> None:
        if isinstance(entry, WideEvent) and entry.event == "brain.permission":
            raise ValueError("the log is full")
        recorded.append(entry)

    brain, _ = await _in_turn(full, unasked)
    other = {"hook_event_name": "PermissionRequest", "session_id": "b2", "transcript_path": "/brain/projects/-brain-cwd/b2.jsonl", "cwd": "/brain/cwd", "permission_mode": "default", "prompt_id": "p9", "tool_name": "Write", "tool_input": {"file_path": "/tmp/x"}}
    assert await brain.posted("PermissionRequest", json.dumps(other).encode()) == hook_output(Deny(BROKEN))
    assert taken(errors) == ["the brain's own work failed: ValueError('the log is full')"]


# A turn Claude Code opens for a background task's result, as its prompt hook names it (backgrounded_then_notified.jsonl, p2).
NOTICE = "<task-notification>\n<task-id>b74jw7b5h</task-id>\n<status>completed</status>\n</task-notification>"


def _owning(clock: VirtualClock, record: Record = lambda _entry: None, launched: Span | None = None) -> Brain:
    """A brain built on `clock` (`_built`), heard only through what its model hands over: the words each prompt hook filed
    (`_took`), and each telling (`_told`)."""
    brain, _ = _built(clock, record, launched=launched)
    return brain


async def _typed_turn(brain: Brain, words: str, asks: Callable[[Asked], None] = unasked) -> _Turn:  # pyright: ignore[reportPrivateUsage]
    """A turn asked of `brain` built by `_built` and pasted at its keyboard, not yet taken by Claude Code. The test reads
    its end off the turn, so its asker's, which is the same end, is let go."""
    clock, keys = cast(VirtualClock, brain._clock), _keys(brain)  # pyright: ignore[reportPrivateUsage]
    asking = asyncio.ensure_future(brain.ask(words, asks))
    asking.add_done_callback(lambda done: done.cancelled() or done.exception())
    pressed = len(keys.pressed)
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: f" {words}" in keys.pressed[pressed:])
    turn = brain._turn  # pyright: ignore[reportPrivateUsage]
    assert turn is not None and turn.pasted
    return turn


async def _untaken(brain: Brain, words: str) -> _Turn:  # pyright: ignore[reportPrivateUsage]
    """A turn typed (`_typed_turn`) that Claude Code never took: its take run out on the brain's clock."""
    turn = await _typed_turn(brain, words)
    clock = cast(VirtualClock, brain._clock)  # pyright: ignore[reportPrivateUsage]
    clock.advance_to(turn.began.began + turn.take)
    await asyncio.wait({turn.answered})
    assert isinstance(turn.answered.exception(), Untaken)
    # Its send heard to its end, so the brain keeps no work of the turn's.
    await brain.drained()
    return turn


def _took(brain: Brain, prompt: str, words: str, queued: bool = False) -> None:
    brain._took(Submitted(SessionId("b1"), PromptId(prompt), words, queued))  # pyright: ignore[reportPrivateUsage]


def _told(brain: Brain, prompt: str | None, ids: Sequence[str], waiting: tuple[str, ...] = (), ending: TurnEnding = ToldByStop("Done.")) -> None:
    brain._told(Summarise(SessionId("b1"), None if prompt is None else PromptId(prompt), ending, waiting, ids=frozenset(map(PromptId, ids))))  # pyright: ignore[reportPrivateUsage]


async def test_a_turn_told_under_an_id_its_hook_did_not_name_ends_its_own_turn_by_the_ids_its_telling_carries() -> None:
    # A message queued into the turn flushed it onto p2: its telling goes by p2, and carries p1 too (`Summarise.ids`).
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    assert turn.running is not None and turn.running.ids == {"p1"}
    _told(brain, "p2", ("p1", "p2"))
    assert (turn.answered.result(), turn.running.matched, brain._claude_turns.open) == (BrainAnswered(PromptId("p2"), None), "id", ())  # pyright: ignore[reportPrivateUsage]


async def test_words_claude_code_takes_before_a_turn_pasted_its_own_are_never_that_turns() -> None:
    # An untaken turn's words, taken late as its clear goes, while the same words asked again wait for the input.
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    turn.pasted = False
    _took(brain, "p1", "hello")
    assert (turn.taken.done(), [running.opener for running in brain._claude_turns.open]) == (False, ["other"])  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("queued", [False, True])
async def test_words_of_an_untaken_turn_that_claude_code_takes_next_are_stopped(queued: bool) -> None:
    brain, keys = _built(VirtualClock())
    await _untaken(brain, "hello")
    _took(brain, "p1", "hello", queued=queued)
    (late,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await brain.drained()
    assert (type(late.ended.exception()), keys.pressed, brain._claude_turns.open) == (process.Unran, [" hello", "escape", "ctrl_c"], ())  # pyright: ignore[reportPrivateUsage]


async def test_words_of_an_untaken_turn_taken_after_a_task_s_result_are_still_stopped() -> None:
    # A task's result posts its hook as it is delivered, not read from the input: the untaken words are still next in it.
    brain, keys = _built(VirtualClock())
    await _untaken(brain, "hello")
    _took(brain, "p1", "<task-notification>done</task-notification>")
    _took(brain, "p1", "hello", queued=True)
    (notified,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await brain.drained()
    assert (notified.ended.done(), keys.pressed, brain._claude_turns.open) == (True, [" hello", "escape", "ctrl_c"], ())  # pyright: ignore[reportPrivateUsage]


async def test_words_of_an_untaken_turn_taken_after_a_stop_cleared_the_input_are_a_person_s_and_run_on() -> None:
    # A stop's Escape and Ctrl-C clear Claude Code's queue and input, so the untaken words cannot come late after one.
    brain, keys = _built(VirtualClock())
    await _untaken(brain, "yes")
    await brain._clear(None)  # pyright: ignore[reportPrivateUsage]
    _took(brain, "p1", "yes")
    assert (len(brain._claude_turns.open), keys.pressed, await brain.drained()) == (1, [" yes", "escape", "ctrl_c"], 0)  # pyright: ignore[reportPrivateUsage]


async def test_words_of_an_untaken_turn_a_turn_asked_since_was_filed_into_run_on_to_answer_it() -> None:
    # The next turn's words, read after the late ones, run in the same turn of Claude Code's: stopping it would end a turn
    # its asker asked.
    brain, keys = _built(VirtualClock())
    await _untaken(brain, "hello")
    asked = await _typed_turn(brain, "what's the weather")
    _took(brain, "p1", "hello")
    _took(brain, "p1", "what's the weather", queued=True)
    # A drain would wait on the asked turn's send, which ends only with it: what a stop would press is pressed by now.
    await settled()
    (late,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert (asked.running, asked.answered.done(), keys.pressed) == (late, False, [" hello", " what's the weather"])


async def test_words_of_an_untaken_turn_taken_after_other_words_are_a_person_s_and_run_on() -> None:
    # Claude Code reads its input in order: once it took other words, the untaken ones cannot come late, and the same words
    # typed at the pane later are a person's.
    brain, keys = _built(VirtualClock())
    await _untaken(brain, "yes")
    _took(brain, "p1", "something else")
    _told(brain, "p1", ["p1"])
    _took(brain, "p2", "yes")
    assert (len(brain._claude_turns.open), keys.pressed, await brain.drained()) == (1, [" yes"], 0)  # pyright: ignore[reportPrivateUsage]


async def test_a_telling_that_names_no_prompt_ends_the_turn_its_ids_are() -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, None, ("p1",))
    assert (turn.answered.result(), brain._claude_turns.open) == (BrainAnswered(None, None), ())  # pyright: ignore[reportPrivateUsage]


async def test_a_notification_opened_before_the_last_turns_stop_is_applied_is_a_turn_of_its_own_each_ending_on_its_own_telling() -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
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
    turn = await _typed_turn(brain, "and the tests?")
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


class _Keys:
    """The brain's keyboard, as fritter's typist takes keys: each typing and press, in order; and its screen, as tmux
    shows it, which shows the prompt unless the test says else: every capture it was read for is counted. Its screen is the
    fake Claude Code the terminal contract holds to the pinned one on the chord's two claims (tests/claudescreen.py; the rest
    it plays is the docs' and held by nothing yet), and what hands types reaches it as the pin takes it: a paste, then the
    chord."""

    def __init__(self) -> None:
        # The Claude Code it types into.
        self.pid = 4242
        self.pressed: list[str] = []
        self.claude_code = ClaudeScreen(CHORD)
        self.looked = 0
        # What each key does in the typist's thread once it is pressed and before it is answered: a gate holds it, a
        # raise fails it.
        self.before: dict[Keystroke, Callable[[], object]] = {}
        # Its process's exit, heard as the terminal's stop ends it.
        self.exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()

    def type(self, text: str, submit: str) -> None:
        # Every turn is sent by the brain's own chord, never by Return.
        assert submit == SUBMIT
        self.pressed.append(text)
        self.claude_code.paste(text)
        self.claude_code.key(CHORD)

    def press(self, key: Keystroke) -> None:
        self.pressed.append(key)
        if key in self.before:
            self.before[key]()
        self.claude_code.key(KEYS[key])

    async def stop(self) -> None:
        self.exit.set_result(0)

    async def screen(self) -> str:
        self.looked += 1
        return self.claude_code.shown()

    def shown(self) -> str:
        """The last lines the terminal showed, which the brain's error messages carry."""
        return self.claude_code.shown()


def _keys(brain: Brain) -> _Keys:
    """The keyboard of the test's that `_built` built `brain` at."""
    return cast(_Keys, brain._typist)  # pyright: ignore[reportPrivateUsage]


class _Listener:
    """The brain's hook listener, as the stop lets go of it."""

    cleaned = False

    async def cleanup(self) -> None:
        self.cleaned = True


async def passed(clock: VirtualClock, seconds: float) -> None:
    """Once a pause of `seconds` begun now is waited on, `clock` moved to its end: the brain's look loop waits each out
    in turn, so a test spends exactly the pauses it names."""
    end = clock.at + seconds
    await clock.sleeping(end)
    clock.advance_to(end)


async def test_a_turn_asked_while_a_notification_turn_runs_is_typed_at_once_into_its_queue_and_never_escapes_it() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append, reading=_read_every_hundredth)
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


async def test_a_brain_stop_let_go_undecided_fails_every_turn_open_and_the_turn_it_answers_saying_the_order_was_lost(errors: list[str]) -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append)
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
    assert taken(errors) == ["the brain lost the order of its turns: the Stop of q was let go before it was decided"]


def _heard_late(brain: Brain, clock: VirtualClock) -> None:
    """What the bare brain hears is heard a second past the bound hands has to answer a hook posted now in: its every
    hearing, however many a hook's answer waits on, at that one time."""
    heard, posted = brain._heard, clock.at  # pyright: ignore[reportPrivateUsage]

    async def late() -> None:
        clock.advance_to(max(clock.at, posted + TAKE_SECONDS + 1))
        await heard()

    brain._heard = late  # pyright: ignore[reportPrivateUsage]


async def test_a_brain_prompt_hook_answered_past_its_timeout_fails_every_turn_open(errors: list[str]) -> None:
    # Answered later than Claude Code waits on it, the turn's first request may have left before hands heard its id.
    clock = VirtualClock()
    brain = _owning(clock)
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _heard_late(brain, clock)
    await brain.posted("UserPromptSubmit", _hooked("UserPromptSubmit", "p2", prompt=NOTICE))
    with pytest.raises(process.OrderLost):
        turn.answered.result()
    assert (turn.order_lost, brain._claude_turns.open) == (f"the UserPromptSubmit hook of p2 was answered in {TAKE_SECONDS + 1:.1f}s, past the {TAKE_SECONDS:g}s hands has to answer it in", ())  # pyright: ignore[reportPrivateUsage]
    assert taken(errors) == [f"the brain lost the order of its turns: {turn.order_lost}"]


async def test_a_permission_hook_claude_code_gave_up_on_fails_no_turn() -> None:
    # Abandoned is a permission's, not a Stop's: the order of the brain's turns does not rest on it.
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await brain._apply(Abandoned(SessionId("b1"), RequestId("r1"), 1.0))  # pyright: ignore[reportPrivateUsage]
    assert (turn.answered.done(), len(brain._claude_turns.open)) == (False, 1)  # pyright: ignore[reportPrivateUsage]


async def test_a_task_result_queued_behind_a_turn_runs_as_its_own_turn_behind_it_never_as_the_typed_turns() -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
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
    turn = await _typed_turn(brain, "and the tests?")
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
    brain, keys = _built(VirtualClock(), recorded.append)
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    # No turn of hands' is in flight: the notification's is stopped all the same, so the user is never locked out by it.
    assert brain.interrupt(notified) is True
    await until(lambda: "escape" in keys.pressed)
    # A turn of no kind Claude Code opens itself, and none at all, is not the brain's to stop on its own.
    assert brain.interrupt(None) is False
    # Told, the turn the Escape stopped is over, and the stop's Ctrl-C goes.
    _told(brain, "n1", ["n1"])
    await brain.drained()
    assert keys.pressed == ["escape", "ctrl_c"]


# The two StopFailures Claude Code 2.1.308 ends a turn with when the brain's login expired (docs/claude-code-behavior.md).
UNREFRESHED = StopFailed("invalid_request", "Login expired · Run /login to sign in again, or re-authenticate your Anthropic profile")
REFRESH_REFUSED = StopFailed("authentication_failed", "Login expired · Please run /login")


@pytest.mark.parametrize("stop", [UNREFRESHED, REFRESH_REFUSED], ids=["no refresh", "refresh refused"])
async def test_a_turn_told_with_an_expired_login_ends_as_one_and_the_brain_reports_it(stop: StopFailed) -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, "p1", ("p1",), ending=ToldByStop("Done.", stop))
    assert (turn.answered.result(), brain.login) == (BrainAnswered(PromptId("p1"), LoginExpired(stop)), LoginExpired(stop))


async def test_a_turn_told_with_another_failure_ends_with_it_as_claude_code_said_it() -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, "p1", ("p1",), ending=ToldByStop("Done.", lost := StopFailed("server_error", "API Error: Connection lost mid-response.")))
    assert (turn.answered.result(), brain.login) == (BrainAnswered(PromptId("p1"), lost), None)


async def test_the_brain_reports_the_login_as_the_latest_turn_claude_code_ended_said_it() -> None:
    """A turn Claude Code ended, answered or failed another way, says the login is no longer reported expired; one no
    telling ended shows nothing of it, so the report before it stands."""
    brain = _owning(VirtualClock())
    reports: list[LoginExpired | None] = []
    _took(brain, "n1", NOTICE)
    _told(brain, "n1", ("n1",), ending=ToldByStop("Done.", UNREFRESHED))
    reports.append(brain.login)
    _took(brain, "n2", NOTICE)
    _told(brain, "n2", ("n2",))
    reports.append(brain.login)
    _took(brain, "n3", NOTICE)
    _told(brain, "n3", ("n3",), ending=ToldByStop("Done.", REFRESH_REFUSED))
    reports.append(brain.login)
    # Ended with no telling, as the brain's exit ends every turn open.
    _took(brain, "n4", NOTICE)
    brain._claude_turns.failed(BrainGone("the brain exited"))  # pyright: ignore[reportPrivateUsage]
    reports.append(brain.login)
    _took(brain, "n5", NOTICE)
    _told(brain, "n5", ("n5",), ending=ToldByStop("Done.", StopFailed("server_error", "API Error: 529 Overloaded")))
    reports.append(brain.login)
    assert reports == [LoginExpired(UNREFRESHED), None, LoginExpired(REFRESH_REFUSED), LoginExpired(REFRESH_REFUSED), None]


async def test_hands_check_says_the_login_the_running_brain_reported_expired_until_a_later_turn_says_otherwise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the heartbeat, the one way `hands check` learns of the running daemon: with every child refused, so no
    Claude Code is started to learn it."""
    home = Home(tmp_path / "home")
    laid_out(home)
    clock = VirtualClock()
    heart = heartbeat.Heart(home.status, os.getpid(), clock.wall(), heartbeat.HEARTBEAT, kick_watchdog=lambda: None, clock=clock)
    brain = _owning(clock)

    def checked() -> readiness.Finding:
        heart.beat("running", None, 0, listening=False, degraded=run._login(brain.login))  # pyright: ignore[reportPrivateUsage]
        # Judged as `look` judges the hands it finds running, on the brain's own clock.
        return readiness.configured(Unstarting(clock, lambda _: None), home, LAID_OUT, {}, heartbeat.judge(home.status, heartbeat.read(home.status), clock.wall(), alive=True))

    _took(brain, "n1", NOTICE)
    _told(brain, "n1", ("n1",), ending=ToldByStop("Done.", REFRESH_REFUSED))
    assert checked() == readiness.Missing("brain's login expired", "the brain's login has expired, so every turn fails", "`hands login`, then `hands restart`")
    # Answered since: its Claude Code is asked again, as with no hands running.
    asked: list[Runnable] = []

    def asking(_children: object, _llm: object, _home: object, _environment: object, brain: Runnable) -> object:
        asked.append(brain)
        raise Rejected("asked")

    monkeypatch.setattr(readiness, "resolve", asking)
    _took(brain, "n2", NOTICE)
    _told(brain, "n2", ("n2",))
    assert (checked(), len(asked)) == (readiness.Missing("brain cannot reach its model", "hands has no model to talk with", "asked"), 1)


async def test_a_turn_only_the_transcripts_record_told_leaves_the_report_as_it_was_and_each_change_is_a_line() -> None:
    """An interrupt's record, whoever pressed the key, says nothing of how Claude Code would have ended the turn."""
    recorded: list[Entry] = []
    brain = _owning(VirtualClock(), recorded.append)
    reports: list[LoginExpired | None] = []
    _took(brain, "n1", NOTICE)
    _told(brain, "n1", ("n1",), ending=ToldByStop("Done.", UNREFRESHED))
    _took(brain, "n2", NOTICE)
    _told(brain, "n2", ("n2",), ending=ToldByRecord())
    reports.append(brain.login)
    _took(brain, "n3", NOTICE)
    _told(brain, "n3", ("n3",))
    reports.append(brain.login)
    assert reports == [LoginExpired(UNREFRESHED), None]
    assert [entry for entry in recorded if isinstance(entry, LoginReported)] == [LoginReported(("n1",), str(LoginExpired(UNREFRESHED))), LoginReported(("n3",), None)]


async def test_a_task_result_ctrl_c_cleared_from_claude_codes_queue_ends_its_turn_unran_and_holds_no_turn_after_it() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append)
    _took(brain, "n1", NOTICE)
    # A second result waits as the first is told: it runs as the turn behind, which no hook opens.
    _told(brain, "n1", ["n1"], waiting=(NOTICE,))
    (behind,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert brain.interrupt(behind) is True
    await brain.drained()
    # Ctrl-C cleared it before it ran: no telling will end it, so the clear does, and nothing is left open to wait on.
    assert ("ctrl_c" in keys.pressed, type(behind.ended.exception()), brain._claude_turns.open) == (True, process.Unran, ())  # pyright: ignore[reportPrivateUsage]
    turn = asyncio.ensure_future(brain.ask("and now?", unasked))
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: " and now?" in keys.pressed)
    turn.cancel()


async def test_a_telling_of_no_turn_names_no_turn_behind() -> None:
    brain = _owning(VirtualClock())
    _took(brain, "n1", NOTICE)
    turn = await _typed_turn(brain, "hello")
    _took(brain, "n1", "hello", queued=True)
    _told(brain, "n1", ["n1"], waiting=("hello",))
    (behind,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    assert brain.answering() is behind
    # A telling that goes by no id ends nothing: the typed words still wait to be answered by the turn they run as.
    _told(brain, None, [])
    assert (behind.ended.done(), behind.ids, turn.answered.done()) == (False, frozenset(), False)


async def test_a_brain_prompt_hook_that_does_not_parse_fails_every_turn_open_as_the_order_lost(errors: list[str]) -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    await brain.posted("UserPromptSubmit", b"{not json")
    with pytest.raises(process.OrderLost):
        turn.answered.result()
    assert brain._claude_turns.open == ()  # pyright: ignore[reportPrivateUsage]
    assert taken(errors) == [f"the brain lost the order of its turns: {turn.answered.exception()}"]


async def test_a_brain_stop_answered_past_its_bound_fails_every_turn_open(errors: list[str]) -> None:
    # Decided in its hold, but answered later than hands has to: Claude Code may have gone on, so its order is not trusted.
    clock = VirtualClock()
    brain = _owning(clock)
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    _heard_late(brain, clock)
    await brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Hi.", stop_hook_active=False))
    # The turn it told is answered; the one still open is not trusted to be the one Claude Code runs next.
    assert turn.answered.result() == BrainAnswered(PromptId("p1"), None)
    with pytest.raises(process.OrderLost, match="the Stop hook of p1 was answered in"):
        notified.ended.result()
    assert taken(errors) == [f"the brain lost the order of its turns: {notified.ended.exception()}"]


async def test_a_notification_that_opens_while_a_turn_waits_for_the_input_is_never_escaped_for_it() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append)
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
    brain, keys = _built(VirtualClock(), recorded.append)
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    # Typed behind the notification, and not yet taken, as the user stops the notification's work shown: the stop's keys
    # wait for the input, which is the turn's until Claude Code takes it.
    turn = await _typed_turn(brain, "and now?")
    assert brain.interrupt(notified) is True
    _took(brain, "n1", "and now?", queued=True)
    # Its words waited behind the turn stopped, and Ctrl-C cleared them: the turn ends stopped, never left waiting.
    await brain.drained()
    assert (turn.answered.result(), "escape" in keys.pressed, brain._claude_turns.open) == (BrainStopped(), True, ())  # pyright: ignore[reportPrivateUsage]


async def test_a_telling_of_a_turn_already_over_ends_no_turn_behind() -> None:
    brain = _owning(VirtualClock())
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _told(brain, "p1", ["p1"], waiting=(NOTICE,))
    (behind,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    # Told again, as another Stop hook's block re-tells it: the turn is over, and the result waiting is not named by it.
    _told(brain, "p1", ["p1"])
    assert (turn.answered.result(), behind.ended.done(), behind.ids) == (BrainAnswered(PromptId("p1"), None), False, frozenset())


async def test_words_queued_while_which_turn_runs_is_unknown_take_no_turn_for_the_one_running(errors: list[str]) -> None:
    brain = _owning(VirtualClock())
    await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    brain._lost("a test")  # pyright: ignore[reportPrivateUsage]
    # Claude Code may still run p1: a task result queued now runs behind it, never as the turn its requests are of.
    _took(brain, "p1", NOTICE, queued=True)
    assert brain.current() is None
    _took(brain, "p2", "next")
    assert brain.current() is not None and brain.current().ids == {"p2"}  # pyright: ignore[reportOptionalMemberAccess]
    assert taken(errors) == ["the brain lost the order of its turns: a test"]


async def test_a_turn_is_typed_only_once_the_brains_screen_shows_its_prompt_in_place_of_a_dialog() -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append)
    keys.claude_code.at_dialog()
    asking = asyncio.ensure_future(brain.ask("and now?", unasked))
    # Looked at four times over, the screen still shows the dialog: a dialog would drop the paste, so nothing goes in.
    for _ in range(3):
        await passed(clock, LOOK_SECONDS)
    await clock.sleeping(clock.at + LOOK_SECONDS)
    assert (keys.looked, keys.pressed) == (4, [])
    # The dialog is answered at Claude Code's own keyboard, and the prompt takes its place.
    keys.claude_code.key("Escape")
    clock.advance(LOOK_SECONDS)
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: keys.pressed == [" and now?"])
    asking.cancel()


def _session_hook(event: str, session: str, **fields: object) -> bytes:
    """A start or an end of `session` in the brain's Claude Code, as Claude Code fills it."""
    return json.dumps({"hook_event_name": event, "session_id": session, "transcript_path": f"/brain/projects/-brain-cwd/{session}.jsonl", "cwd": "/brain/cwd", **fields}).encode()


async def _moving(config_dir: Path, recorded: list[Entry]) -> Brain:
    """A brain launched in b1 long since, holding its conversation under `config_dir`, with a turn in flight as p1."""
    brain, _ = await _in_turn(recorded.append, unasked, config_dir)
    return brain


@pytest.mark.parametrize(("reason", "entered"), [("clear", "b2"), ("resume", "b0")])
async def test_a_clear_or_a_resume_ends_the_turn_running_and_the_brain_answers_in_the_session_it_moved_to(tmp_path: Path, reason: str, entered: str) -> None:
    recorded: list[Entry] = []
    brain = await _moving(tmp_path, recorded)
    (running,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await brain.posted("SessionEnd", _session_hook("SessionEnd", "b1", reason=reason))
    # The leaving ends what ran there, visibly, before Claude Code starts the next session.
    with pytest.raises(process.SessionLeft, match=f"ended its session b1 \\({reason}\\)"):
        running.ended.result()
    await brain.posted("SessionStart", _session_hook("SessionStart", entered, source=reason))
    assert brain._session.id() == SessionId(entered)  # pyright: ignore[reportPrivateUsage]
    assert process.held(tmp_path).read_text() == f"{entered}\n"
    [moved] = events(recorded, "brain.session")
    assert (moved.outcome, moved.facts) == ("ok", {"left": "b1", "entered": entered, "source": reason})
    # What the session left still posts is the brain's all the same: answered as its hooks are, and nothing is lost by it.
    assert await brain.posted("Stop", _hooked("Stop", "p1", last_assistant_message="Late.", stop_hook_active=False)) == {}
    # The session left stays ended: the brain's process is in the one it moved to alone.
    assert brain._sessions.live_ids() == [SessionId(entered)]  # pyright: ignore[reportPrivateUsage]
    # The turn the leaving ended, then the leaving.
    assert [entry.event for entry in recorded if isinstance(entry, WideEvent) and entry.event != "applied"] == ["brain.turn", "brain.session"]


async def test_a_start_of_the_session_the_brain_is_in_leaves_the_turn_running_there(tmp_path: Path) -> None:
    # A compaction starts the session again: nothing is left, so nothing ends.
    recorded: list[Entry] = []
    brain = await _moving(tmp_path, recorded)
    (running,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await brain.posted("SessionStart", _session_hook("SessionStart", "b1", source="compact"))
    assert brain._claude_turns.open == (running,) and not running.ended.done()  # pyright: ignore[reportPrivateUsage]
    [started] = events(recorded, "brain.session")
    assert started.facts == {"left": "b1", "entered": "b1", "source": "compact"}


async def test_a_start_that_does_not_parse_is_one_event_and_leaves_the_brain_adrift(tmp_path: Path, errors: list[str]) -> None:
    recorded: list[Entry] = []
    brain = await _moving(tmp_path, recorded)
    assert await brain.posted("SessionStart", b"{not json") == {}
    [unread] = events(recorded, "brain.unread")
    assert unread.facts == {"hook": "SessionStart"} and unread.outcome == "failed"
    # Which conversation Claude Code is in is not guessed: the brain's run ends, saying the start was unread.
    with pytest.raises(process.Adrift, match="SessionStart hook does not parse"):
        brain._adrift.result()  # pyright: ignore[reportPrivateUsage]
    [adrift] = taken(errors)
    assert adrift.startswith("the brain is adrift: the brain's SessionStart hook does not parse")


async def test_a_start_from_a_source_hands_does_not_know_still_moves_the_brain(tmp_path: Path, errors: list[str]) -> None:
    recorded: list[Entry] = []
    brain = await _moving(tmp_path, recorded)
    await brain.posted("SessionStart", _session_hook("SessionStart", "b2", source="teleport"))
    assert brain._session.id() == SessionId("b2")  # pyright: ignore[reportPrivateUsage]
    [moved] = events(recorded, "brain.session")
    assert moved.facts["source"] == UnknownSource("teleport")
    assert taken(errors) == ["SessionStart source 'teleport' is not one hands knows: the session joins as one started anew"]


async def test_a_start_the_model_fails_to_apply_still_moves_the_brain(tmp_path: Path, errors: list[str]) -> None:
    # Claude Code moved whatever the model made of it.
    recorded: list[Entry] = []
    brain = await _moving(tmp_path, recorded)

    async def broken(hook: object) -> None:
        raise RuntimeError("the model broke")

    brain._reader.hooked = broken  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    await brain.posted("SessionStart", _session_hook("SessionStart", "b2", source="clear"))
    assert brain._session.id() == SessionId("b2")  # pyright: ignore[reportPrivateUsage]
    assert process.held(tmp_path).read_text() == "b2\n"
    assert taken(errors) == ["the brain's own work failed: RuntimeError('the model broke')"]


async def test_an_end_that_is_no_move_leaves_the_turn_running_to_fail_as_the_brain_exits(tmp_path: Path) -> None:
    # Its Claude Code ending, not leaving for another conversation: what it ran fails as the brain is gone.
    recorded: list[Entry] = []
    brain = await _moving(tmp_path, recorded)
    (running,) = brain._claude_turns.open  # pyright: ignore[reportPrivateUsage]
    await brain.posted("SessionEnd", _session_hook("SessionEnd", "b1", reason="other"))
    assert not running.ended.done()


async def test_a_conversation_that_cannot_be_held_is_said_and_leaves_the_brain_adrift_in_it(tmp_path: Path, errors: list[str]) -> None:
    recorded: list[Entry] = []
    unwritable = tmp_path / "a file"
    unwritable.write_text("")
    brain = await _moving(unwritable, recorded)
    await brain.posted("SessionEnd", _session_hook("SessionEnd", "b1", reason="clear"))
    await brain.posted("SessionStart", _session_hook("SessionStart", "b2", source="clear"))
    [moved] = events(recorded, "brain.session")
    assert moved.outcome == "failed" and "could not be held, so the next start of hands would resume the one held before it" in str(moved.error)
    assert brain._session.id() == SessionId("b2")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(process.Adrift, match="b2 could not be held"):
        brain._adrift.result()  # pyright: ignore[reportPrivateUsage]
    [adrift] = taken(errors)
    assert adrift.startswith("the brain is adrift: the brain's conversation b2 could not be held")


async def test_an_end_answered_past_its_bound_fails_every_turn_open_as_the_order_lost(errors: list[str]) -> None:
    # Claude Code may have entered the next session before the model heard this one end.
    clock = VirtualClock()
    brain = _owning(clock)
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _heard_late(brain, clock)
    await brain.posted("SessionEnd", _session_hook("SessionEnd", "b1", reason="clear"))
    # The leaving ended the turn; answered late, which turn Claude Code opens next is not guessed either.
    with pytest.raises(process.SessionLeft):
        turn.answered.result()
    said = taken(errors)
    assert [error.split(":")[0] for error in said] == ["the brain lost the order of its turns"]
    assert "the end of session b1 was answered in" in said[0]


async def _stopping_held(brain: Brain, keys: _Keys, gate: threading.Event) -> _Turn:  # pyright: ignore[reportPrivateUsage]
    """A turn Claude Code took as p1, stopped by hands: its Escape gone, and its Ctrl-C pressed and held at `gate`."""
    keys.before["ctrl_c"] = gate.wait
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    assert brain.interrupt(brain.answering()) is True
    await until(lambda: keys.pressed == [" hello", "escape", "ctrl_c"])
    return turn


async def test_a_stop_s_ctrl_c_held_in_the_keyboard_holds_the_drain() -> None:
    brain, keys = _built(VirtualClock())
    gate = threading.Event()
    try:
        turn = await _stopping_held(brain, keys, gate)
        draining = asyncio.ensure_future(brain.drained())
        await settled()
        assert not draining.done()
    finally:
        gate.set()
    # The stop's work, and the stopped turn's send, which ends with it.
    assert (await draining, turn.answered.result()) == (2, BrainStopped())


async def test_the_model_still_performing_holds_the_drain_and_work_it_keeps_meanwhile_is_waited_on() -> None:
    brain, keys = _built(VirtualClock())
    gate = threading.Event()
    keys.before["ctrl_c"] = gate.wait
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    await _untaken(brain, "hello")
    # The drain begins with no work of the brain's own, as the model decides the untaken words taken late and has not yet
    # performed it: performed, it keeps their stop (`_took`), held at its Ctrl-C.
    draining = asyncio.ensure_future(brain.drained())
    brain._sessions.hear(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "hello"))  # pyright: ignore[reportPrivateUsage]
    try:
        await until(lambda: keys.pressed == [" hello", "escape"])
        # The Escape stopped them, as the transcript's record says.
        await brain._apply(Interrupted(SessionId("b1"), PromptId("p1"), 1.0))  # pyright: ignore[reportPrivateUsage]
        await until(lambda: keys.pressed == [" hello", "escape", "ctrl_c"])
        assert not draining.done()
    finally:
        gate.set()
    assert (await draining, brain._claude_turns.open) == (1, ())  # pyright: ignore[reportPrivateUsage]


async def test_a_key_that_fails_is_said_by_the_time_the_drain_returns(errors: list[str]) -> None:
    brain, keys = _built(VirtualClock())
    jammed = RuntimeError("the key jammed")

    def jam() -> None:
        raise jammed

    keys.before["ctrl_c"] = jam
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    assert brain.interrupt(brain.answering()) is True
    await brain.drained()
    assert taken(errors) == [f"the brain's own work failed: {jammed!r}"]
    # The turn the break ends was stopped already: no key is pressed for it again.
    assert (keys.pressed, turn.answered.exception()) == ([" hello", "escape", "ctrl_c"], jammed)


@pytest.mark.parametrize("words", ["hello", NOTICE])
async def test_a_key_the_terminal_refuses_while_the_brain_runs_is_said_as_a_failure(errors: list[str], words: str) -> None:
    brain, keys = _built(VirtualClock())
    refusal = Untyped("the fritter refused")

    def refused() -> None:
        raise refusal

    keys.before["escape"] = refused
    # The stop of a turn in flight, which ends with what broke it; or of one Claude Code opened itself, which holds none.
    turn = await _typed_turn(brain, words) if words == "hello" else None
    _took(brain, "p1", words)
    assert brain.interrupt(brain.answering() or brain.current()) is True
    await brain.drained()
    assert taken(errors) == [f"the brain's own work failed: {refusal!r}"]
    assert turn is None or turn.answered.exception() is refusal


async def test_a_turn_in_flight_whose_escape_is_refused_as_claude_code_s_own_turn_is_halted_ends_with_the_refusal(errors: list[str]) -> None:
    brain, keys = _built(VirtualClock())
    refusal = Untyped("the fritter refused")

    def refused() -> None:
        raise refusal

    keys.before["escape"] = refused
    # The turn in flight taken, and a turn Claude Code opened itself behind it, whose work is stopped as it is shown.
    turn = await _typed_turn(brain, "hello")
    _took(brain, "p1", "hello")
    _took(brain, "n1", NOTICE)
    (notified,) = brain._claude_turns.opened_itself  # pyright: ignore[reportPrivateUsage]
    assert brain.interrupt(notified) is True
    await brain.drained()
    assert (taken(errors), turn.answered.exception(), brain.answering()) == ([f"the brain's own work failed: {refusal!r}"], refusal, None)


async def test_a_brain_stopped_while_a_stop_of_its_own_holds_ctrl_c_ends_its_terminal_only_once_the_key_has_gone(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    listener = _Listener()
    (tmp_path / "sockets").mkdir()
    brain, keys = _built(VirtualClock(), recorded.append, listener=listener, sockets=tmp_path / "sockets")
    gate = threading.Event()
    try:
        turn = await _stopping_held(brain, keys, gate)
        stopping = asyncio.ensure_future(brain.stop())
        await settled()
        # The terminal still takes keys: the stop waits on its own work first.
        assert (stopping.done(), keys.exit.done(), listener.cleaned) == (False, False, False)
        # No turn is asked of a brain being stopped, the turn in flight still running or not.
        with pytest.raises(BrainGone, match="being stopped"):
            await brain.ask("and now?", unasked)
    finally:
        gate.set()
    await stopping
    assert (keys.pressed, turn.answered.result(), keys.exit.done()) == ([" hello", "escape", "ctrl_c"], BrainStopped(), True)
    assert (listener.cleaned, (tmp_path / "sockets").exists(), brain._reading.cancelled()) == (True, False, True)  # pyright: ignore[reportPrivateUsage]
    [shutdown] = events(recorded, "brain.shutdown")
    assert (shutdown.outcome, shutdown.facts["interrupted"], shutdown.facts["waited_after_exit"]) == ("ok", True, 0)


async def test_a_brain_stop_cancelled_while_it_drains_still_ends_the_terminal_and_lets_go_of_everything(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    listener = _Listener()
    (tmp_path / "sockets").mkdir()
    brain, keys = _built(VirtualClock(), recorded.append, listener=listener, sockets=tmp_path / "sockets")
    gate = threading.Event()
    try:
        await _stopping_held(brain, keys, gate)
        stopping = asyncio.ensure_future(brain.stop())
        await settled()
        # The caller gives up on the stop while its drain waits on the held Ctrl-C.
        stopping.cancel()
        await settled()
        assert keys.exit.done() is False
    finally:
        gate.set()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    # The shutdown runs on, whole, however its caller left it: a stop made again awaits that one.
    await brain.stop()
    assert (keys.exit.done(), listener.cleaned, (tmp_path / "sockets").exists(), brain._reading.cancelled()) == (True, True, False, True)  # pyright: ignore[reportPrivateUsage]
    assert [shutdown.outcome for shutdown in events(recorded, "brain.shutdown")] == ["ok"]


async def test_a_turn_in_flight_as_the_brain_stops_ends_stopped_before_its_terminal_is_ended(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    _ = _Listener()
    (tmp_path / "sockets").mkdir()
    brain, keys = _built(clock, recorded.append, listener=_, sockets=tmp_path / "sockets")
    asking = asyncio.ensure_future(brain.ask("hello", unasked))
    await passed(clock, SETTLE_SECONDS)
    await until(lambda: keys.pressed == [" hello"])
    _took(brain, "p1", "hello")
    ended: list[bool] = []
    keys.before["ctrl_c"] = lambda: ended.append(keys.exit.done())
    await brain.stop()
    assert (await asking, ended) == (BrainStopped(), [False])
    # No turn is asked of a brain that has ended.
    with pytest.raises(BrainGone, match="had ended"):
        await brain.ask("and now?", unasked)
    [turn] = events(recorded, "brain.turn")
    assert (turn.outcome, turn.facts["stopped"], turn.facts["taken"]) == ("ok", True, "p1")



async def test_a_stopped_turn_ends_with_the_id_it_was_told_under_however_far_the_model_reads_ahead_of_its_telling() -> None:
    # [LAW:no-ambient-temporal-coupling] the model has decided the stopped turn told, and the telling is still to be
    # performed: the turn is over only once the telling is heard, under its id, and the stop's Ctrl-C goes only then.
    recorded: list[Entry] = []
    brain, keys = _built(VirtualClock(), recorded.append)
    performing = asyncio.Event()
    # Each stop key as it is pressed, with whether the telling was performed by then.
    pressed: list[tuple[Keystroke, bool]] = []
    keys.before["escape"] = lambda: pressed.append(("escape", performing.is_set()))
    keys.before["ctrl_c"] = lambda: pressed.append(("ctrl_c", performing.is_set()))
    await brain._apply(Located(Membership(SessionId("b1"), 4242, Path("/brain/cwd"), Path(TRANSCRIPT))))  # pyright: ignore[reportPrivateUsage]
    turn = await _typed_turn(brain, "wait")
    await brain._apply(Prompted(SessionId("b1"), 0.0, "default", PromptId("p1"), "wait"))  # pyright: ignore[reportPrivateUsage]
    perform = brain._sessions._perform  # pyright: ignore[reportPrivateUsage]

    async def told_late(effect: Effect) -> None:
        # A telling the model decided is performed only once the test lets it, and what is decided after it waits.
        if isinstance(effect, Summarise):
            await performing.wait()
        await perform(effect)

    brain._sessions._perform = told_late  # pyright: ignore[reportPrivateUsage]
    assert brain.interrupt(brain.answering()) is True
    await until(lambda: "escape" in keys.pressed)
    # Claude Code stops what the Escape stopped: the model decides the turn told, and its telling waits to be performed.
    interrupted = asyncio.ensure_future(_interrupted(brain, "p1"))
    await until(brain._all_told)  # pyright: ignore[reportPrivateUsage]
    await settled()
    assert not turn.answered.done()
    performing.set()
    await interrupted
    await brain.drained()
    # The stop's Ctrl-C, emptying the input, went once Claude Code had acted on the Escape, as the model heard it: after
    # the telling, never while it waited, and before the turn was answered.
    assert (turn.answered.result(), pressed) == (BrainStopped(), [("escape", False), ("ctrl_c", True)])
    [ended] = events(recorded, "brain.turn")
    assert (ended.outcome, ended.facts["stopped"], ended.facts["taken"], ended.facts["told"]) == ("ok", True, "p1", "p1")
    # [LAW:nothing-unseen] the stop's clear is its own event: for a turn, no wait to space its Ctrl-C from one before, and
    # how long Claude Code took to stop what it ran.
    [clear] = events(recorded, "brain.clear")
    assert (clear.outcome, clear.facts["for_turn"], clear.facts["spaced"]) == ("ok", True, 0.0)
    assert isinstance(clear.facts["stopped_in"], float)


async def test_no_stop_presses_ctrl_c_within_claude_codes_exit_window_of_the_last() -> None:
    # A Ctrl-C that finds the input empty arms Claude Code's exit, which a second within 800ms takes: a stop straight
    # after another waits out the window from the last Ctrl-C before it presses its own.
    clock = VirtualClock()
    brain, keys = _built(clock)
    cleared: list[float] = []
    keys.before["ctrl_c"] = lambda: cleared.append(clock.now())
    await brain._clear(None)  # pyright: ignore[reportPrivateUsage]
    again = asyncio.ensure_future(brain._clear(None))  # pyright: ignore[reportPrivateUsage]
    await clock.sleeping(EXIT_SECONDS)
    assert (keys.pressed, cleared) == (["escape", "ctrl_c", "escape"], [0.0])
    clock.advance_to(EXIT_SECONDS)
    await again
    assert cleared == [0.0, EXIT_SECONDS]


async def test_a_turn_whose_typing_fails_unexpectedly_fails_its_asker_says_so_once_and_the_next_turn_is_its_own(errors: list[str]) -> None:
    recorded: list[Entry] = []
    clock = VirtualClock()
    brain, keys = _built(clock, recorded.append)
    typing = keys.type

    def jammed(_text: str, _submit: str) -> None:
        # A typist that fails as a terminal gone from under it does: not the Untyped a stale socket raises.
        raise OSError("jammed")

    keys.type = jammed  # pyright: ignore[reportAttributeAccessIssue]
    asking = asyncio.ensure_future(brain.ask("what is running?", unasked))
    await passed(clock, SETTLE_SECONDS)
    with pytest.raises(OSError, match="jammed"):
        await asking
    await brain.drained()
    keys.type = typing  # pyright: ignore[reportAttributeAccessIssue]
    turn = await _typed_turn(brain, "and now?")
    _took(brain, "p1", "and now?")
    _told(brain, "p1", ["p1"])
    assert turn.answered.result() == BrainAnswered("p1", None)
    await brain.drained()
    assert taken(errors) == ["the brain's own work failed: OSError('jammed')"]
    failed, _ = events(recorded, "brain.turn")
    assert (failed.outcome, failed.error) == ("failed", "OSError: jammed")


async def test_a_stop_whose_keys_fail_unexpectedly_ends_its_turn_says_so_once_and_is_never_pressed_again(errors: list[str]) -> None:
    def jammed() -> None:
        raise OSError("jammed")

    def asks(_held: Asked) -> None:
        # The turn breaks once it is taken, and the keys that would stop it are jammed.
        keys.before["escape"] = keys.before["ctrl_c"] = jammed
        raise RuntimeError("no voice")

    brain, turn = await _in_turn(lambda _entry: None, asks)
    keys = _keys(brain)
    await brain.posted("PermissionRequest", _permission_hook("p1"))
    await brain.drained()
    with pytest.raises(RuntimeError, match="no voice"):
        turn.answered.result()
    # A second stop would press Escape again, and its Ctrl-C would land inside Claude Code's exit window.
    assert (keys.pressed, taken(errors)) == ([" write", "escape"], ["the brain's own work failed: RuntimeError('no voice')", "the brain's own work failed: OSError('jammed')"])
