"""Side questions: a /btw forked from the brain's newest main turn, sent through the proxy, answered by its own reply."""

import asyncio
import json
from datetime import UTC, datetime
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest
from conftest import patiently
from aiohttp import web
from loguru import logger

from hands.brain.asides import AsideFailed, AsideKind, Asides, Deadline, TimeLimit, Unanswered
from hands.brain.fork import REMINDER, RUNNING, Credential, Newest, Turn, Unmarked, fork
from hands.brain.session import BrainSession
from hands.virtualclock import VirtualClock
from hands.core.session import SessionId
from hands.core.wire import (
    API_HOST,
    CLASS_HEADER,
    CONTINUATION_CLOSING,
    CONTINUATION_OPENING,
    Block,
    Ending,
    Exchanged,
    Fork,
    MainTurn,
    Message,
    Observed,
    Reached,
    RedactedThinking,
    Route,
    Send,
    Sent,
    Synthetic,
    Received,
    Text,
    Thinking,
    ToolUse,
    UnreadableRequest,
    Written,
    block_param,
    classify,
)
from hands.sessions.audit import Entry
from hands.sessions.authority import mint
from hands.sessions.proxy import Wire, serve_proxy
from hands.core.trace import Span
from hands.sessions.wide import WideEvent, root

BRAIN = SessionId("b1")
# The loop's own marker, as the brain's main turns carry it; a side question's names no ttl, so it takes the default five
# minutes the brain's subagent ttl is pinned to.
MARK = {"type": "ephemeral", "ttl": "1h"}
FORKED = {"type": "ephemeral"}
HEADERS = (("authorization", "Bearer the-brains-own"), ("x-claude-code-session-id", "b1"), (CLASS_HEADER, "main"), ("content-type", "application/json"))


def body(*messages: dict[str, object]) -> dict[str, object]:
    return {"model": "claude-opus-5-5", "system": [{"type": "text", "text": "You are hands."}], "tools": [{"name": "speak"}], "stream": True, "messages": list(messages)}


def user(*blocks: dict[str, object]) -> dict[str, object]:
    return {"role": "user", "content": list(blocks)}


def text(said: str, **more: object) -> dict[str, object]:
    return {"type": "text", "text": said, **more}


def reply(*content: Block, stop: str = "end_turn") -> Message:
    return Message("msg_1", "claude-opus-5-5", content, stop, {})


def logged_in(authorization: str, named: str = "authorization") -> tuple[tuple[str, str], ...]:
    """HEADERS with the brain's login as `authorization`, its header named as `named`."""
    return tuple((named, authorization) if name == "authorization" else (name, value) for name, value in HEADERS)


def forked(turn: Turn, question: str, credential: Credential | None = None) -> tuple[dict[str, str], dict[str, object]]:
    question_asked = fork(turn, question, credential)
    return dict(question_asked.headers), question_asked.body


def marks(sent: dict[str, object]) -> list[tuple[int, object]]:
    """Each message of a body that carries a cache marker, by its index, with the marker on its last block."""
    messages = cast(list[dict[str, object]], sent["messages"])
    blocks = [cast(list[dict[str, object]], content) if isinstance(content := message["content"], list) else [] for message in messages]
    return [(index, said[-1].get("cache_control")) for index, said in enumerate(blocks) if any("cache_control" in block for block in said)]


def as_sent(history: dict[str, object]) -> bytes:
    return json.dumps(history).encode()


ASKED = [text(REMINDER), text("what now?")]


def test_a_question_before_any_reply_joins_the_last_user_message_and_its_one_marker_goes_on_the_message_before() -> None:
    history = body(user(text("hi", cache_control=MARK)), {"role": "assistant", "content": [text("hello")]}, user(text("and?", cache_control=MARK)))
    headers, sent = forked(Turn("x1", "/v1/messages?beta=true", HEADERS, as_sent(history), None), "what now?")
    assert sent["messages"] == [
        user(text("hi")),
        {"role": "assistant", "content": [text("hello", cache_control=FORKED)]},
        user(text("and?"), *ASKED),
    ]
    # Everything else is the brain's request as it went upstream; the class is Claude Code's for a side question.
    assert {key: value for key, value in sent.items() if key != "messages"} == {key: value for key, value in history.items() if key != "messages"}
    assert headers == {**dict(HEADERS), CLASS_HEADER: "auxiliary"}
    assert classify("/v1/messages", sent, headers) == Fork()


def test_a_question_after_a_finished_reply_follows_it_with_no_placeholder() -> None:
    history = body(user(text("hi", cache_control=MARK)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Thinking("hm", "sig"), Text("Done."))), "what now?")
    assert sent["messages"] == [
        user(text("hi")),
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "sig"}, text("Done.", cache_control=FORKED)]},
        user(*ASKED),
    ]


def test_each_call_of_a_reply_still_running_stands_in_with_claude_codes_placeholder() -> None:
    history = body(user(text("hi", cache_control=MARK)))
    calls = reply(Text("Looking."), ToolUse("t1", "list", {}), ToolUse("t2", "read", {"id": 2}), stop="tool_use")
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), calls), "what now?")
    [*_, (asking)] = cast(list[dict[str, object]], sent["messages"])
    assert asking == user(
        {"type": "tool_result", "tool_use_id": "t1", "content": RUNNING}, {"type": "tool_result", "tool_use_id": "t2", "content": RUNNING}, *ASKED
    )
    assert classify("/v1/messages", sent, {CLASS_HEADER: "auxiliary"}) == Fork()


def test_none_of_the_loops_markers_survive_and_a_lone_message_carries_none() -> None:
    history = body(user(text("one", cache_control=MARK)), {"role": "assistant", "content": [text("two")]}, user(text("three", cache_control=MARK)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Text("four."))), "q")
    assert marks(sent) == [(3, FORKED)]
    _, alone = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(body(user(text("hi", cache_control=MARK)))), None), "q")
    assert alone["messages"] == [user(text("hi"), text(REMINDER), text("q"))]


def test_a_question_merged_into_the_refusal_continuation_is_no_carrier_and_the_marker_goes_on_the_carrier_before_the_last() -> None:
    continuation = f"{CONTINUATION_OPENING}\n<partial-response>\nHalf\n</partial-response>\n{CONTINUATION_CLOSING}"
    history = body(user(text("hi")), {"role": "assistant", "content": [text("hello", cache_control=MARK)]}, user(text(continuation)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), None), "q")
    # Ephemeral, the continuation keeps the flag the question merges into: the carriers are the prompt and the reply.
    assert marks(sent) == [(0, FORKED)]


def test_a_question_forked_from_a_turn_that_carried_no_marker_carries_none() -> None:
    history = body(user(text("hi")), {"role": "assistant", "content": [text("hello")]}, user(text("and?")))
    question = fork(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Text("So."))), "q", None)
    assert marks(question.body) == [] and question.marked == Unmarked.TURN_UNMARKED


# What the brain's reply ends in, and the message the question then marks: the reply when it can carry a marker, else the
# user message before it.
REPLY_KINDS = {
    "text": (reply(Text("So.")), 1),
    # Stellar drift is off in the brain: text of whitespace alone carries one.
    "whitespace text": (reply(Text("So."), Text(" \n")), 1),
    "a call": (reply(Text("Looking."), ToolUse("t1", "list", {}), stop="tool_use"), 1),
    "empty text": (reply(Thinking("hm", "sig"), Text("")), 0),
    "thinking": (reply(Text("So."), Thinking("hm", "sig")), 0),
    "redacted thinking": (reply(Text("So."), RedactedThinking("data")), 0),
}


@pytest.mark.parametrize(("replied", "at"), REPLY_KINDS.values(), ids=REPLY_KINDS.keys())
def test_a_reply_carries_the_marker_by_its_last_block(replied: Message, at: int) -> None:
    question = fork(Turn("x1", "/v1/messages", HEADERS, as_sent(body(user(text("hi", cache_control=MARK)))), replied), "q", None)
    assert marks(question.body) == [(at, FORKED)] and question.marked == at


def test_a_reply_the_loop_sent_as_a_string_carries_the_marker_as_one_text_block() -> None:
    history = body(user(text("hi")), {"role": "assistant", "content": "hello"}, user(text("and?", cache_control=MARK)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), None), "q")
    assert cast(list[object], sent["messages"])[1] == {"role": "assistant", "content": [text("hello", cache_control=FORKED)]}


def test_no_marker_of_a_side_question_names_a_ttl_and_each_keeps_the_rest_of_its_terms() -> None:
    # As Claude Code marks them: a system prompt block of global scope, and the request under the beta that evicts it.
    history = {
        **body(user(text("hi", cache_control=MARK))),
        "cache_control": {**MARK, "evict_on_complete": True},
        "system": [text("billing"), text("You are hands.", cache_control={**MARK, "scope": "global"})],
    }
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Text("So."))), "q")
    assert sent["cache_control"] == {"type": "ephemeral", "evict_on_complete": True}
    assert sent["system"] == [text("billing"), text("You are hands.", cache_control={"type": "ephemeral", "scope": "global"})]
    assert marks(sent) == [(1, FORKED)]


def test_a_turn_that_sent_no_system_prompt_forks_a_question_without_one() -> None:
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, json.dumps({"model": "claude-opus-5-5", "messages": [user(text("hi", cache_control=MARK))]}).encode(), reply(Text("So."))), "q")
    assert "system" not in sent and marks(sent) == [(1, FORKED)]


NO_MESSAGE = "is not a user, assistant or system message of text or blocks"
UNREADABLE = {
    "bytes that are not UTF-8": (b"\xff\xfe{", "its body is not JSON"),
    "text that is not JSON": (b"{", "its body is not JSON"),
    "JSON that is no object": (b"[]", "its body is not a JSON object"),
    "messages that are no list": (as_sent({**body(), "messages": {}}), "its messages are not a list but dict"),
    "content neither text nor blocks": (as_sent(body(user(text("hi", cache_control=MARK)), {"role": "assistant", "content": None})), f"its messages.1 {NO_MESSAGE}"),
    "no blocks": (as_sent(body(user(text("hi", cache_control=MARK)), {"role": "assistant", "content": []})), f"its messages.1 {NO_MESSAGE}"),
    "a block that is no object": (as_sent(body({"role": "user", "content": ["hi"]})), f"its messages.0 {NO_MESSAGE}"),
    "a role the API has no message of": (as_sent(body({"role": "tool", "content": "hi"})), f"its messages.0 {NO_MESSAGE}"),
    "a system prompt neither text nor blocks": (as_sent({**body(user(text("hi"))), "system": 3}), "its system prompt is neither text nor blocks but int"),
    "a system marker that is no object": (as_sent({**body(user(text("hi"))), "system": [text("You are hands.", cache_control="1h")]}), "a cache marker on it, its system prompt or a message is not an object"),
}


@pytest.mark.parametrize(("sent", "why"), UNREADABLE.values(), ids=UNREADABLE.keys())
async def test_a_turn_hands_cannot_read_fails_the_question_unforkable(sent: bytes, why: str) -> None:
    unreadable = Turn("x1", "/v1/messages", HEADERS, sent, None)

    class Unforkable(Newest):
        async def turn(self) -> Turn:
            return unreadable

    with pytest.raises(UnreadableRequest, match=why):
        fork(unreadable, "q", None)
    # Failed as a question fails, never past the asker that waits on its answer.
    asides = Asides(VirtualClock(), "http://127.0.0.1:9", Unforkable(BrainSession(BRAIN)), lambda _entry: None)
    try:
        with pytest.raises(AsideFailed, match=why) as failed:
            await asides.ask(AsideKind.LINE, "q", Deadline(10))
        assert failed.value.why == Unanswered.UNFORKABLE
    finally:
        await asides.close()


def main_exchanged(exchange: str, reply: Ending, session: SessionId = BRAIN) -> Exchanged:
    return Exchanged(exchange, session, MainTurn(None), "main", "POST", "/v1/messages", 0, (), 0.0, 0.0, reply, False, root(VirtualClock()))


def answered(message: Message) -> Reached:
    return Reached(200, 0.0, 0.0, 0, Written(message, True))


async def test_the_newest_is_the_brains_last_main_turn_as_forwarded_with_its_reply_once_read() -> None:
    newest = Newest(BrainSession(BRAIN))
    waiting = asyncio.create_task(newest.turn())
    message = reply(Text("Done."))

    # Another session's main turn, and a fork of the brain's, are never what a question forks from.
    for observed in (
        Sent("o1", SessionId("other"), MainTurn(None), "main", {}),
        Received("o1", "/v1/messages", HEADERS, b"{}", built=False),
        Sent("f1", BRAIN, Fork(), "auxiliary", {}),
        Received("f1", "/v1/messages", HEADERS, b"{}", built=False),
    ):
        newest.hear(observed)
    await asyncio.sleep(0)
    assert not waiting.done()
    newest.hear(Sent("x1", BRAIN, MainTurn(None), "main", {"messages": []}))
    newest.hear(Received("x1", "/v1/messages?beta=true", HEADERS, b'{"messages": [1]}', built=False))
    assert await waiting == Turn("x1", "/v1/messages?beta=true", HEADERS, b'{"messages": [1]}', None)
    newest.hear(main_exchanged("o1", answered(message), SessionId("other")))
    newest.hear(main_exchanged("x1", answered(message)))
    assert (await newest.turn()).reply == message


async def test_a_question_is_sent_with_the_newest_login_the_brain_sent_on_any_request_and_none_of_its_own() -> None:
    newest = Newest(BrainSession(BRAIN))
    main_turn(newest, "x1", OPENED, answered(reply(Text("Done."))))
    assert newest.credential() == Credential("x1", "authorization", "Bearer the-brains-own")
    # The idle brain's own request renews it, whatever the request is, the casing of its header included.
    newest.hear(Received("i1", "/api/oauth/organizations/o/skills/list-skills", logged_in("Bearer renewed", "Authorization"), b"", built=False))
    credential = newest.credential()
    assert credential == Credential("i1", "authorization", "Bearer renewed")
    # The turn stays as the proxy received it: the login is the fork's own.
    turn = await newest.turn()
    assert turn.headers == HEADERS and turn.exchange == "x1"
    question = fork(turn, "what now?", credential)
    assert dict(question.headers)["authorization"] == "Bearer renewed" and question.login_from == "i1"
    # A side question the brain's own Claude Code sends is the brain's request, and renews it.
    newest.hear(Sent("b1", BRAIN, Fork(), "auxiliary", {"messages": []}))
    newest.hear(Received("b1", "/v1/messages", logged_in("Bearer renewed again"), b"{}", built=False))
    assert newest.credential() == Credential("b1", "authorization", "Bearer renewed again")
    # A request hands built carries the login it was given: it renews nothing, or one refusal would outlive its cause.
    newest.hear(Received("f1", "/v1/messages", logged_in("Bearer stale"), b"{}", built=True))
    # A request that carries none, or an empty one, leaves it.
    newest.hear(Received("i2", "/api/hello", (("x-forwarded-for", "h"),), b"", built=False))
    newest.hear(Received("i3", "/api/hello", logged_in(""), b"", built=False))
    assert newest.credential() == Credential("b1", "authorization", "Bearer renewed again")


async def test_a_question_is_sent_with_the_one_login_the_brain_sent_last_whichever_kind_it_is() -> None:
    newest = Newest(BrainSession(BRAIN))
    main_turn(newest, "x1", OPENED, answered(reply(Text("Done."))))
    # The brain's login changes kind, as /login in its pane changes it: the question goes out with the key alone.
    newest.hear(Received("i1", "/api/hello", (("x-api-key", "k"),), b"", built=False))
    credential = newest.credential()
    assert credential == Credential("i1", "x-api-key", "k")
    question = fork(await newest.turn(), "what now?", credential)
    assert {name: value for name, value in question.headers if name in ("authorization", "x-api-key")} == {"x-api-key": "k"}
    # ...and back, from a turn that carried a key.
    keyed = Turn("x2", "/v1/messages", (("x-api-key", "k"), *HEADERS[1:]), as_sent(OPENED), None)
    question = fork(keyed, "what now?", Credential("i2", "authorization", "Bearer renewed"))
    assert {name: value for name, value in question.headers if name in ("authorization", "x-api-key")} == {"authorization": "Bearer renewed"}


async def test_a_question_to_a_brain_that_sent_no_login_goes_out_as_the_turn_it_forks_did() -> None:
    newest = Newest(BrainSession(BRAIN))
    newest.hear(Sent("x1", BRAIN, MainTurn(None), "main", {"messages": []}))
    newest.hear(Received("x1", "/v1/messages", HEADERS[1:], as_sent(OPENED), built=False))
    assert newest.credential() is None
    question = fork(await newest.turn(), "what now?", newest.credential())
    assert "authorization" not in dict(question.headers) and question.login_from == "x1"


# The brain's held turn: its stay_silent call answered, and the proxy answering the request that carries the answer.
SILENT_CALL = ToolUse("t1", "stay_silent", {})
OPENED = body(user(text("hi")))
HELD = body(user(text("hi")), {"role": "assistant", "content": [block_param(SILENT_CALL)]}, user({"type": "tool_result", "tool_use_id": "t1", "content": "held"}))
STAYED_SILENT = Message("msg_hands_1", "claude-opus-5-5", (Text("(stayed silent)"),), "end_turn", {"input_tokens": 0, "output_tokens": 0})


def main_turn(newest: Newest, exchange: str, sent: dict[str, object], reply: Ending) -> None:
    """A main turn of the brain's as the proxy tells it, from its request to its end."""
    newest.hear(Sent(exchange, BRAIN, MainTurn(None), "main", sent))
    newest.hear(Received(exchange, "/v1/messages", HEADERS, as_sent(sent), built=False))
    newest.hear(main_exchanged(exchange, reply))


async def test_a_question_after_a_turn_the_proxy_answered_itself_forks_from_that_turn_with_its_answer() -> None:
    newest = Newest(BrainSession(BRAIN))
    main_turn(newest, "x1", OPENED, answered(reply(SILENT_CALL, stop="tool_use")))
    main_turn(newest, "x2", HELD, Synthetic(Written(STAYED_SILENT, True), 0.0))
    turn = await newest.turn()
    assert turn == Turn("x2", "/v1/messages", HEADERS, as_sent(HELD), STAYED_SILENT)
    # The stay_silent call goes with the result Claude Code holds for it, and no call is left running.
    _, asked = forked(turn, "what now?")
    assert asked["messages"] == [*cast(list[object], HELD["messages"]), {"role": "assistant", "content": [text("(stayed silent)")]}, user(*ASKED)]


async def test_a_question_after_a_held_turn_and_then_one_sent_upstream_forks_from_the_one_sent_upstream() -> None:
    newest = Newest(BrainSession(BRAIN))
    main_turn(newest, "x1", HELD, Synthetic(Written(STAYED_SILENT, True), 0.0))
    asking = body(*cast(list[dict[str, object]], HELD["messages"]), {"role": "assistant", "content": [text("(stayed silent)")]}, user(text("read it")))
    reading = ToolUse("t2", "Read", {"file_path": "/a"})
    main_turn(newest, "x2", asking, answered(reply(reading, stop="tool_use")))
    turn = await newest.turn()
    assert turn.exchange == "x2"
    # Only the call Claude Code holds no result for stands in as running.
    _, asked = forked(turn, "what now?")
    running: dict[str, object] = {"type": "tool_result", "tool_use_id": "t2", "content": RUNNING}
    assert cast(list[object], asked["messages"])[-1] == user(running, *ASKED)


async def test_a_question_asked_as_the_brain_moves_forks_from_the_conversation_it_moved_to() -> None:
    held = BrainSession(BRAIN)
    newest = Newest(held)
    newest.hear(Sent("x1", BRAIN, MainTurn(None), "main", {}))
    newest.hear(Received("x1", "/v1/messages", HEADERS, b'{"messages": [1]}', built=False))
    first = Turn("x1", "/v1/messages", HEADERS, b'{"messages": [1]}', None)
    cleared = SessionId("cleared")
    held.enter(cleared)
    # A /clear's conversation holds nothing of the one it left: a question waits for its first turn.
    waiting = asyncio.create_task(newest.turn())
    await asyncio.sleep(0)
    assert not waiting.done()
    newest.hear(Sent("x2", cleared, MainTurn(None), "main", {}))
    newest.hear(Received("x2", "/v1/messages", HEADERS, b'{"messages": [2]}', built=False))
    assert await waiting == Turn("x2", "/v1/messages", HEADERS, b'{"messages": [2]}', None)
    # Resumed back, it forks from the turn the conversation last sent before it was left.
    held.enter(BRAIN)
    assert await newest.turn() == first


async def test_a_question_waiting_in_a_new_conversation_wakes_as_the_brain_resumes_one_it_has_a_turn_of() -> None:
    held = BrainSession(BRAIN)
    newest = Newest(held)
    newest.hear(Sent("x1", BRAIN, MainTurn(None), "main", {}))
    newest.hear(Received("x1", "/v1/messages", HEADERS, b'{"messages": [1]}', built=False))
    held.enter(SessionId("cleared"))
    waiting = asyncio.create_task(newest.turn())
    await asyncio.sleep(0)
    assert not waiting.done()
    # Nothing is sent: the move alone wakes it.
    held.enter(BRAIN)
    assert await waiting == Turn("x1", "/v1/messages", HEADERS, b'{"messages": [1]}', None)


def sse(event: dict[str, object]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def stream(said: str, stop: str = "end_turn") -> list[bytes]:
    return [
        sse({"type": "message_start", "message": {"id": "msg_9", "model": "m", "usage": {"input_tokens": 3, "cache_read_input_tokens": 900}}}),
        sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": said}}),
        sse({"type": "content_block_stop", "index": 0}),
        sse({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 4}}),
        sse({"type": "message_stop"}),
    ]


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]
Answer = web.StreamResponse | list[bytes] | Handler


def held(gate: asyncio.Event, said: str) -> Handler:
    async def answer(request: web.Request) -> web.StreamResponse:
        await gate.wait()
        return await written(request, stream(said))

    return answer


async def broken(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    await response.write(stream("Half")[0])
    cast(asyncio.Transport, request.transport).abort()
    return response


async def written(request: web.Request, chunks: list[bytes]) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "request-id": "req_1"})
    await response.prepare(request)
    for chunk in chunks:
        await response.write(chunk)
    return response


class Brain:
    """hands' proxy before a stand-in for the API, with Asides listening on its wire, and the brain's main turn sent through it."""

    def __init__(self, clock: VirtualClock, proxy_url: str, asides: Asides, asked: list[tuple[dict[str, str], dict[str, object]]], answers: list[Answer], routed: list[Span]) -> None:
        # The one clock the proxy, its wire, and Asides run on, moved only by the test.
        self.clock = clock
        self.url = proxy_url
        self.asides = asides
        self.asked = asked
        self.answers = answers
        self.routed = routed

    async def main_turn(self) -> None:
        async with self.asides._client.post(self.url + "/v1/messages", headers=list(HEADERS), data=json.dumps(body(user(text("hi", cache_control=MARK))))) as response:  # pyright: ignore[reportPrivateUsage]
            assert response.status == 200
            await response.read()

    async def idle_request(self, authorization: str) -> None:
        """What the idle brain's Claude Code sends on its own, renewing its token before it builds the request."""
        async with self.asides._client.get(self.url + "/api/oauth/organizations/o/skills/list-skills", headers=list(logged_in(authorization))) as response:  # pyright: ignore[reportPrivateUsage]
            assert response.status == 200
            await response.read()


@pytest.fixture
async def brain(tmp_path: Path) -> AsyncIterator[Callable[[list[Entry]], Awaitable[Brain]]]:
    cleanups: list[Callable[[], Awaitable[None]]] = []

    async def start(recorded: list[Entry]) -> Brain:
        asked: list[tuple[dict[str, str], dict[str, object]]] = []
        answers: list[Answer] = []
        routed: list[Span] = []

        async def api(request: web.Request) -> web.StreamResponse:
            # What the brain's Claude Code asks of the account while idle: no answer of the model's.
            if request.path.endswith("/skills/list-skills"):
                return web.json_response({})
            asked.append((dict(request.headers), await request.json()))
            answer = answers.pop(0) if answers else stream("Hello.")
            match answer:
                case web.StreamResponse():
                    return answer
                case list():
                    return await written(request, answer)
                case _:
                    return await answer(request)

        upstream = web.AppRunner(web.Application())
        upstream.app.router.add_route("*", "/{path:.*}", api)
        await upstream.setup()
        await web.TCPSite(upstream, "127.0.0.1", 0).start()
        clock = VirtualClock(epoch=datetime.fromtimestamp(0, UTC))
        wire = Wire(clock, lambda _observed: None)
        proxy = await serve_proxy(f"http://127.0.0.1:{upstream.addresses[0][1]}", wire.observe, wire.route, clock, {}, mint(API_HOST, clock), tmp_path / "trusted.pem")
        asides = Asides(clock, proxy.url, Newest(BrainSession(BRAIN)), recorded.append)

        class Listening:
            def route(self, sent: Sent) -> Route:
                # As Kept routes: the question's own request by the question, the rest as the stage routes every request
                # of the brain's that is not a main turn.
                owned = asides.own(sent)
                route = Send(refusal="final", span=root(clock)) if owned is None else owned
                routed.append(route.span)
                return route

            def hear(self, observed: Observed) -> None:
                asides.hear(observed)

        joined = wire.joined(Listening())
        joined.__enter__()

        async def leave() -> None:
            joined.__exit__(None, None, None)

        cleanups.extend([asides.close, proxy.close, upstream.cleanup, leave])
        return Brain(clock, proxy.url, asides, asked, answers, routed)

    yield start
    for cleanup in cleanups:
        await cleanup()


def events(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "brain.aside"]


@pytest.mark.outside
async def test_a_side_question_is_the_brains_turn_forked_through_the_proxy_and_answered_by_its_own_reply(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    rig.answers.append(stream("It said four."))
    assert await rig.asides.ask(AsideKind.SUMMARY, "what did it say?", Deadline(10)) == "It said four."
    # The API was asked the brain's turn, then the question: on the brain's credentials, as Claude Code's /btw.
    [(_, turn), (headers, question)] = rig.asked
    assert headers["Authorization"] == "Bearer the-brains-own" and headers["x-claude-code-request-class"] == "auxiliary"
    messages = cast(list[dict[str, list[dict[str, object]]]], question["messages"])
    assert messages[-1]["content"][-2:] == ASKED[:1] + [text("what did it say?")]
    assert {key: value for key, value in question.items() if key != "messages"} == {key: value for key, value in turn.items() if key != "messages"}
    [event] = events(recorded)
    assert event.outcome == "ok"
    assert {key: event.facts[key] for key in ("kind", "reply", "status", "request_id", "input_tokens", "cache_read_input_tokens", "output_tokens")} == {
        "kind": "summary", "reply": "It said four.", "status": 200, "request_id": "req_1", "input_tokens": 3, "cache_read_input_tokens": 900, "output_tokens": 4,
    }
    assert isinstance(event.facts["forked"], str)
    # Which message carries the question's marker, read off the body it was sent as.
    assert marks(question) == [(event.facts["marked"], FORKED)]


@pytest.mark.outside
async def test_a_side_question_is_answered_on_the_brains_renewed_login_after_the_token_of_the_turn_it_forks_has_expired(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    await rig.idle_request("Bearer renewed")

    async def accepts_the_renewed_login(request: web.Request) -> web.StreamResponse:
        if request.headers["Authorization"] != "Bearer renewed":
            return web.Response(status=401, text='{"type":"error","error":{"type":"authentication_error","message":"OAuth access token has expired"}}', headers={"Content-Type": "application/json"})
        return await written(request, stream("Still here."))

    rig.answers.append(accepts_the_renewed_login)
    assert await rig.asides.ask(AsideKind.NAME, "who?", Deadline(10)) == "Still here."
    [event] = events(recorded)
    assert event.facts["status"] == 200
    # The event says the login came from the idle request after the turn it forks, not the turn's own.
    renewed = rig.asides._newest.credential()  # pyright: ignore[reportPrivateUsage]
    assert renewed is not None and event.facts["credential_from"] == renewed.exchange != event.facts["forked"]


@pytest.mark.outside
async def test_a_side_question_with_no_answer_fails_saying_why(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    # Before the brain has sent a main turn there is nothing to fork: the question waits for one, inside its asker's time.
    early = asyncio.create_task(rig.asides.ask(AsideKind.NAME, "early", Deadline(0.3)))
    await rig.clock.scheduled(0.3)
    rig.clock.advance(0.3)
    with pytest.raises(AsideFailed, match=r"the brain sent no main turn to fork within Deadline\(seconds=0.3\)"):
        await early
    await rig.main_turn()
    rig.answers.append(stream("", stop="tool_use"))
    with pytest.raises(AsideFailed, match="ended 'tool_use'"):
        await rig.asides.ask(AsideKind.LINE, "silent", Deadline(10))
    assert [(event.facts["question"], event.facts.get("unanswered")) for event in events(recorded)] == [
        ("early", Unanswered.UNFORKED), ("silent", Unanswered.WORDLESS),
    ]


# A refusal fails the question at once, saying what the API answered: nothing asks again. Those on record: a login at
# its limit told to wait days, a login expired, and an organization with subscription access disabled; and an overload.
# The limit and the overload are the two the proxy would hold final, and pass on as its own 502, for Claude Code.
@pytest.mark.outside
@pytest.mark.parametrize(("status", "error", "headers"), [
    (403, "permission_error", {}),
    (529, "overloaded_error", {}),
    (429, "rate_limit_error", {"retry-after": "240711", "anthropic-ratelimit-unified-status": "rejected"}),
    (401, "authentication_error", {}),
])
async def test_a_refused_side_question_fails_at_once_with_what_the_api_said(brain: Callable[[list[Entry]], Awaitable[Brain]], status: int, error: str, headers: dict[str, str]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    said = f'{{"type":"error","error":{{"type":"{error}","message":"no"}}}}'
    rig.answers.append(web.Response(status=status, text=said, headers={"request-id": "req_9", "Content-Type": "application/json", **headers}))
    with pytest.raises(AsideFailed, match=rf"answered {status} \(request-id req_9\).*{error}"):
        await rig.asides.ask(AsideKind.NAME, "refused", Deadline(5))
    [event] = events(recorded)
    assert (event.facts["unanswered"], event.facts["status"]) == (Unanswered.REFUSED, status)
    assert len(rig.asked) == 2


@pytest.mark.outside
async def test_a_side_question_the_api_refuses_is_logged_by_the_proxy_as_hands_bug_naming_the_message(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    said = '{"type":"error","error":{"type":"invalid_request_error","message":"messages.0.content.1: unexpected tool_result block"}}'
    rig.answers.append(web.Response(status=400, text=said, headers={"request-id": "req_9", "Content-Type": "application/json"}))
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    try:
        with pytest.raises(AsideFailed, match=r"answered 400 \(request-id req_9\)"):
            await rig.asides.ask(AsideKind.NAME, "malformed", Deadline(5))
    finally:
        logger.remove(sink)
    [error] = errors
    assert error.startswith("hands' bug: hands built exchange") and "messages.0 is role 'user';" in error and "req_9" in error


def refused(messages: list[dict[str, object]]) -> list[str]:
    """What the Messages API refuses in the order of `messages`: none, for a request it takes."""
    roles = [message["role"] for message in messages]
    spoken = [role for role in roles if role != "system"]
    # A directive-only system message (no content, with output_config) is taken at any position.
    misplaced = [
        f"messages.{index}: system before {next((later for later in roles[index:] if later != 'system'), 'the end')}"
        for index, message in enumerate(messages)
        if message["role"] == "system" and message["content"] != [] and next((later for later in roles[index:] if later != "system"), "assistant") != "assistant"
    ]
    alternating = [] if spoken == ["user", "assistant"] * (len(spoken) // 2) + ["user"] * (len(spoken) % 2) else [f"roles do not alternate: {spoken}"]
    return misplaced + alternating


def said(message: dict[str, object]) -> list[object]:
    return [{key: value for key, value in cast(dict[str, object], block).items() if key != "cache_control"} for block in cast(list[object], message["content"])]


SYSTEM_MESSAGE: dict[str, object] = {"role": "system", "content": [text("Stay brief.")]}
DIRECTIVE: dict[str, object] = {"role": "system", "content": [], "output_config": {"effort": "low"}}
SHAPES: dict[str, list[dict[str, object]]] = {
    "a system message last": [user(text("hi", cache_control=MARK)), SYSTEM_MESSAGE],
    "a system message before an assistant message": [user(text("hi")), SYSTEM_MESSAGE, {"role": "assistant", "content": [text("hello")]}, user(text("and?", cache_control=MARK))],
    "a marked system message last": [user(text("hi")), {"role": "system", "content": [text("Stay brief.", cache_control=MARK)]}],
    "a directive-only system message before the question": [user(text("hi")), {"role": "assistant", "content": [text("hello")]}, DIRECTIVE, user(text("and?", cache_control=MARK))],
    "two marked system messages last": [user(text("hi")), {"role": "system", "content": [text("Stay brief.", cache_control=MARK)]}, {"role": "system", "content": [text("Be kind.", cache_control=MARK)]}],
    "a directive-only system message last":[user(text("hi", cache_control=MARK)), DIRECTIVE],
    "two system messages last": [user(text("hi")), {"role": "assistant", "content": [text("hello")]}, user(text("and?", cache_control=MARK)), SYSTEM_MESSAGE, DIRECTIVE],
}
REPLIES = {"no reply yet": None, "an empty reply": reply(), "a reply of thinking alone": reply(Thinking("hmm", "sig"), stop="max_tokens"), "a reply still calling": reply(Text("Looking."), ToolUse("t1", "list", {}), stop="tool_use")}
# The message each shape's question marks, after each reply: the last that can carry a marker before the question, never a
# system message nor a reply of thinking alone.
NONE = Unmarked.NO_FORK_POINT
MARKED: dict[str, dict[str, int | Unmarked]] = {
    "a system message last": {"no reply yet": NONE, "an empty reply": NONE, "a reply of thinking alone": 0, "a reply still calling": 2},
    "a system message before an assistant message": {"no reply yet": 2, "an empty reply": 2, "a reply of thinking alone": 3, "a reply still calling": 4},
    "a marked system message last": {"no reply yet": NONE, "an empty reply": NONE, "a reply of thinking alone": 0, "a reply still calling": 2},
    "a directive-only system message before the question": {"no reply yet": 1, "an empty reply": 1, "a reply of thinking alone": 3, "a reply still calling": 4},
    "two marked system messages last": {"no reply yet": NONE, "an empty reply": NONE, "a reply of thinking alone": 0, "a reply still calling": 3},
    "a directive-only system message last": {"no reply yet": NONE, "an empty reply": NONE, "a reply of thinking alone": 0, "a reply still calling": 2},
    "two system messages last": {"no reply yet": 1, "an empty reply": 1, "a reply of thinking alone": 2, "a reply still calling": 5},
}


@pytest.mark.parametrize("replied", REPLIES.keys())
@pytest.mark.parametrize("shape", SHAPES.keys())
def test_a_question_forked_from_a_turn_holding_system_messages_is_one_the_api_takes(shape: str, replied: str) -> None:
    question = fork(Turn("x1", "/v1/messages", HEADERS, as_sent(body(*SHAPES[shape])), REPLIES[replied]), "what now?", None)
    sent, at = question.body, MARKED[shape][replied]
    assert marks(sent) == ([] if isinstance(at, Unmarked) else [(at, FORKED)]) and question.marked == at
    messages = cast(list[dict[str, object]], sent["messages"])
    assert refused(messages) == []
    [*_, asking] = [message for message in messages if message["role"] == "user"]
    assert cast(list[object], asking["content"])[-2:] == ASKED
    # Every system message the turn held is still there, in its order: what it says, whatever marker moved off it.
    assert [said(message) for message in messages if message["role"] == "system"] == [said(message) for message in SHAPES[shape] if message["role"] == "system"]
    # The question is what the model answers: the last message that is not a system message is the user's.
    assert [message for message in messages if message["role"] != "system"][-1] is asking


def test_the_checker_refuses_the_order_that_was_refused_live() -> None:
    assert "messages.1: system before user" in refused([user(text("hi")), SYSTEM_MESSAGE, user(text("q"))])


def test_an_empty_reply_is_left_out() -> None:
    _, empty = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(body(user(text("hi", cache_control=MARK)))), reply()), "q")
    assert empty["messages"] == [user(text("hi"), text(REMINDER), text("q"))]


@pytest.mark.outside
async def test_a_reply_that_breaks_off_fails_the_question_and_nothing_else(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    rig.answers.append(broken)
    with pytest.raises(AsideFailed) as failed:
        await rig.asides.ask(AsideKind.LINE, "cut", Deadline(10))
    assert failed.value.why == Unanswered.BROKEN
    # Answered 200 and then cut off mid-frame, the connection ending cleanly: broken too, not refused.
    rig.answers.append([*stream("Half")[:2], b"event: message_del"])
    with pytest.raises(AsideFailed, match="its reply could not be read") as cut:
        await rig.asides.ask(AsideKind.LINE, "cut short", Deadline(10))
    assert cut.value.why == Unanswered.BROKEN
    assert await rig.asides.ask(AsideKind.LINE, "again", Deadline(10)) == "Hello."


@pytest.mark.outside
async def test_questions_wait_for_a_main_turn_side_by_side_and_a_time_limit_runs_from_its_turn(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    # Both wait for the brain's first main turn at once: neither holds the other back while there is nothing to fork, so
    # the one 0.3 s both were given ends them both.
    early = [asyncio.create_task(rig.asides.ask(AsideKind.NAME, question, Deadline(0.3))) for question in ("one", "two")]
    await rig.clock.scheduled(0.3, count=2)
    assert not any(task.done() for task in early)
    rig.clock.advance(0.3)
    for task in early:
        with pytest.raises(AsideFailed, match="no main turn"):
            await task
    await rig.main_turn()
    gate = asyncio.Event()
    rig.answers.extend([held(gate, "Slow."), stream("Patient.")])
    slow = asyncio.create_task(rig.asides.ask(AsideKind.SUMMARY, "slow", Deadline(10)))
    # Its request is at the API, so it holds the turn.
    await patiently(lambda: len(rig.asked) == 2, lambda: len(rig.asked))
    # Queued behind the slow one for longer than its own seconds: they start at its turn, so it is still answered.
    patient = asyncio.create_task(rig.asides.ask(AsideKind.LINE, "patient", TimeLimit(0.3)))
    left = asyncio.create_task(rig.asides.ask(AsideKind.LINE, "left", Deadline(10)))
    # The slow one's deadline and the one left's, both begun now: the patient one's runs from its turn, not yet come.
    await rig.clock.scheduled(rig.clock.now() + 10, count=2)
    rig.clock.advance(0.4)
    left.cancel()
    gate.set()
    assert await slow == "Slow."
    assert await patient == "Patient."
    with pytest.raises(asyncio.CancelledError):
        await left
    by = {event.facts["question"]: event for event in events(recorded)}
    assert by["patient"].facts["queued_ms"] == pytest.approx(400)
    assert by["left"].outcome == "cancelled" and "queued_ms" not in by["left"].facts


@pytest.mark.outside
async def test_a_questions_own_request_is_part_of_its_unit_of_work(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    await rig.asides.ask(AsideKind.SUMMARY, "what?", Deadline(10))
    [event] = events(recorded)
    # The brain's main turn is routed as the stage routes it; the question's request inside the question.
    assert [span.parent_id for span in rig.routed] == [None, event.span_id]


async def test_a_timeout_that_is_not_its_askers_fails_the_question_broken_not_timed_out() -> None:
    class TimesOut(Newest):
        async def turn(self) -> Turn:
            raise TimeoutError("a wait of its own")

    asides = Asides(VirtualClock(), "http://127.0.0.1:9", TimesOut(BrainSession(BRAIN)), lambda _entry: None)
    try:
        with pytest.raises(AsideFailed, match="not its asker's time") as failed:
            await asides.ask(AsideKind.LINE, "q", Deadline(10))
        assert failed.value.why == Unanswered.BROKEN
    finally:
        await asides.close()
