"""Side questions: a /btw forked from the brain's newest main turn, sent through the proxy, answered by its own reply."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

import pytest
from aiohttp import web

from hands.brain.asides import AsideFailed, AsideKind, Asides, Deadline, Unanswered
from hands.brain.fork import REMINDER, RUNNING, Newest, Turn, fork
from hands.core.session import SessionId
from hands.core.wire import CLASS_HEADER, Exchanged, Fork, Forwarded, MainTurn, Message, Observed, Reached, Route, Send, Sent, Text, Thinking, ToolUse, Written, classify
from hands.sessions.audit import Entry
from hands.sessions.proxy import Wire, serve_proxy
from hands.sessions.wide import WideEvent, root

BRAIN = SessionId("b1")
MARK = {"type": "ephemeral", "ttl": "1h"}
HEADERS = (("authorization", "Bearer the-brains-own"), ("x-claude-code-session-id", "b1"), (CLASS_HEADER, "main"), ("content-type", "application/json"))


def body(*messages: dict[str, object]) -> dict[str, object]:
    return {"model": "claude-opus-5-5", "system": [{"type": "text", "text": "You are hands."}], "tools": [{"name": "speak"}], "stream": True, "messages": list(messages)}


def user(*blocks: dict[str, object]) -> dict[str, object]:
    return {"role": "user", "content": list(blocks)}


def text(said: str, **more: object) -> dict[str, object]:
    return {"type": "text", "text": said, **more}


def reply(*content: Text | Thinking | ToolUse, stop: str = "end_turn") -> Message:
    return Message("msg_1", "claude-opus-5-5", content, stop, {})


def forked(turn: Turn, question: str) -> tuple[dict[str, str], dict[str, object]]:
    headers, sent = fork(turn, question)
    return dict(headers), cast(dict[str, object], json.loads(sent))


ASKED = [text(REMINDER), text("what now?")]


def test_a_question_before_any_reply_joins_the_last_user_message_and_the_newest_marker_moves_one_message_earlier() -> None:
    history = body(user(text("hi", cache_control=MARK)), {"role": "assistant", "content": [text("hello")]}, user(text("and?", cache_control=MARK)))
    headers, sent = forked(Turn("x1", "/v1/messages?beta=true", HEADERS, history, None), "what now?")
    assert sent["messages"] == [
        user(text("hi", cache_control=MARK)),
        {"role": "assistant", "content": [text("hello", cache_control=MARK)]},
        user(text("and?"), *ASKED),
    ]
    # Everything else is the brain's request as it went upstream; the class is Claude Code's for a side question.
    assert {key: value for key, value in sent.items() if key != "messages"} == {key: value for key, value in history.items() if key != "messages"}
    assert headers == {**dict(HEADERS), CLASS_HEADER: "auxiliary"}
    assert classify("/v1/messages", sent, headers) == Fork()


def test_a_question_after_a_finished_reply_follows_it_with_no_placeholder() -> None:
    history = body(user(text("hi", cache_control=MARK)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, history, reply(Thinking("hm", "sig"), Text("Done."))), "what now?")
    assert sent["messages"] == [
        user(text("hi")),
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "sig"}, text("Done.", cache_control=MARK)]},
        user(*ASKED),
    ]


def test_each_call_of_a_reply_still_running_stands_in_with_claude_codes_placeholder() -> None:
    history = body(user(text("hi", cache_control=MARK)))
    calls = reply(Text("Looking."), ToolUse("t1", "list", {}), ToolUse("t2", "read", {"id": 2}), stop="tool_use")
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, history, calls), "what now?")
    [*_, (asking)] = cast(list[dict[str, object]], sent["messages"])
    assert asking == user(
        {"type": "tool_result", "tool_use_id": "t1", "content": RUNNING}, {"type": "tool_result", "tool_use_id": "t2", "content": RUNNING}, *ASKED
    )
    assert classify("/v1/messages", sent, {CLASS_HEADER: "auxiliary"}) == Fork()


def test_markers_the_loop_put_on_earlier_messages_stay_and_a_lone_message_has_none_moved() -> None:
    history = body(user(text("one", cache_control=MARK)), {"role": "assistant", "content": [text("two")]}, user(text("three", cache_control=MARK)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, history, reply(Text("four."))), "q")
    messages = cast(list[dict[str, list[dict[str, object]]]], sent["messages"])
    assert [[block.get("cache_control") for block in message["content"]] for message in messages] == [[MARK], [None], [None], [MARK], [None, None]]
    _, alone = forked(Turn("x1", "/v1/messages", HEADERS, body(user(text("hi", cache_control=MARK))), None), "q")
    assert alone["messages"] == [user(text("hi"), text(REMINDER), text("q"))]


async def test_the_newest_is_the_brains_last_main_turn_as_forwarded_with_its_reply_once_read() -> None:
    newest = Newest(BRAIN)
    waiting = asyncio.create_task(newest.turn())
    message = reply(Text("Done."))

    def exchanged(exchange: str, session: SessionId) -> Exchanged:
        return Exchanged(exchange, session, MainTurn(None), "main", "POST", "/v1/messages", 0, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 0, Written(message, True)), False, root())

    # Another session's main turn, and a fork of the brain's, are never what a question forks from.
    for observed in (
        Sent("o1", SessionId("other"), MainTurn(None), "main", {}),
        Forwarded("o1", "/v1/messages", HEADERS, b"{}"),
        Sent("f1", BRAIN, Fork(), "auxiliary", {}),
        Forwarded("f1", "/v1/messages", HEADERS, b"{}"),
    ):
        newest.hear(observed)
    await asyncio.sleep(0)
    assert not waiting.done()
    newest.hear(Sent("x1", BRAIN, MainTurn(None), "main", {"messages": []}))
    newest.hear(Forwarded("x1", "/v1/messages?beta=true", HEADERS, b'{"messages": [1]}'))
    assert await waiting == Turn("x1", "/v1/messages?beta=true", HEADERS, {"messages": [1]}, None)
    newest.hear(exchanged("o1", SessionId("other")))
    newest.hear(exchanged("x1", BRAIN))
    assert (await newest.turn()).reply == message


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


class Brain:
    """hands' proxy before a stand-in for the API, with Asides listening on its wire, and the brain's main turn sent through it."""

    def __init__(self, proxy_url: str, asides: Asides, asked: list[tuple[dict[str, str], dict[str, object]]], answers: list[web.StreamResponse | list[bytes]]) -> None:
        self.url = proxy_url
        self.asides = asides
        self.asked = asked
        self.answers = answers

    async def main_turn(self) -> None:
        async with self.asides._client.post(self.url + "/v1/messages", headers=list(HEADERS), data=json.dumps(body(user(text("hi", cache_control=MARK))))) as response:  # pyright: ignore[reportPrivateUsage]
            assert response.status == 200
            await response.read()


@pytest.fixture
async def brain() -> AsyncIterator[Callable[[list[Entry]], Awaitable[Brain]]]:
    cleanups: list[Callable[[], Awaitable[None]]] = []

    async def start(recorded: list[Entry]) -> Brain:
        asked: list[tuple[dict[str, str], dict[str, object]]] = []
        answers: list[web.StreamResponse | list[bytes]] = []

        async def api(request: web.Request) -> web.StreamResponse:
            asked.append((dict(request.headers), await request.json()))
            answer = answers.pop(0) if answers else stream("Hello.")
            if isinstance(answer, web.StreamResponse):
                return answer
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "request-id": "req_1"})
            await response.prepare(request)
            for chunk in answer:
                await response.write(chunk)
            return response

        upstream = web.AppRunner(web.Application())
        upstream.app.router.add_route("*", "/{path:.*}", api)
        await upstream.setup()
        await web.TCPSite(upstream, "127.0.0.1", 0).start()
        wire = Wire(lambda _observed: None)
        proxy = await serve_proxy(f"http://127.0.0.1:{upstream.addresses[0][1]}", wire.observe, wire.route, clock=lambda: 0.0)
        asides = Asides(proxy.url, Newest(BRAIN), recorded.append)

        class Listening:
            def route(self, sent: Sent) -> Route:
                return Send(span=root())

            def hear(self, observed: Observed) -> None:
                asides.hear(observed)

        joined = wire.joined(Listening())
        joined.__enter__()

        async def leave() -> None:
            joined.__exit__(None, None, None)

        cleanups.extend([asides.close, proxy.close, upstream.cleanup, leave])
        return Brain(proxy.url, asides, asked, answers)

    yield start
    for cleanup in cleanups:
        await cleanup()


def events(recorded: list[Entry]) -> list[WideEvent]:
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "brain.aside"]


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


async def test_a_side_question_with_no_answer_fails_saying_why(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    # Before the brain has sent a main turn there is nothing to fork: the question waits for one, inside its asker's time.
    with pytest.raises(AsideFailed, match=r"the brain sent no main turn to fork within Deadline\(seconds=0.3\)"):
        await rig.asides.ask(AsideKind.NAME, "early", Deadline(0.3))
    await rig.main_turn()
    rig.answers.append(stream("", stop="tool_use"))
    with pytest.raises(AsideFailed, match="ended 'tool_use'"):
        await rig.asides.ask(AsideKind.LINE, "silent", Deadline(10))
    # A refusal fails the question at once, saying what the API answered: nothing asks again.
    rig.answers.append(web.Response(status=400, text='{"type":"error","error":{"type":"invalid_request_error","message":"no"}}', headers={"request-id": "req_2", "Content-Type": "application/json"}))
    with pytest.raises(AsideFailed, match=r"answered 400 \(request-id req_2\)"):
        await rig.asides.ask(AsideKind.LINE, "refused", Deadline(10))
    assert [(event.facts["question"], event.facts.get("unanswered")) for event in events(recorded)] == [
        ("early", Unanswered.UNFORKED), ("silent", Unanswered.WORDLESS), ("refused", Unanswered.REFUSED),
    ]
