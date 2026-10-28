"""Side questions: a /btw forked from the brain's newest main turn, sent through the proxy, answered by its own reply."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

import pytest
from aiohttp import web

from hands.brain.asides import AsideFailed, AsideKind, Asides, Deadline, TimeLimit, Unanswered
from hands.brain.fork import REMINDER, RUNNING, Newest, Turn, fork
from hands.core.session import SessionId
from hands.core.wire import CLASS_HEADER, Exchanged, Fork, Forwarded, MainTurn, Message, Observed, Reached, Route, Send, Sent, Text, Thinking, ToolUse, Written, classify
from hands.sessions.audit import Entry
from hands.sessions.proxy import Wire, serve_proxy
from hands.core.trace import Span
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
    return dict(headers), sent


def as_sent(history: dict[str, object]) -> bytes:
    return json.dumps(history).encode()


ASKED = [text(REMINDER), text("what now?")]


def test_a_question_before_any_reply_joins_the_last_user_message_and_the_newest_marker_moves_one_message_earlier() -> None:
    history = body(user(text("hi", cache_control=MARK)), {"role": "assistant", "content": [text("hello")]}, user(text("and?", cache_control=MARK)))
    headers, sent = forked(Turn("x1", "/v1/messages?beta=true", HEADERS, as_sent(history), None), "what now?")
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
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Thinking("hm", "sig"), Text("Done."))), "what now?")
    assert sent["messages"] == [
        user(text("hi")),
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "sig"}, text("Done.", cache_control=MARK)]},
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


def test_markers_the_loop_put_on_earlier_messages_stay_and_a_lone_message_has_none_moved() -> None:
    history = body(user(text("one", cache_control=MARK)), {"role": "assistant", "content": [text("two")]}, user(text("three", cache_control=MARK)))
    _, sent = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Text("four."))), "q")
    messages = cast(list[dict[str, list[dict[str, object]]]], sent["messages"])
    assert [[block.get("cache_control") for block in message["content"]] for message in messages] == [[MARK], [None], [None], [MARK], [None, None]]
    _, alone = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(body(user(text("hi", cache_control=MARK)))), None), "q")
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
    assert await waiting == Turn("x1", "/v1/messages?beta=true", HEADERS, b'{"messages": [1]}', None)
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

    def __init__(self, proxy_url: str, asides: Asides, asked: list[tuple[dict[str, str], dict[str, object]]], answers: list[Answer], routed: list[Span]) -> None:
        self.url = proxy_url
        self.asides = asides
        self.asked = asked
        self.answers = answers
        self.routed = routed

    async def main_turn(self) -> None:
        async with self.asides._client.post(self.url + "/v1/messages", headers=list(HEADERS), data=json.dumps(body(user(text("hi", cache_control=MARK))))) as response:  # pyright: ignore[reportPrivateUsage]
            assert response.status == 200
            await response.read()


@pytest.fixture
async def brain() -> AsyncIterator[Callable[[list[Entry]], Awaitable[Brain]]]:
    cleanups: list[Callable[[], Awaitable[None]]] = []

    async def start(recorded: list[Entry]) -> Brain:
        asked: list[tuple[dict[str, str], dict[str, object]]] = []
        answers: list[Answer] = []
        routed: list[Span] = []

        async def api(request: web.Request) -> web.StreamResponse:
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
        wire = Wire(lambda _observed: None)
        proxy = await serve_proxy(f"http://127.0.0.1:{upstream.addresses[0][1]}", wire.observe, wire.route, clock=lambda: 0.0)
        asides = Asides(proxy.url, Newest(BRAIN), recorded.append)

        class Listening:
            def route(self, sent: Sent) -> Route:
                # As Kept routes: the question's own request by the question, the rest as the stage routes every request
                # of the brain's that is not a main turn.
                owned = asides.own(sent)
                route = Send(refusal="final", span=root()) if owned is None else owned
                routed.append(route.span)
                return route

            def hear(self, observed: Observed) -> None:
                asides.hear(observed)

        joined = wire.joined(Listening())
        joined.__enter__()

        async def leave() -> None:
            joined.__exit__(None, None, None)

        cleanups.extend([asides.close, proxy.close, upstream.cleanup, leave])
        return Brain(proxy.url, asides, asked, answers, routed)

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
    assert [(event.facts["question"], event.facts.get("unanswered")) for event in events(recorded)] == [
        ("early", Unanswered.UNFORKED), ("silent", Unanswered.WORDLESS),
    ]


# A refusal fails the question at once, saying what the API answered: nothing asks again. Those on record: a login at
# its limit told to wait days, a login expired, and an organization with subscription access disabled; and an overload.
# The limit and the overload are the two the proxy would hold final, and pass on as its own 502, for Claude Code.
@pytest.mark.parametrize(("status", "error", "headers"), [
    (400, "invalid_request_error", {}),
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


def test_an_empty_reply_is_left_out_and_no_marker_lands_on_a_thinking_block() -> None:
    history = body(user(text("hi", cache_control=MARK)))
    _, empty = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply()), "q")
    assert empty["messages"] == [user(text("hi"), text(REMINDER), text("q"))]
    _, thought = forked(Turn("x1", "/v1/messages", HEADERS, as_sent(history), reply(Text("So."), Thinking("hm", "sig"))), "q")
    assert cast(list[object], thought["messages"])[1] == {"role": "assistant", "content": [text("So.", cache_control=MARK), {"type": "thinking", "thinking": "hm", "signature": "sig"}]}


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
    with pytest.raises(AsideFailed, match="its reply broke off") as cut:
        await rig.asides.ask(AsideKind.LINE, "cut short", Deadline(10))
    assert cut.value.why == Unanswered.BROKEN
    assert await rig.asides.ask(AsideKind.LINE, "again", Deadline(10)) == "Hello."


async def test_questions_wait_for_a_main_turn_side_by_side_and_a_time_limit_runs_from_its_turn(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    loop = asyncio.get_running_loop()
    # Both wait for the brain's first main turn at once: neither holds the other back while there is nothing to fork.
    started = loop.time()
    early = [asyncio.create_task(rig.asides.ask(AsideKind.NAME, question, Deadline(0.3))) for question in ("one", "two")]
    for task in early:
        with pytest.raises(AsideFailed, match="no main turn"):
            await task
    assert loop.time() - started < 0.55
    await rig.main_turn()
    gate = asyncio.Event()
    rig.answers.extend([held(gate, "Slow."), stream("Patient.")])
    slow = asyncio.create_task(rig.asides.ask(AsideKind.SUMMARY, "slow", Deadline(10)))
    await asyncio.sleep(0.05)
    # Queued behind the slow one for longer than its own seconds: they start at its turn, so it is still answered.
    patient = asyncio.create_task(rig.asides.ask(AsideKind.LINE, "patient", TimeLimit(0.3)))
    left = asyncio.create_task(rig.asides.ask(AsideKind.LINE, "left", Deadline(10)))
    await asyncio.sleep(0.4)
    left.cancel()
    gate.set()
    assert await slow == "Slow."
    assert await patient == "Patient."
    with pytest.raises(asyncio.CancelledError):
        await left
    by = {event.facts["question"]: event for event in events(recorded)}
    assert cast(float, by["patient"].facts["queued_ms"]) >= 350
    assert by["left"].outcome == "cancelled" and "queued_ms" not in by["left"].facts


async def test_a_questions_own_request_is_part_of_its_unit_of_work(brain: Callable[[list[Entry]], Awaitable[Brain]]) -> None:
    recorded: list[Entry] = []
    rig = await brain(recorded)
    await rig.main_turn()
    await rig.asides.ask(AsideKind.SUMMARY, "what?", Deadline(10))
    [event] = events(recorded)
    # The brain's main turn is routed as the stage routes it; the question's request inside the question.
    assert [span.parent_id for span in rig.routed] == [None, event.span_id]
