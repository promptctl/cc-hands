"""The brain's token usage, read off the wire's reply frames, and the tool that reports it."""

import asyncio
from typing import cast

from hands.brain.usage import Context, Spent, Tally, Usage
from hands.clock import SYSTEM
from hands.core.session import SessionId
from hands.core.wire import Compaction, Exchanged, Heard, Kind, MainTurn, MessageDelta, MessageStarted, Message, Reached, Sent, Written
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, root
from hands.voice.tools import Called, audited, usage_tool
from brains import Holder

BRAIN = SessionId("brain")


def sent(exchange: str, kind: Kind = MainTurn(None), session: SessionId = BRAIN) -> Sent:
    return Sent(exchange, session, kind, None, {})


def started(exchange: str, input: int, cached: int = 0, created: int = 0, model: str = "claude-opus-5-5") -> Heard:
    usage = {"input_tokens": input, "cache_read_input_tokens": cached, "cache_creation_input_tokens": created, "output_tokens": 1}
    return Heard(exchange, MessageStarted(f"msg_{exchange}", model, usage))


def delta(exchange: str, output: int) -> Heard:
    return Heard(exchange, MessageDelta("end_turn", {"output_tokens": output}))


def whole(exchange: str, kind: Kind = MainTurn(None)) -> Exchanged:
    message = Message("m", "claude-opus-5-5", (), "end_turn", {})
    return Exchanged(exchange, BRAIN, kind, None, "POST", "/v1/messages", 1, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 1, Written(message, True)), False, root(SYSTEM))


def heard(usage: Usage, *observed: Sent | Heard | Exchanged) -> Usage:
    for each in observed:
        usage.hear(each)
    return usage


def test_nothing_is_counted_before_a_main_turn_reply_begins() -> None:
    assert heard(Usage(Holder(BRAIN)), sent("a")).reading() == Tally(None, Spent())


def test_the_context_is_the_latest_main_turn_request_with_its_reply_however_it_was_cached() -> None:
    usage = heard(Usage(Holder(BRAIN)), sent("a"), started("a", 2, cached=14800, created=98), delta("a", 40))
    assert usage.reading() == Tally(Context("claude-opus-5-5", 2 + 14800 + 98 + 40), Spent(2 + 14800 + 98, 40, 1))


def test_a_reply_still_streaming_is_counted_so_a_tool_called_in_it_reads_its_own_request() -> None:
    # The tool runs once the reply ends in its call; its Exchanged comes after the last byte, so it may not be whole yet.
    usage = heard(Usage(Holder(BRAIN)), sent("a"), started("a", 10, cached=1000), delta("a", 5), whole("a"), sent("b"), started("b", 3, cached=1015))
    reading = usage.reading()
    assert in_context(reading) == 3 + 1015 + 1
    assert reading.spent == Spent(10 + 1000 + 3 + 1015, 5 + 1, 2)


def test_a_compaction_is_spent_but_is_not_the_conversation_and_the_turn_after_it_is() -> None:
    usage = heard(
        Usage(Holder(BRAIN)),
        sent("a"), started("a", 1, cached=90000), delta("a", 10), whole("a"),
        sent("c", Compaction()), started("c", 1, cached=90010), delta("c", 2000), whole("c", Compaction()),
    )
    assert in_context(usage.reading()) == 1 + 90000 + 10
    heard(usage, sent("b"), started("b", 2500), delta("b", 20))
    reading = usage.reading()
    assert in_context(reading) == 2500 + 20
    assert reading.spent == Spent(90001 + 90011 + 2500, 10 + 2000 + 20, 3)


def test_a_count_a_delta_gives_as_null_keeps_the_one_before() -> None:
    nulled = Heard("a", MessageDelta("end_turn", {"output_tokens": 40, "cache_read_input_tokens": None}))
    usage = heard(Usage(Holder(BRAIN)), sent("a"), started("a", 2, cached=90000), nulled)
    assert in_context(usage.reading()) == 2 + 90000 + 40


def test_a_late_frame_of_an_earlier_main_turn_does_not_stand_for_the_conversation() -> None:
    usage = heard(Usage(Holder(BRAIN)), sent("a"), started("a", 100), sent("b"), started("b", 200), delta("a", 7))
    assert in_context(usage.reading()) == 200 + 1


def in_context(reading: Tally) -> int:
    return cast(Context, reading.context).tokens


def test_a_cleared_conversation_holds_nothing_yet_and_what_was_spent_is_still_read() -> None:
    brain = Holder(BRAIN)
    usage = heard(Usage(brain), sent("a"), started("a", 10), delta("a", 5), whole("a"))
    brain.session = SessionId("cleared")
    assert usage.reading() == Tally(None, Spent(10, 5, 1))


def test_a_conversation_resumed_reads_its_own_latest_turn_again() -> None:
    brain = Holder(BRAIN)
    usage = heard(Usage(brain), sent("a"), started("a", 10), whole("a"))
    brain.session = SessionId("next")
    heard(usage, sent("b", session=SessionId("next")), started("b", 300), whole("b"))
    assert in_context(usage.reading()) == 300 + 1
    brain.session = BRAIN
    assert usage.reading() == Tally(Context("claude-opus-5-5", 10 + 1), Spent(310, 2, 2))


def test_another_sessions_replies_are_not_the_brains() -> None:
    other = SessionId("someone")
    usage = heard(Usage(Holder(BRAIN)), sent("x", session=other), started("x", 5000), sent("a"), started("a", 10))
    assert usage.reading() == Tally(Context("claude-opus-5-5", 11), Spent(10, 1, 1))


async def test_each_call_is_one_event_holding_what_it_reported() -> None:
    recorded: list[Entry] = []
    usage = Usage(Holder(BRAIN))
    given: asyncio.Future[Usage] = asyncio.get_running_loop().create_future()
    given.set_result(usage)
    context_usage = audited(SYSTEM, usage_tool(given), recorded.append)
    nothing = {"in_context": "nothing counted yet: this conversation has no reply of yours", "spent": {"input_tokens": 0, "output_tokens": 0, "replies": 0}}
    assert await context_usage.body() == nothing
    heard(usage, sent("a"), started("a", 2, cached=14800, created=98), delta("a", 40))
    reported = {"model": "claude-opus-5-5", "in_context_tokens": 14940, "spent": {"input_tokens": 14900, "output_tokens": 40, "replies": 1}}
    assert await context_usage.body() == reported
    events = [entry for entry in recorded if isinstance(entry, WideEvent)]
    assert [cast(Called, event.facts["called"]).result for event in events] == [nothing, reported]
