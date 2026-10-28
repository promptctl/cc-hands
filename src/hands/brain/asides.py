"""Side questions: what hands asks in the background, each a /btw forked from the brain's newest main turn.

A summary of a backlog or of a session's turns, and the line an old tool result goes as, are asked here and never typed
into the brain. The brain's input is the user's: a question that is slow, or stuck, or never answered holds nothing a
spoken turn waits on, and nothing asked here enters the brain's conversation.

Each question is one request through hands' proxy, built as the brain's Claude Code builds a side question
(`hands.brain.fork`): no Claude Code is started to ask it. Its answer is read from that request's own reply.
"""

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, replace
from enum import StrEnum

import aiohttp

from hands.brain.fork import Newest, fork
from hands.core.trace import Span
from hands.core.wire import Fork, Garbled, Observed, Route, Send, Sent, Written
from hands.core.wire import Text as Said
from hands.sessions.audit import Record
from hands.sessions.replies import reply_reader
from hands.sessions.wide import annotate, here, since, unit, within


# What a question's reply cost, as its usage names it: what it read of the brain's cached prefix, and what it wrote.
USAGE = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")


class AsideKind(StrEnum):
    """What a side question is asked for."""

    SUMMARY = "summary"  # a backlog's tickets or a session's turns, each as a sentence to be said
    EXPLANATION = "explanation"  # what a session is doing, from its progress
    NAME = "name"  # a session's name, judged from a turn it finished
    LINE = "line"  # the line an old tool result goes as in the brain's context


class Unanswered(StrEnum):
    """Why a side question has no answer."""

    UNFORKED = "unforked"  # the brain had sent no main turn to fork it from within its asker's time
    REFUSED = "refused"  # its request was answered with something other than a reply
    BROKEN = "broken"  # its request or its reply broke off on the way
    TIMED_OUT = "timed_out"  # no answer within what its asker gave it
    WORDLESS = "wordless"  # the model's reply did not end in words


class AsideFailed(Exception):
    """A side question has no answer, `why` says which way, and its message what was seen of it."""

    def __init__(self, why: Unanswered, seen: str) -> None:
        super().__init__(seen)
        self.why = why


@dataclass(frozen=True)
class Deadline:
    """An asker waiting on the answer: `seconds` from when it is asked, its turn behind the questions before it included."""

    seconds: float


@dataclass(frozen=True)
class TimeLimit:
    """An asker in no hurry for its turn: `seconds` from when it has it, however long it waited for it."""

    seconds: float


# How long a side question's asker waits on its answer, its turn and its reply both inside it.
Within = Deadline | TimeLimit


class Asides:
    """What answers hands' side questions: `ask` forks the brain's newest main turn with the question, through the proxy."""

    def __init__(self, proxy_url: str, newest: Newest, record: Record) -> None:
        self._proxy_url = proxy_url
        self._newest = newest
        self._record = record
        # One at a time: nothing asked here is waited on by the user, and the brain's account answers the user's turns first.
        self._one = asyncio.Lock()
        # The question being sent now, by its unit of work's span and the body it is sent as: one at a time, under `_one`.
        self._sending: tuple[Span, object] | None = None
        # As the proxy's own client: the reply read as sent, and no header added that the brain's Claude Code left out.
        self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), auto_decompress=False, skip_auto_headers=("Accept-Encoding", "Content-Type", "User-Agent"))

    async def close(self) -> None:
        await self._client.close()

    def hear(self, observed: Observed) -> None:
        self._newest.hear(observed)

    def adopted(self, sent: Sent, routed: Route) -> Route:
        """`routed` as the question being sent now has it: its own request, inside the question's unit of work."""
        sending = self._sending
        # [LAW:one-source-of-truth] its request is told by the very body it was sent as; a fork's kind narrows the compare.
        if sending is None or not isinstance(sent.kind, Fork) or sent.body != sending[1]:
            return routed
        match routed:
            # Its client is hands, which never asks again: a refusal held final would reach it as the proxy's own 502,
            # and the question would fail saying that, not what the API answered.
            case Send():
                return replace(routed, span=within(sending[0]), refusal="retried")
            case _:
                return replace(routed, span=within(sending[0]))

    async def ask(self, kind: AsideKind, question: str, within: Within) -> str:
        """The answer to `question`, asked beside the brain's conversation; raises AsideFailed when it has none,
        TIMED_OUT once its asker's time is up."""
        # [LAW:nothing-unseen] one event for each question, however it ends: answered, failed, or left by its asker, in its
        # turn or still waiting for it. Asked inside another unit of work, it is that one's part.
        with unit("brain.aside", self._record):
            annotate(kind=kind, question=question)
            try:
                reply = await self._in_turn(question, within)
            except AsideFailed as failure:
                annotate(unanswered=failure.why)
                raise
            annotate(reply=reply)
            return reply

    async def _in_turn(self, question: str, within: Within) -> str:
        queued = time.monotonic()
        # What its asker's time running out means where it stands: waiting for a main turn to fork, or for anything else.
        waiting = (Unanswered.TIMED_OUT, "no answer")
        async with contextlib.AsyncExitStack() as taken:
            try:
                # [LAW:single-enforcer] the one clock its asker's time runs on, from asking or from its turn.
                async with asyncio.timeout(within.seconds if isinstance(within, Deadline) else None) as clock:
                    # Waited for before its turn: nothing is asked of a brain that has sent nothing, so no asker holds the
                    # others back on that.
                    waiting = (Unanswered.UNFORKED, "the brain sent no main turn to fork")
                    await self._newest.turn()
                    waiting = (Unanswered.TIMED_OUT, "no answer")
                    await self._one.acquire()
                    taken.callback(self._one.release)
                    # queued_ms only once it has its turn: the rest of its duration_ms is its request answered.
                    annotate(queued_ms=since(queued))
                    match within:
                        case TimeLimit(seconds=seconds):
                            clock.reschedule(asyncio.get_running_loop().time() + seconds)
                        case Deadline():
                            pass
                    turn = await self._newest.turn()
                    return await self._answered(*fork(turn, question), turn.path, turn.exchange)
            except TimeoutError as error:
                why, seen = waiting
                raise AsideFailed(why, f"{seen} within {within}") from error

    async def _answered(self, headers: tuple[tuple[str, str], ...], body: dict[str, object], path: str, forked: str) -> str:
        annotate(forked=forked)
        self._sending = (here(), body)
        try:
            async with self._client.post(self._proxy_url + path, headers=list(headers), data=json.dumps(body).encode()) as response:
                # [LAW:one-source-of-truth] read by the reader the proxy reads every reply with.
                reader = reply_reader(Fork(), response.headers, lambda _event: None)
                async for chunk in response.content.iter_any():
                    reader.feed(chunk)
                read = reader.finish(None)
        except aiohttp.ClientError as error:
            raise AsideFailed(Unanswered.BROKEN, f"its request broke off: {error!r}") from error
        finally:
            self._sending = None
        request_id = response.headers.get("request-id")
        annotate(status=response.status, request_id=request_id)
        match read:
            case Written(message=message) if response.status == 200:
                annotate(**{name: tokens for name in USAGE if isinstance(tokens := message.usage.get(name), int)})
                said = " ".join(block.text for block in message.content if isinstance(block, Said)).strip()
                if message.stop_reason == "end_turn" and said:
                    return said
                raise AsideFailed(Unanswered.WORDLESS, f"the model's reply ended {message.stop_reason!r} with {said!r}")
            case Garbled(reason=reason) if response.status == 200:
                # Answered, then cut off on the way: the stream ended before its reply was whole.
                raise AsideFailed(Unanswered.BROKEN, f"its reply broke off (request-id {request_id}): {reason}")
            case other:
                raise AsideFailed(Unanswered.REFUSED, f"its request was answered {response.status} (request-id {request_id}): {other}")
