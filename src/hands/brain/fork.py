"""A side question as the brain's Claude Code asks one: /btw, forked from the brain's newest main turn.

Claude Code 2.1.308 asks a side question (DTt, in its side-question chunk) as one request beside its loop: the loop's own
model, system, tools, and settings, its conversation so far, its newest reply included, then one user message holding a
placeholder result for each call still running, its reminder, and the question. It sets skipCacheWrite, so the request
reads the loop's cached prefix and writes none of its own. hands builds that request from what its proxy forwarded for
the brain, and sends it back through the proxy: no Claude Code is started to ask it.
"""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from hands.brain.pinned import STELLAR_DRIFT
from hands.brain.session import SessionView
from hands.core.session import SessionId
from hands.core.wire import (
    CLASS_HEADER,
    cache_marked,
    fork_point,
    read_request,
    Block,
    Exchanged,
    Forwarded,
    MainTurn,
    Message,
    Observed,
    Reached,
    RedactedThinking,
    Request,
    Said,
    Sent,
    Text,
    Thinking,
    ToolUse,
    UnreadableRequest,
    Written,
)

# Verbatim from DTt: the text block Claude Code opens every side question's prompt with, the question in a block after it.
REMINDER = """<system-reminder>This is a side question from the user. You must answer this question directly in a single response.

IMPORTANT CONTEXT:
- You are a separate, lightweight agent spawned to answer this one question
- The main agent is NOT interrupted - it continues working independently in the background
- You share the conversation context but are a completely separate instance
- Do NOT reference being interrupted or what you were "previously doing" - that framing is incorrect

CRITICAL CONSTRAINTS:
- You have NO tools available - you cannot read files, run commands, search, or take any actions
- Do NOT write tool calls or tool output as text (for example invoke or function_calls XML blocks) - nothing you write here is executed; if answering would need reading files, running commands, or searching, say that can't be checked from a side question and suggest asking in the main conversation
- This is a one-off response - there will be no follow-up turns
- You can ONLY provide information based on what you already know from the conversation context
- NEVER say things like "Let me try...", "I'll now...", "Let me check...", or promise to take any action
- If you don't know the answer, say so - do not offer to look it up or investigate

Simply answer the question with the information you have.</system-reminder>

"""
# Verbatim from DTt's x(): the result a call of the loop's newest reply stands in with while it is still running.
RUNNING = "[No result yet — this call is still in progress in the main conversation (running, awaiting approval, or queued)]"
# The class Claude Code names a side question in (es() of querySource "side_question").
SIDE_QUESTION_CLASS = "auxiliary"


class Unmarked(StrEnum):
    """Why a side question carries no cache marker on its messages."""

    UNCACHED = "uncached"  # the brain's turn marked none of its messages: its prompt caching is off
    NO_FORK_POINT = "no_fork_point"  # no message before the question's own can carry one


@dataclass(frozen=True)
class Turn:
    """The brain's newest main turn: its request as the proxy forwarded it, and the reply's message once read whole."""

    exchange: str
    path: str
    headers: tuple[tuple[str, str], ...]
    # The bytes as forwarded, parsed only when a question is forked from them: never on the way upstream.
    body: bytes
    reply: Message | None


@dataclass(frozen=True)
class Forked:
    """A side question's request: its headers, its body, and which of the body's messages carries its cache marker, or
    why none does."""

    headers: tuple[tuple[str, str], ...]
    body: dict[str, object]
    marked: int | Unmarked


def fork(turn: Turn, question: str) -> Forked:
    """The side question asked of `turn`, as Claude Code's /btw builds it; UnreadableRequest when the turn's request is no
    messages request hands can read."""
    sent = _read(turn.body)
    # An empty reply is no message the API takes: Claude Code's history has none for it.
    replied = () if turn.reply is None or not turn.reply.content else (Said("assistant", tuple(_block(block) for block in turn.reply.content)),)
    running = () if turn.reply is None else tuple({"type": "tool_result", "tool_use_id": block.id, "content": RUNNING} for block in turn.reply.content if isinstance(block, ToolUse))
    asked = (*running, {"type": "text", "text": REMINDER}, {"type": "text", "text": question})
    # Claude Code marks the history it sends afresh on every request: none of the loop's markers is the fork's.
    shared = [_unmarked(message) for message in (*sent.messages, *replied)]
    # The API takes a system message only before an assistant message or at the end: a run of them closing a turn with no
    # reply yet to follow it stays at the end, and the question goes before it.
    at = next((index + 1 for index in reversed(range(len(shared))) if shared[index].role != "system"), 0)
    history, closing = shared[:at], shared[at:]
    # Claude Code's normalisation merges adjacent user messages: the question joins a last message that is the user's.
    match history:
        case [*_, Said(role="user", content=content) as last]:
            history[-1] = replace(last, content=(*_blocks(content), *asked))
        case _:
            history.append(Said("user", asked))
    history.extend(closing)
    # [LAW:one-source-of-truth] the loop marks its messages only when its prompt caching is on, and a fork of it marks as
    # it does: one marker, on its fork point, so it reads the loop's cached prefix and writes none of its own.
    point = fork_point(history, STELLAR_DRIFT) if any(cache_marked(message) for message in sent.messages) else Unmarked.UNCACHED
    marked = Unmarked.NO_FORK_POINT if point is None else point
    messages = [_as_sent(_marked(message) if index == marked else message) for index, message in enumerate(history)]
    headers = tuple((name, SIDE_QUESTION_CLASS if name.lower() == CLASS_HEADER else value) for name, value in turn.headers)
    body = {**_retimed(sent.fields), **_prompt(sent.system), "messages": messages}
    return Forked(headers, body, marked)


class Newest:
    """The brain's newest main turn in each conversation, heard off the wire: the one thing a side question is forked from."""

    def __init__(self, session: SessionView) -> None:
        # Which conversation the brain is in now: a question forks only from a turn of it.
        self._session = session
        # Main turns of the brain's told and not yet forwarded, by the session each is of: its request is held once its
        # bytes go upstream.
        self._main: dict[str, SessionId | None] = {}
        # [LAW:one-source-of-truth] the newest turn held of each conversation, so a /resume back forks from where it was left.
        self._turns: dict[SessionId | None, Turn] = {}
        # [LAW:no-ambient-temporal-coupling] set, and replaced, as each turn is held and as the brain moves: a question
        # waits on it for a turn of the conversation the brain is in, which after a /clear is none until it sends its first.
        self._changed = asyncio.Event()
        session.watch(self._change)

    async def turn(self) -> Turn:
        """The brain's newest main turn in the conversation it is in, once that conversation has sent one."""
        while (turn := self._turns.get(self._session.id())) is None:
            await self._changed.wait()
        return turn

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(exchange=exchange, session=session, kind=MainTurn()):
                self._main[exchange] = session
            case Forwarded(exchange=exchange, path=path, headers=headers, body=body) if exchange in self._main:
                self._turns[self._main.pop(exchange)] = Turn(exchange, path, headers, body, None)
                self._change()
            case Exchanged(exchange=exchange, reply=reply):
                # A main turn held or refused before it was forwarded is never waited on again.
                self._main.pop(exchange, None)
                match reply:
                    case Reached(body=Written(message=message)):
                        self._turns = {of: replace(turn, reply=message) if turn.exchange == exchange else turn for of, turn in self._turns.items()}
                    case _:
                        pass
            case _:
                pass

    def _change(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()


def _block(block: Block) -> dict[str, object]:
    match block:
        case Text(text=text):
            return {"type": "text", "text": text}
        case Thinking(thinking=thinking, signature=signature):
            return {"type": "thinking", "thinking": thinking, "signature": signature}
        case RedactedThinking(data=data):
            return {"type": "redacted_thinking", "data": data}
        case ToolUse(id=call, name=name, input=given):
            return {"type": "tool_use", "id": call, "name": name, "input": given}


def _read(body: bytes) -> Request:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise UnreadableRequest(f"its body is not JSON: {error}") from error
    return read_request(decoded)


def _blocks(content: str | tuple[Mapping[str, object], ...]) -> tuple[Mapping[str, object], ...]:
    """A message's content as blocks, a string as the one text block Claude Code sends it as."""
    return ({"type": "text", "text": content},) if isinstance(content, str) else content


def _unmarked(message: Said) -> Said:
    match message.content:
        case str():
            # A string carries no marker.
            return message
        case blocks:
            return replace(message, content=tuple({term: value for term, value in block.items() if term != "cache_control"} for block in blocks))


def _marked(message: Said) -> Said:
    # On its last block, as Claude Code marks a message: a reply's is one that can carry it, or it is no fork point. The
    # marker is the one a side question's ttl gives it, which names none (`pinned.CACHE_TTL`).
    *before, last = _blocks(message.content)
    return replace(message, content=(*before, {**last, "cache_control": {"type": "ephemeral"}}))


def _as_sent(message: Said) -> dict[str, object]:
    return {"role": message.role, "content": message.content if isinstance(message.content, str) else list(message.content), **message.rest}


def _retimed(marked: Mapping[str, object]) -> dict[str, object]:
    """A block, or a request, whose marker, if it has one, takes a side question's ttl, which names none
    (`pinned.CACHE_TTL`): the rest of it, a system prompt's scope and a request's eviction among them, stays as sent."""
    marker = marked.get("cache_control")
    if not isinstance(marker, Mapping):
        return dict(marked)
    return {**marked, "cache_control": {term: value for term, value in cast(Mapping[str, object], marker).items() if term != "ttl"}}


def _prompt(system: str | tuple[Mapping[str, object], ...] | None) -> dict[str, object]:
    """The system prompt a side question sends: the turn's, if it sent one, its markers taking a side question's ttl."""
    match system:
        case None:
            return {}
        case str():
            return {"system": system}
        case blocks:
            return {"system": [_retimed(block) for block in blocks]}
