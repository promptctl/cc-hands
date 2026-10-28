"""A side question as the brain's Claude Code asks one: /btw, forked from the brain's newest main turn.

Claude Code 2.1.308 asks a side question (DTt, in its side-question chunk) as one request beside its loop: the loop's own
model, system, tools, and settings, its conversation so far, its newest reply included, then one user message holding a
placeholder result for each call still running, its reminder, and the question. It sets skipCacheWrite, so the request
reads the loop's cached prefix and writes none of its own. hands builds that request from what its proxy forwarded for
the brain, and sends it back through the proxy: no Claude Code is started to ask it.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from hands.core.session import SessionId
from hands.core.wire import (
    CLASS_HEADER,
    Block,
    Exchanged,
    Forwarded,
    MainTurn,
    Message,
    Observed,
    Reached,
    RedactedThinking,
    Sent,
    Text,
    Thinking,
    ToolUse,
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


@dataclass(frozen=True)
class Turn:
    """The brain's newest main turn: its request as the proxy forwarded it, and the reply's message once read whole."""

    exchange: str
    path: str
    headers: tuple[tuple[str, str], ...]
    body: Mapping[str, object]
    reply: Message | None


def fork(turn: Turn, question: str) -> tuple[tuple[tuple[str, str], ...], bytes]:
    """The headers and body of a side question asked of `turn`, as Claude Code's /btw builds them."""
    messages = [dict(cast(Mapping[str, object], message)) for message in cast(Sequence[object], turn.body["messages"])]
    replied = [] if turn.reply is None else [{"role": "assistant", "content": [_block(block) for block in turn.reply.content]}]
    running = [] if turn.reply is None else [{"type": "tool_result", "tool_use_id": block.id, "content": RUNNING} for block in turn.reply.content if isinstance(block, ToolUse)]
    asked = [*running, {"type": "text", "text": REMINDER}, {"type": "text", "text": question}]
    # [LAW:one-source-of-truth] the loop's own newest marker is the one moved: same cache_control, on the last message the
    # fork shares with the loop. Markers it put on earlier messages stay where they are.
    newest = next((index for index in reversed(range(len(messages))) if _marks(messages[index])), None)
    marker = None if newest is None else _marks(messages[newest])[-1]
    history = [_unmarked(message) if index == newest else message for index, message in enumerate((*messages, *replied))]
    # Claude Code's normalisation merges adjacent user messages: the question joins a last message that is the user's.
    match history[-1]:
        case {"role": "user", "content": content}:
            history[-1] = {**history[-1], "content": [*_blocks(content), *asked]}
        case _:
            history.append({"role": "user", "content": asked})
    # skipCacheWrite: the marker sits on the last message the fork shares with the loop, so nothing new is written.
    if marker is not None and len(history) > 1:
        history[-2] = _marked(history[-2], marker)
    headers = tuple((name, SIDE_QUESTION_CLASS if name.lower() == CLASS_HEADER else value) for name, value in turn.headers)
    return headers, json.dumps({**turn.body, "messages": history}).encode()


class Newest:
    """The brain's newest main turn, heard off the wire: the one thing a side question is forked from."""

    def __init__(self, session: SessionId) -> None:
        self._session = session
        # Main turns of the brain's told and not yet forwarded: its request is held once its bytes go upstream.
        self._main: set[str] = set()
        self._turn: Turn | None = None
        # [LAW:no-ambient-temporal-coupling] set once, by the first turn held: a question asked before it awaits this.
        self._first = asyncio.Event()

    async def turn(self) -> Turn:
        """The brain's newest main turn, once it has sent one."""
        await self._first.wait()
        return cast(Turn, self._turn)

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(exchange=exchange, session=session, kind=MainTurn()) if session == self._session:
                self._main.add(exchange)
            case Forwarded(exchange=exchange, path=path, headers=headers, body=body) if exchange in self._main:
                self._main.discard(exchange)
                self._turn = Turn(exchange, path, headers, cast(Mapping[str, object], json.loads(body)), None)
                self._first.set()
            case Exchanged(exchange=exchange, reply=Reached(body=Written(message=message))) if self._turn is not None and exchange == self._turn.exchange:
                self._turn = replace(self._turn, reply=message)
            case Exchanged(exchange=exchange):
                # A main turn held or refused before it was forwarded is never waited on again.
                self._main.discard(exchange)
            case _:
                pass


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


def _blocks(content: object) -> list[object]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(cast(Sequence[object], content))


def _marks(message: Mapping[str, object]) -> list[object]:
    return [mark for block in _blocks(message["content"]) if isinstance(block, Mapping) and (mark := cast(Mapping[str, object], block).get("cache_control")) is not None]


def _unmarked(message: Mapping[str, object]) -> dict[str, object]:
    blocks = [{key: value for key, value in cast(Mapping[str, object], block).items() if key != "cache_control"} if isinstance(block, Mapping) else block for block in _blocks(message["content"])]
    return {**message, "content": blocks}


def _marked(message: Mapping[str, object], marker: object) -> dict[str, object]:
    blocks = _blocks(message["content"])
    return {**message, "content": [*blocks[:-1], {**cast(Mapping[str, object], blocks[-1]), "cache_control": marker}]}
