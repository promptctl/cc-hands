"""A side question as the brain's Claude Code asks one: /btw, forked from the brain's newest main turn.

Claude Code 2.1.308 asks a side question (DTt, in its side-question chunk) as one request beside its loop: the loop's own
model, system, tools, and settings, its conversation so far, its newest reply included, then one user message holding a
placeholder result for each call still running, its reminder, and the question. It sets skipCacheWrite, so the request
reads the loop's cached prefix and writes none of its own. hands builds that request from the brain's newest main turn as
its proxy received it, and sends it back through the proxy: no Claude Code is started to ask it.
"""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal, cast

from hands.brain.pinned import CACHE_TTL, STELLAR_DRIFT
from hands.brain.session import SessionView
from hands.core.session import SessionId
from hands.core.wire import (
    CLASS_HEADER,
    login_of,
    cache_marked,
    content_blocks,
    Content,
    fork_point,
    read_request,
    block_param,
    Exchanged,
    MainTurn,
    Message,
    MessageParam,
    Observed,
    Reached,
    Request,
    Sent,
    Synthetic,
    Received,
    ToolUse,
    UnreadableRequest,
    Written,
    with_header,
    with_login,
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

    TURN_UNMARKED = "turn_unmarked"  # the brain's turn marked none of its messages, as it does with prompt caching off
    NO_FORK_POINT = "no_fork_point"  # no message before the question's own can carry one


@dataclass(frozen=True)
class Turn:
    """The brain's newest main turn: its request as the proxy received it, and the reply's message once it is whole."""

    exchange: str
    path: str
    headers: tuple[tuple[str, str], ...]
    # The bytes the conversation goes on from: as forwarded, hands' changes made, or as the client sent a request hands
    # answered itself. Parsed only when a question is forked from them: never on the way upstream.
    body: bytes
    reply: Message | None


@dataclass(frozen=True)
class Credential:
    """The login the brain's Claude Code sent on `exchange`."""

    exchange: str
    # The header the login was in, and its value: a live secret, kept out of every repr, and never empty (`login_of`).
    header: str
    value: str = field(repr=False)


@dataclass(frozen=True)
class Forked:
    """A side question's request: its headers, its body, and which of the body's messages carries its cache marker, or
    why none does."""

    headers: tuple[tuple[str, str], ...]
    body: dict[str, object]
    marked: int | Unmarked
    # The exchange the login in `headers` was sent on: the credential's, or the turn's own when the brain has sent none.
    login_from: str


def fork(turn: Turn, question: str, credential: Credential | None) -> Forked:
    """The side question asked of `turn`, as Claude Code's /btw builds it and sent with `credential`, when the brain has sent
    one, in place of the turn's own login, which may have expired since; UnreadableRequest when the turn's request is no
    messages request hands can read."""
    sent = _read(turn.body)
    # An empty reply is no message the API takes: Claude Code's history has none for it.
    replied = () if turn.reply is None or not turn.reply.content else (MessageParam("assistant", tuple(block_param(block) for block in turn.reply.content)),)
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
        case [*_, MessageParam(role="user", content=content) as last]:
            history[-1] = replace(last, content=(*content_blocks(content), *asked))
        case _:
            history.append(MessageParam("user", asked))
    history.extend(closing)
    # [LAW:one-source-of-truth] the loop marks its messages only when its prompt caching is on, and a fork of it marks as
    # it does: one marker, on its fork point, so it reads the loop's cached prefix and writes none of its own.
    turn_marked = any(cache_marked(message) for message in sent.messages)
    point = fork_point(history, STELLAR_DRIFT) if turn_marked else None
    messages = [_as_sent(_marked(message) if index == point else message) for index, message in enumerate(history)]
    marked = point if point is not None else Unmarked.NO_FORK_POINT if turn_marked else Unmarked.TURN_UNMARKED
    headers = with_header(turn.headers, CLASS_HEADER, SIDE_QUESTION_CLASS)
    if credential is not None:
        headers = with_login(headers, (credential.header, credential.value))
    login_from = turn.exchange if credential is None else credential.exchange
    marker = {} if sent.marker is None else {"cache_control": _retimed(sent.marker)}
    body = {"model": sent.model, **sent.rest, **marker, **_prompt(sent.system), "messages": messages}
    return Forked(headers, body, marked, login_from)


class Newest:
    """The brain's newest main turn in each conversation, heard off the wire: the one thing a side question is forked from."""

    def __init__(self, session: SessionView) -> None:
        # Which conversation the brain is in now: a question forks only from a turn of it.
        self._session = session
        # Main turns of the brain's told and not yet received, by the session each is of: its request is held once the
        # proxy receives it, to go upstream or to be answered by hands.
        self._main: dict[str, SessionId | None] = {}
        # [LAW:one-source-of-truth] the newest turn held of each conversation, so a /resume back forks from where it was left.
        self._turns: dict[SessionId | None, Turn] = {}
        # [LAW:one-source-of-truth] the login a question is sent with: the newest, of either kind, the brain's Claude Code sent on any request
        # through the proxy, a question's own excepted. It renews its token when it expires within five minutes of a request
        # it builds, and an idle one sends a request every ten minutes (docs/claude-code-behavior.md): so this one is live
        # when the turn a question forks carries one long expired, bar the minutes between a token's expiry and that
        # request. None while the brain has sent none, as one on an API key does.
        self._credential: Credential | None = None
        # [LAW:no-ambient-temporal-coupling] set, and replaced, as each turn is held and as the brain moves: a question
        # waits on it for a turn of the conversation the brain is in, which after a /clear is none until it sends its first.
        self._changed = asyncio.Event()
        session.watch(self._change)

    async def turn(self) -> Turn:
        """The brain's newest main turn in the conversation it is in, once that conversation has sent one."""
        while (turn := self._turns.get(self._session.id())) is None:
            await self._changed.wait()
        return turn

    def credential(self) -> Credential | None:
        """The newest login the brain has sent."""
        return self._credential

    def hear(self, observed: Observed) -> None:
        match observed:
            case Sent(exchange=exchange, session=session, kind=MainTurn()):
                self._main[exchange] = session
            case Received(exchange=exchange, path=path, headers=headers, body=body, built=built):
                # A question's request carries the login it was given, which is not one the brain has sent.
                if not built and (login := login_of(headers)) is not None:
                    self._credential = Credential(exchange, *login)
                if exchange in self._main:
                    self._turns[self._main.pop(exchange)] = Turn(exchange, path, headers, body, None)
                    self._change()
            case Exchanged(exchange=exchange, reply=reply):
                # A main turn that ended before the proxy received it is never waited on again.
                self._main.pop(exchange, None)
                match reply:
                    # The model's answer, or the one hands gave in its place: either is the reply Claude Code holds.
                    case Reached(body=Written(message=message)) | Synthetic(written=Written(message=message)):
                        self._turns = {of: replace(turn, reply=message) if turn.exchange == exchange else turn for of, turn in self._turns.items()}
                    case _:
                        pass
            case _:
                pass

    def _change(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()


def _read(body: bytes) -> Request:
    try:
        decoded = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UnreadableRequest(f"its body is not JSON: {error}") from error
    return read_request(decoded)


# [LAW:one-source-of-truth] the terms a marker gives its ttl by, as Claude Code writes one: five minutes is the API's
# default, so only an hour is named (docs/claude-code-behavior.md, "side question system and request markers").
_TTL_TERMS: Mapping[Literal["5m", "1h"], Mapping[str, str]] = {"5m": {}, "1h": {"ttl": "1h"}}


def _unmarked(message: MessageParam) -> MessageParam:
    match message.content:
        case str():
            # A string carries no marker.
            return message
        case blocks:
            return replace(message, content=tuple({term: value for term, value in block.items() if term != "cache_control"} for block in blocks))


def _marked(message: MessageParam) -> MessageParam:
    # On its last block, as Claude Code marks a message: a reply's is one that can carry it, or it is no fork point.
    *before, last = content_blocks(message.content)
    return replace(message, content=(*before, {**last, "cache_control": _retimed({"type": "ephemeral"})}))


def _as_sent(message: MessageParam) -> dict[str, object]:
    return {"role": message.role, "content": message.content if isinstance(message.content, str) else list(message.content), **message.rest}


def _retimed(marker: Mapping[str, object]) -> dict[str, object]:
    """A marker given a side question's ttl: the rest of it, a system prompt's scope and a request's eviction among
    them, stays as sent."""
    return {**{term: value for term, value in marker.items() if term != "ttl"}, **_TTL_TERMS[CACHE_TTL]}


def _retimed_block(block: Mapping[str, object]) -> dict[str, object]:
    if "cache_control" not in block:
        return dict(block)
    # An object, as `read_request` reads every marker on a system prompt.
    return {**block, "cache_control": _retimed(cast(Mapping[str, object], block["cache_control"]))}


def _prompt(system: Content | None) -> dict[str, object]:
    """The system prompt a side question sends: the turn's, if it sent one, its markers taking a side question's ttl."""
    match system:
        case None:
            return {}
        case str():
            return {"system": system}
        case blocks:
            return {"system": [_retimed_block(block) for block in blocks]}
