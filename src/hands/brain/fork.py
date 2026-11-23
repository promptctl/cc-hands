"""A side question as the brain's Claude Code asks one: /btw, forked from the brain's newest main turn.

Claude Code asks a side question (DTt, in its side-question chunk) as one request beside its loop: the loop's own model,
system, tools, and settings, its conversation so far, its newest reply included, then one user message holding a
placeholder result for each call still running, its reminder, and the question. It sets skipCacheWrite, so the request
writes nothing to the cache and reads it up to its fork point (docs/claude-code-behavior.md, side question markers), each
of its markers with a side question's own ttl. hands builds that request from what its proxy forwarded for the brain, and
sends it back through the proxy: no Claude Code is started to ask it.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from hands.brain.session import SessionView
from hands.core.session import SessionId
from hands.core.wire import (
    CLASS_HEADER,
    cache_marked,
    fork_point,
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
# The values Claude Code reads an environment variable as on by (Fe).
ON = frozenset({"1", "true", "yes", "on"})
# Whether the brain runs with stellar drift (eT): the brain's --settings env says so (hands.brain.process.SETTINGS_ENV).
STELLAR_DRIFT = False


@dataclass(frozen=True)
class CacheTtls:
    """The ttls Claude Code gives a side question's cache markers, None for the API's default of five minutes: its system
    prompt's, and its messages' and the request's own."""

    system: Literal["1h"] | None
    messages: Literal["1h"] | None


def side_question_ttls(environment: Mapping[str, str], setting: Literal["5m", "1h"] | None) -> CacheTtls:
    """The ttls Claude Code gives a side question in `environment`, its settings.json naming `setting` as its
    subagentPromptCacheTtl (Fwe, ain and UZ for querySource side_question): the hour the loop's markers have goes only to
    the sources its 1h allowlist names, and that list never names a side question."""
    named = environment.get("CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL", "").strip()
    chosen = "5m" if _on(environment, "FORCE_PROMPT_CACHING_5M") else named if named in ("5m", "1h") else setting or ("1h" if _on(environment, "ENABLE_PROMPT_CACHING_1H") else "5m")
    system: Literal["1h"] | None = "1h" if chosen == "1h" else None
    return CacheTtls(system, None if environment.get("CLAUDE_CODE_MESSAGES_PROMPT_CACHE_TTL", "").strip() == "5m" else system)


def _on(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "").strip().lower() in ON


@dataclass(frozen=True)
class Forked:
    """A side question's request, and the message of its body that carries its one cache marker, None when none does."""

    headers: tuple[tuple[str, str], ...]
    body: dict[str, object]
    marked: int | None


@dataclass(frozen=True)
class Turn:
    """The brain's newest main turn: its request as the proxy forwarded it, and the reply's message once read whole."""

    exchange: str
    path: str
    headers: tuple[tuple[str, str], ...]
    # The bytes as forwarded, parsed only when a question is forked from them: never on the way upstream.
    body: bytes
    reply: Message | None


def fork(turn: Turn, question: str, ttls: CacheTtls) -> Forked:
    """A side question asked of `turn`, as Claude Code's /btw builds it, its markers given `ttls`."""
    sent = cast(dict[str, object], json.loads(turn.body))
    messages = [dict(cast(Mapping[str, object], message)) for message in cast(Sequence[object], sent["messages"])]
    # An empty reply is no message the API takes: Claude Code's history has none for it.
    replied = [] if turn.reply is None or not turn.reply.content else [{"role": "assistant", "content": [_block(block) for block in turn.reply.content]}]
    running = [] if turn.reply is None else [{"type": "tool_result", "tool_use_id": block.id, "content": RUNNING} for block in turn.reply.content if isinstance(block, ToolUse)]
    asked = [*running, {"type": "text", "text": REMINDER}, {"type": "text", "text": question}]
    shared = [_unmarked(message) for message in (*messages, *replied)]
    # The API takes a system message only before an assistant message or at the end: a run of them closing a turn with no
    # reply yet to follow it stays at the end, and the question goes before it.
    at = next((index + 1 for index in reversed(range(len(shared))) if shared[index].get("role") != "system"), 0)
    history, closing = shared[:at], shared[at:]
    # Claude Code's normalisation merges adjacent user messages: the question joins a last message that is the user's.
    match history:
        case [*_, {"role": "user", "content": content}]:
            history[-1] = {**history[-1], "content": [*_blocks(content), *asked]}
        case _:
            history.append({"role": "user", "content": asked})
    history.extend(closing)
    # [LAW:one-source-of-truth] Claude Code places a side question's markers afresh, none carried from the loop's: one on
    # its fork point, found once the question is merged in, so a question merged into the loop's last message reads that
    # message uncached; and one only when its caching marks messages at all, as the loop's markers say it does
    # (docs/claude-code-behavior.md, side question markers).
    point = fork_point(history, STELLAR_DRIFT) if any(cache_marked(message) for message in messages) else None
    marker = _ttl({"type": "ephemeral"}, ttls.messages)
    history = [_marked(message, marker) if index == point else message for index, message in enumerate(history)]
    headers = tuple((name, SIDE_QUESTION_CLASS if name.lower() == CLASS_HEADER else value) for name, value in turn.headers)
    # [LAW:one-source-of-truth] the marked message is read off the messages sent, whose loop markers are all taken off.
    return Forked(headers, {**sent, **_retimed(sent, ttls), "messages": history}, next((index for index, message in enumerate(history) if cache_marked(message)), None))


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


def _blocks(content: object) -> list[object]:
    match content:
        case str():
            return [{"type": "text", "text": content}]
        case list():
            return cast(list[object], content)
        case _:
            raise ValueError(f"a message whose content is neither text nor blocks, but {type(content).__name__}")


def _unmarked(message: Mapping[str, object]) -> dict[str, object]:
    # The message as the loop sent it, only its markers taken off.
    content = message["content"]
    match content:
        case list():
            return {**message, "content": [_fields(block, {key: value for key, value in _mapping(block).items() if key != "cache_control"}) for block in cast(list[object], content)]}
        case _:
            return dict(message)


def _marked(message: Mapping[str, object], marker: Mapping[str, object]) -> dict[str, object]:
    # As Claude Code marks a message (bhr, whr): on its last block, text content made a block to carry it.
    blocks = _blocks(message["content"])
    return {**message, "content": [*blocks[:-1], *(_fields(last, {**_mapping(last), "cache_control": marker}) for last in blocks[-1:])]}


def _mapping(block: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], block) if isinstance(block, Mapping) else {}


def _fields(block: object, fields: dict[str, object]) -> object:
    # A block that is a JSON object, as `fields` remake it; any other kept as it is.
    return fields if isinstance(block, Mapping) else block


def _retimed(sent: Mapping[str, object], ttls: CacheTtls) -> dict[str, object]:
    """The request's own marker and its system prompt's, each as the loop's is but for its ttl, the side question's."""
    system = sent.get("system")
    return {
        **({"system": [_retimed_block(block, ttls.system) for block in cast(list[object], system)]} if isinstance(system, list) else {}),
        **({"cache_control": _ttl(cast(Mapping[str, object], marker), ttls.messages)} if isinstance(marker := sent.get("cache_control"), Mapping) else {}),
    }


def _retimed_block(block: object, ttl: Literal["1h"] | None) -> object:
    marker = _mapping(block).get("cache_control")
    return {**_mapping(block), "cache_control": _ttl(cast(Mapping[str, object], marker), ttl)} if isinstance(marker, Mapping) else block


def _ttl(marker: Mapping[str, object], ttl: Literal["1h"] | None) -> dict[str, object]:
    # d9: a marker names its ttl only when it is an hour, five minutes being the API's default.
    return {**{key: value for key, value in marker.items() if key != "ttl"}, **({"ttl": ttl} if ttl is not None else {})}
