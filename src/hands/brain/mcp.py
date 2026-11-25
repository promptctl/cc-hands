"""hands' MCP server: the intermediary's tools, served over MCP's streamable HTTP transport to the brain.

    claude --mcp-config '{"mcpServers": {"hands": {"type": "http", "url": "<the server's url>"}}}'

The adapter over the tools in hands.voice.tools, reading each through hands.voice.tool. It speaks the part of MCP a
tools-only server needs: initialize, ping, tools/list, and tools/call, each POSTed as JSON-RPC and answered as JSON. It
opens no event stream of its own, so a GET is refused, as the transport allows.

Its tools type into sessions and answer permissions, and a port on loopback is reachable by every local process and
every web page, so each request carries a token only the brain's --mcp-config holds.
"""

import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from aiohttp import web
from loguru import logger

from hands.clock import Clock
from hands.core import loopback
from hands.core.bearer import bearer, bears
from hands.core.trace import Span
from hands.sessions.audit import Record, said
from hands.sessions.payload import NotJson, Payload, Rejected
from hands.sessions.wide import annotate, begun, continuing, fail, unit
from hands.voice.answered import answered_then
from hands.voice.tool import After, Answer, Result, Tool, audited

# What the brain's --mcp-config names this server; its tools reach the model as mcp__hands__<tool>.
SERVER_NAME = "hands"
PATH = "/mcp"

# JSON-RPC's codes for a message that is no request, notification or answer, for a request that names no method this
# server has, and for one whose params do not parse.
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
PARSE_ERROR = -32700

# The protocol versions this server speaks, newest first: those with streamable HTTP and no JSON-RPC batches, in which a
# tools-only server says the same in every one.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")

# Where Claude Code puts, in a tools/call's _meta, the id of the tool_use block the call runs.
TOOL_USE_ID = "claudecode/toolUseId"


class CallSpans:
    """The span of each call the brain's replies are running, by its tool_use id: the brain's stage mints it as it hears
    the call's block close, and the call's run here is its child. The proxy hands the stage each byte before Claude Code
    has it, so a call's span is here before its request is [LAW:no-ambient-temporal-coupling]."""

    def __init__(self) -> None:
        # [LAW:no-shared-mutable-globals] owned here: written by `opened` and `ended`, read by `span`.
        self._spans: dict[str, Span] = {}

    def opened(self, call: str, span: Span) -> None:
        self._spans[call] = span

    def ended(self, calls: Iterable[str]) -> None:
        for call in calls:
            self._spans.pop(call, None)

    def span(self, call: str | None) -> Span | None:
        """The span of a call, None for one no turn is running: a call made outside a voice turn is a trace of its own."""
        return None if call is None else self._spans.get(call)


@dataclass(frozen=True)
class McpServer:
    url: str
    token: str
    runner: web.AppRunner

    async def close(self) -> None:
        await self.runner.cleanup()

    def config(self) -> str:
        """The --mcp-config that points a Claude Code at this server, beside any its own setup names, with the token that lets it in."""
        return json.dumps({"mcpServers": {SERVER_NAME: {"type": "http", "url": self.url, "headers": {"Authorization": bearer(self.token)}}}})


async def serve_mcp(clock: Clock, tools: Sequence[Tool[Answer]], record: Record, spans: CallSpans) -> McpServer:
    """Listen on a free local port until the returned server is closed. [LAW:single-enforcer] each call of a tool it
    serves is one `tool.run` event, here, whichever tools a door hands it."""
    by_name = {tool.name: audited(clock, tool, record) for tool in tools}
    token = clock.token(32)

    @web.middleware
    async def admitted(request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> web.StreamResponse:
        # [LAW:single-enforcer] the one door: nothing reaches a tool without the token.
        if not bears(request.headers.get("Authorization", ""), token):
            logger.warning(f"the MCP server refused a {request.method} without its token, from {request.remote} (Origin {request.headers.get('Origin')})")
            return web.Response(status=401)
        return await handler(request)

    async def post(request: web.Request) -> web.StreamResponse:
        match parse(await request.read()):
            case Request(id, "tools/call", Payload({"name": str(name)} as params), meta) if name in by_name and (arguments := _arguments(params)) is not None:
                used = _text(meta.fields.get(TOOL_USE_ID))
                match await call(by_name[name], arguments, used):
                    case After(result=result) as after:
                        # [LAW:nothing-unseen] its own event, in the trace of the call it followed: the call's event
                        # ended with its answer.
                        with continuing(spans.span(used)), unit("tool.after", record, begun(clock)):
                            annotate(tool=name)
                            return await answered_then(request, web.json_response(_answered(id, result)), after, name)
                    case result:
                        return web.json_response(_answered(id, result))
            case Request(id, method, Payload(params)):
                # [LAW:nothing-unseen] every request but a tool's run, which is the tool's own event, is one event.
                with unit("mcp.request", record, begun(clock)):
                    annotate(method=method)
                    answered = answer(id, method, params)
                return web.json_response(answered)
            case Unanswerable(code, why, id, method):
                # Answered, by none where no id could be read from it: a 202 would leave a client that sent a request
                # waiting on an answer that never comes.
                with unit("mcp.request", record, begun(clock)):
                    annotate(method=method)
                    fail(why)
                return web.json_response(_error(id, code, why))
            case Unanswered():
                return web.Response(status=202)

    def answer(id: RequestId, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        match method:
            case "initialize":
                # The client's version where this server speaks it, else the newest it speaks, as MCP's negotiation has it.
                asked = params.get("protocolVersion")
                version = asked if isinstance(asked, str) and asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                client = _object(params.get("clientInfo"))
                annotate(client=_text(client.get("name")), client_version=_text(client.get("version")), asked=_text(asked), protocol=version)
                return _result(id, {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": SERVER_NAME, "version": "0"}})
            case "ping":
                return _result(id, {})
            case "tools/list":
                return _result(id, {"tools": [{"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema} for tool in tools]})
            case "tools/call":
                # A call naming no tool of this server's, or none at all, or with arguments that are no object.
                why = f"no tool {params.get('name')} taking arguments {params.get('arguments')!r}"
                fail(why)
                return _error(id, INVALID_PARAMS, why)
            case "server/discover":
                # Claude Code asks for it first and, refused, opens with initialize: a known question with a known no.
                return _error(id, METHOD_NOT_FOUND, f"no method {method}")
            case _:
                fail(f"no method {method}")
                return _error(id, METHOD_NOT_FOUND, f"no method {method}")

    async def call(tool: Tool[Answer], arguments: Mapping[str, object], used: str | None) -> Answer | Failed:
        if used is None:
            # [LAW:no-silent-failure] Claude Code names the tool_use block each call runs: a call naming none is
            # run as a trace of its own, cut off from the turn that made it.
            logger.warning(f"the brain called {tool.name} naming no tool_use in {TOOL_USE_ID}; its run is not joined to its turn")
        try:
            # The run is a part of the turn's call to the tool, where a turn is running it.
            with continuing(spans.span(used)):
                return await tool.body(**arguments)
        except Exception as error:
            # The tool's event holds what it raised; the model hears that the tool failed. [LAW:no-silent-failure] the
            # exception ends here, so it is said here.
            failed = f"{tool.name} failed: {type(error).__name__}: {error}"
            said(failed)
            return Failed(failed)

    async def refused(_request: web.Request) -> web.Response:
        return web.Response(status=405, headers={"Allow": "POST"})

    app = web.Application(middlewares=[admitted])
    app.router.add_post(PATH, post)
    app.router.add_route("*", PATH, refused)
    # [LAW:no-ambient-temporal-coupling] a client that hangs up mid-call does not cancel it: a tool whose effect is half
    # done is worse than one whose result nobody reads, and the draft tools say so of themselves.
    runner = web.AppRunner(app, access_log=None, handler_cancellation=False)
    await runner.setup()
    site = web.TCPSite(runner, loopback.HOST, 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return McpServer(url=f"http://{host}:{port}{PATH}", token=token, runner=runner)


type RequestId = str | int


@dataclass(frozen=True)
class Request:
    id: RequestId
    method: str
    params: Payload
    # The params' _meta, where a client says what the request is part of: empty where it names none.
    meta: Payload


@dataclass(frozen=True)
class Unanswered:
    """A notification, or the client's answer to a request this server never makes: nothing to say back."""


@dataclass(frozen=True)
class Unanswerable:
    """A message refused before there is a request to answer: not JSON, no message, an id JSON-RPC takes none of, or
    params that are no object."""

    code: int
    why: str
    id: RequestId | None
    method: str | None


type Message = Request | Unanswered | Unanswerable


def parse(body: bytes) -> Message:
    """A POSTed body as the one JSON-RPC message it is. [LAW:parse-dont-validate] the id, the params and their _meta
    are parsed here, once, by hands' one reader of JSON, so no request reaches a method with an id the client could
    not be answered by, or with params that are no object."""
    try:
        message = Payload.parse(body)
    except NotJson as error:
        return Unanswerable(PARSE_ERROR, str(error), None, None)
    except Rejected as error:
        # A batch among them: no version this server speaks takes one.
        return Unanswerable(INVALID_REQUEST, f"no JSON-RPC message: {error}", None, None)
    match message.fields:
        case {"method": str(method), "id": id}:
            match _request_id(id):
                case None:
                    return Unanswerable(INVALID_REQUEST, f"a request's id is a string or an integer, never {json.dumps(id)}", None, method)
                case parsed:
                    try:
                        params = Payload.of(message.fields.get("params", {}), "its params")
                        return Request(parsed, method, params, Payload.of(params.fields.get("_meta", {}), "its params' _meta"))
                    except Rejected as error:
                        return Unanswerable(INVALID_PARAMS, str(error), parsed, method)
        case {"method": str()} | {"id": str() | int(), "result": _} | {"id": str() | int(), "error": _}:
            return Unanswered()
        case fields:
            # Refused by its id where one can be read from it, so a client that sent a request learns which.
            why = "no JSON-RPC message: an object with no method named in text and no answer"
            return Unanswerable(INVALID_REQUEST, why, _request_id(fields.get("id")), None)


def _request_id(value: object) -> RequestId | None:
    match value:
        # JSON's true and false, which Python's int pattern would take.
        case bool():
            return None
        case str() | int():
            return value
        case _:
            return None


def _object(value: object) -> Mapping[str, object]:
    match value:
        case dict():
            return cast(dict[str, object], value)
        case _:
            return {}


def _arguments(params: Mapping[str, object]) -> Mapping[str, object] | None:
    """A call's arguments, none given being none at all; None for arguments that are no object."""
    match arguments := params.get("arguments", {}):
        case dict():
            return cast(dict[str, object], arguments)
        case _:
            return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


@dataclass(frozen=True)
class Failed:
    """A tool's body that raised, as the model hears it."""

    said: str


def _answered(id: RequestId, result: Result | Failed) -> Mapping[str, object]:
    match result:
        case Failed(said=failed):
            return _result(id, {"content": [{"type": "text", "text": failed}], "isError": True})
        case _:
            # A refusal is a result like any other: Claude Code turns an isError result into an error of its own
            # wording, out of which the brain's stage could not read the refusal back.
            return _result(id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": False})


def _result(id: RequestId, result: object) -> Mapping[str, object]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: RequestId | None, code: int, message: str) -> Mapping[str, object]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}
