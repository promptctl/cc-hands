"""The brain on the phone's page: what it is doing now, live, and one Stop that ends its turn in flight.

Served beside the phone's page, on its port and under its key (`hands.voice.phonepage`, `hands.voice.keyed`). What the
brain is doing is its stage's to say (`hands.brain.stage.BrainStage.work`), and stopping it is its stage's to do: nothing
here reaches a working session.
"""

import json
from typing import Literal, Protocol, TypedDict

from aiohttp import web

from hands.clock import Clock
from hands.core.brainwork import AlreadyEnded, BrainWork, Idle, NotWorking, Running, StopOutcome, Stopping, Working
from hands.core.wire import Unreadable
from hands.sessions.audit import Record
from hands.sessions.payload import Rejected
from hands.sessions.wide import annotate, count, fail
from hands.voice.keyed import keyed
from hands.voice.longpoll import version_of, waited


class Watched(Protocol):
    """The brain as the page sees it: what it is doing, and its turn stopped."""

    def work(self) -> BrainWork: ...

    def stop(self, turn: str) -> StopOutcome: ...


class IdleShown(TypedDict):
    doing: Literal["idle"]
    calls: list[dict[str, object]]


class WorkingShown(TypedDict):
    doing: Literal["working"]
    turn: str
    since: float
    calls: list[dict[str, object]]


def shown(work: BrainWork) -> IdleShown | WorkingShown:
    """The brain's work as the page draws it: its calls are whatever it is running, none at all while it is idle
    [LAW:dataflow-not-control-flow]."""
    match work:
        case Idle():
            return {"doing": "idle", "calls": []}
        case Working(turn=turn, since=since, calls=calls):
            return {"doing": "working", "turn": turn, "since": since, "calls": [_call(call) for call in calls]}


def _call(call: Running) -> dict[str, object]:
    match call.input:
        case Unreadable(reason=reason):
            return {"tool": call.tool, "since": call.since, "unreadable": reason}
        case given:
            return {"tool": call.tool, "since": call.since, "input": given}


def readback(outcome: StopOutcome) -> str:
    """What the page says of a Stop."""
    match outcome:
        case Stopping():
            return "Stopping the brain's turn."
        case NotWorking():
            return "The brain is not working on anything."
        case AlreadyEnded():
            return "That turn had already ended."


def parse_stop(body: bytes) -> str:
    """The turn a page's Stop names, as its request's body; raises Rejected naming what is wrong with it."""
    try:
        sent: object = json.loads(body)
    except ValueError as error:
        # Not JSON, or not text at all: JSONDecodeError and UnicodeDecodeError are both ValueErrors.
        raise Rejected(f"a Stop is JSON: {error}") from error
    match sent:
        case {"turn": str(turn)} if turn:
            return turn
        case _:
            raise Rejected("a Stop is {turn}, the turn the page was shown")


def brain_routes(clock: Clock, brain: Watched, key: str, record: Record) -> list[web.RouteDef]:
    """The read of what the brain is doing, and a Stop.

    [LAW:nothing-unseen] each read is one `brain.read` event: what the brain was doing and how many calls it was running,
    what the page had seen, how long the read waited for that to change and how many looks that took; each Stop one
    `brain.stop`: the turn it named and what came of it, or why it was refused, which fails it.
    """

    async def look() -> tuple[str, IdleShown | WorkingShown]:
        # [LAW:effects-at-boundaries] a look reads what the stage holds in memory, as the page draws it, at every poll.
        work = brain.work()
        drawn = shown(work)
        return version_of(drawn), drawn

    @keyed(clock, "brain.read", record, key, counts=("calls",))
    async def read(request: web.Request) -> web.Response:
        read = await waited(clock, look, request.query.get("seen"))
        annotate(doing=read["shown"]["doing"])
        count(calls=len(read["shown"]["calls"]))
        return web.json_response(read)

    @keyed(clock, "brain.stop", record, key)
    async def stop(request: web.Request) -> web.Response:
        try:
            turn = parse_stop(await request.read())
        except Rejected as error:
            fail(str(error))
            return web.Response(status=400, text=str(error))
        outcome = brain.stop(turn)
        annotate(turn=turn, outcome=type(outcome).__name__)
        return web.json_response({"readback": readback(outcome)})

    return [web.get("/brain", read), web.post("/brain/stop", stop)]
