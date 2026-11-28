"""The process runs MLX on one thread, one piece of work at a time: the desk's and the phone's Smart Turns, loading
together as hands starts and judging turns from an executor thread each, never run on MLX side by side.

Two threads in MLX at once killed the daemon with a null pointer read in `eval_impl`, the desk's Smart Turn warming up on
its executor's thread while the phone's warmed up on its own (promptctl-voice-oalb.8vn). Each test holds one port inside
a verdict, as a model is held evaluating, gives the other's work while it is held, and sees it wait."""

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from pipecat.audio.turn.smart_turn import base_smart_turn

from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import engaged, mlxthread
from hands.voice.engaged import Models
from hands.voice.mlxthread import OnMLX

RATE = 16000
# 20 ms of a person speaking, as Silero says it is: what a verdict is asked on.
SPEECH = bytes(2 * RATE // 50)


@dataclass
class Ports:
    """Smart Turn's port for each analyzer, numbered in the order they load, as the test holds them: each load and each
    verdict's start and end, by port, and the threads they ran on; a port told to hold keeps its next verdict inside until
    the test lets it go."""

    seen: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    threads: set[int] = field(default_factory=set[int])
    holding: dict[int, threading.Event] = field(default_factory=dict[int, threading.Event])

    def ported(self, _on: OnMLX) -> "Port":
        number = sum(step == "loaded" for step, _ in self.seen)
        self.ran("loaded", number)
        return Port(self, number)

    def hold(self, number: int) -> threading.Event:
        """Hold port `number`'s next verdict inside it until the event is set."""
        self.holding[number] = threading.Event()
        return self.holding[number]

    def ran(self, step: str, number: int) -> None:
        self.seen.append((step, number))
        self.threads.add(threading.get_ident())


@dataclass
class Port:
    ports: Ports
    number: int

    def predict_endpoint(self, audio: np.ndarray, *, sample_rate: int) -> SimpleNamespace:
        self.ports.ran("in", self.number)
        held = self.ports.holding.pop(self.number, None)
        if held is not None:
            held.wait()
        self.ports.ran("out", self.number)
        return SimpleNamespace(prediction=0, probability=0.0)


async def until(done: Callable[[], bool]) -> None:
    """Until `done` holds, looked at again each turn of the loop, on no clock: a wait that never ends is the suite's
    timeout's to end."""
    while not done():
        await asyncio.sleep(0)


@pytest.fixture
def ports(monkeypatch: pytest.MonkeyPatch) -> Ports:
    ports = Ports()
    monkeypatch.setattr(engaged, "ported", ports.ported)
    return ports


@asynccontextmanager
async def ears(name: Literal["desk.loaded", "phone.loaded"], events: list[WideEvent]) -> AsyncGenerator[Models]:
    async with engaged.loaded(VirtualClock(), RATE, events.append, name) as models:
        yield models


async def loaded(name: Literal["desk.loaded", "phone.loaded"], events: list[WideEvent]) -> None:
    async with ears(name, events):
        pass


async def test_the_desks_and_the_phones_smart_turn_loading_as_hands_starts_run_on_mlx_one_after_the_other(ports: Ports) -> None:
    events = list[WideEvent]()
    # The desk's warm-up, held inside the port as the phone's load is given.
    let_go = ports.hold(0)
    loads = list[asyncio.Task[None]]()
    try:
        loads.append(asyncio.create_task(loaded("desk.loaded", events)))
        await until(lambda: ports.seen == [("loaded", 0), ("in", 0)])
        # Nothing else waits for the MLX thread, so the work that comes to wait next is the phone's load.
        assert mlxthread.waiting() == 0
        loads.append(asyncio.create_task(loaded("phone.loaded", events)))
        await until(lambda: mlxthread.waiting() == 1 or len(ports.seen) > 2)
        assert ports.seen == [("loaded", 0), ("in", 0)]
    finally:
        let_go.set()
    await asyncio.gather(*loads)
    assert ports.seen == [("loaded", 0), ("in", 0), ("out", 0), ("loaded", 1), ("in", 1), ("out", 1)]
    assert len(ports.threads) == 1
    assert sorted((event.event, event.outcome) for event in events) == [("desk.loaded", "ok"), ("phone.loaded", "ok")]


async def test_a_verdict_asked_while_the_other_analyzers_is_inside_mlx_waits_for_it(ports: Ports, monkeypatch: pytest.MonkeyPatch) -> None:
    events, clock = list[WideEvent](), VirtualClock()
    # Pipecat's analyzer keeps the speech it judges, and times its verdict, by the `time` module's clocks.
    monkeypatch.setattr(base_smart_turn, "time", SimpleNamespace(monotonic=clock.now, perf_counter=clock.now))
    async with ears("desk.loaded", events) as desk, ears("phone.loaded", events) as phone:
        warmed = len(ports.seen)
        # Pipecat asks each analyzer's verdict on that analyzer's own executor thread.
        let_go = ports.hold(0)
        judging = list[asyncio.Task[object]]()
        for models in (desk, phone):
            models.heard(SPEECH, True)
        try:
            judging.append(asyncio.create_task(desk.judge()))
            await until(lambda: ports.seen[warmed:] == [("in", 0)])
            assert mlxthread.waiting() == 0
            judging.append(asyncio.create_task(phone.judge()))
            await until(lambda: mlxthread.waiting() == 1 or len(ports.seen) > warmed + 1)
            assert ports.seen[warmed:] == [("in", 0)]
        finally:
            let_go.set()
        await asyncio.gather(*judging)
    assert ports.seen[warmed:] == [("in", 0), ("out", 0), ("in", 1), ("out", 1)]
    assert len(ports.threads) == 1
