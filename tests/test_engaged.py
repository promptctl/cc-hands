"""The desk's driver, engaged: a triple tap engages, the user's voice opens each turn, end-of-turn detection closes it,
and another triple tap disengages."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import pytest
from conftest import hearing_with
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.virtualclock import VirtualClock
from hands.sessions.wide import WideEvent
from hands.voice.desk import Level, Leveled, Made
from hands.voice.engaged import Ears, Hears, drive
from hands.voice.trigger import Choice
from hands.voice.hold import KeyEvent, Move, Pressed, Released, Ripe

# Right Shift's triple tap: what engages, and what disengages.
TAP: tuple[KeyEvent, ...] = tuple(e for i in range(3) for e in (Pressed(1.0 + i / 10), Released()))
# What the gate is handed for the taps themselves: each press arms the microphone and its release disarms it.
TAPPED: list[Move] = ["arm", "disarm", "arm", "disarm", "arm"]


class ScriptedEars:
    """The models, as a script: each buffer of audio names the detector's state, and the verdicts are given in turn."""

    def __init__(self, verdicts: Sequence[bool]) -> None:
        self._verdicts = list(verdicts)
        self.cleared = 0
        # Set to keep Smart Turn thinking until it is cleared, as audio goes on being heard.
        self.thinking: asyncio.Event | None = None

    async def detect(self, audio: bytes) -> VADState:
        return {b"q": VADState.QUIET, b"s": VADState.STARTING, b"S": VADState.SPEAKING, b"x": VADState.QUIET}[audio]

    def heard(self, audio: bytes, speech: bool) -> bool:
        # b"x" is quiet run past Smart Turn's stop_secs.
        return audio == b"x"

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        if self.thinking is not None:
            await self.thinking.wait()
        complete = self._verdicts.pop(0)
        return complete, TurnMetricsData(processor="test", is_complete=complete, probability=0.9 if complete else 0.1, e2e_processing_time_ms=40.0)

    def clear(self) -> None:
        self.cleared += 1

    def afresh(self) -> None:
        pass


class Rig:
    """The desk's driver with a fake talk key and a fake microphone."""

    def __init__(self, verdicts: Sequence[bool]) -> None:
        self.ears = ScriptedEars(verdicts)
        self.keys: list[Callable[[KeyEvent], None]] = []
        self.audio: asyncio.Queue[bytes] = asyncio.Queue()
        self.made: list[Move] = []
        self.made_by: list[Made] = []
        self.levels: list[Leveled] = []
        self.chosen: asyncio.Queue[Choice] = asyncio.Queue()
        self.events: list[WideEvent] = []
        self.clock = VirtualClock()
        # Told each time the driver makes a move or takes a buffer, so a test waits on what it did, not on time passing.
        self.changed = asyncio.Condition()

    async def _told(self) -> None:
        async with self.changed:
            self.changed.notify_all()

    async def until(self, done: Callable[[], bool]) -> None:
        async with self.changed:
            await self.changed.wait_for(done)

    @asynccontextmanager
    async def tapped(self, into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
        self.keys.append(into)
        yield

    @asynccontextmanager
    async def overheard(self) -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            while True:
                each = await self.audio.get()
                await self._told()
                yield each

        yield buffers()

    async def on_move(self, made: Made) -> None:
        self.made_by.append(made)
        self.made.append(made.move)
        await self._told()

    async def on_level(self, leveled: Leveled) -> None:
        self.levels.append(leveled)

    def start(self, ears: Ears | None = None) -> asyncio.Task[None]:
        return asyncio.create_task(drive(self.clock, self.tapped, self.overheard, hearing_with(ears or self.ears, unwoken), self.chosen, self.on_move, self.on_level, self.events.append))

    async def tap(self) -> None:
        while not self.keys:
            await asyncio.sleep(0)
        for event in TAP:
            self.keys[0](event)

    async def choose(self, level: Level) -> Level:
        """Choose `level` as the tool does, and get back the level the driver stepped it from."""
        was: asyncio.Future[Level] = asyncio.get_running_loop().create_future()
        self.chosen.put_nowait(Choice(level, was))
        return await was

    async def hear(self, *audio: bytes) -> None:
        for each in audio:
            self.audio.put_nowait(each)

    async def settle(self, made: int) -> None:
        await self.until(lambda: len(self.made) >= made)
        for _ in range(20):
            await asyncio.sleep(0)

    async def drained(self) -> None:
        await self.until(lambda: not self.audio.qsize())


async def unwoken(audio: bytes, begun: bool) -> bool:
    return False


async def stopped(driving: asyncio.Task[None]) -> None:
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving


async def test_three_turns_with_no_key_between_them_then_a_disengage() -> None:
    rig = Rig(verdicts=[False, True, True, False])
    driving = rig.start()
    await rig.tap()
    # A pause mid-thought held open, then the thought finished; a second turn; a third held open by its verdict, then
    # ended by the silence after it running on.
    await rig.hear(b"s", b"S", b"q", b"s", b"S", b"q")
    await rig.settle(10)
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(13)
    await rig.hear(b"s", b"S", b"q", b"x")
    await rig.settle(16)
    await rig.tap()
    await rig.hear(b"s", b"S", b"q")
    for _ in range(20):
        await asyncio.sleep(0)
    await stopped(driving)
    assert rig.made == [*TAPPED, "disarm", "listen", *["arm", "start", "stop"] * 3, *TAPPED, "disarm", "deafen"]
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", "engaged", "tap"), ("engaged", "off", "tap")]
    # One event for each stint at a level, emitted as the level changes: the stint engaged counts its turns.
    stints = [event for event in rig.events if event.event == "desk"]
    assert [(event.facts["level"], event.outcome) for event in stints] == [("off", "ok"), ("engaged", "ok"), ("off", "cancelled")]
    desk = stints[1]
    counts = dict(desk.counts)
    assert {name: counts[name] for name in ("start", "stop", "held_open", "ended_on_silence", "by_engaged_conversation", "to_engaged", "to_off")} == {
        "start": 3, "stop": 3, "held_open": 2, "ended_on_silence": 1, "by_engaged_conversation": 10, "to_engaged": 0, "to_off": 1,
    }
    judged = [event for event in rig.events if event.event == "desk.judged"]
    assert [event.facts["complete"] for event in judged] == [False, True, True, False]
    assert all(event.parent_id == desk.span_id for event in judged)
    leveled = [event.facts for event in rig.events if event.event == "desk.leveled"]
    assert [(facts["was"], facts["now"], facts["by"]) for facts in leveled] == [("off", "engaged", "tap"), ("engaged", "off", "tap")]


async def test_off_the_room_opens_nothing_and_the_held_key_still_talks() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    await rig.hear(b"s", b"S", b"q", b"x")
    while not rig.keys:
        await asyncio.sleep(0)
    for event in (Pressed(1.0), Ripe(1.0), Released()):
        rig.keys[0](event)
    await rig.settle(3)
    await stopped(driving)
    assert rig.made_by == [Made("arm", "held key"), Made("start", "held key"), Made("stop", "held key")]


async def test_a_level_the_tool_chose_is_stepped_as_a_tap_is() -> None:
    rig = Rig(verdicts=[True])
    driving = rig.start()
    was = [await rig.choose("engaged")]
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(4)
    was += [await rig.choose("engaged"), await rig.choose("off")]
    await rig.settle(5)
    await stopped(driving)
    assert rig.made == ["listen", "arm", "start", "stop", "deafen"]
    # The level each choice found, as the driver stepped it; one chosen again changes nothing.
    assert was == ["off", "engaged", "engaged"]
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", "engaged", "tool"), ("engaged", "off", "tool")]


async def test_the_key_talks_and_a_choice_is_stepped_while_the_models_load() -> None:
    rig = Rig(verdicts=[])

    @asynccontextmanager
    async def loading() -> AsyncGenerator[tuple[Ears, Hears]]:
        await asyncio.Event().wait()
        yield rig.ears, unwoken

    driving = asyncio.create_task(drive(rig.clock, rig.tapped, rig.overheard, loading, rig.chosen, rig.on_move, rig.on_level, rig.events.append))
    while not rig.keys:
        await asyncio.sleep(0)
    for event in (Pressed(1.0), Ripe(1.0), Released()):
        rig.keys[0](event)
    await rig.settle(3)
    assert await rig.choose("engaged") == "off"
    await stopped(driving)
    assert rig.made_by == [Made("arm", "held key"), Made("start", "held key"), Made("stop", "held key"), Made("listen", "engaged conversation")]


async def test_a_model_that_cannot_be_loaded_fails_the_drive() -> None:
    rig = Rig(verdicts=[])

    @asynccontextmanager
    async def unloadable() -> AsyncGenerator[tuple[Ears, Hears]]:
        raise OSError("no Smart Turn in the hub's cache, and no network")
        yield rig.ears, unwoken

    with pytest.raises(ExceptionGroup) as failed:
        await drive(rig.clock, rig.tapped, rig.overheard, unloadable, rig.chosen, rig.on_move, rig.on_level, rig.events.append)
    assert failed.group_contains(OSError, match="no network")


async def test_a_choice_whose_tool_call_was_cancelled_is_stepped_and_the_desk_goes_on() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    cancelled: asyncio.Future[Level] = asyncio.get_running_loop().create_future()
    cancelled.cancel()
    rig.chosen.put_nowait(Choice("engaged", cancelled))
    assert await rig.choose("off") == "engaged"
    await stopped(driving)
    assert rig.made == ["listen", "deafen"]


async def test_each_level_hears_from_where_it_started() -> None:
    rig = Rig(verdicts=[])
    begun: list[bool] = []

    async def woken(audio: bytes, afresh: bool) -> bool:
        begun.append(afresh)
        return False

    driving = asyncio.create_task(drive(rig.clock, rig.tapped, rig.overheard, hearing_with(rig.ears, woken), rig.chosen, rig.on_move, rig.on_level, rig.events.append))
    # Off, nothing hears the room; the wake word's model hears from its level's first buffer, afresh each time it starts.
    await rig.hear(b"q")
    for level in ("wake word", "off", "wake word"):
        await rig.choose(level)
        await rig.hear(b"q", b"q")
        await rig.drained()
        for _ in range(20):
            await asyncio.sleep(0)
    await stopped(driving)
    assert begun == [True, False, True, False]


async def test_speech_begun_under_a_tap_of_shift_opens_a_turn_once_the_key_is_up() -> None:
    rig = Rig(verdicts=[True])
    driving = rig.start()
    await rig.choose("engaged")
    rig.keys[0](Pressed(1.0))
    await rig.hear(b"s", b"S")
    await rig.drained()
    rig.keys[0](Released())
    await rig.hear(b"S", b"q")
    await rig.settle(6)
    await stopped(driving)
    assert rig.made == ["listen", "arm", "disarm", "arm", "start", "stop"]


async def test_a_verdict_ends_the_turn_it_judged_however_long_smart_turn_thinks() -> None:
    rig = Rig(verdicts=[True, True])
    rig.ears.thinking = asyncio.Event()
    driving = rig.start()
    await rig.choose("engaged")
    # A pause is judged; while Smart Turn thinks, the user speaks again, and the silence after that runs on.
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(3)
    await rig.hear(b"s", b"S", b"q", b"x")
    for _ in range(20):
        await asyncio.sleep(0)
    rig.ears.thinking.set()
    await rig.settle(6)
    await stopped(driving)
    # The first verdict sends the first turn; the speech heard while it was given opens the next, which is judged too.
    assert rig.made[:7] == ["listen", "arm", "start", "stop", "arm", "start", "stop"]
