"""The desk's driver, engaged: a triple tap engages, the user's voice opens each turn, end-of-turn detection closes it,
and another triple tap disengages."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from conftest import hearing_with, model_facts, spoken
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent
from hands.voice.desk import Level, Leveled, Made
from hands.voice.engaged import SMART_TURN, SMART_TURN_REVISION, STOP_SECS, Choice, Ears, drive, loaded
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
        raise AssertionError("engaged conversation has no wake word to hear afresh from")


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

    @asynccontextmanager
    async def tapped(self, into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
        self.keys.append(into)
        yield

    @asynccontextmanager
    async def overheard(self) -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            while True:
                yield await self.audio.get()

        yield buffers()

    async def on_move(self, made: Made) -> None:
        self.made_by.append(made)
        self.made.append(made.move)

    async def on_level(self, leveled: Leveled) -> None:
        self.levels.append(leveled)

    def start(self, ears: Ears | None = None) -> asyncio.Task[None]:
        return asyncio.create_task(drive(self.tapped, self.overheard, hearing_with(ears or self.ears, unwoken), self.chosen, self.on_move, self.on_level, self.events.append))

    async def tap(self) -> None:
        while not self.keys:
            await asyncio.sleep(0)
        for event in TAP:
            self.keys[0](event)

    async def choose(self, level: Level) -> Level:
        """Choose `level` as the tool does, and get back the level the driver stepped it from."""
        was: asyncio.Future[Level] = asyncio.get_running_loop().create_future()
        self.chosen.put_nowait(Choice(level, was))
        async with asyncio.timeout(1.0):
            return await was

    async def hear(self, *audio: bytes) -> None:
        for each in audio:
            self.audio.put_nowait(each)

    async def settle(self, made: int) -> None:
        async with asyncio.timeout(1.0):
            while len(self.made) < made:
                await asyncio.sleep(0.001)
        for _ in range(20):
            await asyncio.sleep(0)


async def unwoken(audio: bytes) -> bool:
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
    async def loading() -> AsyncGenerator[tuple[Ears, Callable[[bytes], Awaitable[bool]]]]:
        await asyncio.Event().wait()
        yield rig.ears, unwoken

    driving = asyncio.create_task(drive(rig.tapped, rig.overheard, loading, rig.chosen, rig.on_move, rig.on_level, rig.events.append))
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
    async def unloadable() -> AsyncGenerator[tuple[Ears, Callable[[bytes], Awaitable[bool]]]]:
        raise OSError("no Smart Turn in the hub's cache, and no network")
        yield rig.ears, unwoken

    with pytest.raises(ExceptionGroup) as failed:
        await drive(rig.tapped, rig.overheard, unloadable, rig.chosen, rig.on_move, rig.on_level, rig.events.append)
    assert failed.group_contains(OSError, match="no network")


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


async def test_loading_the_models_is_its_own_unit_of_work() -> None:
    events: list[WideEvent] = []
    async with loaded(16000, events.append) as ears:
        assert await ears.detect(bytes(1024)) == VADState.QUIET
    assert [(event.event, event.outcome, model_facts(event)) for event in events] == [
        ("desk.loaded", "ok", {"smart_turn": SMART_TURN, "smart_turn_revision": SMART_TURN_REVISION, "vad_stop_secs": STOP_SECS})
    ]


def quiet(seconds: float) -> bytes:
    return bytes(round(16_000 * seconds) * 2)


@pytest.mark.parametrize(("said", "complete"), [("create a ticket to read session history.", True), ("can you tell me about the", False)])
async def test_smart_turn_judges_a_finished_sentence_complete_and_one_broken_off_not(said: str, complete: bool, tmp_path: Path) -> None:
    # As the edge gives it: the speech as speech, then the silence the detector ends it at, then the question.
    audio = spoken(said, tmp_path / "said.wav")
    async with loaded(16000, lambda event: None) as models:
        for at in range(0, len(audio), 640):
            models.heard(audio[at : at + 640], True)
        models.heard(quiet(STOP_SECS), False)
        assert (await models.judge())[0] is complete


async def test_a_pause_between_words_is_inside_the_turn_heard_by_the_real_models(tmp_path: Path) -> None:
    rig = Rig(verdicts=[])
    # Two phrases with a breath between them, then silence past Smart Turn's own stop: one turn, whatever it judges.
    audio = spoken("create a ticket", tmp_path / "asked.wav") + quiet(0.4) + spoken("to read session history", tmp_path / "more.wav") + quiet(4.0)
    async with loaded(16000, rig.events.append) as models:
        driving = rig.start(models)
        await rig.choose("engaged")
        await rig.hear(*(audio[at : at + 640] for at in range(0, len(audio), 640)))
        async with asyncio.timeout(10.0):
            # Disengaging is heard after every buffer, so the engagement ends once all of it is heard.
            while rig.audio.qsize():
                await asyncio.sleep(0.01)
            await rig.choose("off")
            while "deafen" not in rig.made:
                await asyncio.sleep(0.01)
        await stopped(driving)
    assert rig.made == ["listen", "arm", "start", "stop", "deafen"]
    # The turn ended on its own, by a verdict or by the silence after it, not by the level that disengaged.
    desk = next(event for event in rig.events if event.event == "desk" and event.facts["level"] == "engaged")
    verdicts = [event.facts["complete"] for event in rig.events if event.event == "desk.judged"]
    assert verdicts.count(True) + desk.counts["ended_on_silence"] == 1
