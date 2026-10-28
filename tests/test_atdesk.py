"""The desk served as the daemon serves it: the talk key's events through the tap's own handling, the gate, the level's
owner and the brain's tools, the speaker's cues, and Silero, Smart Turn, and the wake word's model, real, on macOS's own
voice. macOS's key tap, the microphone's device, and the speaker's device are stood in for."""

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from pathlib import Path

import pytest
from conftest import quiet, patiently, spoken

from hands.sessions.audit import Entry, Played
from hands.sessions.wide import WideEvent
from hands.voice.atdesk import serve_desk
from hands.voice.cues import ENGAGED, OFF, WAKE_WORD, Cue
from hands.voice.hold import HOLD_SECONDS, CommandPressed, CommandReleased, KeyEvent, Move, Pressed, Released, Typed
from hands.voice.keys import tapped
from hands.voice.ptt import PushToTalk
from hands.voice.tools import listening_tools
from hands.voice.trigger import Levels
from hands.voice.wake import SAMPLE_RATE
from hands.voice.wakeword import Pretrained

# Twenty milliseconds of the desk's microphone, as it hands audio over.
BUFFER = 640


class Speaker:
    """The speaker attached: it keeps each cue played, and hands is never speaking."""

    def __init__(self) -> None:
        self.cues: list[Cue] = []

    @property
    def hands_speaking(self) -> bool:
        return False

    def cue(self, cue: Cue) -> Played:
        self.cues.append(cue)
        return "desk"


class Microphone:
    """The desk's microphone, heard from a queue the test fills, in the buffers the real one hands over."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.audio: asyncio.Queue[bytes] = asyncio.Queue()

    @asynccontextmanager
    async def overheard(self) -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            while True:
                yield await self.audio.get()

        yield buffers()

    def say(self, audio: bytes) -> None:
        for start in range(0, len(audio), BUFFER):
            self.audio.put_nowait(audio[start : start + BUFFER].ljust(BUFFER, b"\0"))


class Keyboard:
    """macOS's key tap, stood in for: each key the test presses is heard on a thread of its own, as the tap hears it."""

    def __init__(self) -> None:
        self._heard: Callable[[KeyEvent], None] | None = None

    @property
    def listening(self) -> bool:
        return self._heard is not None

    def tap(self, heard: Callable[[KeyEvent], None]) -> Callable[[], None]:
        self._heard = heard
        return lambda: None

    def tapped(self, into: Callable[[KeyEvent], None]) -> AbstractAsyncContextManager[None]:
        return tapped(into, self.tap)

    def press(self, *events: KeyEvent) -> None:
        heard = self._heard
        assert heard is not None, "the desk reads the key from the start"
        for event in events:
            thread = threading.Thread(target=heard, args=(event,))
            thread.start()
            thread.join()

    def triple_tap(self, command: bool) -> None:
        for _ in range(3):
            self.press(*((CommandPressed(time.monotonic()), CommandReleased()) if command else (Pressed(time.monotonic()), Released())))


async def test_the_desk_served_talks_by_the_key_switches_by_taps_and_tools_and_hears_each_level(models: Path, tmp_path: Path) -> None:
    events: list[WideEvent] = []
    entries: list[Entry] = []
    key, levels, speaker, microphone, keyboard = PushToTalk(entries.append), Levels(), Speaker(), Microphone(), Keyboard()
    taken: list[Move] = []
    beats: list[None] = []

    async def took(move: Move) -> None:
        taken.append(move)

    # Rendered before the desk runs, so macOS's `say` never holds the loop the desk, the tap, and its timers run on.
    off, wake, engaged, deaf = (spoken(said, tmp_path / f"{name}.wav") for name, said in [("off", "create a ticket to read session history."), ("wake", "Hey Jarvis. What is the status of the build?"), ("engaged", "create a ticket to read session history."), ("deaf", "create a ticket to read session history.")])

    level, choose = listening_tools(levels, Pretrained())
    serving = asyncio.create_task(serve_desk(key=key, levels=levels, tapped=keyboard.tapped, microphone=microphone, speaker=lambda: speaker, models=models, word=Pretrained(), taken=took, beat=partial(beats.append, None), record=events.append))

    async def until(done: Callable[[], bool]) -> None:
        # A desk that ended is the failure itself, raised, never a wait that runs out on it.
        await patiently(lambda: serving.done() or done(), lambda: (len(taken), len(speaker.cues), microphone.audio.qsize(), len(events)))
        if serving.done():
            serving.result()

    async def settled() -> None:
        # The desk steps a choice only once everything queued before it is stepped, so a choice of the level it is at
        # returns once every buffer and key heard so far has been stepped.
        await until(lambda: microphone.audio.empty())
        assert await levels.choose(levels.level) == levels.level

    await until(lambda: keyboard.listening)

    # Off: the room opens nothing, and Right Shift held alone talks, opened once it is held long enough.
    microphone.say(off + quiet(3.0))
    keyboard.press(Pressed(time.monotonic()))
    await until(lambda: "start" in taken)
    keyboard.press(Released())
    await until(lambda: taken[-1] == "stop")
    assert taken == ["arm", "start", "stop"]
    await settled()
    assert taken == ["arm", "start", "stop"] and key.gate.key == "up"

    # Right Shift typing a capital opens nothing.
    keyboard.press(Pressed(time.monotonic()), Typed())
    # Held past the hold, so its Ripe has been told: still nothing opens.
    await asyncio.sleep(HOLD_SECONDS * 2)
    await settled()
    assert taken[3:] == ["arm", "disarm"]
    keyboard.press(Released())
    await settled()
    assert taken[3:] == ["arm", "disarm"]

    # Right Command's triple tap: the wake word, cued, the level its owner and the tool say, and the desk listening.
    keyboard.triple_tap(command=True)
    await until(lambda: levels.level == "wake word")
    assert speaker.cues == [WAKE_WORD] and key.gate.key == "listening"
    assert str((await level.body())["readback"]).startswith("The wake word: say Hey Jarvis")
    await settled()
    del taken[:]

    # Said, the wake word opens a turn, and Smart Turn's verdict on what was asked sends it.
    microphone.say(quiet(0.5) + wake + quiet(4.0))
    await until(lambda: "stop" in taken)
    assert taken[:3] == ["arm", "start", "stop"]

    # The tool chooses engaged as a tap does, and is answered with the level the desk was at.
    await settled()
    chosen = await choose.body(level="engaged")
    assert chosen["was"] == "wake word"
    # The choice is answered as it is dequeued, before its step cues it.
    await until(lambda: len(speaker.cues) == 2)
    assert speaker.cues == [WAKE_WORD, ENGAGED] and levels.level == "engaged" and (await level.body())["level"] == "engaged"
    await settled()
    del taken[:]

    # Engaged, speech alone opens a turn, and its end sends it.
    microphone.say(quiet(0.5) + engaged + quiet(4.0))
    await until(lambda: "stop" in taken)
    assert "start" in taken and taken.index("start") < taken.index("stop")

    # Right Shift's triple tap from engaged: off, cued, and the desk deaf to the room again.
    await settled()
    keyboard.triple_tap(command=False)
    await until(lambda: levels.level == "off")
    assert speaker.cues == [WAKE_WORD, ENGAGED, OFF] and key.gate.key == "up"
    await settled()
    del taken[:]
    microphone.say(deaf + quiet(3.0))
    await settled()
    assert taken == []

    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving
    assert len(beats) == 3, "each change of level is beaten, so the indicator reads it at once"
    fetched = [event for event in events if event.event == "desk.fetched"]
    assert [(event.outcome, event.facts["fetched"]) for event in fetched] == [("ok", "")]
    leveled = [(event.facts["was"], event.facts["now"], event.facts["by"]) for event in events if event.event == "desk.leveled"]
    assert leveled == [("off", "wake word", "tap"), ("wake word", "engaged", "tool"), ("engaged", "off", "tap")]
