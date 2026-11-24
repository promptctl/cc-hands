"""The wake word: saying it opens a turn with no key, end-of-turn detection closes it, and the detector hears the
room while hands speaks as while it is quiet."""

import ast
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from conftest import hearing_with, model_facts
from speech import quiet, spoken
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.clock import SYSTEM
from hands.sessions.wide import WideEvent, begun, unit
from hands.fanout import Fanout
from hands.voice.desk import AnswerWindowOpened, Leveled, Made, Served, Woken
from hands.voice.handsvoice import HandsVoice
from hands.voice.engaged import drive
from hands.voice.listening import Levels
from hands.voice.shortcut import KeyEvent, Move
from hands.voice.engaged import loaded as ears_loaded
from hands.voice.wake import SAMPLE_RATE, WakeWord, listening, loaded
from hands.voice.wakemodels import PRETRAINED, Files, WakeModels
from hands.voice.wakeword import THRESHOLD, Pretrained, Trained, Word


class ScriptedEars:
    """Silero and Smart Turn as a script: each buffer names the detector's state, and every verdict is complete."""

    async def detect(self, audio: bytes) -> VADState:
        return {b"q": VADState.QUIET, b"S": VADState.SPEAKING, b"w": VADState.SPEAKING}[audio]

    def heard(self, audio: bytes, speech: bool) -> bool:
        return False

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        return True, TurnMetricsData(processor="test", is_complete=True, probability=0.9, e2e_processing_time_ms=40.0)

    def clear(self) -> None:
        pass

    def afresh(self) -> None:
        pass


@asynccontextmanager
async def untapped(into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
    yield


async def wake_word_level() -> Levels:
    # The run starts at the wake word, which the desk listens at once its models are loaded.
    return Levels("wake word")


class Serving:
    """What the desk does, served: each move it makes, and each level it arrives at, the audio heard once it is at the
    wake word's. Hands never speaks here, so no answer window opens."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.made: list[Move] = []

    async def __call__(self, act: Served) -> None:
        match act:
            case Made(move=move):
                self.made.append(move)
            case Leveled():
                self.reached.set()
            case AnswerWindowOpened():
                raise AssertionError(f"an answer window opened though hands never spoke: {act}")


async def test_the_wake_word_opens_turns_with_no_key_and_off_stops_the_desk_listening() -> None:
    audio: asyncio.Queue[bytes] = asyncio.Queue()
    served = Serving()
    events: list[WideEvent] = []

    @asynccontextmanager
    async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            await served.reached.wait()
            while True:
                yield await audio.get()

        yield buffers()

    async def woken(heard: bytes, begun: bool) -> Woken | None:
        return Woken(SYSTEM.now(), over_hands=False, built_in=False) if heard == b"w" else None

    levels = await wake_word_level()
    driving = asyncio.create_task(drive(SYSTEM, untapped, overheard, hearing_with(ScriptedEars(), woken), Fanout[HandsVoice]().subscribed, levels, served, events.append))
    # Talk in the room opens nothing; the wake word opens a turn, the pause after it ends nothing, and the verdict on
    # the stop after what was asked sends it; and again.
    for each in (b"S", b"q", b"w", b"q", b"S", b"q", b"S", b"q", b"w", b"q", b"S", b"q"):
        audio.put_nowait(each)
    async with asyncio.timeout(1.0):
        while len(served.made) < 7:
            await asyncio.sleep(0.001)
    deafening = asyncio.create_task(levels.choose("off", "tool"))
    async with asyncio.timeout(1.0):
        while len(served.made) < 8:
            await asyncio.sleep(0.001)
    assert await deafening == "wake word"
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert served.made == ["listen", "arm", "start", "stop", "arm", "start", "stop", "deafen"]
    desk = [event for event in events if event.event == "desk"]
    assert [(event.facts["level"], event.outcome, dict(event.counts)["start"], dict(event.counts)["by_wake_word"]) for event in desk] == [
        ("off", "ok", 0, 1), ("wake word", "ok", 2, 7), ("off", "cancelled", 0, 0),
    ]


class FakeWord(WakeWord):
    """The wake word's model as a script: it is sure of the wake word in any buffer that is not silence."""

    def __init__(self) -> None:
        self.heard: list[bytes] = []
        self.resets = 0

    def score(self, audio: bytes) -> float:
        self.heard.append(audio)
        return 0.0 if audio.count(0) == len(audio) else 0.9

    def reset(self) -> None:
        self.resets += 1


def test_the_detector_hears_the_room_while_hands_speaks_and_each_wake_says_whether_it_was_over_hands_on_the_macs_own_pair() -> None:
    word = FakeWord()
    # Whether hands' voice may be in the second the turn would keep, and whether the desk is on the Mac's own pair.
    now = [(True, False)]
    events: list[WideEvent] = []
    woken = listening(SYSTEM, word, lambda: now[0][0], lambda: now[0][1], events.append)

    async def heard() -> list[tuple[bool, bool] | None]:
        with unit("desk", events.append, begun(SYSTEM), ("wake_words",)):
            echoes: list[tuple[bool, bool] | None] = []
            for facts, audio in [((True, False), b"\x01\x02" * 160), ((False, True), b"\x01\x02" * 160), ((True, True), b"\x01\x02" * 160), ((True, True), bytes(320))]:
                now[0] = facts
                wake = await woken(audio, False)
                echoes.append(None if wake is None else (wake.over_hands, wake.built_in))
            return echoes

    # Each wake says what the two facts were as it was heard, for the desk to take it by; silence is no wake at all.
    assert asyncio.run(heard()) == [(True, False), (False, True), (True, True), None]
    assert word.resets == 3
    hearings = [event for event in events if event.event == "desk.wake_word_heard"]
    assert [(event.facts["over_hands"], event.facts["built_in"]) for event in hearings] == [(True, False), (False, True), (True, True)]
    assert dict(events[-1].counts) == {"wake_words": 3}
    assert {event.parent_id for event in hearings} == {events[-1].span_id}


async def files_of(models: Path, word: Word) -> Files:
    """The files `word` is heard with, from the models the run fetched."""
    return (await WakeModels(models, SYSTEM).ensured(word)).files


def said(asked: str) -> bytes:
    """What is `asked`, with a second of silence either side."""
    return quiet(1.0) + spoken(asked) + quiet(1.0)


def wakes(word: WakeWord, audio: bytes) -> bool:
    """Whether `word` wakes on `audio`, heard as the desk's microphone hands it over, 20 ms at a time."""
    word.reset()
    return any(word.score(audio[start : start + 640]) >= 0.5 for start in range(0, len(audio), 640))


@pytest.mark.outside
async def test_the_model_wakes_on_hey_jarvis_and_not_on_jarvis_named_in_passing(models: Path) -> None:
    events: list[WideEvent] = []
    async with loaded(SYSTEM, SAMPLE_RATE, await files_of(models, Pretrained()), Pretrained(), events.append) as word:
        assert wakes(word, said("Hey Jarvis. What is the status of the build?"))
        assert not wakes(word, said("I was talking with Jarvis earlier about the build, and it was fine."))
    assert [(event.event, event.outcome, model_facts(event)["phrase"]) for event in events] == [("desk.wake_word_loaded", "ok", "Hey Jarvis")]


@pytest.mark.outside
async def test_another_wake_word_wakes_on_itself_and_not_on_hey_jarvis(models: Path) -> None:
    events: list[WideEvent] = []
    async with loaded(SYSTEM, SAMPLE_RATE, await files_of(models, Pretrained("Hey Mycroft")), Pretrained("Hey Mycroft"), events.append) as word:
        assert wakes(word, said("Hey Mycroft. What is the status of the build?"))
        assert not wakes(word, said("Hey Jarvis. What is the status of the build?"))
    assert [(event.facts["phrase"], event.facts["model"]) for event in events] == [("Hey Mycroft", str(models / PRETRAINED["Hey Mycroft"].name))]


@pytest.mark.outside
async def test_a_wake_word_the_user_trained_is_heard_with_their_model_wherever_they_keep_it(tmp_path: Path, models: Path) -> None:
    # openWakeWord's Hey Mycroft model stands in for one the user trained, under a name of their own.
    trained = tmp_path / "mine" / "hey_computer.onnx"
    trained.parent.mkdir()
    trained.write_bytes((models / PRETRAINED["Hey Mycroft"].name).read_bytes())
    events: list[WideEvent] = []
    word_trained = Trained("Hey Computer", trained)
    async with loaded(SYSTEM, SAMPLE_RATE, await files_of(models, word_trained), word_trained, events.append) as word:
        assert wakes(word, said("Hey Mycroft. What is the status of the build?"))
    assert [(event.facts["phrase"], event.facts["model"]) for event in events] == [("Hey Computer", str(trained))]


async def test_a_microphone_at_another_rate_is_refused_out_loud() -> None:
    events: list[WideEvent] = []
    with pytest.raises(ValueError, match="16000 Hz"):
        async with loaded(SYSTEM, 48_000, Files(Path("unread"), Path("unread"), Path("unread")), Pretrained(), events.append):
            pass
    assert [(event.event, event.outcome) for event in events] == [("desk.wake_word_loaded", "failed")]


@pytest.mark.outside
@pytest.mark.parametrize(
    "asked",
    [
        "Hey Jarvis what is the status of the build",
        "Hey Jarvis, what is the status of the build?",
        "Hey Jarvis. [[slnc 1500]] What is the status of the build?",
        # A one-word ask.
        "Hey Jarvis stop",
    ],
)
async def test_what_is_asked_after_the_wake_word_is_judged_whether_asked_in_one_breath_or_after_a_pause(asked: str, models: Path) -> None:
    """Silero, Smart Turn, and the wake word's model, real, on macOS's own voice: Smart Turn's verdict on what was asked
    sends the turn, not the silence after it, and not the pause after the wake word."""
    audio = said(asked) + quiet(4.0)
    served = Serving()
    events: list[WideEvent] = []

    @asynccontextmanager
    async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            await served.reached.wait()
            for start in range(0, len(audio), 640):
                yield audio[start : start + 640]
            await asyncio.Event().wait()

        yield buffers()

    async with ears_loaded(SYSTEM, SAMPLE_RATE, events.append, "desk.loaded") as ears, loaded(SYSTEM, SAMPLE_RATE, await files_of(models, Pretrained()), Pretrained(), events.append) as word:
        driving = asyncio.create_task(drive(SYSTEM, untapped, overheard, hearing_with(ears, listening(SYSTEM, word, lambda: False, lambda: False, events.append)), Fanout[HandsVoice]().subscribed, await wake_word_level(), served, events.append))
        async with asyncio.timeout(60.0):
            while "stop" not in served.made:
                await asyncio.sleep(0.01)
        driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driving
    assert served.made == ["listen", "arm", "start", "stop"]
    # The verdict before the silence it spared: a turn that ended on silence was judged incomplete, by the probability
    # its facts carry.
    judged = [event.facts for event in events if event.event == "desk.judged"]
    assert [facts["complete"] for facts in judged] == [True], judged
    [desk] = [event for event in events if event.event == "desk" and event.facts["level"] == "wake word"]
    assert dict(desk.counts)["ended_on_silence"] == 0


def test_wake_training_scores_its_models_at_the_threshold_hands_wakes_at() -> None:
    # wake-training does not import hands, so it keeps hands' threshold as a copy; this is the copy's check.
    source = Path(__file__).parent.parent / "wake-training" / "src" / "wake_training" / "evaluate.py"
    copied = next(
        node.value for node in ast.parse(source.read_text()).body if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["THRESHOLD"]
    )
    assert ast.literal_eval(copied) == THRESHOLD
