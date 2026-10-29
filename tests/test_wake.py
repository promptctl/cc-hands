"""The wake word: saying it opens a turn with no key, end-of-turn detection closes it, and while hands speaks the
detector hears silence."""

import ast
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.http_parser import HttpResponseParserPy, RawResponseMessage
from aiohttp.streams import StreamReader
from conftest import hearing_with, model_facts
from speech import quiet, spoken
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent, unit
from hands.voice.desk import Level, Leveled, Made
from hands.voice.engaged import drive
from hands.voice.trigger import Choice
from hands.voice.hold import KeyEvent, Move
from hands.voice.engaged import loaded as ears_loaded
from hands.voice.wake import EMBEDDING, MELSPECTROGRAM, SAMPLE_RATE, THRESHOLD, WakeWord, fetched, listening, loaded
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained


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


def choice(level: Level) -> Choice:
    return Choice(level, asyncio.get_running_loop().create_future())


async def wake_word_level() -> asyncio.Queue[Choice]:
    chosen: asyncio.Queue[Choice] = asyncio.Queue()
    chosen.put_nowait(choice("wake word"))
    return chosen


class AtLevel:
    """The desk's report of each level it arrives at: the audio is heard once it is at the wake word's."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()

    async def __call__(self, _leveled: Leveled) -> None:
        self.reached.set()


async def test_the_wake_word_opens_turns_with_no_key_and_off_stops_the_desk_listening() -> None:
    audio: asyncio.Queue[bytes] = asyncio.Queue()
    level = AtLevel()
    made: list[Move] = []
    events: list[WideEvent] = []

    @asynccontextmanager
    async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            await level.reached.wait()
            while True:
                yield await audio.get()

        yield buffers()

    async def woken(heard: bytes, begun: bool) -> bool:
        return heard == b"w"

    async def on_move(move: Made) -> None:
        made.append(move.move)

    chosen = await wake_word_level()
    driving = asyncio.create_task(drive(untapped, overheard, hearing_with(ScriptedEars(), woken), chosen, on_move, level, events.append))
    # Talk in the room opens nothing; the wake word opens a turn, the pause after it ends nothing, and the verdict on
    # the stop after what was asked sends it; and again.
    for each in (b"S", b"q", b"w", b"q", b"S", b"q", b"S", b"q", b"w", b"q", b"S", b"q"):
        audio.put_nowait(each)
    async with asyncio.timeout(1.0):
        while len(made) < 7:
            await asyncio.sleep(0.001)
    chosen.put_nowait(choice("off"))
    async with asyncio.timeout(1.0):
        while len(made) < 8:
            await asyncio.sleep(0.001)
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert made == ["listen", "arm", "start", "stop", "arm", "start", "stop", "deafen"]
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


def test_while_hands_speaks_the_detector_hears_silence_and_each_waking_is_an_event() -> None:
    word = FakeWord()
    speaking = [True]
    events: list[WideEvent] = []
    woken = listening(word, lambda: speaking[0], events.append)

    async def heard() -> list[bool]:
        with unit("desk", events.append, counts=("heard", "muted")):
            deaf = await woken(b"\x01\x02" * 160, False)
            speaking[0] = False
            return [deaf, await woken(b"\x01\x02" * 160, False)]

    assert asyncio.run(heard()) == [False, True]
    assert word.heard == [bytes(320), b"\x01\x02" * 160]
    assert word.resets == 1
    assert [(event.event, event.facts.get("score")) for event in events] == [("desk.wake_word_heard", 0.9), ("desk", None)]
    assert dict(events[1].counts) == {"heard": 1, "muted": 1}
    assert events[0].parent_id == events[1].span_id


def said(asked: str) -> bytes:
    """What is `asked`, with a second of silence either side."""
    return quiet(1.0) + spoken(asked) + quiet(1.0)


def wakes(word: WakeWord, audio: bytes) -> bool:
    """Whether `word` wakes on `audio`, heard as the desk's microphone hands it over, 20 ms at a time."""
    word.reset()
    return any(word.score(audio[start : start + 640]) >= 0.5 for start in range(0, len(audio), 640))


async def test_the_model_wakes_on_hey_jarvis_and_not_on_jarvis_named_in_passing(models: Path) -> None:
    events: list[WideEvent] = []
    async with loaded(SAMPLE_RATE, models, Pretrained(), events.append) as word:
        assert wakes(word, said("Hey Jarvis. What is the status of the build?"))
        assert not wakes(word, said("I was talking with Jarvis earlier about the build, and it was fine."))
    assert [(event.event, event.outcome, model_facts(event)["phrase"]) for event in events] == [("desk.wake_word_loaded", "ok", "Hey Jarvis")]


async def test_another_wake_word_wakes_on_itself_and_not_on_hey_jarvis(models: Path) -> None:
    events: list[WideEvent] = []
    async with loaded(SAMPLE_RATE, models, Pretrained("Hey Mycroft"), events.append) as word:
        assert wakes(word, said("Hey Mycroft. What is the status of the build?"))
        assert not wakes(word, said("Hey Jarvis. What is the status of the build?"))
    assert [(event.facts["phrase"], event.facts["model"]) for event in events] == [("Hey Mycroft", str(models / PRETRAINED["Hey Mycroft"]))]


async def test_a_wake_word_the_user_trained_is_heard_with_their_model_wherever_they_keep_it(tmp_path: Path, models: Path) -> None:
    # openWakeWord's Hey Mycroft model stands in for one the user trained, under a name of their own.
    trained = tmp_path / "mine" / "hey_computer.onnx"
    trained.parent.mkdir()
    trained.write_bytes((models / PRETRAINED["Hey Mycroft"]).read_bytes())
    events: list[WideEvent] = []
    async with loaded(SAMPLE_RATE, models, Trained("Hey Computer", trained), events.append) as word:
        assert wakes(word, said("Hey Mycroft. What is the status of the build?"))
    assert [(event.facts["phrase"], event.facts["model"]) for event in events] == [("Hey Computer", str(trained))]


async def test_a_microphone_at_another_rate_is_refused_out_loud() -> None:
    events: list[WideEvent] = []
    with pytest.raises(ValueError, match="16000 Hz"):
        async with loaded(48_000, Path("unread"), Pretrained(), events.append):
            pass
    assert [(event.event, event.outcome) for event in events] == [("desk.wake_word_loaded", "failed")]


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
    level = AtLevel()
    made: list[Move] = []
    events: list[WideEvent] = []

    @asynccontextmanager
    async def overheard() -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            await level.reached.wait()
            for start in range(0, len(audio), 640):
                yield audio[start : start + 640]
            await asyncio.Event().wait()

        yield buffers()

    async def on_move(move: Made) -> None:
        made.append(move.move)

    async with ears_loaded(SAMPLE_RATE, events.append) as ears, loaded(SAMPLE_RATE, models, Pretrained(), events.append) as word:
        driving = asyncio.create_task(drive(untapped, overheard, hearing_with(ears, listening(word, lambda: False, events.append)), await wake_word_level(), on_move, level, events.append))
        async with asyncio.timeout(60.0):
            while "stop" not in made:
                await asyncio.sleep(0.01)
        driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driving
    assert made == ["listen", "arm", "start", "stop"]
    # The verdict before the silence it spared: a turn that ended on silence was judged incomplete, by the probability
    # its facts carry.
    judged = [event.facts for event in events if event.event == "desk.judged"]
    assert [facts["complete"] for facts in judged] == [True], judged
    [desk] = [event for event in events if event.event == "desk" and event.facts["level"] == "wake word"]
    assert dict(desk.counts)["ended_on_silence"] == 0


JARVIS = PRETRAINED["Hey Jarvis"]


@asynccontextmanager
async def release(model: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> AsyncGenerator[str]:
    """A release served from this machine, each file in it answered by `model`: the address it is fetched from."""
    app = web.Application()
    app.router.add_get("/{name}", model)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


async def test_the_models_are_fetched_whole_or_not_at_all_and_never_twice(tmp_path: Path) -> None:
    served: list[str] = []

    async def model(request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        served.append(name)
        match name:
            case "embedding_model.onnx":
                raise web.HTTPNotFound()
            case "hey_jarvis_v0.1.onnx":
                # The connection drops with the body half sent.
                response = web.StreamResponse(headers={"Content-Length": "100"})
                await response.prepare(request)
                await response.write(b"half")
                request.transport.close() if request.transport else None
                return response
            case _:
                return web.Response(body=b"features")

    async with release(model) as address:
        with pytest.raises(aiohttp.ClientResponseError):
            await fetched(tmp_path, Pretrained(), address)
        assert sorted(path.name for path in tmp_path.iterdir()) == [MELSPECTROGRAM]
        (tmp_path / EMBEDDING).write_bytes(b"features")
        with pytest.raises(aiohttp.ClientPayloadError):
            await fetched(tmp_path, Pretrained(), address)
        assert not (tmp_path / JARVIS).exists()
        (tmp_path / JARVIS).write_bytes(b"word")
        assert await fetched(tmp_path, Pretrained(), address) == ()
        assert served == [MELSPECTROGRAM, EMBEDDING, JARVIS]


async def test_the_models_are_fetched_from_a_release_whose_headers_are_far_longer_than_githubs(tmp_path: Path) -> None:
    # GitHub's content-security-policy alone was 9243 bytes on 2026-10-23; this is three and a half times that.
    long = "x" * (32 * 1024)

    async def model(_request: web.Request) -> web.Response:
        return web.Response(body=b"features", headers={"Content-Security-Policy": long})

    async with release(model) as address:
        assert await fetched(tmp_path, Pretrained(), address) == (MELSPECTROGRAM, EMBEDDING, JARVIS)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == dict.fromkeys((MELSPECTROGRAM, EMBEDDING, JARVIS), b"features")


async def test_a_long_header_line_still_arriving_when_a_read_ends_is_waited_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """aiohttp's own parser, the one it falls back to where its compiled one was not built, holds a line it has not had
    the end of against a limit of its own, so the header is split: its end is sent once its start has been read."""
    begun = asyncio.Event()

    class Reading(HttpResponseParserPy):
        def feed_data(self, data: bytes, *args: Any, **kwargs: Any) -> tuple[list[tuple[RawResponseMessage, StreamReader]], bool, bytes]:
            begun.set()
            return super().feed_data(data, *args, **kwargs)

    monkeypatch.setattr("aiohttp.client_proto.HttpResponseParser", Reading)
    head = b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nContent-Security-Policy: " + b"x" * (32 * 1024)

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(head[: len(head) // 2])
        await writer.drain()
        await begun.wait()
        writer.write(head[len(head) // 2 :] + b"\r\n\r\nfeatures")
        await writer.drain()
        writer.close()

    for name in (MELSPECTROGRAM, EMBEDDING):
        (tmp_path / name).write_bytes(b"features")
    async with await asyncio.start_server(answer, "127.0.0.1", 0) as server:
        assert await fetched(tmp_path, Pretrained(), f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}") == (JARVIS,)
    assert (tmp_path / JARVIS).read_bytes() == b"features"


def test_wake_training_scores_its_models_at_the_threshold_hands_wakes_at() -> None:
    # wake-training does not import hands, so it keeps hands' threshold as a copy; this is the copy's check.
    source = Path(__file__).parent.parent / "wake-training" / "src" / "wake_training" / "evaluate.py"
    copied = next(
        node.value for node in ast.parse(source.read_text()).body if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["THRESHOLD"]
    )
    assert ast.literal_eval(copied) == THRESHOLD
