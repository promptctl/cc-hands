"""The system channel: what hands says about itself, and where it goes when speech or the model is what failed."""

import asyncio
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory

from conftest import running
from hands.sessions import heartbeat
from hands.daemon.cli import crashed_before
from hands.sessions.audit import Announced, Entry
from hands.daemon import notify
from hands.daemon.notify import notification_command, post_notification
from hands.sessions.home import Home
from hands.voice.system import (
    NoMicrophone,
    StopFailed,
    TurnTimedOut,
    told,
    AudioMoved,
    BURST_SECONDS,
    ModelFailed,
    ModelFault,
    UsageLimitReached,
    ModelReplyEmpty,
    ModelUnanswered,
    ModelActedUntold,
    ModelUnreachable,
    Post,
    Say,
    Started,
    SystemChannel,
    SystemFact,
    TranscriptionFailed,
    Unrouted,
    alarm,
    system_text,
)
from hands.voice.microphone import Devices
from hands.voice.hold import Move
from hands.voice.ptt import Gate
from hands.voice.turnstop import TurnResolved, Words
from hands.voice.holdowner import HoldOwner
from hands.voice.holds import Hold


BUILT_IN = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
DEAF = Devices(input=None, output="Mac mini Speakers")


@pytest.mark.parametrize(
    ("fact", "said"),
    [
        (Started(after_crash=False, devices=BUILT_IN), "hands is up."),
        (Started(after_crash=True, devices=BUILT_IN), "hands is back after a crash."),
        (Started(after_crash=False, devices=DEAF), "hands is up, but there is no microphone, so it cannot hear you."),
        (Started(after_crash=True, devices=DEAF), "hands is back after a crash, but there is no microphone, so it cannot hear you."),
        (AudioMoved(DEAF), "No microphone: hands cannot hear you. Speaking on Mac mini Speakers."),
        (NoMicrophone(), "There is no microphone, so hands cannot hear you."),
        (ModelUnreachable(), "The language model is unreachable."),
        (ModelFailed(ErrorCategory.RATE_LIMIT), "The language model failed: rate limit."),
        (StopFailed("invalid_request", "Login expired · Run /login to sign in again"), "The brain failed: Login expired · Run /login to sign in again."),
        (StopFailed("server_error", "API Error: Request timed out."), "The brain failed: API Error: Request timed out."),
        (StopFailed("authentication_failed", None), "The brain failed: authentication failed."),
        (StopFailed(None, None), "The brain failed, and Claude Code did not say why."),
        (ModelReplyEmpty(), "The language model sent back nothing."),
        (ModelUnanswered(), "The language model gave no answer to that turn."),
        (ModelActedUntold(), "The language model did not say how that went."),
        (UsageLimitReached(None), "The language model's usage limit is reached."),
        (TranscriptionFailed(), "Speech recognition failed for that turn."),
        (TurnTimedOut(), "That turn ran 10 minutes, so hands sent it and stopped recording."),
    ],
)
def test_each_fact_is_said_from_its_template(fact: SystemFact, said: str) -> None:
    assert system_text(fact) == said


@pytest.mark.parametrize(
    ("move", "devices", "said"),
    [
        ("start", DEAF, (NoMicrophone(),)),
        ("listen", DEAF, (NoMicrophone(),)),
        ("listen", BUILT_IN, ()),
        ("stop", DEAF, ()),
        ("drop", DEAF, ()),
        ("start", BUILT_IN, ()),
        ("timeout", BUILT_IN, (TurnTimedOut(),)),
        ("timeout", DEAF, ()),
    ],
)
def test_a_move_the_tone_alone_would_leave_unexplained_is_said(move: Move, devices: Devices, said: tuple[SystemFact, ...]) -> None:
    assert told(move, devices) == said


class Services:
    def __init__(self) -> None:
        self.stt, self.llm, self.tts, self.transport = FrameProcessor(), FrameProcessor(), FrameProcessor(), FrameProcessor()

    def alarm(self, error: ErrorFrame) -> object:
        return alarm(error, stt=self.stt, llm=self.llm, tts=self.tts)


RETURNS = datetime(2026, 10, 1, tzinfo=UTC).timestamp()


@pytest.mark.parametrize(
    ("exception", "category", "fact"),
    [
        # The brain's stage has read the fact off the wire already, and it is said as read.
        (ModelFault(UsageLimitReached(RETURNS)), ErrorCategory.UNKNOWN, UsageLimitReached(RETURNS)),
        (ModelFault(ModelUnreachable()), ErrorCategory.UNKNOWN, ModelUnreachable()),
        (ModelFault(ModelReplyEmpty()), ErrorCategory.UNKNOWN, ModelReplyEmpty()),
        # Anything else keeps its category, and none of its own text is said: it can carry ids, counts and URLs.
        (RuntimeError("boom"), ErrorCategory.UNKNOWN, ModelFailed(ErrorCategory.UNKNOWN)),
        (RuntimeError("overloaded"), ErrorCategory.SERVER, ModelFailed(ErrorCategory.SERVER)),
    ],
)
def test_a_fault_the_brains_stage_read_is_said_as_itself_and_any_other_failure_by_its_category(exception: Exception, category: ErrorCategory, fact: SystemFact) -> None:
    services = Services()
    error = ErrorFrame(f"Unknown error occurred: {exception}", exception=exception, processor=services.llm, category=category)
    assert services.alarm(error) == Say(fact)


def test_the_limit_is_said_to_lift_in_the_listeners_own_time() -> None:
    # The zone is the process's, so it is put back exactly as found, and re-read, before any other test runs.
    before = os.environ.get("TZ")
    os.environ["TZ"] = "America/Denver"
    time.tzset()
    try:
        assert system_text(UsageLimitReached(RETURNS)) == "The language model's usage limit is reached, until September 30 at 6:00 PM."
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


def test_an_error_is_told_by_the_processor_that_raised_it() -> None:
    services = Services()
    assert services.alarm(ErrorFrame("bad key", processor=services.llm, category=ErrorCategory.AUTHENTICATION)) == Say(ModelFailed(ErrorCategory.AUTHENTICATION))
    assert services.alarm(ErrorFrame("boom", processor=services.stt)) == Say(TranscriptionFailed())
    assert services.alarm(ErrorFrame("no voice", processor=services.tts)) == Post("no voice")
    # An error that names no processor is not the model's.
    assert services.alarm(ErrorFrame("lost")) == Unrouted("no processor", "lost")
    assert services.alarm(ErrorFrame("device gone", processor=services.transport)) == Unrouted(str(services.transport), "device gone")


class Recorder(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self.frames: list[Frame] = []

    async def queue_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM, callback: object = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        await asyncio.sleep(0)  # a real processor's queue is awaited, and the channel must hold across it
        self.frames.append(frame)


async def test_a_fact_goes_to_speech_while_it_works_and_to_the_screen_when_it_does_not() -> None:
    tts, posted = Recorder(), list[str]()

    async def notify(text: str) -> bool:
        posted.append(text)
        return True

    recorded: list[Entry] = []
    channel = SystemChannel(tts, notify, recorded.append)
    await channel.say(ModelUnreachable())
    # Kept out of the model's context: there it would read as the model's own reply.
    assert [(type(frame), getattr(frame, "text", None), getattr(frame, "append_to_context", None)) for frame in tts.frames] == [
        (TTSSpeakFrame, "The language model is unreachable.", False)
    ]
    await tts.set_usable(False)
    await channel.say(TranscriptionFailed())
    await channel.sound(Post("no voice"))
    assert len(tts.frames) == 1
    assert posted == ["hands cannot speak, so: Speech recognition failed for that turn.", "hands cannot speak: no voice"]
    assert recorded == [
        Announced("The language model is unreachable.", "speech"),
        Announced("Speech recognition failed for that turn.", "screen"),
        Announced("hands cannot speak: no voice", "screen"),
    ]


class Clock:
    """A clock the test winds, so crossing a burst window costs no wall time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def speaking(clock: Clock) -> tuple[Recorder, list[Entry], SystemChannel]:
    """A channel whose speech works, so nothing it says should reach the screen."""
    tts, recorded = Recorder(), list[Entry]()

    async def notify(_text: str) -> bool:
        raise AssertionError("speech works, so nothing goes to the screen")

    return tts, recorded, SystemChannel(tts, notify, recorded.append, clock)


def said(tts: Recorder) -> list[str | None]:
    return [getattr(frame, "text", None) for frame in tts.frames]


async def test_a_fact_repeated_through_one_burst_is_said_once() -> None:
    """A key held down on 2026-09-22 queued hundreds of empty turns, and this channel reported each one every 0.43 s
    for as long as they drained. A fault that recurs recurs in bursts, and a burst that fills the only channel the
    daemon has is not a loud failure but a jammed one."""
    clock = Clock()
    tts, recorded, channel = speaking(clock)
    for _ in range(200):
        await channel.say(TranscriptionFailed())
        clock.now += 0.43
    assert said(tts) == ["Speech recognition failed for that turn."] * 9  # 200 turns over 86 s, not 200 sentences
    # The audit says a thing was announced only where it was: what a burst costs is the saying, not the knowing.
    assert recorded == [Announced("Speech recognition failed for that turn.", "speech")] * 9


async def test_a_brain_failing_one_way_through_one_burst_is_said_once_however_its_words_differ() -> None:
    """What Claude Code shows for a failure can name a request id or a time, so it differs every time the same failure
    happens; the burst is the failure's kind, and a different kind is news."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    await channel.say(StopFailed("server_error", "API Error: Connection lost mid-response. req_1"))
    await channel.say(StopFailed("server_error", "API Error: Connection lost mid-response. req_2"))
    await channel.say(StopFailed("invalid_request", "Login expired · Run /login to sign in again"))
    assert said(tts) == [
        "The brain failed: API Error: Connection lost mid-response. req_1.",
        "The brain failed: Login expired · Run /login to sign in again.",
    ]


async def test_a_fault_that_is_still_happening_is_said_again_once_its_burst_has_passed() -> None:
    """Nothing else is ever said while a model is down, so only time can end the burst. Suppressing until
    something else was said would leave a user pressing a key at a daemon that has gone permanently silent."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    await channel.say(TranscriptionFailed())
    clock.now += BURST_SECONDS - 0.01
    await channel.say(TranscriptionFailed())
    assert said(tts) == ["Speech recognition failed for that turn."]
    clock.now += 0.01
    await channel.say(TranscriptionFailed())
    assert said(tts) == ["Speech recognition failed for that turn."] * 2


async def test_two_faults_taking_turns_do_not_between_them_defeat_the_burst() -> None:
    """Two faults can interleave, as a turn that fails and a press with no microphone. Were the window one slot
    wide, each would be news to the other and the pair would speak at the full jammed cadence."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    for fact in (TranscriptionFailed(), NoMicrophone()) * 20:
        await channel.say(fact)
        clock.now += 0.43
    assert said(tts) == [
        "Speech recognition failed for that turn.",
        "There is no microphone, so hands cannot hear you.",
        "Speech recognition failed for that turn.",
        "There is no microphone, so hands cannot hear you.",
    ]


async def test_two_model_failures_of_different_kinds_are_both_heard() -> None:
    """The facts carry what differs, so sameness is the type's answer and not a comparison of sentences."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    for fact in (ModelFailed(ErrorCategory.CONNECTIVITY), ModelFailed(ErrorCategory.CONNECTIVITY), ModelFailed(ErrorCategory.RATE_LIMIT)):
        await channel.say(fact)
    assert len(tts.frames) == 2
    for fact in (UsageLimitReached(RETURNS), UsageLimitReached(RETURNS)):
        await channel.say(fact)
    assert len(tts.frames) == 3


async def test_the_screen_is_not_filled_by_a_burst_either() -> None:
    """A TTS that fails raises one error frame per frame it was handed, and the screen jams exactly as the ear
    did — the same window, taken from the other path. Pipecat's text for a silent utterance carries a fresh
    context id each time, so a window taken on the sentence rather than on the fault would never once close."""
    clock, posted = Clock(), list[str]()

    async def notify(text: str) -> bool:
        posted.append(text)
        return True

    channel = SystemChannel(Recorder(), notify, lambda _: None, clock)
    for turn in range(200):
        await channel.sound(Post(f"TTS context 0000-{turn:04d} completed with no audio"))
        clock.now += 0.43
    assert len(posted) == 9  # 200 turns over 86 s, not 200 notifications
    assert posted[0] == "hands cannot speak: TTS context 0000-0000 completed with no audio"


async def test_a_post_the_screen_refused_is_not_taken_for_one_the_user_saw() -> None:
    """A run started over ssh may have no GUI session to post into, so osascript refuses every time. The refusal is no
    announcement and goes on no audit — and it still costs one attempt, because a screen that always says no is
    the one case where recording only what landed would hammer it at the jammed cadence forever."""
    clock, recorded, attempts = Clock(), list[Entry](), list[str]()

    async def notify(text: str) -> bool:
        attempts.append(text)
        return False

    tts = Recorder()
    await tts.set_usable(False)
    channel = SystemChannel(tts, notify, recorded.append, clock)
    for _ in range(200):
        await channel.say(TranscriptionFailed())
        clock.now += 0.43
    assert attempts == ["hands cannot speak, so: Speech recognition failed for that turn."] * 9
    assert recorded == []


async def test_a_burst_that_arrives_all_at_once_is_still_said_once() -> None:
    """Pipecat registers `on_pipeline_error` async (`sync: bool = False`), so it dispatches each error on its own
    task and a dead TTS puts one of these in flight per queued frame. A window taken after the delivery rather
    than at the decision is a window that none of them would have seen, and the jam comes straight back."""
    posted, tts = list[str](), Recorder()

    async def notify(text: str) -> bool:
        await asyncio.sleep(0)  # the await they are all in flight across
        posted.append(text)
        return True

    channel = SystemChannel(tts, notify, lambda _: None, Clock())
    await asyncio.gather(*(channel.sound(Post(f"TTS context 0000-{turn:04d} completed with no audio")) for turn in range(200)))
    assert posted == ["hands cannot speak: TTS context 0000-0000 completed with no audio"]
    await asyncio.gather(*(channel.say(TranscriptionFailed()) for _ in range(200)))
    assert said(tts) == ["Speech recognition failed for that turn."]


class Said:
    """A transcriber that says, for each hold in turn, the words given, nothing, or fails."""

    def __init__(self, *said: str | None | Exception) -> None:
        self.said = list(said)

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        match self.said.pop(0):
            case Exception() as error:
                raise error
            case text:
                return text


class Told(FrameProcessor):
    """What comes out of the hold owner for the turn: each hold's frames from its words, if any, to its resolution."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.holds: list[list[type[Frame]]] = [[]]

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Words() | TurnResolved():
                self.holds[-1].append(type(frame))
                if isinstance(frame, TurnResolved):
                    self.holds.append([])
            case _:
                pass
        await self.push_frame(frame, direction)


async def test_the_hold_owner_is_done_with_every_hold_and_says_nothing_of_one_it_heard_nothing_in() -> None:
    owner, told = HoldOwner(Said(None, "what time is it", RuntimeError("model failed")), (), lambda _: None), Told()
    async with running([owner, told]) as run:
        # Three turns sent, one for each transcription.
        gate = Gate()
        for _ in range(3):
            for gate in (gate.after("start", "held key"), gate.after("start", "held key").after("stop", "held key")):
                await run.worker.queue_frame(gate.framed(b"\x00\x00", b"\x00\x00", None, 16000, 1, "desk"))
        async with asyncio.timeout(10):
            while len(told.holds) < 4:
                await asyncio.sleep(0.01)
        # Every transcription ends with the owner done with its hold, and one it heard nothing in yields nothing else: no
        # frame that could reach the speaker; one that failed is said aloud as an error from the owner.
        assert told.holds[:3] == [[TurnResolved], [Words, TurnResolved], [TurnResolved]]
        [failed] = run.errors
        assert failed.processor is owner and "Hold 3 could not be transcribed: RuntimeError: model failed" == failed.error
        assert failed.category is not None and not failed.category.is_permanent and owner.is_usable


def test_the_notification_text_is_an_argument_not_part_of_the_script() -> None:
    command = notification_command('-say "hi" \\ now')
    assert command[-2:] == ["--", '-say "hi" \\ now']
    assert all('"hi"' not in part for part in command[:-1])


async def test_a_notification_osascript_never_posts_is_refused_and_said(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stuck(*_argv: str, timeout: float) -> None:
        raise TimeoutError

    monkeypatch.setattr(notify, "run", stuck)
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    try:
        assert await post_notification("hello") is False
    finally:
        logger.remove(sink)
    assert errors == [f"osascript did not post 'hello' in {notify.POST_TIMEOUT_SECONDS:.0f}s"]


GONE = "a pid no process holds"


@pytest.mark.parametrize(
    ("pipeline", "pid", "written_ago", "crashed"),
    [
        (None, GONE, 0, False),  # never ran
        ("stopped", GONE, 0, False),
        ("running", GONE, 0, True),
        ("starting", GONE, 0, True),
        ("running", os.getpid(), 0, False),  # up: this is not a restart
        ("running", os.getpid(), 60, True),  # hung, and something now holds its pid
    ],
)
def test_a_run_crashed_when_its_last_heartbeat_was_not_a_stop_and_it_is_not_beating(
    pipeline: heartbeat.PipelineState | None, pid: int | str, written_ago: int, crashed: bool, tmp_path: Path, dead_pid: Callable[[], int]
) -> None:
    home = Home(tmp_path)
    now = datetime.now(UTC)
    if pipeline is not None:
        heartbeat.write(home.status, heartbeat.Status(dead_pid() if pid == GONE else int(pid), now, now - timedelta(seconds=written_ago), heartbeat.HEARTBEAT, pipeline, None, 0, False, (), None))
    assert crashed_before(home) is crashed


def test_a_heartbeat_whose_pid_a_later_process_took_is_a_crash(tmp_path: Path) -> None:
    home = Home(tmp_path)
    day_ago = datetime.now(UTC) - timedelta(days=1)
    # This process holds the pid, but it started long after the run that wrote the heartbeat.
    heartbeat.write(home.status, heartbeat.Status(os.getpid(), day_ago, day_ago, heartbeat.HEARTBEAT, "running", None, 0, False, (), None))
    assert crashed_before(home) is True


def test_a_heartbeat_that_does_not_parse_is_not_taken_for_a_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    home.status.write_text("{")
    assert crashed_before(home) is False
    assert "not counted as a crash" in capsys.readouterr().err
