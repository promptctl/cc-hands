"""The rules of a voice turn's end, and an owner of one turn that is not the desk: the same rules and the same life of the
models end its turns."""

import asyncio
from collections import deque
from dataclasses import dataclass

import pytest
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice.turnlife import Life
from hands.voice.voiceturn import ASK_SECONDS, Ended, Going, NothingAsked, Speaking, SpeechStarted, SpeechStarting, SpeechStopped, Told, Turn, TurnEnded, TurnTooLong, Waiting, deadlines, opened, step

WAITING = Waiting(1.0)
SPEAKING = Speaking(1.0)


@pytest.mark.parametrize(
    ("turn", "told", "after"),
    [
        # What is asked starts: the wake's turn has speech in it, since it opened.
        (WAITING, SpeechStarted(1.5), Going(Speaking(1.0))),
        # Before what is asked has started, a stop, a silence and a verdict end nothing: they are the end of the wake word.
        (WAITING, SpeechStarting(), Going(WAITING)),
        (WAITING, SpeechStopped(), Going(WAITING)),
        (WAITING, TurnEnded("silence", 1.0), Going(WAITING)),
        (WAITING, TurnEnded("verdict", 1.0), Going(WAITING)),
        # A wake that nothing asked after is thrown away, for the wake it was set for alone.
        (WAITING, NothingAsked(1.0), Ended("drop", 1.0)),
        (WAITING, NothingAsked(0.5), Going(WAITING)),
        (WAITING, TurnTooLong(1.0), Ended("timeout", 1.0)),
        (WAITING, TurnTooLong(0.5), Going(WAITING)),
        # Speech that stops is judged; a verdict or silence ends the turn it names and no other.
        (SPEAKING, SpeechStopped(), Going(SPEAKING, judge=True)),
        (SPEAKING, SpeechStarting(), Going(SPEAKING)),
        (SPEAKING, SpeechStarted(2.0), Going(SPEAKING)),
        (SPEAKING, TurnEnded("verdict", 1.0), Ended("stop", 1.0)),
        (SPEAKING, TurnEnded("silence", 1.0), Ended("stop", 1.0)),
        (SPEAKING, TurnEnded("verdict", 0.5), Going(SPEAKING)),
        (SPEAKING, TurnEnded("silence", 0.5), Going(SPEAKING)),
        # What is asked past it is no part of a wake's ask, and a limit counts for its own turn alone.
        (SPEAKING, NothingAsked(1.0), Going(SPEAKING)),
        (SPEAKING, TurnTooLong(1.0), Ended("timeout", 1.0)),
        (SPEAKING, TurnTooLong(0.5), Going(SPEAKING)),
    ],
)
def test_a_turn_told_something_goes_on_or_ends_by_the_one_table(turn: Turn, told: Told, after: Going | Ended) -> None:
    assert step(turn, told) == after


def test_a_turn_is_opened_by_the_step_that_makes_it_a_different_turn_and_by_no_other() -> None:
    assert opened(None, WAITING) == WAITING
    assert opened(None, SPEAKING) == SPEAKING
    assert opened(Waiting(0.5), WAITING) == WAITING
    # The wake's turn gaining speech is the same turn going on; a turn that is gone opens nothing.
    assert opened(WAITING, Speaking(1.0)) is None
    assert opened(SPEAKING, None) is None
    assert opened(None, None) is None


def test_a_turn_asks_for_its_limit_and_a_wake_for_its_ask_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.shortcut.TURN_LIMIT_SECONDS", 30.0)
    assert deadlines(WAITING) == ((31.0, TurnTooLong(1.0)), (1.0 + ASK_SECONDS, NothingAsked(1.0)))
    assert deadlines(SPEAKING) == ((31.0, TurnTooLong(1.0)),)


class FakeEars:
    """The models, as a script: each buffer of audio names the detector's state, and the verdicts are given in turn."""

    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = verdicts
        self.cleared = 0
        # Set to keep Smart Turn thinking, and set by it once it is.
        self.thinking: asyncio.Event | None = None
        self.judging = asyncio.Event()

    async def detect(self, audio: bytes) -> VADState:
        return {b"q": VADState.QUIET, b"s": VADState.STARTING, b"S": VADState.SPEAKING, b"x": VADState.QUIET}[audio]

    def heard(self, audio: bytes, speech: bool) -> bool:
        # b"x" is quiet run past Smart Turn's stop_secs.
        return audio == b"x"

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        self.judging.set()
        if self.thinking is not None:
            await self.thinking.wait()
        complete = self._verdicts.pop(0)
        return complete, TurnMetricsData(processor="test", is_complete=complete, probability=0.9 if complete else 0.1, e2e_processing_time_ms=40.0)

    def clear(self) -> None:
        self.cleared += 1

    def afresh(self) -> None:
        pass


@dataclass(frozen=True)
class Cut:
    """The owner's gate took the turn, whatever the voice was doing."""


@dataclass(frozen=True)
class Open:
    """The owner's gate opened `turn`, however it came to."""

    turn: Turn


class Owner:
    """An owner of one turn at a time that is no desk: it holds which turn is open, steps what it is told in the order it
    was told, and judges inline, queuing the verdict behind whatever was told while it thought."""

    def __init__(self, clock: VirtualClock, ears: FakeEars) -> None:
        self.ears = ears
        self.queue: deque[Told | Cut | Open] = deque()
        self.turn: Turn | None = None
        self.ended: list[Ended] = []
        self.events: list[WideEvent] = []
        self.life = Life(clock, ears, self.queue.append, self.events.append, "owner")

    def _open(self, turn: Turn) -> None:
        self.turn = turn
        self.life.opened(turn)

    async def hear(self, *audio: bytes) -> None:
        for each in audio:
            self.queue.extend((await self.life.hear(each, True, self.turn)).told())
            await self.run()

    async def run(self) -> None:
        while self.queue:
            each = self.queue.popleft()
            if isinstance(each, Cut):
                self.turn = None
                self.life.ended()
            elif isinstance(each, Open):
                self._open(each.turn)
            elif self.turn is None:
                # A turn the owner has not opened is not told anything by the voice but where speech starts.
                if isinstance(each, SpeechStarted):
                    self._open(Speaking(each.at))
            else:
                match step(self.turn, each):
                    case Going(turn=turn, judge=judge):
                        self.turn = turn
                        if judge and (verdict := await self.life.judge(turn)) is not None:
                            self.queue.append(verdict)
                    case Ended() as ended:
                        self.ended.append(ended)
                        self.turn = None
                        self.life.ended()


async def test_speech_judged_complete_ends_its_turn_and_clears_smart_turn() -> None:
    owner = Owner(VirtualClock(), FakeEars([True]))
    await owner.hear(b"s", b"S", b"q")
    assert owner.ended == [Ended("stop", 0.0)] and owner.turn is None
    assert owner.ears.cleared == 1
    assert [event.event for event in owner.events] == ["owner.judged"]


async def test_a_pause_judged_a_thought_still_going_holds_the_turn_until_the_silence_after_it_ends_it() -> None:
    owner = Owner(VirtualClock(), FakeEars([False]))
    await owner.hear(b"s", b"S", b"q")
    assert owner.ended == [] and owner.turn == Speaking(0.0)
    await owner.hear(b"x")
    assert owner.ended == [Ended("stop", 0.0)]


async def test_a_silence_with_no_turn_open_says_nothing() -> None:
    owner = Owner(VirtualClock(), FakeEars([]))
    detected = await owner.life.hear(b"x", True, None)
    assert detected.silence is None and detected.told() == ()
    await owner.hear(b"x")
    assert owner.ended == [] and owner.turn is None and owner.ears.cleared == 0


async def test_a_wake_nothing_is_asked_after_is_dropped_when_the_ask_runs_out() -> None:
    clock = VirtualClock()
    owner = Owner(clock, FakeEars([]))
    owner.queue.append(Open(WAITING))
    await owner.run()
    await clock.scheduled(1.0 + ASK_SECONDS)
    clock.advance_to(1.0 + ASK_SECONDS)
    await asyncio.sleep(0)
    await owner.run()
    assert owner.ended == [Ended("drop", 1.0)] and owner.ears.cleared == 1


async def test_a_turn_open_to_the_limit_is_sent_as_it_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.shortcut.TURN_LIMIT_SECONDS", 3.0)
    clock = VirtualClock()
    owner = Owner(clock, FakeEars([]))
    await owner.hear(b"S")
    await clock.scheduled(3.0)
    clock.advance_to(3.0)
    await asyncio.sleep(0)
    await owner.run()
    assert owner.ended == [Ended("timeout", 0.0)] and owner.ears.cleared == 1


async def test_a_wake_that_was_asked_after_is_not_dropped_when_the_ask_runs_out() -> None:
    clock = VirtualClock()
    owner = Owner(clock, FakeEars([]))
    owner.queue.append(Open(WAITING))
    await owner.run()
    await owner.hear(b"S")
    await clock.scheduled(1.0 + ASK_SECONDS)
    clock.advance_to(1.0 + ASK_SECONDS)
    await asyncio.sleep(0)
    await owner.run()
    assert owner.ended == [] and owner.turn == Speaking(1.0)


async def test_a_verdict_given_late_for_a_turn_the_gate_has_since_replaced_ends_nothing() -> None:
    ears = FakeEars([True])
    ears.thinking = asyncio.Event()
    owner = Owner(VirtualClock(), ears)
    hearing = asyncio.create_task(owner.hear(b"s", b"S", b"q"))
    await ears.judging.wait()
    # While Smart Turn thinks, the gate takes the turn the verdict is for and opens another: both wait behind the judge.
    owner.queue.extend((Cut(), Open(Speaking(5.0))))
    ears.thinking.set()
    await hearing
    # The verdict, queued behind them, names the first turn: it is no end of the second.
    assert owner.ended == [] and owner.turn == Speaking(5.0)
    assert ears.cleared == 1
    assert step(Speaking(5.0), TurnEnded("verdict", 0.0)) == Going(Speaking(5.0))


async def test_the_models_are_refused_a_use_begun_beside_another() -> None:
    ears = FakeEars([True])
    ears.thinking = asyncio.Event()
    owner = Owner(VirtualClock(), ears)
    judging = asyncio.create_task(owner.life.judge(SPEAKING))
    await ears.judging.wait()
    with pytest.raises(RuntimeError, match="asked to hear a buffer while they judge"):
        await owner.life.hear(b"q", True, SPEAKING)
    with pytest.raises(RuntimeError, match="asked to clear while they judge"):
        owner.life.ended()
    ears.thinking.set()
    assert await judging == TurnEnded("verdict", 1.0)
    # What was refused left the models as the judge held them: free again once it is done.
    owner.life.ended()
    assert ears.cleared == 1

