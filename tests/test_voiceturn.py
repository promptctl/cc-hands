"""The life of a turn the voice opened, run with no desk: the rules of how it ends (`hands.voice.voiceturn`) on fake ears
and a runner that owns the one open turn (`hands.voice.turnlife`), as anything that hears a person speak runs them."""

import asyncio
from collections.abc import Sequence

import pytest
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import shortcut
from hands.voice.turnlife import Life, Runner
from hands.voice.voiceturn import ASK_SECONDS, Ended, Going, NothingAsked, SpeechStarted, SpeechStopped, Speaking, SpeechStarting, Told, Turn, TurnEnded, TurnTooLong, Waiting, deadlines, opened, step


@pytest.mark.parametrize(
    ("turn", "told", "after"),
    [
        # What is asked starts: the turn has speech in it, since the wake.
        (Waiting(1.0), SpeechStarted(1.5), Going(Speaking(1.0))),
        # The wake word's own pause, and the silence after it, end nothing.
        (Waiting(1.0), SpeechStarting(), Going(Waiting(1.0))),
        (Waiting(1.0), SpeechStopped(), Going(Waiting(1.0))),
        (Waiting(1.0), TurnEnded("silence", 1.0), Going(Waiting(1.0))),
        # A wake nothing was asked after is thrown away, once, for the wake it was set for.
        (Waiting(1.0), NothingAsked(1.0), Ended("drop")),
        (Waiting(1.0), NothingAsked(0.5), Going(Waiting(1.0))),
        (Speaking(1.0), NothingAsked(1.0), Going(Speaking(1.0))),
        (Speaking(1.0), SpeechStarting(), Going(Speaking(1.0))),
        (Speaking(1.0), SpeechStarted(2.0), Going(Speaking(1.0))),
        # Speech that stopped is judged; the verdict or the silence sends the turn it names, and no other.
        (Speaking(1.0), SpeechStopped(), Going(Speaking(1.0), judge=True)),
        (Speaking(1.0), TurnEnded("verdict", 1.0), Ended("stop")),
        (Speaking(1.0), TurnEnded("silence", 1.0), Ended("stop")),
        (Speaking(1.0), TurnEnded("verdict", 0.5), Going(Speaking(1.0))),
        # The limit sends a turn as it stands, for the turn it was set for.
        (Waiting(1.0), TurnTooLong(1.0), Ended("timeout")),
        (Speaking(1.0), TurnTooLong(1.0), Ended("timeout")),
        (Speaking(1.0), TurnTooLong(0.5), Going(Speaking(1.0))),
    ],
)
def test_a_voice_turn_ends_by_these_rules(turn: Turn, told: Told, after: Going | Ended) -> None:
    assert step(turn, told) == after


def test_a_turn_that_just_opened_asks_for_its_deadlines_and_one_that_went_on_asks_for_none() -> None:
    assert opened(None, Waiting(1.0)) == Waiting(1.0)
    assert opened(Waiting(1.0), Speaking(1.0)) is None
    assert opened(Speaking(1.0), None) is None
    assert deadlines(Waiting(1.0)) == ((1.0 + shortcut.TURN_LIMIT_SECONDS, TurnTooLong(1.0)), (1.0 + ASK_SECONDS, NothingAsked(1.0)))
    assert deadlines(Speaking(2.0)) == ((2.0 + shortcut.TURN_LIMIT_SECONDS, TurnTooLong(2.0)),)


STATES = {b"q": VADState.QUIET, b"s": VADState.STARTING, b"S": VADState.SPEAKING, b"x": VADState.QUIET}


class FakeEars:
    """The models, as a script: each buffer names the detector's state, b"x" is quiet run past Smart Turn's stop_secs, and
    the verdicts are given in turn. Detecting and judging are held at their gate while it is set; `detecting` and `asked`
    say a use has reached it. `log` is every use of them, in order."""

    def __init__(self, verdicts: Sequence[bool]) -> None:
        self._verdicts = list(verdicts)
        self.log: list[str] = []
        self.gate: asyncio.Event | None = None
        self.asked = asyncio.Event()
        self.detect_gate: asyncio.Event | None = None
        self.detecting = asyncio.Event()

    async def detect(self, audio: bytes) -> VADState:
        self.log.append(f"detect {audio!r}")
        self.detecting.set()
        if self.detect_gate is not None:
            await self.detect_gate.wait()
        return STATES[audio]

    def heard(self, audio: bytes, speech: bool) -> bool:
        self.log.append(f"fed {audio!r}")
        return audio == b"x"

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        self.log.append("judge asked")
        self.asked.set()
        if self.gate is not None:
            await self.gate.wait()
        complete = self._verdicts.pop(0)
        self.log.append("judge answered")
        return complete, None

    def clear(self) -> None:
        self.log.append("clear")

    def afresh(self) -> None:
        self.log.append("afresh")

    def hold_detecting(self) -> asyncio.Event:
        """Hold the next buffer's detector at a gate, and return the gate, for the test to open."""
        self.detecting.clear()
        self.detect_gate = asyncio.Event()
        return self.detect_gate


class Rig:
    """A runner with no desk, on fake ears and a virtual clock, with the deadlines its turns ask for told to it as they come
    due, and the events of the units of work its life makes."""

    def __init__(self, verdicts: Sequence[bool] = ()) -> None:
        self.clock = VirtualClock()
        self.ears = FakeEars(verdicts)
        self.due: list[Told] = []
        self.events: list[WideEvent] = []
        self.runner = Runner(Life(self.clock, self.ears, self.due.append, self.events.append, "test"))

    async def until(self, at: float) -> Ended | None:
        """The clock at `at`, and each deadline that came due stepped in the order it came."""
        self.clock.advance_to(at)
        await asyncio.sleep(0)
        ended = None
        while self.due:
            ended = await self.runner.advance(self.due.pop(0)) or ended
        return ended


async def test_a_turn_handed_open_that_nothing_is_asked_after_is_thrown_away_at_ask_seconds() -> None:
    rig = Rig()
    rig.clock.advance_to(1.0)
    await rig.runner.open(Waiting(1.0))
    assert rig.ears.log == ["afresh"]
    assert await rig.until(1.0 + ASK_SECONDS - 0.1) is None and rig.runner.turn == Waiting(1.0)
    assert await rig.until(1.0 + ASK_SECONDS) == Ended("drop")
    assert rig.runner.turn is None and rig.ears.log == ["afresh", "clear"]
    # The limit it set for itself comes due on a turn that is over, and ends nothing.
    assert await rig.until(1.0 + shortcut.TURN_LIMIT_SECONDS) is None


async def test_a_turn_handed_open_is_sent_on_smart_turns_verdict_once_speech_has_stopped() -> None:
    rig = Rig(verdicts=[False, True])
    await rig.runner.open(Waiting(0.0))
    # The pause after the wake word, and its silence, are no end: nothing is judged.
    assert await rig.runner.hear(b"s") is None and await rig.runner.hear(b"q") is None
    assert "judge asked" not in rig.ears.log and rig.runner.turn == Waiting(0.0)
    # What is asked starts and stops: a pause Smart Turn takes for a thought still going holds the turn open, and the
    # speech after it, judged complete, sends it.
    assert await rig.runner.hear(b"S") is None and rig.runner.turn == Speaking(0.0)
    assert await rig.runner.hear(b"q") is None and rig.runner.turn == Speaking(0.0)
    assert await rig.runner.hear(b"S") is None
    assert await rig.runner.hear(b"q") == Ended("stop")
    assert rig.runner.turn is None and rig.ears.log[-1] == "clear"
    # Each verdict is its own unit of work, and says what Smart Turn answered.
    assert [(each.event, each.facts["complete"]) for each in rig.events] == [("test.judged", False), ("test.judged", True)]


async def test_a_turn_handed_open_is_sent_on_the_silence_after_speech_that_smart_turn_held_open() -> None:
    rig = Rig(verdicts=[False])
    await rig.runner.open(Waiting(0.0))
    await rig.runner.hear(b"S")
    await rig.runner.hear(b"q")
    assert await rig.runner.hear(b"x") == Ended("stop")


async def test_a_turn_handed_open_is_sent_at_the_limit_as_it_stands() -> None:
    rig = Rig()
    await rig.runner.open(Waiting(0.0))
    await rig.runner.hear(b"S")
    assert await rig.until(shortcut.TURN_LIMIT_SECONDS - 1.0) is None
    assert await rig.until(shortcut.TURN_LIMIT_SECONDS) == Ended("timeout")


async def test_the_limit_and_the_ask_are_the_numbers_the_desk_has_where_they_are_defined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.shortcut.TURN_LIMIT_SECONDS", 3.0)
    monkeypatch.setattr("hands.voice.voiceturn.ASK_SECONDS", 2.0)
    rig = Rig()
    await rig.runner.open(Waiting(0.0))
    assert await rig.until(2.0) == Ended("drop")
    await rig.runner.open(Waiting(2.0))
    await rig.runner.hear(b"S")
    assert await rig.until(5.0) == Ended("timeout")


async def test_a_cut_the_next_turn_and_its_words_wait_for_the_judge_and_the_verdict_ends_only_the_turn_it_judged() -> None:
    rig = Rig(verdicts=[True])
    await rig.runner.open(Waiting(1.0))
    await rig.runner.hear(b"S")
    rig.ears.gate = asyncio.Event()
    judging = asyncio.create_task(rig.runner.hear(b"q"))
    await rig.ears.asked.wait()
    # The gate takes the turn, the next opens and its first words come, all while Smart Turn thinks.
    cutting = asyncio.create_task(rig.runner.cut())
    opening = asyncio.create_task(rig.runner.open(Waiting(2.0)))
    hearing = asyncio.create_task(rig.runner.hear(b"S"))
    # One turn of the loop runs the three as far as they go: to the lock, which the judge holds. Had one not been held
    # off, it would have run through the fakes, and the log would show it before the answer.
    await asyncio.sleep(0)
    assert rig.ears.log[-1] == "judge asked" and rig.runner.turn == Speaking(1.0)
    rig.ears.gate.set()
    # The verdict is for the turn that was open when it was asked, and sends it.
    assert await judging == Ended("stop")
    await asyncio.gather(cutting, opening)
    assert await hearing is None
    # The cut found the turn already sent. The next turn was heard afresh, and what it said started it.
    assert rig.runner.turn == Speaking(2.0)
    assert rig.ears.log[rig.ears.log.index("judge answered") + 1 :] == ["clear", "afresh", "detect b'S'", "fed b'S'"]


async def test_a_verdict_that_names_another_turn_ends_nothing() -> None:
    rig = Rig()
    await rig.runner.open(Speaking(1.0))
    assert await rig.runner.advance(TurnEnded("verdict", 0.5)) is None
    assert rig.runner.turn == Speaking(1.0) and rig.ears.log == []


async def test_a_cut_and_the_next_turn_wait_for_the_buffer_being_heard_and_its_stop_is_judged_for_the_turn_it_was_heard_in() -> None:
    rig = Rig(verdicts=[False])
    await rig.runner.open(Speaking(1.0))
    await rig.runner.hear(b"S")
    # The speech stops in a buffer the detector is still thinking about, as the gate takes the turn and opens the next.
    open_gate = rig.ears.hold_detecting()
    hearing = asyncio.create_task(rig.runner.hear(b"q"))
    await rig.ears.detecting.wait()
    cutting = asyncio.create_task(rig.runner.cut())
    opening = asyncio.create_task(rig.runner.open(Speaking(2.0)))
    await asyncio.sleep(0)
    open_gate.set()
    assert await hearing is None
    await asyncio.gather(cutting, opening)
    # Smart Turn judged the stop before it was cleared for the next turn, which has heard nothing.
    assert rig.ears.log[rig.ears.log.index("detect b'q'") :] == ["detect b'q'", "fed b'q'", "judge asked", "judge answered", "clear"]
    assert rig.runner.turn == Speaking(2.0)


async def test_what_a_buffer_said_before_a_turn_opened_is_no_part_of_it() -> None:
    rig = Rig()
    # The tail of the wake word, in a buffer the detector is still thinking about, as the gate opens the turn it woke.
    open_gate = rig.ears.hold_detecting()
    hearing = asyncio.create_task(rig.runner.hear(b"S"))
    await rig.ears.detecting.wait()
    opening = asyncio.create_task(rig.runner.open(Waiting(2.0)))
    await asyncio.sleep(0)
    open_gate.set()
    assert await hearing is None
    await opening
    # Its speech is not what was asked, and the models were told to hear afresh after it: the turn still waits.
    assert rig.runner.turn == Waiting(2.0)
    assert rig.ears.log == ["detect b'S'", "fed b'S'", "afresh"]


async def test_silence_with_no_turn_open_ends_nothing_and_a_turn_is_not_opened_over_another() -> None:
    rig = Rig()
    assert await rig.runner.hear(b"S") is None and await rig.runner.hear(b"x") is None
    await rig.runner.open(Waiting(0.0))
    with pytest.raises(ValueError, match="is open"):
        await rig.runner.open(Waiting(1.0))
