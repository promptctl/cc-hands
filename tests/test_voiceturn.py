"""The rules of a voice turn's end, and an owner of one turn that is not the desk: the same rules and the same life of the
models end its turns."""

import asyncio
import itertools
from collections import deque
from dataclasses import dataclass

import pytest
from conftest import ScriptedEars

from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import voiceturn
from hands.voice.turnlife import Life
from hands.voice.voiceturn import Ended, Going, Judging, NothingAsked, Speaking, SpeechStarted, SpeechStarting, SpeechStopped, Told, Turn, TurnEnded, TurnTooLong, Waiting, asked, closed, deadlines, opened, step

# Turn 1, opened at 1.0, and the turn the owner opened after it, which names itself apart from it.
WAITING = Waiting(1, 1.0)
SPEAKING = Speaking(1, 1.0)
NEXT = 2


@pytest.mark.parametrize(
    ("turn", "told", "after"),
    [
        # What is asked starts: the wake's turn has speech in it, since it opened.
        (WAITING, SpeechStarted(1.5), Going(SPEAKING)),
        # Before what is asked has started, a stop, a silence and a verdict end nothing: they are the end of the wake word.
        (WAITING, SpeechStarting(), Going(WAITING)),
        (WAITING, SpeechStopped(), Going(WAITING)),
        (WAITING, TurnEnded("silence", 1), Going(WAITING)),
        (WAITING, TurnEnded("verdict", 1), Going(WAITING)),
        # A wake that nothing asked after is thrown away, for the wake it was set for alone.
        (WAITING, NothingAsked(1), Ended("drop", 1)),
        (WAITING, NothingAsked(NEXT), Going(WAITING)),
        (WAITING, TurnTooLong(1), Ended("timeout", 1)),
        (WAITING, TurnTooLong(NEXT), Going(WAITING)),
        # Speech that stops is judged; a verdict or silence ends the turn it names and no other.
        (SPEAKING, SpeechStopped(), Judging(SPEAKING)),
        (SPEAKING, SpeechStarting(), Going(SPEAKING)),
        (SPEAKING, SpeechStarted(2.0), Going(SPEAKING)),
        (SPEAKING, TurnEnded("verdict", 1), Ended("stop", 1)),
        (SPEAKING, TurnEnded("silence", 1), Ended("stop", 1)),
        (SPEAKING, TurnEnded("verdict", NEXT), Going(SPEAKING)),
        (SPEAKING, TurnEnded("silence", NEXT), Going(SPEAKING)),
        # What is asked past it is no part of a wake's ask, and a limit counts for its own turn alone.
        (SPEAKING, NothingAsked(1), Going(SPEAKING)),
        (SPEAKING, TurnTooLong(1), Ended("timeout", 1)),
        (SPEAKING, TurnTooLong(NEXT), Going(SPEAKING)),
        # A turn opened at the same instant as the one before it is not that turn: only its number tells them apart.
        (Speaking(NEXT, 1.0), TurnEnded("verdict", 1), Going(Speaking(NEXT, 1.0))),
        (Speaking(NEXT, 1.0), TurnTooLong(1), Going(Speaking(NEXT, 1.0))),
    ],
)
def test_a_turn_told_something_goes_on_or_ends_by_the_one_table(turn: Turn, told: Told, after: Going | Judging | Ended) -> None:
    assert step(turn, told) == after


def test_a_turn_is_opened_by_the_step_that_makes_it_a_different_turn_and_by_no_other() -> None:
    assert opened(None, WAITING) == WAITING
    assert opened(None, SPEAKING) == SPEAKING
    assert opened(Waiting(NEXT, 0.5), WAITING) == WAITING
    # A turn that replaces one opened at the same instant is a different turn.
    assert opened(SPEAKING, Speaking(NEXT, 1.0)) == Speaking(NEXT, 1.0)
    # The wake's turn gaining speech is the same turn going on; a turn that is gone opens nothing.
    assert opened(WAITING, SPEAKING) is None
    assert opened(SPEAKING, None) is None
    assert opened(None, None) is None


def test_a_turn_is_closed_by_the_step_that_leaves_it_gone_or_replaced_and_by_no_other() -> None:
    assert closed(SPEAKING, None) == SPEAKING
    assert closed(WAITING, Speaking(NEXT, 1.0)) == WAITING
    assert closed(WAITING, SPEAKING) is None
    assert closed(None, SPEAKING) is None
    assert closed(None, None) is None


def test_a_turn_asks_for_its_limit_and_a_wake_for_its_ask_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each number is moved once, in its one place.
    monkeypatch.setattr("hands.voice.shortcut.TURN_LIMIT_SECONDS", 30.0)
    monkeypatch.setattr("hands.voice.voiceturn.ASK_SECONDS", 2.0)
    assert deadlines(WAITING) == ((31.0, TurnTooLong(1)), (3.0, NothingAsked(1)))
    assert deadlines(SPEAKING) == ((31.0, TurnTooLong(1)),)


def test_a_move_of_the_turn_asks_for_what_the_turn_it_opened_asks_for() -> None:
    assert asked(None, WAITING) == deadlines(WAITING)
    assert asked(WAITING, SPEAKING) == ()
    assert asked(SPEAKING, None) == ()


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

    def __init__(self, clock: VirtualClock, ears: ScriptedEars) -> None:
        self.ears = ears
        self.queue: deque[Told | Cut | Open] = deque()
        self.turn: Turn | None = None
        self.ended: list[Ended] = []
        self.events: list[WideEvent] = []
        self._minted = itertools.count(1)
        # Set as a deadline arrives, so a test waits on that and not on the loop's turns.
        self._arrived = asyncio.Event()
        self.life = Life(clock, ears, self._arrive, self.events.append, "owner")

    def _arrive(self, told: Told) -> None:
        self.queue.append(told)
        self._arrived.set()

    def _move(self, now: Turn | None) -> None:
        was, self.turn = self.turn, now
        self.life.moved(was, now)

    async def hear(self, *audio: bytes) -> None:
        for each in audio:
            self.queue.extend((await self.life.hear(each, True, self.turn)).told())
            await self.run()

    async def due(self) -> None:
        """Step what a deadline told, once one has."""
        await self._arrived.wait()
        self._arrived.clear()
        await self.run()

    async def run(self) -> None:
        while self.queue:
            each = self.queue.popleft()
            if isinstance(each, Cut):
                self._move(None)
            elif isinstance(each, Open):
                self._move(each.turn)
            elif self.turn is None:
                # A turn the owner has not opened is not told anything by the voice but where speech starts.
                if isinstance(each, SpeechStarted):
                    self._move(Speaking(next(self._minted), each.at))
            else:
                match step(self.turn, each):
                    case Going(turn=turn):
                        self._move(turn)
                    case Judging(turn=turn):
                        self._move(turn)
                        if (verdict := await self.life.judge(turn)) is not None:
                            self.queue.append(verdict)
                    case Ended() as ended:
                        self.ended.append(ended)
                        self._move(None)


async def test_speech_judged_complete_ends_its_turn_and_clears_smart_turn() -> None:
    owner = Owner(VirtualClock(), ScriptedEars([True]))
    await owner.hear(b"s", b"S", b"q")
    assert owner.ended == [Ended("stop", 1)] and owner.turn is None
    assert owner.ears.cleared == 1
    assert [event.event for event in owner.events] == ["owner.judged"]


async def test_a_pause_judged_a_thought_still_going_holds_the_turn_until_the_silence_after_it_ends_it() -> None:
    owner = Owner(VirtualClock(), ScriptedEars([False]))
    await owner.hear(b"s", b"S", b"q")
    assert owner.ended == [] and owner.turn == Speaking(1, 0.0)
    await owner.hear(b"x")
    assert owner.ended == [Ended("stop", 1)]


async def test_a_silence_with_no_turn_open_says_nothing() -> None:
    owner = Owner(VirtualClock(), ScriptedEars([]))
    detected = await owner.life.hear(b"x", True, None)
    assert detected.silence is None and detected.told() == ()
    await owner.hear(b"x")
    assert owner.ended == [] and owner.turn is None and owner.ears.cleared == 0


async def test_a_wake_nothing_is_asked_after_is_dropped_when_the_ask_runs_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.voiceturn.ASK_SECONDS", 3.0)
    clock = VirtualClock()
    owner = Owner(clock, ScriptedEars([]))
    owner.queue.append(Open(WAITING))
    await owner.run()
    # The wake word's own speech is no part of what is asked: the turn is heard afresh from where it opened.
    assert owner.ears.afreshed == 1
    await clock.scheduled(1.0 + voiceturn.ASK_SECONDS)
    clock.advance_to(1.0 + voiceturn.ASK_SECONDS)
    await owner.due()
    assert owner.ended == [Ended("drop", 1)] and owner.ears.cleared == 1


async def test_a_turn_open_to_the_limit_is_sent_as_it_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.shortcut.TURN_LIMIT_SECONDS", 3.0)
    clock = VirtualClock()
    owner = Owner(clock, ScriptedEars([]))
    await owner.hear(b"S")
    # A turn opened by speech has no wake word in it to hear afresh from.
    assert owner.ears.afreshed == 0
    await clock.scheduled(3.0)
    clock.advance_to(3.0)
    await owner.due()
    assert owner.ended == [Ended("timeout", 1)] and owner.ears.cleared == 1


async def test_a_wake_that_was_asked_after_is_not_dropped_when_the_ask_runs_out() -> None:
    clock = VirtualClock()
    owner = Owner(clock, ScriptedEars([]))
    owner.queue.append(Open(WAITING))
    await owner.run()
    await owner.hear(b"S")
    await clock.scheduled(1.0 + voiceturn.ASK_SECONDS)
    clock.advance_to(1.0 + voiceturn.ASK_SECONDS)
    await owner.due()
    assert owner.ended == [] and owner.turn == SPEAKING


async def test_a_verdict_given_late_for_a_turn_the_gate_has_since_replaced_ends_nothing() -> None:
    ears = ScriptedEars([True])
    ears.thinking = asyncio.Event()
    owner = Owner(VirtualClock(), ears)
    hearing = asyncio.create_task(owner.hear(b"s", b"S", b"q"))
    await ears.judging.wait()
    # While Smart Turn thinks, the gate takes the turn the verdict is for and opens another at the very same instant: both
    # wait behind the judge.
    second = Speaking(NEXT, 0.0)
    owner.queue.extend((Cut(), Open(second)))
    ears.thinking.set()
    await hearing
    # The verdict, queued behind them, names the first turn: it is no end of the second.
    assert owner.ended == [] and owner.turn == second
    assert ears.cleared == 1


async def test_the_models_are_refused_a_use_begun_beside_another() -> None:
    ears = ScriptedEars([True])
    ears.thinking = asyncio.Event()
    owner = Owner(VirtualClock(), ears)
    judging = asyncio.create_task(owner.life.judge(SPEAKING))
    await ears.judging.wait()
    with pytest.raises(RuntimeError, match="asked to hear a buffer while they judge"):
        await owner.life.hear(b"q", True, SPEAKING)
    with pytest.raises(RuntimeError, match="asked to clear while they judge"):
        owner.life.moved(SPEAKING, None)
    ears.thinking.set()
    assert await judging == TurnEnded("verdict", 1)
    # What was refused left the models as the judge held them: free again once it is done.
    owner.life.moved(SPEAKING, None)
    assert ears.cleared == 1
