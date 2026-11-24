"""The phone's driver: a turn a wake opened at the phone is heard with the models and ended by the one table of how a voice
turn ends, through the gate, which takes an end only for the turn it is at, and told as the gate ends one."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, NoReturn

import pytest
from conftest import ScriptedEars

from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice import engaged, shortcut, voiceturn
from hands.voice.atphone import drive
from hands.voice.engaged import loaded
from hands.voice.ptt import Gate, GateEnded, GateOpened, KeyedAudio, PushToTalk
from hands.voice.shortcut import Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.voiceturn import TurnId

RATE = 16000


@dataclass
class Rig:
    """The gate at the phone listening for its wake word, the driver hearing the turns it opens, and what the driver took."""

    ears: ScriptedEars
    clock: VirtualClock = field(default_factory=VirtualClock)
    taken: list[Move] = field(default_factory=list[Move])
    events: list[WideEvent] = field(default_factory=list[WideEvent])
    turns: asyncio.Queue[GateOpened | GateEnded | KeyedAudio] = field(default_factory=asyncio.Queue[GateOpened | GateEnded | KeyedAudio])

    def __post_init__(self) -> None:
        self.key = PushToTalk(self.clock, lambda _entry: None)
        # As the phone does: the driver is told each turn the gate opens and ends.
        self.key.watch(self.turns.put_nowait)
        self.key.go("phone")
        self.key.move("listen", PHONE_WAKE)

    def took(self, moved: Move | None) -> None:
        """As the phone's own does: a move the gate took as nothing is nothing."""
        if moved is not None:
            self.taken.append(moved)

    def wake(self) -> TurnId:
        """The page's wake, as the phone steps it: the gate opens a turn, and tells the driver the number it gives it."""
        self.key.move("arm", PHONE_WAKE)
        self.key.move("start", PHONE_WAKE)
        assert self.key.gate.wake_turn is not None
        return self.key.gate.turn

    async def hear(self, *audio: bytes, of: Gate | None = None) -> None:
        """The frames the phone keyed as the gate stood, `of` it if it is a gate from before, then the driver's turn at them."""
        gate = self.key.gate if of is None else of
        for each in audio:
            self.turns.put_nowait(gate.framed(each, each, None, RATE, 1, "phone"))
        await settle()

    async def due(self, when: float) -> None:
        """Move the clock to `when`, when a deadline is set for it, and the driver's turn at what that told."""
        await self.clock.scheduled(when)
        self.clock.advance_to(when)
        await settle()


async def settle() -> None:
    # What was put is stepped once the task that carries it into the driver and the driver have each had a turn on the loop;
    # nothing here waits on time.
    for _ in range(4):
        await asyncio.sleep(0)


@asynccontextmanager
async def driving(ears: ScriptedEars) -> AsyncGenerator[Rig]:
    rig = Rig(ears)
    task = asyncio.create_task(drive(rig.clock, rig.key, rig.turns, ears, rig.took, rig.events.append))
    try:
        yield rig
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_speech_after_the_wake_and_the_quiet_after_it_send_the_turn_through_the_gate() -> None:
    ears = ScriptedEars([True])
    async with driving(ears) as rig:
        rig.wake()
        # The wake word's own pause ends nothing; what is asked starts the turn's speech, and stopping asks Smart Turn.
        await rig.hear(b"q", b"s", b"S", b"S")
        assert (rig.taken, rig.key.gate.turn_open) == ([], True)
        before = rig.key.gate
        await rig.hear(b"q")
        assert rig.taken == ["stop"]
        assert (rig.key.gate.turn_open, rig.key.gate.ended, rig.key.gate.key) == (False, ("sent",), "listening")
        assert [event.event for event in rig.events] == ["phone.judged", "phone.turn"]
        # Heard afresh from the wake, and Smart Turn cleared as the turn went.
        assert (ears.afreshed, ears.cleared) == (1, 1)
        # Audio the phone had queued for the turn before it ended is heard by no one.
        await rig.hear(b"S", of=before)
        assert (ears.afreshed, ears.cleared, rig.taken) == (1, 1, ["stop"])
        turn = rig.events[-1]
        assert (turn.facts["end"], turn.facts["frames"], turn.outcome) == ("stop", 5, "ok")


async def test_a_wake_nothing_is_asked_after_is_dropped_when_the_ask_runs_out_counted_from_the_wake() -> None:
    async with driving(ScriptedEars([])) as rig:
        # The page's audio never comes: the ask runs from the wake, not from audio.
        rig.wake()
        await settle()
        await rig.due(voiceturn.ASK_SECONDS)
        assert rig.taken == ["drop"]
        assert (rig.key.gate.turn_open, rig.key.gate.ended) == (False, ("dropped",))


async def test_a_turn_open_to_the_limit_is_sent_as_it_stands() -> None:
    async with driving(ScriptedEars([])) as rig:
        rig.wake()
        await rig.hear(b"q", b"s", b"S")
        await rig.due(shortcut.TURN_LIMIT_SECONDS)
        assert rig.taken == ["timeout"]
        assert (rig.key.gate.turn_open, rig.key.gate.ended) == (False, ("sent",))


async def test_a_verdict_for_a_turn_a_press_took_ends_nothing_of_the_buttons_turn() -> None:
    ears = ScriptedEars([True])
    ears.thinking = asyncio.Event()
    async with driving(ears) as rig:
        turn = rig.wake()
        await rig.hear(b"q", b"s", b"S", b"q")
        await ears.judging.wait()
        # The press sends the wake's turn at the gate and opens the button's own, while Smart Turn is still thinking.
        assert rig.key.end(turn, "stop", PHONE_WAKE) == "stop"
        rig.key.move("arm", "phone button")
        rig.key.move("start", "phone button")
        ears.thinking.set()
        await settle()
        assert rig.taken == []
        assert (rig.key.gate.turn_open, rig.key.gate.opened, rig.key.gate.ended) == (True, "phone button", ("sent",))
        # What the driver followed was over before its verdict: the gate ended it first.
        assert [(event.event, event.facts.get("end"), event.facts.get("by")) for event in rig.events] == [("phone.judged", None, None), ("phone.turn", "gate", "phone wake word")]


@pytest.mark.parametrize("cause", ["phone wake word", "shortcut", "call"])
async def test_a_turn_the_gate_ended_is_let_go_as_it_ends_it_and_says_what_ended_it(cause: Literal["phone wake word", "shortcut", "call"]) -> None:
    ears = ScriptedEars([True])
    async with driving(ears) as rig:
        turn = rig.wake()
        await rig.hear(b"q")
        match cause:
            case "phone wake word":
                # A press or the level going off: the phone sends the turn its wake opened.
                assert rig.key.end(turn, "stop", PHONE_WAKE) == "stop"
            case "shortcut":
                # A hand at the desk takes hands there, and the clip open at the phone is sent as it stands.
                rig.key.move("start", "shortcut")
            case "call":
                rig.key.go("desk")
        await settle()
        # No frame follows: the call is gone, or hands is elsewhere. The driver's Smart Turn is cleared all the same.
        assert (ears.afreshed, ears.cleared) == (1, 1)
        [event] = rig.events
        assert (event.event, event.facts["end"], event.facts["by"], event.facts["frames"]) == ("phone.turn", "gate", cause, 1)
        # Its deadlines end nothing, and no later frame is heard.
        await rig.due(shortcut.TURN_LIMIT_SECONDS)
        await rig.hear(b"S")
        assert (rig.taken, len(rig.events), ears.cleared) == ([], 1, 1)


async def test_the_next_wake_opens_a_turn_after_one_the_gate_ended_and_the_old_ones_deadlines_end_nothing_of_it() -> None:
    ears = ScriptedEars([True])
    async with driving(ears) as rig:
        first = rig.wake()
        await rig.hear(b"q")
        # The call's page takes the turn at the gate (a press, a hand at the desk taking hands there, the level going off).
        assert rig.key.end(first, "stop", PHONE_WAKE) == "stop"
        rig.clock.advance_to(1.0)
        second = rig.wake()
        assert second == first + 1
        await rig.hear(b"q")
        # The first's ask runs out while the second waits on its own: it is not the second's to end.
        await rig.due(voiceturn.ASK_SECONDS)
        assert (rig.taken, rig.key.gate.turn_open) == ([], True)
        await rig.hear(b"s", b"S", b"q")
        assert rig.taken == ["stop"]
        assert (rig.key.gate.turn_open, rig.key.gate.ended) == (False, ("sent", "sent"))
        # Heard afresh for each wake, and each turn gone cleared.
        assert (ears.afreshed, ears.cleared) == (2, 2)
        assert [event.facts.get("end") for event in rig.events if event.event == "phone.turn"] == ["gate", "stop"]


async def test_ears_that_cannot_load_fail_the_phones_load_and_say_which_unit_and_why(monkeypatch: pytest.MonkeyPatch) -> None:
    def cannot(clock: VirtualClock, sample_rate: int) -> NoReturn:
        raise RuntimeError("no weights for Smart Turn")

    monkeypatch.setattr(engaged, "Models", cannot)
    events: list[WideEvent] = []
    with pytest.raises(RuntimeError, match="no weights for Smart Turn"):
        async with loaded(VirtualClock(), 16000, events.append, "phone"):
            pytest.fail("the phone heard with ears that did not load")
    [event] = events
    assert (event.event, event.outcome) == ("phone.loaded", "failed")
    assert event.error is not None and "no weights for Smart Turn" in event.error
