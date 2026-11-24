"""The phone's driver: the turns a wake opens at the phone are heard and ended by the rules of any turn the voice opened, and the
gate stays the one owner of the turn. The models are a script (`conftest.ScriptedEars`: each buffer names the detector's
state), and the clock is one the test moves."""

import asyncio
from collections.abc import Callable

import pytest
from conftest import ScriptedEars
from pipecat.audio.vad.vad_analyzer import VADState

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.virtualclock import VirtualClock
from hands.voice.atphone import drive
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.voice.shortcut import TURN_LIMIT_SECONDS, Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.voiceturn import ASK_SECONDS


class Phoned:
    """The phone as the driver follows it: the audio it gives the pipeline, and the moves it cues."""

    def __init__(self) -> None:
        self.heard: Callable[[KeyedAudio], None] | None = None
        self.cued: list[Move] = []

    def follow(self, heard: Callable[[KeyedAudio], None]) -> None:
        self.heard = heard

    def took(self, taken: tuple[Move, ...]) -> None:
        self.cued.extend(taken)


class Listening(ScriptedEars):
    """Scripted models that remember what the detector was given."""

    def __init__(self, verdicts: list[bool]) -> None:
        super().__init__(verdicts)
        self.detected: list[bytes] = []

    async def detect(self, audio: bytes) -> VADState:
        self.detected.append(audio)
        return await super().detect(audio)


class Rig:
    """The driver following a gate whose phone listens for its wake word, hands at the phone."""

    def __init__(self, verdicts: list[bool] | None = None) -> None:
        self.clock = VirtualClock()
        self.recorded: list[Entry] = []
        self.phone = Phoned()
        self.ears = Listening([] if verdicts is None else verdicts)
        self.key = PushToTalk(self.clock, self.recorded.append)
        self.key.go("phone")
        self.key.move("listen", PHONE_WAKE)
        self.driving: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.driving = asyncio.create_task(drive(self.clock, self.key, self.phone, self.ears, self.recorded.append))
        await self.settle()

    async def stop(self) -> None:
        assert self.driving is not None
        self.driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await self.driving

    async def settle(self) -> None:
        for _ in range(30):
            await asyncio.sleep(0)

    def wake(self) -> None:
        """The page's wake, as `Phone` takes it: armed and started, each cued."""
        self.phone.took(self.key.move("arm", PHONE_WAKE))
        self.phone.took(self.key.move("start", PHONE_WAKE))

    async def hear(self, *sounds: bytes) -> None:
        """Each sound, as a frame of the phone's audio under the gate as it stands, one at a time."""
        assert self.phone.heard is not None
        for sound in sounds:
            self.phone.heard(self.key.gate.framed(sound, sound, None, 16000, 1, "phone"))
        await self.settle()

    def events(self, name: str) -> list[WideEvent]:
        return [entry for entry in self.recorded if isinstance(entry, WideEvent) and entry.event == name]


async def test_a_wakes_turn_hears_only_from_the_wake_on_and_is_sent_by_smart_turns_verdict() -> None:
    rig = Rig([True])
    await rig.start()
    # The second the page sent before the wake is the clip's, kept as the desk keeps its own: the driver never hears it.
    await rig.hear(b"S", b"q")
    rig.wake()
    await rig.settle()
    assert rig.ears.detected == [] and rig.ears.afreshed == 1
    await rig.hear(b"q", b"s", b"S", b"q")
    await rig.stop()
    assert rig.ears.detected == [b"q", b"s", b"S", b"q"]
    assert (rig.key.gate.ended, rig.key.gate.turn_open) == (("sent",), False)
    assert rig.phone.cued == ["arm", "start", "stop"]
    [followed] = rig.events("phone.turn")
    assert (followed.outcome, followed.facts) == ("ok", {"how": "sent", "ended_by": PHONE_WAKE})
    assert [event.facts["complete"] for event in rig.events("phone.judged")] == [True]
    assert rig.ears.cleared == 1


async def test_a_wake_nothing_was_asked_after_is_thrown_away_once_ask_seconds_have_passed() -> None:
    rig = Rig()
    await rig.start()
    rig.wake()
    await rig.hear(b"q", b"q")
    rig.clock.advance(ASK_SECONDS - 0.5)
    await rig.settle()
    assert rig.key.gate.turn_open
    rig.clock.advance(0.5)
    await rig.settle()
    await rig.stop()
    assert (rig.key.gate.ended, rig.phone.cued[-1]) == (("dropped",), "drop")
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "dropped", "ended_by": PHONE_WAKE}]


async def test_a_wakes_turn_that_runs_to_the_limit_is_sent_as_it_stands_there() -> None:
    rig = Rig()
    await rig.start()
    rig.wake()
    await rig.hear(b"s", b"S")
    rig.clock.advance(TURN_LIMIT_SECONDS)
    await rig.settle()
    await rig.stop()
    assert (rig.key.gate.ended, rig.phone.cued[-1]) == (("sent",), "timeout")


async def test_a_verdict_that_comes_back_after_a_press_took_the_turn_ends_nothing_of_the_button_turn() -> None:
    rig = Rig([True])
    await rig.start()
    rig.wake()
    rig.ears.thinking = asyncio.Event()
    await rig.hear(b"s", b"S", b"q")
    await rig.settle()
    assert rig.ears.judging.is_set()
    # The page's button is pressed while Smart Turn is still thinking: the gate sends the wake's turn and opens the button's.
    rig.phone.took(rig.key.move("start", "phone button"))
    assert (rig.key.gate.ended, rig.key.gate.opened) == (("sent",), "phone button")
    rig.ears.thinking.set()
    await rig.settle()
    await rig.stop()
    assert rig.key.gate.turn_open and rig.key.gate.ended == ("sent",)
    assert rig.phone.cued == ["arm", "start", "stop", "start"]
    # The turn the verdict was for was told ended by the button, and the driver followed it to that end.
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "sent", "ended_by": "phone button"}]


def call_ends(key: PushToTalk) -> object:
    return key.go("desk")


def level_goes_off(key: PushToTalk) -> object:
    return key.move("deafen", PHONE_WAKE)


def hand_at_the_desk(key: PushToTalk) -> object:
    return key.move("start", "shortcut")


@pytest.mark.parametrize(
    ("ends", "by"),
    [(call_ends, "call"), (level_goes_off, PHONE_WAKE), (hand_at_the_desk, "shortcut")],
    ids=["the call ends", "the level goes off", "a hand at the desk takes hands there"],
)
async def test_whatever_ends_a_wakes_turn_ends_the_drivers_following_of_it_and_the_next_wake_opens_a_turn(ends: Callable[[PushToTalk], object], by: str) -> None:
    rig = Rig([True])
    await rig.start()
    rig.wake()
    await rig.hear(b"s", b"S")
    ends(rig.key)
    await rig.settle()
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "sent", "ended_by": by}]
    assert rig.ears.cleared == 1
    # Nothing the models hear after it belongs to it: the phone's audio under the gate as it stands now is no turn of the wake's.
    heard = list(rig.ears.detected)
    await rig.hear(b"S", b"q")
    assert rig.ears.detected == heard
    # Hands at the phone again, listening: the next wake is a turn of its own, and is heard from its start.
    rig.key.go("phone")
    rig.key.move("listen", PHONE_WAKE)
    rig.key.move("start", "phone button")
    rig.key.end(rig.key.gate.turn, "stop", "phone button")
    rig.wake()
    await rig.hear(b"s", b"S", b"q")
    await rig.stop()
    assert (rig.ears.detected[len(heard) :], rig.ears.afreshed) == ([b"s", b"S", b"q"], 2)
    assert rig.key.gate.ended[-1:] == ("sent",) and not rig.key.gate.turn_open


async def test_a_wake_the_gate_refused_leaves_the_driver_with_no_turn() -> None:
    rig = Rig()
    await rig.start()
    # Hands is at the desk: the gate refuses, so no turn opens and the driver hears nothing.
    rig.key.go("desk")
    rig.key.move("listen", PHONE_WAKE)
    assert rig.key.gate.refusal(PHONE_WAKE) == "elsewhere"
    rig.wake()
    await rig.hear(b"s", b"S", b"q")
    await rig.stop()
    assert rig.ears.detected == [] and rig.events("phone.turn") == [] and rig.key.gate.ended == ()
