"""The phone's driver: the turns a wake opens at the phone are heard and ended by the rules of any turn the voice opened, and the
gate stays the one owner of the turn. The page is played at `Phone`: it sends the messages a page sends, and its audio arrives
under the gate as it stands. The models are a script (`conftest.ScriptedEars`: each buffer names the detector's state), and the
clock is one the test moves."""

import asyncio
from collections.abc import Callable
from unittest.mock import Mock

import pytest
from aiortc import RTCPeerConnection
from av import AudioResampler
from conftest import ScriptedEars, events, settled
from pipecat.audio.vad.vad_analyzer import VADState

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, begun
from hands.virtualclock import VirtualClock
from hands.voice.atphone import drive
from hands.voice.phone import Asked, Call, Level, Outbound, Phone, Said, Wake
from hands.voice.ptt import PushToTalk
from hands.voice.shortcut import TURN_LIMIT_SECONDS, Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.voiceturn import ASK_SECONDS


class Page(Phone):
    """The phone, with the call of a page the test plays."""

    def __init__(self, clock: VirtualClock, key: PushToTalk, record: Callable[[Entry], None]) -> None:
        super().__init__(clock, key, 16000, 24000, record, speaking=lambda: False)
        # The connection and the speech out are doubles: the page's messages touch neither, and building either mints ids.
        self.call = Call(Mock(spec=RTCPeerConnection), Mock(spec=Outbound), AudioResampler(format="s16", layout="mono", rate=16000), 16000, "test", Asked("test page", "take"), begun(clock), asyncio.Event())
        self._cued: list[Move] = []

    def says(self, said: Said) -> None:
        """The page sends `said`, as its channel brings it."""
        self.hear(self.call, said)

    def sounds(self, sound: bytes) -> None:
        """A frame of the page's audio, under the gate as it stands, given to the pipeline and to those who follow it."""
        frame = self._key.gate.framed(sound, sound, None, 16000, 1, "phone")
        for follower in self._followers:
            follower(frame)

    def cued(self) -> list[Move]:
        """Every move the page's tones and words have been told to follow so far."""
        while not self.moves.empty():
            self._cued.append(self.moves.get_nowait())
        return self._cued


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
        self.ears = Listening([] if verdicts is None else verdicts)
        self.key = PushToTalk(self.clock, self.recorded.append)
        self.page = Page(self.clock, self.key, self.recorded.append)
        self.key.go("phone")
        self.page.says(Level("wake word"))
        self.driving: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.driving = asyncio.create_task(drive(self.clock, self.key, self.page, self.ears, self.recorded.append))
        await settled()

    async def stop(self) -> None:
        assert self.driving is not None
        self.driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await self.driving

    async def hear(self, *sounds: bytes) -> None:
        """Each sound, as a frame of the phone's audio, one at a time."""
        for sound in sounds:
            self.page.sounds(sound)
        await settled()

    def events(self, name: str) -> list[WideEvent]:
        return events(self.recorded, name)


async def test_a_wakes_turn_hears_only_from_the_wake_on_and_is_sent_by_smart_turns_verdict() -> None:
    rig = Rig([True])
    await rig.start()
    # The second the page sent before the wake is the clip's, kept as the desk keeps its own: the driver never hears it.
    await rig.hear(b"S", b"q")
    rig.page.says(Wake(0.9))
    await settled()
    assert rig.ears.detected == [] and rig.ears.afreshed == 1
    # The page's wake is the press the turn is the answer to.
    assert rig.page.call.pressing == {0: 1}
    await rig.hear(b"q", b"s", b"S", b"q")
    await rig.stop()
    assert rig.ears.detected == [b"q", b"s", b"S", b"q"]
    assert (rig.key.gate.ended, rig.key.gate.turn_open) == (("sent",), False)
    assert rig.page.cued() == ["listen", "arm", "start", "stop"]
    [followed] = rig.events("phone.turn")
    assert (followed.outcome, followed.facts) == ("ok", {"how": "sent", "move": "stop", "ended_by": PHONE_WAKE})
    assert [event.facts["complete"] for event in rig.events("phone.judged")] == [True]
    assert rig.ears.cleared == 1


async def test_a_wake_nothing_was_asked_after_is_thrown_away_once_ask_seconds_have_passed() -> None:
    rig = Rig()
    await rig.start()
    rig.page.says(Wake(0.9))
    await rig.hear(b"q", b"q")
    rig.clock.advance(ASK_SECONDS - 0.5)
    await settled()
    assert rig.key.gate.turn_open
    rig.clock.advance(0.5)
    await settled()
    await rig.stop()
    assert (rig.key.gate.ended, rig.page.cued()[-1]) == (("dropped",), "drop")
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "dropped", "move": "drop", "ended_by": PHONE_WAKE}]


async def test_a_wakes_turn_that_runs_to_the_limit_is_sent_as_it_stands_there() -> None:
    rig = Rig()
    await rig.start()
    rig.page.says(Wake(0.9))
    await rig.hear(b"s", b"S")
    rig.clock.advance(TURN_LIMIT_SECONDS)
    await settled()
    await rig.stop()
    assert (rig.key.gate.ended, rig.page.cued()[-1]) == (("sent",), "timeout")
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "sent", "move": "timeout", "ended_by": PHONE_WAKE}]


async def test_a_verdict_that_comes_back_after_a_press_took_the_turn_ends_nothing_of_the_button_turn() -> None:
    rig = Rig([True])
    await rig.start()
    rig.page.says(Wake(0.9))
    rig.ears.thinking = asyncio.Event()
    await rig.hear(b"s", b"S", b"q")
    await settled()
    assert rig.ears.judging.is_set()
    # The page's button is pressed while Smart Turn is still thinking: the gate sends the wake's turn and opens the button's.
    rig.page.says("start")
    assert (rig.key.gate.ended, rig.key.gate.opened) == (("sent",), "phone button")
    rig.ears.thinking.set()
    await settled()
    await rig.stop()
    assert rig.key.gate.turn_open and rig.key.gate.ended == ("sent",)
    assert rig.page.cued() == ["listen", "arm", "start", "stop", "start"]
    # The turn the verdict was for was told ended by the button, and the driver followed it to that end.
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "sent", "move": "stop", "ended_by": "phone button"}]


async def test_a_verdict_ends_the_turn_for_the_driver_at_once_so_the_audio_queued_behind_it_is_not_heard_as_the_turn() -> None:
    rig = Rig([True])
    await rig.start()
    rig.page.says(Wake(0.9))
    rig.ears.thinking = asyncio.Event()
    await rig.hear(b"s", b"S", b"q")
    assert rig.ears.judging.is_set()
    # Two frames of the turn reach the driver while Smart Turn thinks; the verdict ends the turn before either is stepped.
    await rig.hear(b"q", b"q")
    rig.ears.thinking.set()
    await settled()
    await rig.stop()
    assert rig.ears.detected == [b"s", b"S", b"q"]
    assert [event.facts["ended_by"] for event in rig.events("phone.turn")] == [PHONE_WAKE]


def call_ends(rig: Rig) -> None:
    rig.key.go("desk")


def level_goes_off(rig: Rig) -> None:
    rig.page.says(Level("off"))


def hand_at_the_desk(rig: Rig) -> None:
    rig.key.move("start", "shortcut")


@pytest.mark.parametrize(
    ("ends", "by"),
    [(call_ends, "call"), (level_goes_off, "level"), (hand_at_the_desk, "shortcut")],
    ids=["the call ends", "the level goes off", "a hand at the desk takes hands there"],
)
async def test_whatever_ends_a_wakes_turn_ends_the_drivers_following_of_it_and_the_next_wake_opens_a_turn(ends: Callable[[Rig], None], by: str) -> None:
    rig = Rig([True])
    await rig.start()
    rig.page.says(Wake(0.9))
    await rig.hear(b"s", b"S")
    ends(rig)
    await settled()
    assert [event.facts for event in rig.events("phone.turn")] == [{"how": "sent", "move": "stop", "ended_by": by}]
    assert rig.ears.cleared == 1
    # Nothing the models hear after it belongs to it: the phone's audio under the gate as it stands now is no turn of the wake's.
    heard = list(rig.ears.detected)
    await rig.hear(b"S", b"q")
    assert rig.ears.detected == heard
    # Hands at the phone again, listening: the next wake is a turn of its own, and is heard from its start.
    rig.key.go("phone")
    rig.page.says(Level("wake word"))
    rig.page.says("start")
    rig.page.says("stop")
    rig.page.says(Wake(0.9))
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
    rig.page.says(Wake(0.9))
    await rig.hear(b"s", b"S", b"q")
    await rig.stop()
    assert rig.ears.detected == [] and rig.events("phone.turn") == [] and rig.key.gate.ended == ()
    assert [(event.facts["opened"], event.facts["press"]) for event in rig.events("phone.wake")] == [(False, 1)]
    assert rig.page.call.pressing == {}


async def test_a_wake_turn_still_open_as_the_driver_stops_is_one_event_that_says_so() -> None:
    rig = Rig()
    await rig.start()
    rig.page.says(Wake(0.9))
    await rig.hear(b"s")
    await rig.stop()
    [followed] = rig.events("phone.turn")
    assert (followed.outcome, followed.facts) == ("failed", {"ended_by": "stop"})


def test_a_second_press_over_the_buttons_own_clip_opens_no_clip_and_leaves_the_first_press_its_turn() -> None:
    rig = Rig()
    rig.page.says("start")
    rig.page.says("start")
    # The page counted two presses; the gate opened one clip, which the first press is the answer to.
    assert (rig.page.call.presses, rig.page.call.pressing, rig.page.cued()) == (2, {0: 1}, ["listen", "start"])
