"""Following the audio devices, with the reopen and the saying replaced."""

import asyncio
from contextlib import suppress
from typing import Literal

import pytest

from hands.clock import SYSTEM
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from conftest import events, patiently
from hands.voice.devicechoice import Auto, Choices, Closed, DefaultDevices, Hardware, Instead, Named, Off, OnDefault, OnNamed, Placement, placed
from hands.voice.devices import follow
from hands.voice.microphone import Devices
from hands.voice.system import AudioMoved, SystemFact, system_text

HEADSET = Devices(input="headset", output="headset")
BUILT_IN = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")


def on(device: int) -> Placement:
    """Both sides on the defaults, which are the one device."""
    return Placement(OnDefault(device), OnDefault(device))


class Notices(asyncio.Event):
    """The device notices, which also tell when the follower has settled: waiting on them, none pending."""

    def __init__(self) -> None:
        super().__init__()
        self.settled = asyncio.Event()

    async def wait(self) -> Literal[True]:
        if not self.is_set():
            self.settled.set()
        return await super().wait()

    def set(self) -> None:
        self.settled.clear()
        super().set()


class Follower:
    def __init__(self, *opened: Devices, at: Placement = on(1)) -> None:
        self.defaults = at
        self.opened_on = self.defaults
        self.changes = Notices()
        self.opened = list(opened)
        self.devices = HEADSET
        self.reopens = 0
        self.failing: Exception | None = None
        self.said: list[SystemFact] = []
        self.recorded: list[Entry] = []
        self.reopening = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.task = asyncio.create_task(follow(SYSTEM, self.changes, self.current, self, self.say, self.recorded.append))

    async def reopen(self) -> Devices:
        self.reopens += 1
        # The transport reads the defaults as PortAudio lists them, which is before it has finished reopening.
        self.opened_on = self.defaults
        self.reopening.set()
        await self.release.wait()
        self.reopening.clear()
        if self.failing is not None:
            raise self.failing
        self.devices = self.opened.pop(0)
        return self.devices

    async def current(self) -> Placement:
        return self.defaults

    def wanted(self) -> Placement:
        return self.defaults

    async def say(self, fact: SystemFact) -> None:
        self.said.append(fact)

    def notice(self, defaults: Placement) -> None:
        self.defaults = defaults
        self.changes.set()

    async def until(self, event: asyncio.Event) -> None:
        """Return once `event` is set, moved by what the follower records; raise what stopped it if it stops first."""
        await patiently(lambda: event.is_set() or self.task.done(), lambda: len(self.recorded))
        if self.task.done():
            self.task.result()
            raise AssertionError("the follower returned, which it never does until cancelled")

    async def settled(self) -> None:
        await self.until(self.changes.settled)

    @property
    def moves(self) -> list[WideEvent]:
        return events(self.recorded, "devices.moved")

    async def stop(self) -> None:
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task


async def test_a_change_of_defaults_reopens_the_transport_and_says_where_the_audio_went() -> None:
    follower = Follower(BUILT_IN)
    follower.release.clear()
    await follower.settled()
    # An unplugged headset changes the default input and output together, as two notices, one while the reopen runs.
    follower.notice(on(2))
    await follower.until(follower.reopening)
    follower.notice(on(2))
    follower.release.set()
    await follower.settled()
    await follower.stop()
    assert follower.reopens == 1
    assert follower.said == [AudioMoved(BUILT_IN)]
    [move] = follower.moves
    moved = on(2)
    assert (move.outcome, move.facts) == ("ok", {"before": HEADSET, "wanted": moved, "after": BUILT_IN, "opened_on": moved})


async def test_a_change_that_comes_while_the_transport_reopens_is_followed_too() -> None:
    follower = Follower(HEADSET, BUILT_IN)
    follower.devices = BUILT_IN
    follower.release.clear()
    await follower.settled()
    follower.notice(on(2))  # plugged in
    await follower.until(follower.reopening)
    follower.notice(on(1))  # and pulled out again before the first reopen finished
    follower.release.set()
    await follower.settled()
    await follower.stop()
    assert follower.said == [AudioMoved(HEADSET), AudioMoved(BUILT_IN)]
    assert [(move.facts["before"], move.facts["after"]) for move in follower.moves] == [(BUILT_IN, HEADSET), (HEADSET, BUILT_IN)]


def test_where_the_audio_went_is_said_by_the_devices_names() -> None:
    assert system_text(AudioMoved(BUILT_IN)) == "Audio moved: listening on MacBook Pro Microphone, speaking on MacBook Pro Speakers."


async def test_a_notice_that_leaves_the_defaults_where_they_were_reopens_nothing() -> None:
    follower = Follower()
    await follower.settled()
    follower.notice(follower.defaults)
    await follower.settled()
    await follower.stop()
    assert (follower.reopens, follower.said, follower.recorded) == (0, [], [])


async def test_a_headset_unplugged_before_the_pipeline_started_is_followed_once_it_has() -> None:
    follower = Follower(BUILT_IN)
    follower.defaults = on(2)  # moved while the models loaded, after PortAudio listed the old ones
    await follower.settled()
    await follower.stop()
    assert follower.said == [AudioMoved(BUILT_IN)]


async def test_a_follower_stopped_during_a_reopen_waits_for_it_to_finish() -> None:
    follower = Follower(BUILT_IN)
    follower.release.clear()
    await follower.settled()
    follower.notice(on(2))
    await follower.until(follower.reopening)
    follower.task.cancel()
    await asyncio.sleep(0.01)
    follower.task.cancel()  # told again, as a run stopped while it stops tells it
    await asyncio.sleep(0.01)
    assert not follower.task.done()  # the reopen holds the streams, so the stop waits for it
    follower.release.set()
    await asyncio.wait({follower.task})
    assert follower.task.cancelled()
    assert follower.opened == []  # the reopen ran to its end
    assert follower.said == []  # and nothing was said by a follower told to stop
    [move] = follower.moves  # and the move it was stopped during says where the streams went
    assert (move.outcome, move.facts["after"]) == ("cancelled", BUILT_IN)


async def test_a_reopen_that_fails_while_the_follower_is_stopped_fails_its_move() -> None:
    follower = Follower(BUILT_IN)
    follower.failing = OSError("no default output device")
    follower.release.clear()
    await follower.settled()
    follower.notice(on(2))
    await follower.until(follower.reopening)
    follower.task.cancel()
    await asyncio.sleep(0.01)
    follower.task.cancel()
    follower.release.set()
    await asyncio.wait({follower.task})
    with pytest.raises(OSError):
        follower.task.result()
    [move] = follower.moves
    assert (move.outcome, move.error) == ("failed", "OSError: no default output device")


async def test_a_reopen_that_fails_stops_the_follower_and_its_move_is_failed() -> None:
    follower = Follower(BUILT_IN)
    follower.failing = OSError("no default output device")
    await follower.settled()
    follower.notice(on(2))
    await asyncio.wait({follower.task})
    with pytest.raises(OSError):
        follower.task.result()
    [move] = follower.moves
    assert (move.outcome, move.error, move.facts) == ("failed", "OSError: no default output device", {"before": HEADSET, "wanted": on(2)})
    assert follower.said == []


DISPLAY = Choices(Named("Studio Display Microphone"), Named("Studio Display Speakers"))
ON_DISPLAY = Devices(input="Studio Display Microphone", output="Studio Display Speakers")


def plugged(default: int, *names: str, generation: int = 1) -> Hardware:
    """The Mac's own microphone and speakers, the headset, and the devices `names`, on `default`. CoreAudio gave each
    device its id as plugged in `generation`: one that left and came back is plugged in at a later one."""

    def ided(listed: tuple[str, ...]) -> dict[str, int]:
        return {name: generation * 100 + index for index, name in enumerate(listed)}

    return Hardware(DefaultDevices(default, default), ided(("MacBook Pro Microphone", "headset", *names)), ided(("MacBook Pro Speakers", "headset", *names)))


def test_auto_follows_the_defaults_and_a_named_device_keeps_to_itself_while_it_is_there() -> None:
    there = plugged(1, "Studio Display Microphone", "Studio Display Speakers")
    assert placed(Choices(), there) == Placement(OnDefault(1), OnDefault(1))
    assert placed(DISPLAY, there) == Placement(OnNamed("Studio Display Microphone", 102), OnNamed("Studio Display Speakers", 103))
    # Missing, a named device puts its side on the default until it is back; a microphone is no speaker.
    assert placed(DISPLAY, plugged(3)) == Placement(OnDefault(3), OnDefault(3))
    assert placed(Choices(Auto(), Named("MacBook Pro Microphone")), there).output == OnDefault(1)


async def test_a_default_that_moves_under_named_devices_reopens_nothing() -> None:
    follower = Follower(at=placed(DISPLAY, plugged(1, "Studio Display Microphone", "Studio Display Speakers")))
    await follower.settled()
    follower.notice(placed(DISPLAY, plugged(2, "Studio Display Microphone", "Studio Display Speakers")))  # a headset plugged in, the new default
    await follower.settled()
    await follower.stop()
    assert (follower.reopens, follower.said) == (0, [])


async def test_a_named_device_that_leaves_and_comes_back_between_two_reads_is_a_new_device_and_reopens_its_side() -> None:
    follower = Follower(ON_DISPLAY, at=placed(DISPLAY, plugged(1, "Studio Display Microphone", "Studio Display Speakers")))
    await follower.settled()
    # The notices of its going and its coming back were both heard before either was read: the list reads the same, the ids do not.
    follower.notice(placed(DISPLAY, plugged(1, "Studio Display Microphone", "Studio Display Speakers", generation=2)))
    await follower.settled()
    await follower.stop()
    assert follower.reopens == 1


async def test_a_side_left_off_is_wanted_nowhere_whatever_the_hardware_does() -> None:
    off = Choices(Off(), Off())
    assert placed(off, plugged(1)) == placed(off, plugged(2, "headset")) == Placement(Closed(), Closed())
    follower = Follower(at=placed(off, plugged(1)))
    await follower.settled()
    follower.notice(placed(off, plugged(2)))
    await follower.settled()
    await follower.stop()
    assert (follower.reopens, follower.said) == (0, [])


async def test_a_named_device_that_goes_is_left_for_the_default_said_so_and_returned_to_once_it_is_back() -> None:
    instead = Devices(input=Instead("Studio Display Microphone", "MacBook Pro Microphone"), output=Instead("Studio Display Speakers", "MacBook Pro Speakers"))
    follower = Follower(instead, ON_DISPLAY, at=placed(DISPLAY, plugged(1, "Studio Display Microphone", "Studio Display Speakers")))
    follower.devices = ON_DISPLAY
    await follower.settled()
    follower.notice(placed(DISPLAY, plugged(1)))  # the display unplugged: no default moved, the device list did
    await follower.settled()
    assert follower.said == [AudioMoved(instead)]
    follower.notice(placed(DISPLAY, plugged(1, "Studio Display Microphone", "Studio Display Speakers")))  # and plugged in again
    await follower.settled()
    await follower.stop()
    assert follower.said == [AudioMoved(instead), AudioMoved(ON_DISPLAY)]
    assert [system_text(fact) for fact in follower.said] == [
        "Audio moved: listening on MacBook Pro Microphone in place of the missing Studio Display Microphone, speaking on MacBook Pro Speakers in place of the missing Studio Display Speakers.",
        "Audio moved: listening on Studio Display Microphone, speaking on Studio Display Speakers.",
    ]
