"""Keeping the voice on the audio devices config.toml chooses: the system's defaults, which an unplugged headset moves,
or a device by name, which leaves and comes back; each move is said aloud, with where to.

[LAW:nothing-unseen] each move is one unit of work, `devices.moved`: the devices it left, where the sides were wanted that
moved it, the devices it reopened on and where it read them wanted as it did, which the next move is measured from, and
how long the reopen took; failed with what raised where the reopen did, whether or not the follower was stopped meanwhile.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, begun, unit
from hands.voice.coreaudio import device_changes
from hands.voice.devicechoice import Placement
from hands.voice.microphone import Devices
from hands.voice.system import AudioMoved, SystemFact
from hands.threads import off_loop


class Transport(Protocol):
    """The streams a follower moves: where they were wanted as they opened and the devices they are open on, where they
    are wanted now, and the reopen that moves them."""

    @property
    def opened_on(self) -> Placement: ...
    @property
    def devices(self) -> Devices: ...
    def wanted(self) -> Placement: ...
    async def reopen(self) -> Devices: ...


async def follow(
    clock: Clock,
    changes: asyncio.Event,
    current: Callable[[], Awaitable[Placement]],
    transport: Transport,
    say: Callable[[SystemFact], Awaitable[None]],
    record: Record,
) -> None:
    """Each time where the streams are wanted moves off where they were as they opened, reopen them there and say
    where; until cancelled."""

    async def reopened() -> Devices:
        # Annotated by the reopen itself, so a move the follower was stopped during still says where it went.
        devices = await transport.reopen()
        annotate(after=devices, opened_on=transport.opened_on)
        return devices

    while True:
        wanted = await moved(changes, current, transport.opened_on)
        # [LAW:no-silent-failure] a reopen that fails raises out of here and stops the run, which reads as down, so
        # the next run opens whatever devices there are, rather than one run going on silently deaf and mute.
        with unit("devices.moved", record, begun(clock)):
            annotate(before=transport.devices, wanted=wanted)
            devices = await finished(reopened())
        await say(AudioMoved(devices))


async def finished[T](work: Awaitable[T]) -> T:
    """The result of work that, once begun, is let finish even when the caller is cancelled; what it raised is raised
    either way."""
    # [LAW:no-ambient-temporal-coupling] a reopen holds the streams, part of it on a thread that cancelling cannot
    # stop. A follower stopped during one waits for it, within its deadline, so the pipeline's cleanup that comes next
    # finds the streams attached and closes them itself, instead of racing a reopen for them.
    # Waited on rather than awaited, since a wait never cancels what it waits on, however often it is cancelled itself.
    running = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while not running.done():
        try:
            await asyncio.wait({running})
        except asyncio.CancelledError as cancel:
            cancelled = cancel
    # [LAW:no-silent-failure] a reopen that failed while the follower was being stopped fails the move.
    result = running.result()
    if cancelled is not None:
        raise cancelled
    return result


async def moved(changes: asyncio.Event, current: Callable[[], Awaitable[Placement]], opened: Placement) -> Placement:
    """Where the streams are wanted, once it is no longer where they were as they opened.

    An unplugged headset changes the input and the output within a millisecond, as two notices; the second finds
    them wanted where the first reopened them. A change while a reopen runs is read after it, and followed. A change
    that moves nothing config.toml chose, a default for a side that keeps to a device by name, reopens nothing.
    """
    while (now := await current()) == opened:
        await changes.wait()
        changes.clear()
    return now


async def follow_devices(clock: Clock, started: asyncio.Event, audio: Transport, say: Callable[[SystemFact], Awaitable[None]], record: Record) -> None:
    """Follow the audio devices once the pipeline has started, until cancelled."""
    # [LAW:no-ambient-temporal-coupling] the streams are Pipecat's to open and start until the pipeline has started.
    # A change before then is still followed: where they are wanted is compared with where they were as PortAudio
    # listed the devices.
    await started.wait()
    loop = asyncio.get_running_loop()
    changes = asyncio.Event()
    with device_changes(lambda: loop.call_soon_threadsafe(changes.set)):
        # Read off the loop: CoreAudio answers under a lock it may be holding while it tears down a device that is gone.
        await follow(clock, changes, lambda: off_loop(audio.wanted, "reading the audio devices"), audio, say, record)
