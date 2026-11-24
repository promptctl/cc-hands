"""The phone's driver: the one task that hears a turn a wake opened at the phone and ends it, by the rules of the desk's wake
word turn (`hands.voice.voiceturn`), with the work around the models that goes with them (`hands.voice.turnlife`).

The gate is the one authority on the phone's turn (`hands.voice.ptt`), and this driver follows it. It is told of a turn as
the gate takes the wake that opens it, hears the phone's audio from then on and no earlier (the second the page sent ahead of
the wake is the clip's, never the models'), and lets the turn go when the audio the gate keyed says it ended the turn: each
frame carries how many turns the gate has ended. It ends a turn only through `PushToTalk.end`, which takes the end only
while that turn is the one the gate is at, so a verdict, a limit or an ask of a turn a press, a hand at the desk, the level
or the call already ended ends nothing of the next one.
"""

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Literal

from hands.clock import Clock
from hands.sessions.wide import Begun, WideEvent, begun, ended
from hands.voice import voiceturn
from hands.voice.phone import WakeOpened
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.voice.shortcut import Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.turnlife import Ears, Life
from hands.voice.voiceturn import Ended, Going, How, Judging, Speaking, Told, Turn, Waiting, step

# How a turn the driver followed ended: by the rules of a voice turn's end, which the gate took, or by the gate first.
End = How | Literal["gate"]


async def drive(clock: Clock, key: PushToTalk, turns: asyncio.Queue[WakeOpened | KeyedAudio], ears: Ears, taken: Callable[[Move | None], None], emit: Callable[[WideEvent], None]) -> None:
    """Follow every turn a wake opens until cancelled, handing each end the gate took from it to `taken`.

    One task steps the turn, in the order things were heard, so the speech, the deadlines and Smart Turn's verdicts never
    race each other: a verdict is stepped as it is given, before any audio heard while the model thought, so it ends the
    turn it judged and no other."""
    events: asyncio.Queue[WakeOpened | KeyedAudio | Told] = asyncio.Queue()
    # What has been told and is yet to be stepped, in the order it is stepped.
    heard: deque[Told] = deque()
    life = Life(clock, ears, events.put_nowait, heard.appendleft, emit, "phone")
    # The turn the driver follows: None between turns.
    turn: Turn | None = None
    # [LAW:nothing-unseen] each turn followed is one `phone.turn`: how it ended, and how many frames it heard. Both are set as
    # a turn opens, and read as it closes.
    began: Begun = begun(clock)
    frames = 0

    def become(was: Turn | None, now: Turn | None, end: End = "gate") -> Turn | None:
        """The turn followed, as it goes from `was` to `now`."""
        nonlocal began, frames
        life.moved(was, now)
        match voiceturn.closed(was, now):
            case None:
                pass
            case gone:
                ended("phone.turn", emit, began, None, None, turn=gone.id, end=end, frames=frames)
        if voiceturn.opened(was, now) is not None:
            began, frames = begun(clock), 0
        return now

    async def following() -> None:
        while True:
            events.put_nowait(await turns.get())

    async with asyncio.TaskGroup() as group:
        group.create_task(following(), name="the phone's turns, followed")
        while True:
            each = heard.popleft() if heard else await events.get()
            match each:
                case WakeOpened(turn=opened, at=at):
                    # A turn the driver still held is over: the gate opens a turn only with none open.
                    turn = become(turn, Waiting(opened, at))
                case KeyedAudio(turn=named, audio=audio):
                    # [LAW:one-source-of-truth] the frame says how many turns the gate has ended, so it says whether the turn
                    # held is over; frames come in the order they were captured, so one of a later turn is the first to say so.
                    match turn:
                        case None:
                            pass
                        case Waiting(id=held) | Speaking(id=held) if held != named:
                            turn = become(turn, None)
                        case held:
                            frames += 1
                            heard.extend((await life.hear(audio, True, held)).told())
                case told if turn is not None:
                    result = step(turn, told)
                    match result:
                        case Going(turn=now) | Judging(turn=now):
                            turn = become(turn, now)
                        case Ended(how=how, of=of):
                            # The gate takes the end only while this is its turn: what it did not take was over already.
                            moved = key.end(of, how, PHONE_WAKE)
                            taken(moved)
                            turn = become(turn, None, "gate" if moved is None else how)
                    match result:
                        case Judging(turn=speaking):
                            await life.judge(speaking)
                        case _:
                            pass
                case _:
                    # Told of a turn that is over, with none followed: nothing is the turn's to step.
                    pass
