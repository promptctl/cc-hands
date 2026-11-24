"""The phone's driver: the one task that hears a turn a wake opened at the phone and ends it, by the rules of the desk's wake
word turn (`hands.voice.voiceturn`), with the work around the models that goes with them (`hands.voice.turnlife`).

The gate is the one authority on the phone's turn (`hands.voice.ptt`), and this driver follows it. It is told of a turn as
the gate opens it for a wake, hears the phone's audio from then on and no earlier (the second the page sent ahead of
the wake is the clip's, never the models'), and is told as the gate ends the turn, whatever ended it: a press, a hand at the
desk, the level or the call. It ends a turn only through `PushToTalk.end`, which takes the end only while that turn is the
one the gate is at, so a verdict, a limit or an ask of a turn the gate already ended ends nothing, and the gate's own end
closes it.
"""

import asyncio
from collections import deque
from collections.abc import Callable

from hands.clock import Clock
from hands.sessions.wide import Begun, WideEvent, begun, ended
from hands.voice import voiceturn
from hands.voice.ptt import GateEnded, GateOpened, KeyedAudio, PushToTalk
from hands.voice.shortcut import Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.turnlife import Ears, Life
from hands.voice.voiceturn import Ended, Going, How, Judging, Told, Turn, Waiting, step

# How a turn the driver followed ended: by the rules of a voice turn's end, which the gate took, or by the gate first, which
# says what ended it.
End = How | GateEnded


async def drive(clock: Clock, key: PushToTalk, turns: asyncio.Queue[GateOpened | GateEnded | KeyedAudio], ears: Ears, taken: Callable[[Move | None], None], emit: Callable[[WideEvent], None]) -> None:
    """Follow every turn a wake opens until cancelled, handing each end the gate took from it to `taken`.

    One task steps the turn, in the order things were heard, so the speech, the deadlines and Smart Turn's verdicts never
    race each other: a verdict is stepped as it is given, before any audio heard while the model thought, so it ends the
    turn it judged and no other."""
    events: asyncio.Queue[GateOpened | GateEnded | KeyedAudio | Told] = asyncio.Queue()
    # What has been told and is yet to be stepped, in the order it is stepped.
    heard: deque[Told] = deque()
    life = Life(clock, ears, events.put_nowait, heard.appendleft, emit, "phone")
    # The turn the driver follows: None between turns.
    turn: Turn | None = None
    # [LAW:nothing-unseen] each turn followed is one `phone.turn`: how it ended, and how many frames it heard. Both are set as
    # a turn opens, and read as it closes.
    began: Begun = begun(clock)
    frames = 0

    def become(was: Turn | None, now: Turn) -> Turn:
        """The turn followed, as it opens or goes on, from `was` to `now`."""
        nonlocal began, frames
        life.moved(was, now)
        if voiceturn.opened(was, now) is not None:
            began, frames = begun(clock), 0
        return now

    def close(was: Turn, end: End) -> None:
        """The turn followed is over, `end`ed."""
        life.moved(was, None)
        match end:
            case GateEnded(by=by):
                ended("phone.turn", emit, began, None, None, turn=was.id, end="gate", by=by, frames=frames)
            case how:
                ended("phone.turn", emit, began, None, None, turn=was.id, end=how, frames=frames)

    async def following() -> None:
        while True:
            events.put_nowait(await turns.get())

    async with asyncio.TaskGroup() as group:
        group.create_task(following(), name="the phone's turns, followed")
        while True:
            each = heard.popleft() if heard else await events.get()
            match each:
                case GateOpened(turn=opened, by=by, at=at) if by == PHONE_WAKE:
                    # The gate opens a turn only with none open, and told us as it ended any before.
                    turn = become(turn, Waiting(opened, at))
                case GateOpened():
                    # A turn a hand opened, the page's button or the desk's: not one a wake word's driver follows.
                    pass
                case GateEnded(turn=gone) if turn is not None and turn.id == gone:
                    close(turn, each)
                    turn = None
                case GateEnded():
                    # A turn that was not the one followed: the desk's, or one the driver ended itself.
                    pass
                case KeyedAudio(audio=audio) if turn is not None:
                    frames += 1
                    heard.extend((await life.hear(audio, True, turn)).told())
                case KeyedAudio():
                    pass
                case told if turn is not None:
                    match step(turn, told):
                        case Going(turn=now):
                            turn = become(turn, now)
                        case Judging(turn=now):
                            turn = become(turn, now)
                            await life.judge(now)
                        case Ended(how=how, of=of):
                            match key.end(of, how, PHONE_WAKE):
                                case None:
                                    # The gate ended this turn first, and told us as it did: that end is queued, and closes it.
                                    pass
                                case moved:
                                    taken(moved)
                                    close(turn, how)
                                    turn = None
                case _:
                    # Told of a turn that is over, with none followed: nothing is the turn's to step.
                    pass
