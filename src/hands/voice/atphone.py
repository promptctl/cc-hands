"""The phone's driver: the one task that hears a turn a wake opened at the phone and ends it, by the rules of the desk's wake
word turn (`hands.voice.voiceturn`), with the work around the models that goes with them (`hands.voice.turnlife`).

The gate is the one authority on the phone's turn (`hands.voice.ptt`): `Phone` opens it as it reads the page's wake, and
this driver follows. It hears the audio the gate tagged as of a wake's turn, from the wake on and no earlier (the second the
page sent ahead of the wake is the clip's, never the models'), opens its own turn on the first frame of a turn it does not
hold, and ends it only through `PushToTalk.end`, which takes the end only while that turn is the one open. So a verdict, a
limit or an ask of a turn a press, a hand at the desk, the level or the call already ended ends nothing of the next one, and
the driver lets go of the turn the gate took.
"""

import asyncio
from collections import deque
from collections.abc import Callable

from hands.clock import Clock
from hands.sessions.wide import WideEvent
from hands.voice.clips import Trigger
from hands.voice.phone import WakeAudio
from hands.voice.ptt import PushToTalk
from hands.voice.shortcut import Move
from hands.voice.turnlife import Ears, Life
from hands.voice.voiceturn import Ended, Going, Judging, Speaking, Told, Turn, TurnId, Waiting, step

# The trigger the driver ends its turns as: the one that opened them.
TRIGGER: Trigger = "phone wake word"


async def drive(clock: Clock, key: PushToTalk, wakes: asyncio.Queue[WakeAudio], ears: Ears, taken: Callable[[Move], None], emit: Callable[[WideEvent], None]) -> None:
    """Hear every turn a wake opens until cancelled, handing each move the gate took from it to `taken`.

    One task steps the turn, in the order things were heard, so the speech, the deadlines and Smart Turn's verdicts never
    race each other: a verdict is stepped as it is given, before any audio heard while the model thought, so it ends the
    turn it judged and no other."""
    events: asyncio.Queue[WakeAudio | Told] = asyncio.Queue()
    # What has been told and is yet to be stepped, in the order it is stepped.
    heard: deque[Told] = deque()
    life = Life(clock, ears, events.put_nowait, heard.appendleft, emit, "phone")
    # The turn the driver holds: None until a wake's first audio, and after the turn ends.
    turn: Turn | None = None
    # The newest turn the driver has let go of. The gate numbers turns in the order it opens them, and `Phone` queues audio
    # in the order it was captured, so audio of a turn at or before it was queued before the driver let that turn go.
    over = TurnId(-1)

    async def hearing() -> None:
        while True:
            events.put_nowait(await wakes.get())

    async with asyncio.TaskGroup() as group:
        group.create_task(hearing(), name="the phone's wakes, heard")
        while True:
            each = heard.popleft() if heard else await events.get()
            was = turn
            match each:
                case WakeAudio(turn=named, audio=audio, at=at):
                    # [LAW:one-source-of-truth] the gate says which turn the audio is of: the one the driver holds goes
                    # on, a newer one is a turn a wake opened and the one held, if any, is over, and an older one or one
                    # the driver ended is heard by no one.
                    match was:
                        case Waiting(id=held) | Speaking(id=held) if held == named:
                            now: Turn | None = was
                        case Waiting(id=held) | Speaking(id=held) if named < held:
                            now = None
                        case _:
                            now = Waiting(named, at) if named > over else None
                    if now is not None:
                        turn = now
                        life.moved(was, turn)
                        heard.extend((await life.hear(audio, True, turn)).told())
                case told if turn is not None:
                    result = step(turn, told)
                    match result:
                        case Going(turn=now) | Judging(turn=now):
                            turn = now
                        case Ended(how=how, of=of):
                            # The gate takes the end only while this is its turn: what it did not take was over already.
                            match key.end(of, how, TRIGGER):
                                case None:
                                    pass
                                case moved:
                                    taken(moved)
                            turn, over = None, of
                    life.moved(was, turn)
                    match result:
                        case Judging(turn=speaking):
                            await life.judge(speaking)
                        case _:
                            pass
                case _:
                    # Told of a turn that is over, with none held: nothing is the turn's to step.
                    pass
