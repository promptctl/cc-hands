"""The phone served: the driver that follows each turn the page's wake word opens, with ears of its own.

A wake the page heard opens a turn made by no hand (`hands.voice.phone`), and the gate says so (`PushToTalk.watch`). The
turn then ends by the rules of any turn the voice opened (`hands.voice.voiceturn`, the work around the models being
`hands.voice.turnlife`), which the desk runs on the room: Smart Turn's verdict, the silence past it, the limit, or, for a wake
nothing was asked after, being thrown away.

The driver owns only how that turn ends. [LAW:one-source-of-truth] The gate owns the turn: the driver opens its own when
told the gate opened one for a wake, hears the phone's audio from then on (never the second the page sent before the wake,
which was captured with the key at rest), and closes when told the gate ended it, whatever ended it: a press, the level
going off, a hand at the desk, the call. When its own rules end the turn it asks the gate to, naming the turn; a refused ask
means the gate ended the turn first and has told it.
"""

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Protocol

from hands.clock import Clock
from hands.sessions.wide import Begun, WideEvent, begun, ended
from hands.voice.ptt import GateEnded, GateOpened, KeyedAudio, PushToTalk
from hands.voice.shortcut import Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.turnlife import Ears, Life
from hands.voice.voiceturn import Ended, Going, Judging, Told, Turn, Waiting, step


class Phoned(Protocol):
    """The phone, as the driver follows it: the audio it gives the pipeline, and the moves the gate made that its tones and words
    follow."""

    def follow(self, heard: Callable[[KeyedAudio], None]) -> None: ...

    def took(self, taken: tuple[Move, ...]) -> None: ...


async def drive(clock: Clock, key: PushToTalk, phone: Phoned, ears: Ears, emit: Callable[[WideEvent], None]) -> None:
    """Follow every turn a wake opens on the phone until cancelled, in the order things happen: the gate's word of a turn
    opening and ending, the phone's audio, and what the turn's life tells of the turn, so a verdict is stepped as it is given,
    ahead of the audio heard while Smart Turn thought, and ends the turn it judged and no other."""
    events: asyncio.Queue[KeyedAudio | GateOpened | GateEnded | Told] = asyncio.Queue()
    # What has been told and is yet to be stepped, in the order it is stepped.
    heard: deque[Told] = deque()
    life = Life(clock, ears, events.put_nowait, heard.appendleft, emit, "phone")
    key.watch(events.put_nowait)
    phone.follow(events.put_nowait)
    # The turn a wake opened, while it is open, and when it opened, for its event.
    turn: Turn | None = None
    opened: Begun = begun(clock)
    while True:
        each = heard.popleft() if heard else await events.get()
        match each:
            case GateOpened(turn=number, by=by, at=at) if by == PHONE_WAKE:
                turn, opened = Waiting(number, at), begun(clock)
                life.moved(None, turn)
            case GateOpened():
                pass
            case GateEnded(turn=number, fate=fate, by=by) if turn is not None and turn.id == number:
                life.moved(turn, None)
                # [LAW:nothing-unseen] each turn followed is one event: how it ended, and what ended it where the gate did.
                ended("phone.turn", emit, opened, None, how=fate, ended_by=by)
                turn = None
            case GateEnded():
                pass
            case KeyedAudio() as frame:
                # Audio of the turn the gate has open for the wake, and no other: what was captured before it opened, with the
                # key at rest, is the clip's and not the driver's.
                if turn is not None and frame.key == "down" and frame.opened == PHONE_WAKE and frame.turn == turn.id:
                    heard.extend((await life.hear(frame.audio, True, turn)).told())
            case told if turn is not None:
                match step(turn, told):
                    case Going(turn=going):
                        life.moved(turn, going)
                        turn = going
                    case Judging(turn=judged):
                        await life.judge(judged)
                    case Ended(how=how, of=of):
                        phone.took(key.end(of, how, PHONE_WAKE))
            case _:
                # A deadline or a verdict of a turn the gate has ended.
                pass
