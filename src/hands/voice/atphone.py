"""The phone served: the driver that follows each turn the page's wake word opens, with ears of its own.

A wake the page heard opens a turn made by no hand (`hands.voice.phone`), and the gate says so (`PushToTalk.watch`). The
turn then ends by the rules of any turn the voice opened (`hands.voice.voiceturn`, the work around the models being
`hands.voice.turnlife`), which the desk runs on the room: Smart Turn's verdict, the silence past it, the limit, or, for a wake
nothing was asked after, being thrown away.

The driver owns only how that turn ends. [LAW:one-source-of-truth] The gate owns the turn: the driver follows the one it is
told the gate opened for a wake, hears the phone's audio from then on (never the second the page sent before the wake,
which was captured with the key at rest), and stops following the moment the gate ends it, whatever ended it: a press, the
level going off, a hand at the desk, the call, or its own rules, which ask the gate to end it, naming the turn.
"""

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Protocol

from hands.clock import Clock
from hands.sessions.wide import Begun, WideEvent, begun, ended
from hands.voice.ptt import Cause, GateEnded, GateOpened, KeyedAudio, PushToTalk, fate_of
from hands.voice.shortcut import Ending, Move
from hands.voice.trigger import PHONE_WAKE
from hands.voice.turnlife import Ears, Life
from hands.voice.voiceturn import Ended, Going, Judging, Told, Turn, Waiting, step


class Phoned(Protocol):
    """The phone, as the driver follows it: the audio it gives the pipeline, and the moves the gate made that its tones and words
    follow."""

    def follow(self, heard: Callable[[KeyedAudio], None]) -> None: ...

    def took(self, taken: tuple[Move, ...]) -> None: ...


def drive(clock: Clock, key: PushToTalk, phone: Phoned, ears: Ears, emit: Callable[[WideEvent], None]) -> Coroutine[None, None, None]:
    """Follow every turn a wake opens on the phone until cancelled, in the order things happen: the gate's word of a turn
    opening and ending, the phone's audio, and what the turn's life tells of the turn, so a verdict is stepped as it is given,
    ahead of the audio heard while Smart Turn thought, and ends the turn it judged and no other.

    [LAW:no-ambient-temporal-coupling] the gate and the phone are followed from here, as it is called, so a wake that comes
    before the returned coroutine first runs is still followed."""
    events: asyncio.Queue[KeyedAudio | GateOpened | GateEnded | Told] = asyncio.Queue()
    key.watch(events.put_nowait)
    phone.follow(events.put_nowait)
    return _follow(clock, key, phone, ears, emit, events)


async def _follow(clock: Clock, key: PushToTalk, phone: Phoned, ears: Ears, emit: Callable[[WideEvent], None], events: asyncio.Queue[KeyedAudio | GateOpened | GateEnded | Told]) -> None:
    # What has been told and is yet to be stepped, in the order it is stepped.
    heard: deque[Told] = deque()
    life = Life(clock, ears, events.put_nowait, heard.appendleft, emit, "phone")
    # The turn a wake opened, while it is open, and when it opened, for its event.
    turn: Turn | None = None
    opened: Begun = begun(clock)

    def closed(move: Ending, by: Cause) -> None:
        # [LAW:nothing-unseen] each turn followed is one event: how it ended, by which move, and what ended it where the gate did.
        nonlocal turn
        life.moved(turn, None)
        ended("phone.turn", emit, opened, None, how=fate_of(move), move=move, ended_by=by)
        turn = None

    while True:
        each = heard.popleft() if heard else await events.get()
        match each:
            case GateOpened(turn=number, by=by, at=at) if by == PHONE_WAKE:
                turn, opened = Waiting(number, at), begun(clock)
                life.moved(None, turn)
            case GateOpened():
                pass
            case GateEnded(turn=number, move=move, by=by) if turn is not None and turn.id == number:
                closed(move, by)
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
                        moves = key.end(of, how, PHONE_WAKE)
                        phone.took(moves)
                        # The gate has ended the turn as it was asked to, and the driver follows it no further: what is queued
                        # of it, and the echo of the gate's word, find no turn. A refused ask means the gate ended the turn first,
                        # and its word, already queued, closes it.
                        if moves:
                            closed(how, PHONE_WAKE)
            case _:
                # A deadline or a verdict of a turn the gate has ended.
                pass
