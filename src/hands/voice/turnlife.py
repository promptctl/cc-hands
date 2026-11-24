"""A voice turn's life on its ears: the work around the models that goes with the rules of how a turn ends
(`hands.voice.voiceturn`), for whoever runs a turn.

Two models listen to a microphone, heard through the echo canceller as every buffer is: Silero's voice activity detector
says where speech starts and stops, and at each stop Smart Turn says whether it is the end of the turn or a pause inside
it, from how the speech ended. `Life` is what is done with them for a turn the voice opened: hearing afresh from a wake,
feeding Smart Turn what it is fed, judging, clearing as a turn ends, and setting the deadlines a turn asks for. It holds
no turn and orders nothing: the turn is the desk's phase, or `Runner.turn` for a runner with no desk, and whoever holds the
turn runs one use of the models at a time, so a verdict is stepped against the turn it judged.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.clock import Clock
from hands.sessions import wide
from hands.sessions.wide import WideEvent, annotate, unit
from hands.voice import voiceturn
from hands.voice.voiceturn import Ended, Going, SpeechStarted, SpeechStarting, SpeechStopped, Told, Turn, TurnEnded


class Ears(Protocol):
    """What a turn's life asks of the models that listen: Silero's detector and Smart Turn, in one."""

    async def detect(self, audio: bytes) -> VADState: ...
    def heard(self, audio: bytes, speech: bool) -> bool: ...
    async def judge(self) -> tuple[bool, TurnMetricsData | None]: ...
    def clear(self) -> None: ...
    def afresh(self) -> None: ...


@dataclass(frozen=True)
class Detected:
    """What one buffer of audio said: where the detector's state moved, and whether Smart Turn's silence ran out in a turn
    that was open."""

    changes: tuple[Told, ...]
    silence: TurnEnded | None

    def told[Between](self, between: tuple[Between, ...] = ()) -> tuple[Told | Between, ...]:
        """What the buffer said, in the one order it is stepped in: the speech, then what else a runner heard of the buffer
        (the desk's wake word), then the silence."""
        return (*self.changes, *between, *([self.silence] if self.silence else []))


class Life:
    """The work around the models for the turns of one stream of audio, with the deadlines each asks of `later`. The models
    take one use at a time, which the holder of the turn sees to: the desk's loop, or `Runner`."""

    def __init__(self, clock: Clock, ears: Ears, later: Callable[[Told], object], emit: Callable[[WideEvent], None], name: str) -> None:
        self._clock = clock
        self._ears = ears
        self._later = later
        self._emit = emit
        # The unit of work each verdict is: `<name>.judged`.
        self._name = name
        # The silence the last buffer was heard in, as the detector said it.
        self._detected = VADState.QUIET

    def afresh(self) -> None:
        """Hear on as if nothing came before: Silero is quiet until sure of speech again, and Smart Turn holds none."""
        self._ears.afresh()
        self._detected = VADState.QUIET

    def quiet(self) -> None:
        """Forget the silence the last buffer was heard in, so the speech the next buffer says starts as any speech does."""
        self._detected = VADState.QUIET

    def opened(self, turn: Turn) -> None:
        """A turn has just opened (`voiceturn.opened` says which): it is heard afresh from where it opened, if a wake
        opened it, and its deadlines are set."""
        # [LAW:types-are-the-program] the wake word's own speech is no part of what is asked: heard afresh from where it
        # opened the turn, what is asked starts as any speech does, and Smart Turn judges it alone.
        if isinstance(turn, voiceturn.Waiting):
            self.afresh()
        for at, told in voiceturn.deadlines(turn):
            self._clock.call_at(at, partial(self._later, told))

    async def hear(self, audio: bytes, speech_allowed: bool, turn: Turn | None) -> Detected:
        """Hear a buffer: detect, and feed Smart Turn where the detector is sure of speech and `speech_allowed`, as Pipecat
        feeds it, so the room heard otherwise is trimmed as silence is. `turn` is the one open as it is heard."""
        state = await self._ears.detect(audio)
        silent_too_long = self._ears.heard(audio, speech_allowed and state == VADState.SPEAKING)
        changes = _changed(self._detected, state, self._clock)
        self._detected = state
        return Detected(changes, TurnEnded("silence", turn.since) if silent_too_long and turn is not None else None)

    async def judge(self, turn: Turn) -> TurnEnded | None:
        """Ask Smart Turn whether the speech that just stopped ends `turn`: its own unit of work, with its verdict. The
        answer names the turn it was asked for."""
        with unit(f"{self._name}.judged", self._emit, wide.begun(self._clock)):
            complete, metrics = await self._ears.judge()
            # [LAW:nothing-unseen] the verdict, and how sure of it the model was; no metrics is a segment with no audio in it.
            annotate(complete=complete, probability=None if metrics is None else metrics.probability, inference_ms=None if metrics is None else metrics.e2e_processing_time_ms)
        return TurnEnded("verdict", turn.since) if complete else None

    def ended(self) -> None:
        """A voice turn is over, by its own rules or by something outside its life, a gate that took it: Smart Turn starts
        the next with no speech of this one in it. Pipecat clears it only as a verdict or the silence completes a turn, so a
        turn thrown away or sent at the limit is cleared here."""
        self._ears.clear()


class Runner:
    """The one turn open for a runner with no desk, and its life: handed a turn already open, as a wake opens one, it ends
    the turn by the rules of any voice turn's life, or is told by whoever owns the gate that something else did.

    [LAW:no-ambient-temporal-coupling] Which turn is open and what the models are fed or reset with have one owner: each
    method is one critical section under one lock, so what is heard is stepped against the turn it was heard in, a verdict
    against the turn it judged, and a cut, an open or a deadline from another task waits behind the models in use."""

    def __init__(self, life: Life) -> None:
        self._life = life
        self._turn: Turn | None = None
        self._ordered = asyncio.Lock()

    @property
    def turn(self) -> Turn | None:
        return self._turn

    async def open(self, turn: Turn) -> None:
        async with self._ordered:
            # [LAW:no-silent-failure] a turn opened over another means the runner and the gate disagree about which is open.
            if self._turn is not None:
                raise ValueError(f"a turn opened at {turn.since} while the one opened at {self._turn.since} is open")
            self._turn = turn
            self._life.opened(turn)

    async def cut(self) -> None:
        """The turn was ended by something outside its life: the life ends with it, and clears the models as any end does.
        A turn the runner ended itself first is already over, and the gate's cut for it ends nothing."""
        async with self._ordered:
            if self._turn is not None:
                self._turn = None
                self._life.ended()

    async def hear(self, audio: bytes, speech_allowed: bool = True) -> Ended | None:
        """Hear a buffer, and step what it said, in order. How the turn ended, if it did."""
        async with self._ordered:
            detected = await self._life.hear(audio, speech_allowed, self._turn)
            for told in detected.told():
                if (ended := await self._step(told)) is not None:
                    return ended
            return None

    async def advance(self, told: Told) -> Ended | None:
        """Step the open turn with what it was told, a deadline come due. How the turn ended, if it did."""
        async with self._ordered:
            return await self._step(told)

    async def _step(self, told: Told) -> Ended | None:
        """Step the open turn with `told`, asking Smart Turn when the speech stopped, and its verdict at once."""
        if self._turn is None:
            return None
        match voiceturn.step(self._turn, told):
            case Ended() as ended:
                self._turn = None
                self._life.ended()
                return ended
            case Going(turn=turn, judge=judge):
                self._turn = turn
                if judge and (verdict := await self._life.judge(turn)) is not None:
                    return await self._step(verdict)
                return None


def _changed(was: VADState, now: VADState, clock: Clock) -> tuple[Told, ...]:
    """What the detector's state moving from `was` to `now` says of the speech."""
    match was, now:
        case VADState.QUIET, VADState.STARTING:
            return (SpeechStarting(),)
        case VADState.QUIET | VADState.STARTING, VADState.SPEAKING:
            return (SpeechStarted(clock.now()),)
        case VADState.STARTING | VADState.SPEAKING | VADState.STOPPING, VADState.QUIET:
            return (SpeechStopped(),)
        case _:
            return ()
