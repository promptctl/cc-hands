"""A voice turn's life on its ears: the work around the models that goes with the rules of how a turn ends
(`hands.voice.voiceturn`), for whoever runs a turn.

Two models listen to a microphone, heard through the echo canceller as every buffer is: Silero's voice activity detector
says where speech starts and stops, and at each stop Smart Turn says whether it is the end of the turn or a pause inside
it, from how the speech ended. `Life` is what is done with them for a turn the voice opened: hearing afresh from a wake,
feeding Smart Turn what it is fed, judging, clearing as a turn ends, and setting the deadlines a turn asks for. It holds
no turn: the turn is the runner's, the desk's phase, or `Runner.turn` for a runner with no desk, and a verdict is stepped
against the turn the runner has open once the judge has answered, never the one it had when it asked.
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
from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice import voiceturn
from hands.voice.voiceturn import Ended, Going, SpeechStarted, SpeechStarting, SpeechStopped, Told, Turn, TurnEnded


class Ears(Protocol):
    """What a turn's life asks of the models that listen: Silero's detector and Smart Turn, in one."""

    async def detect(self, audio: bytes) -> VADState: ...
    def heard(self, audio: bytes, speech: bool) -> bool: ...
    async def judge(self) -> tuple[bool, TurnMetricsData | None]: ...
    def clear(self) -> None: ...
    def afresh(self) -> None: ...


# The counts a life makes on the event of the unit of work open where it runs, which that unit declares.
COUNTED: tuple[str, ...] = ("held_open", "ended_on_silence")


@dataclass(frozen=True)
class Buffer:
    """What one buffer of audio said: where the detector's state moved, and whether Smart Turn's silence ran out in a turn
    that was open. They are apart so a runner puts what else it heard of the buffer between them, in the order the desk
    always has."""

    changes: tuple[SpeechStarting | SpeechStarted | SpeechStopped, ...]
    silence: TurnEnded | None


class Life:
    """The work around the models for the turns of one stream of audio, with the deadlines each asks of `later`."""

    def __init__(self, clock: Clock, ears: Ears, later: Callable[[Told], object], emit: Callable[[WideEvent], None], name: str) -> None:
        self._clock = clock
        self._ears = ears
        self._later = later
        self._emit = emit
        # The unit of work each verdict is: `<name>.judged`.
        self._name = name
        # The silence the last buffer was heard in, as the detector said it.
        self._detected = VADState.QUIET
        # Held while a judge is in flight and while a buffer is heard, so audio is fed to Smart Turn only after a judge that
        # was asked before it has answered, and the models are never reset under their own thread.
        self._models = asyncio.Lock()
        # A reset asked while the models were in use waits for the next buffer, which is the first thing heard after it.
        self._afresh_pending = False
        self._clear_pending = False

    def afresh(self) -> None:
        """Hear on as if nothing came before: Silero is quiet until sure of speech again, and Smart Turn holds none."""
        if self._models.locked():
            self._afresh_pending = True
        else:
            self._afresh_now()

    def _afresh_now(self) -> None:
        self._ears.afresh()
        self._detected = VADState.QUIET

    def quiet(self) -> None:
        """Forget the silence the last buffer was heard in, so the speech the next buffer says starts as any speech does."""
        self._detected = VADState.QUIET

    def opened(self, was: Turn | None, now: Turn | None) -> None:
        """The turn going from `was` to `now`: one that has just opened is heard afresh from where it opened, if a wake
        opened it, and its deadlines are set."""
        # [LAW:single-enforcer] one place says a turn has just opened, and what follows.
        if (turn := voiceturn.opened(was, now)) is None:
            return
        # [LAW:types-are-the-program] the wake word's own speech is no part of what is asked: heard afresh from where it
        # opened the turn, what is asked starts as any speech does, and Smart Turn judges it alone.
        if isinstance(turn, voiceturn.Waiting):
            self.afresh()
        for at, told in voiceturn.deadlines(turn):
            self._clock.call_at(at, partial(self._later, told))

    async def hear(self, audio: bytes, speech_allowed: bool, turn: Turn | None) -> Buffer:
        """Hear a buffer: detect, and feed Smart Turn where the detector is sure of speech and `speech_allowed`, as Pipecat
        feeds it, so the room heard otherwise is trimmed as silence is. `turn` is the one open as it is heard."""
        async with self._models:
            self._settle()
            state = await self._ears.detect(audio)
            silent_too_long = self._ears.heard(audio, speech_allowed and state == VADState.SPEAKING)
        changes = _changed(self._detected, state, self._clock)
        self._detected = state
        return Buffer(changes, TurnEnded("silence", turn.since) if silent_too_long and turn is not None else None)

    async def judge(self, turn: Turn) -> TurnEnded | None:
        """Ask Smart Turn whether the speech that just stopped ends `turn`: its own unit of work, with its verdict. The
        answer names the turn it was asked for, and says nothing of the one open when it comes."""
        async with self._models:
            with unit(f"{self._name}.judged", self._emit, wide.begun(self._clock)):
                complete, metrics = await self._ears.judge()
                # [LAW:nothing-unseen] the verdict, and how sure of it the model was; no metrics is a segment with no audio
                # in it.
                annotate(complete=complete, probability=None if metrics is None else metrics.probability, inference_ms=None if metrics is None else metrics.e2e_processing_time_ms)
        if complete:
            return TurnEnded("verdict", turn.since)
        count(held_open=1)
        return None

    def ended(self, cause: object) -> None:
        """A voice turn is over, by `cause`, or by something outside its life, a gate that took it: Smart Turn starts the
        next with no speech of this one in it, as Pipecat clears it as each turn ends."""
        if isinstance(cause, TurnEnded) and cause.by == "silence":
            count(ended_on_silence=1)
        if self._models.locked():
            self._clear_pending = True
        else:
            self._ears.clear()

    def _settle(self) -> None:
        """Do the resets that waited for the models, before a buffer is heard with them."""
        if self._afresh_pending:
            self._afresh_pending = False
            self._afresh_now()
        if self._clear_pending:
            self._clear_pending = False
            self._ears.clear()


class Runner:
    """The one turn open for a runner with no desk, and its life: handed a turn already open, as a wake opens one, it ends
    the turn by the rules of any voice turn's life, or is told by whoever owns the gate that something else did."""

    def __init__(self, life: Life) -> None:
        self._life = life
        self.turn: Turn | None = None

    def open(self, turn: Turn) -> None:
        # [LAW:no-silent-failure] a turn opened over another means the runner and the gate disagree about which is open.
        if self.turn is not None:
            raise ValueError(f"a turn opened at {turn.since} while the one opened at {self.turn.since} is open")
        self.turn = turn
        self._life.opened(None, turn)

    def cut(self) -> None:
        """The turn was ended by something outside its life: the life ends with it, and clears the models as any end does."""
        if self.turn is not None:
            self.turn = None
            self._life.ended(None)

    async def hear(self, audio: bytes, speech_allowed: bool = True) -> Ended | None:
        """Hear a buffer, and step what it said, in order, while the turn is open."""
        buffer = await self._life.hear(audio, speech_allowed, self.turn)
        for told in (*buffer.changes, *([buffer.silence] if buffer.silence else [])):
            if isinstance(told, SpeechStarting):
                continue
            if (ended := await self.advance(told)) is not None:
                return ended
        return None

    async def advance(self, told: Told) -> Ended | None:
        """Step the open turn with `told`, asking Smart Turn when the speech stopped, and stepping its verdict at once
        against the turn open when it answers. How the turn ended, if it did."""
        asked = self.turn
        if asked is None:
            return None
        match voiceturn.step(asked, told):
            case Ended() as ended:
                self.turn = None
                self._life.ended(told)
                return ended
            case Going(turn=turn, judge=judge):
                self.turn = turn
                if judge and (verdict := await self._life.judge(turn)) is not None:
                    return await self.advance(verdict)
                return None


def _changed(was: VADState, now: VADState, clock: Clock) -> tuple[SpeechStarting | SpeechStarted | SpeechStopped, ...]:
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
