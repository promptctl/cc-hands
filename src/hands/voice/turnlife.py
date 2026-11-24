"""The work around the models for a turn the voice opened, for whoever owns the turn: the rules of how it ends are
`hands.voice.voiceturn`.

Two models listen to a microphone, heard through the echo canceller as every buffer is: Silero's voice activity detector
says where speech starts and stops, and at each stop Smart Turn says whether it is the end of the turn or a pause inside
it, from how the speech ended. `Life` is what is done with them for a turn the voice opened: hearing afresh from a wake,
feeding Smart Turn what it is fed, judging, clearing as a turn ends, and setting the deadlines a turn asks for. It holds
no turn: which turn is open is its owner's, the desk's phase or the gate's, and each thing it is asked names the turn it is
for. What it makes names the turn too, so its owner applies it only to that turn.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.clock import Clock
from hands.sessions import wide
from hands.sessions.wide import WideEvent, annotate, unit
from hands.voice import voiceturn
from hands.voice.voiceturn import SpeechStarted, SpeechStarting, SpeechStopped, Told, Turn, TurnEnded


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
        """What the buffer said, in the one order it is stepped in: the speech, then what else its owner heard of the buffer
        (the desk's wake word), then the silence."""
        return (*self.changes, *between, *([self.silence] if self.silence else []))


class Life:
    """The work around the models for the turns of one stream of audio, with the deadlines each asks of `later`.

    The models take one use at a time: a use begun while another is in flight is refused out loud. The owner of the turn
    orders them, as the desk's loop does by awaiting each use before the next; an owner whose judge runs beside its next
    buffer keeps them apart itself."""

    def __init__(self, clock: Clock, ears: Ears, later: Callable[[Told], None], emit: Callable[[WideEvent], None], name: str) -> None:
        self._clock = clock
        self._ears = ears
        self._later = later
        self._emit = emit
        # The unit of work each verdict is: `<name>.judged`.
        self._name = name
        # The silence the last buffer was heard in, as the detector said it.
        self._detected = VADState.QUIET
        # The use of the models in flight, if one is.
        self._in_use: str | None = None

    @contextmanager
    def _using(self, use: str) -> Generator[None]:
        # [LAW:no-silent-failure] a use of the models beside another would clear Smart Turn under its own judge, or feed it
        # audio the judge was not asked about, and say nothing.
        if self._in_use is not None:
            raise RuntimeError(f"the models were asked to {use} while they {self._in_use}")
        self._in_use = use
        try:
            yield
        finally:
            self._in_use = None

    def afresh(self) -> None:
        """Hear on as if nothing came before: Silero is quiet until sure of speech again, and Smart Turn holds none."""
        with self._using("hear afresh"):
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
        feeds it, so the room heard otherwise is trimmed as silence is. `turn` is the one open as it is heard: the silence
        that runs out in it ends it, and in no turn it ends nothing."""
        with self._using("hear a buffer"):
            state = await self._ears.detect(audio)
            silent_too_long = self._ears.heard(audio, speech_allowed and state == VADState.SPEAKING)
        changes = _changed(self._detected, state, self._clock)
        self._detected = state
        return Detected(changes, TurnEnded("silence", turn.since) if silent_too_long and turn is not None else None)

    async def judge(self, turn: Turn) -> TurnEnded | None:
        """Ask Smart Turn whether the speech that just stopped ends `turn`: its own unit of work, with its verdict. The
        answer names the turn it was asked for."""
        with self._using("judge"), unit(f"{self._name}.judged", self._emit, wide.begun(self._clock)):
            complete, metrics = await self._ears.judge()
            # [LAW:nothing-unseen] the verdict, and how sure of it the model was; no metrics is a segment with no audio in it.
            annotate(complete=complete, probability=None if metrics is None else metrics.probability, inference_ms=None if metrics is None else metrics.e2e_processing_time_ms)
        return TurnEnded("verdict", turn.since) if complete else None

    def ended(self) -> None:
        """A voice turn is over, by its own rules or by something outside its life, a gate that took it: Smart Turn starts
        the next with no speech of this one in it. Pipecat clears it only as a verdict or the silence completes a turn, so a
        turn thrown away or sent at the limit is cleared here."""
        with self._using("clear"):
            self._ears.clear()


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
