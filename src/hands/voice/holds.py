"""What a hold is, and the two ways a model hears one; nothing here loads Pipecat.

A hold is one span of audio the gate opened and closed: the talk key's press to its release, engaged conversation's
detector from speech to the end of the turn, a phone's button, the wake word; or words the user typed, a hold with no
audio. The hold owner (`hands.voice.holdowner`) cuts and numbers every hold and gives each model the same samples of it,
in the order they were captured. A model is one of two things to it:

- a `Listener`, which hears every hold of the key's as it is spoken and says what it heard, tagged by the owner with the
  hold it heard it in: what cuts hands off as the user speaks over it;
- the `Transcriber`, which is given each hold the key sent, whole, once it ends, and says what was said in it: the words
  the turn sends to the brain.

[LAW:locality-or-seam] a model added or replaced is one more listener, or another transcriber, and nothing about the
hold changes for it.
"""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Levels

# What every hold is said in.
LANGUAGE = "en"

# The sample rate every hold is heard at: each model is given mono 16-bit samples at this rate.
RATE = 16_000

# Bytes to a second of a hold's audio.
SECOND = RATE * 2


# [LAW:domain-language] an edge is what moves the gate (docs/architecture.md, "The gate has one owner and several
# edges"). The desk has three (`hands.voice.desk`): Right Shift held alone, at every level (`hands.voice.hold`); speech,
# engaged (`hands.voice.engaged`); and the wake word said, at its level (`hands.voice.wake`). The phone's is its page's
# talk button, there for every call.
Edge = Literal["held key", "engaged conversation", "wake word", "phone button"]
# What opens a hold: an edge of the gate, or words the user typed into the conversation page
# (`hands.voice.conversationpage`), which move no gate and are heard nowhere, so they are a hold of their own.
Opener = Edge | Literal["typed"]


@dataclass(frozen=True)
class Hold:
    """One hold, by the number the hold owner gave it as it opened, and what opened it."""

    number: int
    opener: Opener


# What the gate did with a hold of the key's as it ended: sent it to be transcribed, or threw it away.
Fate = Literal["sent", "dropped"]


def dbfs(audio: bytes) -> float | None:
    """The mean power of 16-bit audio in dB below full scale, to a tenth; None where no sample of it sounds."""
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64) / 32768
    power = float(np.dot(samples, samples))
    # [LAW:types-are-the-program] digital silence has no level in dB: it is said as an absence, not as a floor.
    return round(10 * math.log10(power / samples.size), 1) if power else None


# A hold's audio as a model hears it: mono samples in [-1, 1) at RATE.
type Samples = npt.NDArray[np.float32]


def decoded(audio: bytes) -> Samples:
    """Mono 16-bit `audio` as the samples in [-1, 1) a model hears: the one decoding of a hold, for every model alike."""
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / np.float32(32768)


@dataclass(frozen=True)
class Reference:
    """10 ms of what the desk's speaker played, as the echo canceller was given it: 16-bit at the speaker's own rate and
    channels, which are not a hold's; and each voice of hands' own TTS in it, as the TTS made it, none for silence or a
    cue."""

    audio: bytes
    rate: int
    channels: int
    voices: tuple[str, ...]


@dataclass(frozen=True)
class Sound:
    """Audio of a hold, mono 16-bit at RATE: as the pipeline hears it, after the echo canceller, and as the microphone
    captured it, before; the two the same length, and the same at the phone, which is heard through no canceller. And
    what the desk's speaker was playing as it was captured, the canceller's reference, one for each 10 ms of it; None at
    the phone, whose own canceller is told what it plays, and hands is not."""

    heard: bytes
    captured: bytes
    played: tuple[Reference, ...] | None

    @property
    def seconds(self) -> float:
        return len(self.heard) / SECOND

    @property
    def levels(self) -> Levels:
        """How loud it was on both sides of the echo canceller."""
        return Levels(dbfs(self.captured), dbfs(self.heard))


class Recording:
    """The audio of the hold being gathered, as the pipeline hears it, as the microphone captured it, before the echo
    canceller, and what the speaker played over it: appended and trimmed together, so a hold is measured on both sides
    of the canceller over the very audio it is made of, and lines up with what hands was saying."""

    def __init__(self) -> None:
        self._heard = bytearray()
        self._captured = bytearray()
        # None once a frame with no reference is added: a recording is of one place, cleared as the place moves.
        self._played: list[Reference] | None = []

    def add(self, sound: Sound, gathering: bool) -> None:
        """A frame more; while nothing is `gathering` it, only the last second is kept, the second a hold the desk was
        listening for opens with."""
        self._heard += sound.heard
        self._captured += sound.captured
        match self._played, sound.played:
            case list() as played, tuple() as more:
                played += more
            case _:
                self._played = None
        # [LAW:dataflow-not-control-flow] every frame trims; gathering, it trims nothing.
        cut = 0 if gathering else max(0, len(self._heard) - SECOND)
        del self._heard[:cut]
        del self._captured[:cut]
        match self._played:
            case list() as played:
                # One reference for each 10 ms cut.
                del played[: cut // (SECOND // 100)]
            case None:
                pass

    def clear(self) -> None:
        self._heard.clear()
        self._captured.clear()
        self._played = []

    @property
    def sound(self) -> Sound:
        return Sound(bytes(self._heard), bytes(self._captured), None if self._played is None else tuple(self._played))


@dataclass(frozen=True)
class Heard:
    """What a listener heard of the hold open: the words said in it, None where it heard none."""

    words: str | None


# A hearing: a listener's slow work on a hold, while it is open or after it closed, which the owner runs off the audio
# path, one at a time behind the listener's own hearings returned before it, bounds, and tells as one event. What the
# model made of the audio it puts on that event itself, from inside.
type Hearing = Callable[[], Awaitable[Heard]]

# What a listener has to hear of a hold after it was given more of it, or told it ended: nothing, often.
type Hearings = tuple[Hearing, ...]


class Listener(Protocol):
    """A model hearing every hold of the key's as it is captured, in capture order.

    [LAW:no-ambient-temporal-coupling] each call returns at once: work that takes longer runs in the hearings it returns,
    so no listener holds up the audio, another listener, or the hold, and the owner bounds and tells every piece of it.
    Whatever moves a model's own stream on (a hold's samples given, its end marked) is done in the call, before it
    returns, and a hearing only awaits a result: a lane backed up behind a slow hearing would otherwise let the next
    hold's samples reach the stream ahead of this one's end, and two holds would be heard as one.
    """

    @property
    def name(self) -> str:
        """The listener, as its hearings' events name it."""
        ...

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        """`hold` opens, `before` the audio it keeps from before it opened: what was heard while the press was arming,
        or the second the desk heard while it listened."""
        ...

    def heard(self, sound: Sound) -> Hearings:
        """More of the hold open, as it was captured."""
        ...

    def ended(self, hold: Hold, fate: Fate, sound: Sound) -> Hearings:
        """The hold open ends, sent or thrown away, `sound` all of it: what the listener still has to hear of it, such as
        the last words the streaming recognizer decodes from the silence after it, which settle after it closes."""
        ...


class Transcriber(Protocol):
    """The model whose words a turn sends: given each hold the key sent, whole, as mono 16-bit samples at RATE."""

    async def transcribed(self, hold: Hold, audio: bytes) -> str | None:
        """The words said in `hold`, None where nothing was; it raises where it cannot say."""
        ...
