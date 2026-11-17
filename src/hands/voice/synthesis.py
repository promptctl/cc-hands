"""Syntheses: each sentence hands' TTS makes into speech, followed from its making to the device taking its last frame.

A synthesis is told to the speaker ahead of its audio (`hands.voice.voices.SpeakingIn`) and its making behind it
(`Made`), and the speaker writes every chunk with the syntheses in it. The playout takes all of it in the order it was
written, on the device's clock, and `Tallies` reads off it what the device made of each synthesis: how much of it played,
and the gaps inside it, where the device found nothing written while more of the synthesis was still to come. A gap is
heard as a crackle: written ahead by only `WRITE_AHEAD_SECS`, the speech is broken by any stall longer than that.

[LAW:nothing-unseen] each synthesis is one wide event, `speech`, so a gap is seen in the sentence it broke, beside how
fast the TTS made that sentence: a TTS slower than real time starves the speaker; a gap under a fast one is delivery.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pipecat.frames.frames import DataFrame

from hands.sessions.wide import Begun


@dataclass(frozen=True)
class Synthesis:
    """One sentence hands' TTS made into speech, in `voice`, `begun` as the TTS was handed it."""

    voice: str
    begun: Begun


@dataclass
class Made(DataFrame):
    """Behind a synthesis's audio: how long the TTS took to make it, and how long its audio plays, in seconds."""

    synthesis: Synthesis
    making: float
    audio: float


@dataclass(frozen=True)
class Ended:
    """Hands' speech ended: every synthesis it held is played out, and the silence after it is no gap."""


@dataclass(frozen=True)
class Run:
    """`length` bytes written to the speaker in one write, with the syntheses whose audio is in them: none for the
    transport's own silence and padding, or a cue; two where Pipecat's sender joined the end of one to the start of the
    next in one chunk."""

    length: int
    syntheses: tuple[Synthesis, ...]


# What is written to the speaker, in the order the device takes it.
type Written = Run | Made | Ended


def voices(taken: Iterable[Written]) -> tuple[str, ...]:
    """The voices of hands' TTS in what the device took, in order, each once."""
    return tuple(dict.fromkeys(synthesis.voice for each in taken if isinstance(each, Run) for synthesis in each.syntheses))


@dataclass(frozen=True)
class Delivered:
    """What the device made of one synthesis, once it took the last of it: how much of it played and the gaps inside it,
    in milliseconds; how many callbacks PortAudio said came too late while it played; and its making, None where the TTS
    was stopped before it finished making it."""

    synthesis: Synthesis
    made: Made | None
    played_ms: float
    gaps: int
    gap_ms: float
    late: int


@dataclass
class _Tally:
    made: Made | None = None
    played: int = 0
    gaps: int = 0
    gap: int = 0
    # Bytes the device found missing since this synthesis's last audio: a gap if more of it follows, its end if not.
    missing: int = 0
    late: int = 0


@dataclass
class Tallies:
    """[LAW:single-enforcer] what the device made of each synthesis, read off what it takes; on the one thread that takes
    it. `finished` is handed each synthesis as the device takes the last of it, on that thread."""

    bytes_per_second: int
    finished: Callable[[Delivered], None]
    _open: dict[Synthesis, _Tally] = field(default_factory=dict[Synthesis, _Tally])

    def took(self, taken: Iterable[Written], missing: int, late: bool) -> None:
        """The device took `taken`, in order, then found `missing` bytes more than was written, `late` as PortAudio said."""
        for each in taken:
            match each:
                case Run(syntheses=()):
                    # Sound of no synthesis's: it neither carries one on nor ends one.
                    pass
                case Run(length=length, syntheses=syntheses):
                    # A synthesis ends where the device takes audio of the next and none of it.
                    self._finish([synthesis for synthesis in self._open if synthesis not in syntheses])
                    for synthesis in syntheses:
                        tally = self._open.setdefault(synthesis, _Tally())
                        tally.gaps += tally.missing > 0
                        tally.gap += tally.missing
                        tally.missing = 0
                        tally.played += length
                case Made(synthesis=synthesis):
                    self._open.setdefault(synthesis, _Tally()).made = each
                case Ended():
                    self._finish(list(self._open))
        for tally in self._open.values():
            tally.missing += missing
            tally.late += late

    def _finish(self, syntheses: list[Synthesis]) -> None:
        for synthesis in syntheses:
            tally = self._open.pop(synthesis)
            self.finished(Delivered(synthesis, tally.made, self._ms(tally.played), tally.gaps, self._ms(tally.gap), tally.late))

    def _ms(self, length: int) -> float:
        return round(length * 1000 / self.bytes_per_second, 3)
