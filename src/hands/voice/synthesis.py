"""Syntheses: each sentence hands' TTS makes into speech, followed from its making to the device taking its last byte.

The speaker sees each synthesis whole as the TTS hands it over (`SpeakingIn`, its audio, `Made`), before Pipecat's
sender cuts the audio into chunks that join the end of one synthesis to the start of the next. So it keeps a ledger of
the bytes heard and splits each chunk it writes exactly, into `Piece`s, each the bytes of one synthesis or of none.
The stream that plays a piece says what became of it: taken by the device, with what the device found missing
between, or never played. A synthesis is finished once the TTS has finished with it and every byte of it is
accounted for, and is then delivered.

A gap is heard as a crackle: written ahead by only `WRITE_AHEAD_SECS`, the speech is broken by any stall longer than
that. It is bytes the device found missing after some of a synthesis and before more of it; the silence after its
last byte is nobody's gap.

[LAW:nothing-unseen] each synthesis is one wide event, `speech`, so a gap is seen in the sentence it broke, beside how
fast the TTS made that sentence: a TTS slower than real time starves the speaker; a gap under a fast one is delivery.
"""

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from pipecat.frames.frames import DataFrame

from hands.sessions.wide import Begun


@dataclass(frozen=True)
class Synthesis:
    """One sentence hands' TTS made into speech, in `voice`, `begun` as the TTS was handed it."""

    voice: str
    begun: Begun


@dataclass
class Made(DataFrame):
    """Behind a synthesis's last audio: the TTS has finished with it, having taken `making` seconds to make it."""

    synthesis: Synthesis
    making: float


@dataclass(frozen=True)
class Piece:
    """`length` bytes of sound in order, of one synthesis's audio, or of none: the transport's silence and padding, a cue."""

    synthesis: Synthesis | None
    length: int


def voices(pieces: Iterable[Piece]) -> tuple[str, ...]:
    """The voices of hands' TTS in `pieces`, in order, each once."""
    return tuple(dict.fromkeys(piece.synthesis.voice for piece in pieces if piece.synthesis is not None))


def take(pieces: deque[Piece], length: int) -> list[Piece]:
    """The first `length` bytes of `pieces`, let go of from them, the piece they end in split; fewer where they run out."""
    taken: list[Piece] = []
    while length and pieces:
        piece = pieces.popleft()
        if piece.length > length:
            pieces.appendleft(Piece(piece.synthesis, piece.length - length))
            piece = Piece(piece.synthesis, length)
        taken.append(piece)
        length -= piece.length
    return taken


class Delivery(Protocol):
    """What a stream says became of the pieces written to it."""

    def took(self, pieces: list[Piece], missing: int, late: bool) -> None:
        """The device took `pieces`, in order, then found `missing` bytes more than was written; `late` as PortAudio
        said this callback came too late, the device having played a gap before it."""
        ...

    def unplayed(self, pieces: list[Piece]) -> None:
        """The stream closed, or refused a write, with `pieces` never played."""
        ...


@dataclass(frozen=True)
class Delivered:
    """What became of one synthesis, in bytes: its audio as the TTS made it, then of that, how much the device played,
    the phone was handed, and was never played; the gaps the device found inside it, how many and how long, and how many
    of its callbacks PortAudio said came too late; and `making`, the seconds the TTS took, None where it was cut off
    first."""

    synthesis: Synthesis
    making: float | None
    audio: int
    played: int
    phone: int
    unplayed: int
    gaps: int
    gap: int
    late: int


@dataclass
class _Followed:
    heard: int = 0
    # The TTS is done with it: its making, or cut off, None while it is still making it.
    done: bool = False
    making: float | None = None
    played: int = 0
    phone: int = 0
    unplayed: int = 0
    gaps: int = 0
    gap: int = 0
    late: int = 0
    # What the device found since this synthesis's last byte played: a gap if more of it plays, its end if not.
    missing: int = 0
    missed_late: int = 0

    def accounted(self) -> int:
        return self.played + self.phone + self.unplayed


@dataclass
class Syntheses:
    """[LAW:single-enforcer] where each synthesis is, from the TTS handing it over to the device taking its last byte,
    on the event loop alone: the speaker tells it what the TTS hands over and what it writes, and each stream what
    became of it (`Delivery`). `finished` is handed each synthesis as it is finished."""

    finished: Callable[[Delivered], None]
    # [LAW:one-source-of-truth] the bytes heard and not yet written, in order: Pipecat's sender holds them.
    _ledger: deque[Piece] = field(default_factory=deque[Piece])
    _followed: dict[Synthesis, _Followed] = field(default_factory=dict[Synthesis, _Followed])
    # The synthesis the TTS is making, whose audio is heard next; None between them.
    _making: Synthesis | None = None

    def speaking_in(self, synthesis: Synthesis) -> None:
        # A making the TTS never finished ended where its audio did.
        self._cut_making()
        self._making = synthesis
        self._followed[synthesis] = _Followed()

    def heard(self, length: int) -> None:
        """`length` bytes of the TTS's audio reached the speaker, ahead of its sender."""
        if self._making is not None:
            self._followed[self._making].heard += length
        # Audio of a synthesis already cut off, sent before the cut reached it, plays as nobody's.
        _join(self._ledger, Piece(self._making, length))

    def made(self, made: Made) -> None:
        self._followed[made.synthesis].making = made.making
        self._followed[made.synthesis].done = True
        self._making = None
        self._settle([made.synthesis])

    def interrupted(self) -> None:
        """Pipecat's sender dropped the audio it held: a synthesis heard and not yet written ends where it was written."""
        for piece in self._ledger:
            if piece.synthesis is not None:
                self._followed[piece.synthesis].heard -= piece.length
        self._ledger.clear()
        self._cut_making()
        self._settle(list(self._followed))

    def written(self, length: int) -> list[Piece]:
        """The pieces of `length` bytes of the TTS's audio the sender wrote, in order: its own padding is nobody's."""
        pieces = take(self._ledger, length)
        padding = length - sum(piece.length for piece in pieces)
        return [*pieces, Piece(None, padding)] if padding else pieces

    def phoned(self, pieces: list[Piece]) -> None:
        for synthesis, length in _of_syntheses(pieces):
            self._followed[synthesis].phone += length
        self._settle([synthesis for synthesis, _ in _of_syntheses(pieces)])

    def took(self, pieces: list[Piece], missing: int, late: bool) -> None:
        taken = [synthesis for synthesis, _ in _of_syntheses(pieces)]
        for synthesis, length in _of_syntheses(pieces):
            followed = self._followed[synthesis]
            followed.gaps += followed.missing > 0
            followed.gap += followed.missing
            followed.late += followed.missed_late
            followed.missing = followed.missed_late = 0
            followed.played += length
        for synthesis, followed in self._followed.items():
            match synthesis in taken, followed.played > 0:
                case True, _:
                    followed.late += late
                    followed.missing += missing
                case False, True:
                    followed.missed_late += late
                    followed.missing += missing
                case False, False:
                    # Not yet begun on the device: nothing is missing from it.
                    pass
        self._settle(taken)

    def unplayed(self, pieces: list[Piece]) -> None:
        for synthesis, length in _of_syntheses(pieces):
            self._followed[synthesis].unplayed += length
        self._settle([synthesis for synthesis, _ in _of_syntheses(pieces)])

    def _cut_making(self) -> None:
        if self._making is not None:
            self._followed[self._making].done = True
            self._making = None

    def _settle(self, syntheses: list[Synthesis]) -> None:
        # Finished once the TTS is done with it and every byte heard of it is accounted for; what went missing after its
        # last byte was its end, and is dropped with it.
        for synthesis in dict.fromkeys(syntheses):
            followed = self._followed[synthesis]
            if followed.done and followed.accounted() == followed.heard:
                del self._followed[synthesis]
                self.finished(
                    Delivered(
                        synthesis, followed.making, followed.heard, followed.played, followed.phone, followed.unplayed, followed.gaps, followed.gap, followed.late
                    )
                )


def _of_syntheses(pieces: Iterable[Piece]) -> list[tuple[Synthesis, int]]:
    return [(piece.synthesis, piece.length) for piece in pieces if piece.synthesis is not None]


def _join(pieces: deque[Piece], piece: Piece) -> None:
    # Bytes of one synthesis heard in a row are one piece.
    if pieces and pieces[-1].synthesis == piece.synthesis:
        pieces[-1] = Piece(piece.synthesis, pieces[-1].length + piece.length)
    else:
        pieces.append(piece)
