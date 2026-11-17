"""Syntheses: each sentence hands' TTS makes into speech, followed from its making to the device taking its last byte.

Pipecat's sender cuts the audio it is handed into chunks, and on its own joins one synthesis's end to the next's start,
pads a tail, or drops what it holds. So the speaker cuts the audio first, into the sender's own whole chunks, each of one
synthesis, its tail padded with silence: the sender's buffer is empty at every frame, and each chunk it writes is one
the speaker cut, in order. Each chunk is `Piece`s, the bytes of one synthesis or of none, and the stream that plays a
piece says what became of it: taken by the device, after what the device found missing, or never played. A synthesis
is finished once it has ended and every byte of it is accounted for, and is then delivered.

A gap is heard as a crackle: written ahead by only `WRITE_AHEAD_SECS`, the speech is broken by any stall longer than
that. It is bytes the device found missing between two bytes of one synthesis; the silence between syntheses is
nobody's gap.

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


@dataclass(frozen=True)
class Chunk:
    """One of the sender's whole chunks, as the speaker cut it: its audio, and the pieces that audio is."""

    audio: bytes
    pieces: tuple[Piece, ...]


def voices_of(pieces: Iterable[Piece]) -> tuple[str, ...]:
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

    def took(self, pieces: list[Piece], missing: int, late: int) -> None:
        """The device took `pieces`, in order, having found `missing` bytes more than was written since the last it took,
        over which PortAudio said `late` callbacks came too late, the device having played a gap before each."""
        ...

    def unplayed(self, pieces: list[Piece]) -> None:
        """The stream closed, or refused a write, with `pieces` never played."""
        ...


@dataclass(frozen=True)
class Delivered:
    """What became of one synthesis, in bytes: its audio as the TTS made it, then of that, how much the device played,
    the phone was handed, and was never played; the gaps the device found inside it, how many and how long, and how many
    callbacks PortAudio said came too late inside it; and `making`, the seconds the TTS took, None where it was cut off
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
    audio: int = 0
    ended: bool = False
    making: float | None = None
    played: int = 0
    phone: int = 0
    unplayed: int = 0
    gaps: int = 0
    gap: int = 0
    late: int = 0

    def accounted(self) -> int:
        return self.played + self.phone + self.unplayed


@dataclass
class Syntheses:
    """[LAW:single-enforcer] where each synthesis is, from the TTS handing it over to the device taking its last byte,
    on the event loop alone: the speaker tells it what the TTS hands over and is handed the chunks to pass the sender, and
    each stream says what became of them (`Delivery`). `finished` is handed each synthesis as it is finished."""

    finished: Callable[[Delivered], None]
    # The synthesis the TTS is making, whose audio is cut next, and the end of it short of a whole chunk; None between.
    _open: Synthesis | None = None
    _held: bytes = b""
    # [LAW:one-source-of-truth] the chunks passed to the sender and not yet written: its queue, which keeps their order
    # and loses them only to an interruption.
    _queued: deque[tuple[Piece, ...]] = field(default_factory=deque[tuple[Piece, ...]])
    _followed: dict[Synthesis, _Followed] = field(default_factory=dict[Synthesis, _Followed])
    # The synthesis the device last took bytes of, and what it found missing, and how late, since: a gap if it takes
    # more of the same synthesis next.
    _last: Synthesis | None = None
    _missing: int = 0
    _late: int = 0

    def speaking_in(self, synthesis: Synthesis, size: int) -> list[Chunk]:
        """The TTS begins `synthesis`: the one it was making ended where its audio did, its tail padded to a chunk."""
        chunks = self._end(size)
        self._open = synthesis
        self._followed[synthesis] = _Followed()
        return chunks

    def heard(self, audio: bytes, size: int) -> list[Chunk]:
        """The whole chunks of `size` bytes the TTS's `audio` completes; the rest is held for the next."""
        held = self._held + audio
        whole = len(held) - len(held) % size
        self._held = held[whole:]
        # Audio with no synthesis open plays as nobody's.
        return [self._queue(held[at : at + size], Piece(self._open, size)) for at in range(0, whole, size)]

    def made(self, made: Made, size: int) -> list[Chunk]:
        self._followed[made.synthesis].making = made.making
        return self._end(size)

    def stopped(self, size: int) -> list[Chunk]:
        """The TTS stopped: what it was making ended where its audio did."""
        return self._end(size)

    def interrupted(self) -> None:
        """Pipecat's sender dropped every chunk it held, and the TTS stopped: what was held or queued is never played."""
        held = Piece(self._open, len(self._held))
        self._made([held])
        dropped = [*(piece for pieces in self._queued for piece in pieces), held]
        self._queued.clear()
        self._held = b""
        self._close()
        for synthesis, length in _of_syntheses(dropped):
            self._followed[synthesis].unplayed += length
        self._settle(list(self._followed))

    def written(self) -> tuple[Piece, ...]:
        """The pieces of the next chunk of TTS audio the sender writes."""
        # [LAW:no-silent-failure] a chunk the speaker never cut would be counted as another's.
        assert self._queued, "the sender wrote TTS audio the speaker never passed it"
        return self._queued.popleft()

    def phoned(self, pieces: Iterable[Piece]) -> None:
        # The phone plays on: what the desk finds missing meanwhile is no gap in what it played.
        self._last = None
        for synthesis, length in _of_syntheses(pieces):
            self._followed[synthesis].phone += length
        self._settle([synthesis for synthesis, _ in _of_syntheses(pieces)])

    def took(self, pieces: list[Piece], missing: int, late: int) -> None:
        self._missing += missing
        self._late += late
        for synthesis, length in _of_syntheses(pieces):
            followed = self._followed[synthesis]
            # [LAW:dataflow-not-control-flow] what went missing since the last byte taken is a gap of a synthesis only
            # between two bytes of it; otherwise it is the silence between two.
            inside = synthesis == self._last
            followed.gaps += inside and self._missing > 0
            followed.gap += inside * self._missing
            followed.late += inside * self._late
            followed.played += length
            self._last, self._missing, self._late = synthesis, 0, 0
        self._settle([synthesis for synthesis, _ in _of_syntheses(pieces)])

    def unplayed(self, pieces: Iterable[Piece]) -> None:
        for synthesis, length in _of_syntheses(pieces):
            self._followed[synthesis].unplayed += length
        self._settle([synthesis for synthesis, _ in _of_syntheses(pieces)])

    def _queue(self, audio: bytes, *pieces: Piece) -> Chunk:
        chunk = Chunk(audio, tuple(piece for piece in pieces if piece.length))
        self._made(chunk.pieces)
        self._queued.append(chunk.pieces)
        return chunk

    def _made(self, pieces: Iterable[Piece]) -> None:
        # [LAW:one-source-of-truth] a synthesis's audio is counted once, as its bytes are queued or dropped.
        for synthesis, length in _of_syntheses(pieces):
            self._followed[synthesis].audio += length

    def _end(self, size: int) -> list[Chunk]:
        held, self._held = self._held, b""
        padding = -len(held) % size
        chunks = [self._queue(held + bytes(padding), Piece(self._open, len(held)), Piece(None, padding))] if held else []
        self._close()
        return chunks

    def _close(self) -> None:
        # The open synthesis has ended, all of its audio counted: finished now if all of it is accounted for already.
        match self._open:
            case None:
                pass
            case synthesis:
                self._followed[synthesis].ended = True
                self._settle([synthesis])
        self._open = None

    def _settle(self, syntheses: list[Synthesis]) -> None:
        # Finished once it has ended and every byte of it is accounted for.
        for synthesis in dict.fromkeys(syntheses):
            followed = self._followed[synthesis]
            if followed.ended and followed.accounted() == followed.audio:
                del self._followed[synthesis]
                self.finished(
                    Delivered(
                        synthesis, followed.making, followed.audio, followed.played, followed.phone, followed.unplayed, followed.gaps, followed.gap, followed.late
                    )
                )


def _of_syntheses(pieces: Iterable[Piece]) -> list[tuple[Synthesis, int]]:
    return [(piece.synthesis, piece.length) for piece in pieces if piece.synthesis is not None]
