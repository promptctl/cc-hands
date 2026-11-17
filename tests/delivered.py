"""Stand-ins for what follows each synthesis hands' TTS makes, for tests of the streams that play it."""

from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice.synthesis import Piece, Synthesis


class Unfollowed:
    """A `Delivery` that keeps what a stream told it, in order."""

    def __init__(self) -> None:
        self.told: list[tuple[str, list[Piece], int, int]] = []

    def took(self, pieces: list[Piece], missing: int, late: int) -> None:
        self.told.append(("took", pieces, missing, late))

    def unplayed(self, pieces: list[Piece]) -> None:
        self.told.append(("unplayed", pieces, 0, 0))


def nobody(audio: bytes) -> tuple[Piece, ...]:
    """`audio` as one piece of no synthesis: the transport's silence, a cue."""
    return (Piece(None, len(audio)),)


def said(voice: str) -> Synthesis:
    """A synthesis in `voice`, begun at the start of a clock of its own."""
    return Synthesis(voice, begun(VirtualClock()))
