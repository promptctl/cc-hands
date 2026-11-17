"""Syntheses: each sentence hands' TTS makes into speech, followed from its making to the device taking its last byte.

[LAW:one-source-of-truth] whose each byte is travels with the bytes, in band, through every queue they pass: Pipecat's
sender queue, the speaker's writer thread, and the playout's waiting sound. A synthesis opens with a `SpeakingIn` ahead of
its audio and closes with one `Ended` behind it, and each queue keeps, drops or delays a marker with the bytes around it,
so nothing in hands keeps a copy of a queue it does not own. Pipecat's sender joins the end of one frame's audio to the
start of the next into its chunks, so the speaker cuts the audio into the sender's whole chunks first (`Cutter`): the
sender's buffer is empty at every frame, and no marker overtakes a byte.

A gap is heard as a crackle: written ahead by only `WRITE_AHEAD_SECS`, the speech is broken by any stall longer than
that. It is bytes the device found missing between two bytes of one synthesis; the silence between syntheses is
nobody's gap.

[LAW:nothing-unseen] each synthesis is one wide event, `speech`, so a gap is seen in the sentence it broke, beside how
fast the TTS made that sentence: a TTS slower than real time starves the speaker; a gap under a fast one is delivery.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pipecat.frames.frames import DataFrame, Frame, TTSAudioRawFrame, UninterruptibleFrame

from hands.sessions.wide import Begun, Fact, WideEvent, ended


@dataclass(frozen=True, eq=False)
class Synthesis:
    """One sentence hands' TTS made into speech, in `voice`, `begun` as the TTS was handed it."""

    voice: str
    begun: Begun


@dataclass
class SpeakingIn(DataFrame):
    """Ahead of the audio of `synthesis`, which runs to its `Ended`: sent by hands' TTS, and queued by Pipecat's sender in
    line with the audio, so the speaker knows whose each chunk it plays is."""

    synthesis: Synthesis


@dataclass
class Made(DataFrame):
    """Behind a synthesis's last audio, from the TTS: it took `making` seconds to make `made` seconds of speech, and
    stopped with the `failure` it said, or None once it had made all of it."""

    synthesis: Synthesis
    making: float
    made: float
    failure: str | None


# How a synthesis's making ended: as the TTS said; or cut before it said, by a barge-in or the speaker shutting. What of
# it played is the device's to say, behind it.
How = Made | Literal["cut"]


@dataclass
class Ended(DataFrame, UninterruptibleFrame):
    """Behind every byte of `synthesis` the speaker passed to the sender: one for each synthesis the speaker opened, and
    kept through a barge-in, so it always reaches the output transport behind what of it plays."""

    synthesis: Synthesis
    how: How


@dataclass(frozen=True)
class Chunking:
    """The sender's chunks: `size` bytes each, of audio at `rate`."""

    size: int
    rate: int


@dataclass
class Cutter:
    """[LAW:single-enforcer] the speaker's ingress, the one place a synthesis is opened and ended: it cuts the audio of
    the open synthesis into the sender's whole chunks, holds the rest, and pads that to a chunk as the synthesis ends.

    Audio with no synthesis open is a barge-in's straggler, already given up on, and is dropped."""

    _open: Synthesis | None = None
    # The end of the open synthesis's audio short of a whole chunk, as a frame of the one it came from; None for none.
    _held: TTSAudioRawFrame | None = None

    def speaking_in(self, speaking: SpeakingIn, chunking: Chunking) -> list[Frame]:
        # [LAW:dataflow-not-control-flow] a synthesis still open as the next opens was cut short: one whose making was
        # cancelled before the TTS said it was done.
        ending = self._end("cut", self._padded(chunking))
        self._open = speaking.synthesis
        return [*ending, speaking]

    def heard(self, frame: TTSAudioRawFrame, chunking: Chunking) -> list[Frame]:
        # [LAW:no-silent-failure] the sender resamples audio at another rate, and every cut would be off.
        assert frame.sample_rate == chunking.rate, f"TTS audio at {frame.sample_rate} Hz reached a speaker at {chunking.rate} Hz"
        if self._open is None:
            return []
        audio = (b"" if self._held is None else self._held.audio) + frame.audio
        whole = len(audio) - len(audio) % chunking.size
        self._held = _carrying(frame, audio[whole:]) if whole < len(audio) else None
        return [_carrying(frame, audio[at : at + chunking.size]) for at in range(0, whole, chunking.size)]

    def made(self, made: Made, chunking: Chunking) -> list[Frame]:
        # A Made of a synthesis no longer open is of one a barge-in already ended.
        return self._end(made, self._padded(chunking)) if made.synthesis is self._open else []

    def interrupted(self) -> list[Frame]:
        """A barge-in: what is held is never played, and the open synthesis ends here."""
        self._held = None
        return self._end("cut", [])

    def _padded(self, chunking: Chunking) -> list[Frame]:
        # What is held, padded with silence to a whole chunk, let go of.
        held, self._held = self._held, None
        return [] if held is None else [_carrying(held, held.audio + bytes(-len(held.audio) % chunking.size))]

    def _end(self, how: How, tail: list[Frame]) -> list[Frame]:
        opened, self._open = self._open, None
        return [] if opened is None else [*tail, Ended(opened, how)]


def _carrying(frame: TTSAudioRawFrame, audio: bytes) -> TTSAudioRawFrame:
    """A frame of `audio` in place of `frame`'s: Pipecat's routing and context kept, its frame fields made afresh."""
    carried = TTSAudioRawFrame(audio, frame.sample_rate, frame.num_channels, context_id=frame.context_id)
    carried.transport_destination = frame.transport_destination
    return carried


# Seconds of sound of one synthesis. The sound of none, the transport's own silence and a cue, is no piece of hands'
# speech: a cue inside a synthesis does not end its run.
Piece = tuple[Synthesis, float]


@dataclass(frozen=True)
class Taken:
    """What one of the device's callbacks took, in order: `pieces` of hands' speech, and the `ends` of syntheses whose
    every byte had then been taken or let go of; after `missing` seconds the device found nothing written for since it
    last took any of hands' speech, over which PortAudio said `late` callbacks came too late."""

    pieces: tuple[Piece, ...]
    ends: tuple[Ended, ...]
    missing: float
    late: int


@dataclass
class _Followed:
    played: float = 0.0
    phone: float = 0.0
    gaps: int = 0
    gap: float = 0.0
    late: int = 0


@dataclass
class Syntheses:
    """[LAW:single-enforcer] what became of each synthesis the speaker opened, on the event loop alone: told by the
    device what it took (`took`) and by the speaker what it gave the phone (`phoned`), it emits each synthesis's `speech`
    event as its `Ended` comes out of where its bytes went."""

    emit: Callable[[WideEvent], None]
    # [LAW:no-silent-failure] each synthesis from its opening to its event: a byte or an end of one not followed is a
    # fault in the order its markers travel, and raises rather than starting it again.
    _followed: dict[Synthesis, _Followed] = field(default_factory=dict[Synthesis, _Followed])
    # The synthesis the device last took bytes of: what it found missing since is a gap if it takes more of the same
    # synthesis next.
    _last: Synthesis | None = None

    def opened(self, synthesis: Synthesis) -> None:
        # [LAW:nothing-unseen] followed from its opening, so one whose every marker is lost is still emitted at close.
        self._followed[synthesis] = _Followed()

    def took(self, taken: Taken) -> None:
        # One callback's pieces run on with nothing missing between them: what went missing is before the first.
        missing, late = taken.missing, taken.late
        for synthesis, length in taken.pieces:
            followed = self._followed[synthesis]
            # [LAW:dataflow-not-control-flow] what went missing since the last byte taken is a gap of a synthesis only
            # between two bytes of it; otherwise it is the silence between two.
            inside = synthesis is self._last
            followed.gaps += inside and missing > 0
            followed.gap += inside * missing
            followed.late += inside * late
            followed.played += length
            self._last, missing, late = synthesis, 0.0, 0
        for end in taken.ends:
            self.finished(end)

    def phoned(self, synthesis: Synthesis | None, length: float) -> None:
        # The phone plays on: what the desk finds missing meanwhile is no gap in what it played.
        self.moved()
        if synthesis is not None:
            self._followed[synthesis].phone += length

    def moved(self) -> None:
        """hands' speech went elsewhere than the device last taking it: what that device finds missing is no gap."""
        self._last = None

    def finished(self, end: Ended) -> None:
        """Every byte of `end`'s synthesis has been taken by the device, handed to the phone, or let go of."""
        followed = self._followed.pop(end.synthesis)
        ended(
            "speech",
            self.emit,
            end.synthesis.begun,
            None,
            _failure(end.how),
            voice=end.synthesis.voice,
            played_s=followed.played,
            phone_s=followed.phone,
            gaps=followed.gaps,
            gap_s=followed.gap,
            late=followed.late,
            **_how(end.how),
        )

    def close(self) -> None:
        """The speaker is shutting: every synthesis whose `Ended` never came out was cut by it."""
        for synthesis in list(self._followed):
            self.finished(Ended(synthesis, "cut"))


def _how(how: How) -> dict[str, Fact]:
    """How a synthesis's making ended, as facts of its event: made, with how long the TTS took over how long what it
    made plays, a ratio of nothing for no sound made; or cut."""
    match how:
        case Made(making=making, made=made):
            return {"how": "made", "made_s": made, "rtf": making / made if made else None}
        case "cut":
            return {"how": "cut", "made_s": None, "rtf": None}


def _failure(how: How) -> str | None:
    """[LAW:no-silent-failure] the TTS's own failure, as the event's: Pipecat's catches what its model raises and says so."""
    match how:
        case Made(failure=failure):
            return failure
        case "cut":
            return None
