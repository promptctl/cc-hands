"""Push-to-talk: the key is the voice activity detector and the microphone mute.

Pipecat's turn machinery, and the clip owner that cuts the clips every model
hears, both run off the key. So the key *is* the VAD: every microphone frame
carries the key it was captured under, and the clip owner
(`hands.voice.clipowner`) says the user started or stopped speaking where those
keys change. The key is also the mute: the microphone bytes are silence unless
the key is pressed (applied where they are captured, in
`hands.voice.microphone`), and a clip's audio begins at its press, so what was
said before the key meant talk is kept. Audio frames always flow; only their
content and their key follow the key. The turn strategies open and close the
user turn on those frames (`hands.voice.turnstart`, `hands.voice.turnstop`),
and its cut broadcasts the interruption that flushes queued speech on
barge-in. While the key is up, whatever the microphone hears,
including the pipeline's own speech, is silence to the pipeline.

The key is pressed at a place: the desk, by the talk key, or the phone, by its page's button. The gate holds where the
last turn was opened, and that place is where hands is: the pipeline hears that place's microphone and answers on its
speaker, and the other place's moves cannot touch a turn that is not theirs.
"""

from dataclasses import dataclass, replace
from typing import Literal

from pipecat.frames.frames import InputAudioRawFrame

from hands.clock import Clock
from hands.core.place import Modality, Place, modality_at
from hands.sessions.audit import Record
from hands.sessions.wide import begun, ended
from hands.voice.clips import Played
from hands.voice.shortcut import Move
from hands.voice.trigger import Trigger, place_of

# [LAW:types-are-the-program] the key is up; listening, the desk heard between turns at the wake word or engaged, so a turn the voice
# opens keeps the words said before the detector was sure of them; arming, pressed but not yet meaning talk, which hears
# so the words said before it does are kept; or down, a clip. The edges decide every move, so the gate never sees one
# that is not a transition.
Key = Literal["up", "listening", "arming", "down"]

@dataclass(kw_only=True)
class KeyedAudio(InputAudioRawFrame):
    """Microphone audio, the gate as it was when the audio was captured, and the place whose microphone captured it.

    [LAW:no-ambient-temporal-coupling] the gate travels with the audio, in capture order, so whatever reads it later
    reads the gate as it was when this sound was recorded, never as it is by the time the frame arrives.
    """

    # The same sound as the microphone captured it, before the echo canceller and as long as `audio`, muted by the key
    # as it is; the phone is heard through no canceller, so there it is `audio` itself.
    captured: bytes
    # What the desk's speaker was playing as it was captured, the echo canceller's reference for each 10 ms of `audio`;
    # None at the phone, whose own canceller is told what it plays, and hands is not.
    played: Played
    key: Key
    # How many turns the gate has sent, and thrown away, by this frame.
    sent: int
    dropped: int
    place: Place
    # The trigger that opened the last clip, as it opened it.
    opened: Trigger


@dataclass(frozen=True)
class Gate:
    """The pure push-to-talk state: which position the key is in, the place hands is at, whether the desk listens
    between turns, and how many turns have ended each way."""

    key: Key = "up"
    place: Place = "desk"
    # [LAW:one-source-of-truth] listening between turns is the desk's: at the phone, its button alone opens the microphone.
    listens: bool = False
    # [LAW:no-ambient-temporal-coupling] how each turn ended is counted, not left in a key position: a frame is captured
    # every 20 ms and a turn can end and the next arm between two of them, so the clip owner ends each turn as these move.
    sent: int = 0
    dropped: int = 0
    # [LAW:no-ambient-temporal-coupling] the trigger that opened the last turn travels with the audio, as the key does, so
    # what reads a clip's opening reads the trigger that opened it, whichever is in use by the time its words arrive.
    opened: Trigger = "shortcut"

    def took(self, move: Move, at: Place) -> Move | None:
        """What `move` at `at` did to the turn, as the gate takes it; None where it does nothing to it.

        [LAW:one-source-of-truth] the place is moved only by a turn opening there, or by the phone coming and going: a
        Shift typed at the desk while the user talks on the phone arms nothing and ends nothing of theirs. A turn opened
        at one place while a clip is open at the other is a second hand on a second key, which drops both, as a key
        pressed while the talk key is held drops the turn; and the end of a clip the gate has already thrown away, as a
        call came or went, ends nothing more. The desk's listening is taken wherever hands is.
        """
        match at == self.place, move, self.key:
            case _, "listen" | "deafen", _:
                return move
            case True, "stop" | "drop" | "timeout", "up" | "listening" | "arming":
                return None
            case True, _, _:
                return move
            case False, "start", "arming" | "down":
                return "drop"
            case False, "start", "up" | "listening":
                return "start"
            case False, _, _:
                return None

    def after(self, move: Move, by: Trigger) -> "Gate":
        """The gate once `move`, made by `by`, has moved the turn."""
        at = place_of(by)
        rest = self._rest(at, self.listens)
        match self.took(move, at):
            case None:
                return self
            case "listen" | "deafen" as listening:
                # The desk's listening moves only where the key rests: a press or a turn open is left to end as it ends.
                listens = listening == "listen"
                resting = self.key in ("up", "listening")
                return replace(self, key=self._rest(self.place, listens) if resting else self.key, listens=listens)
            case "arm":
                return replace(self, key="arming", place=at)
            case "disarm":
                return replace(self, key=rest, place=at)
            case "start":
                return replace(self, key="down", place=at, opened=by)
            case "stop" | "timeout":
                return replace(self, key=rest, place=at, sent=self.sent + 1)
            case "drop":
                return replace(self, key=rest, place=at, dropped=self.dropped + (self.key == "down"))

    def moved(self, to: Place) -> "Gate":
        """The gate once hands is at `to`: a clip open at the place it leaves is thrown away, not sent."""
        if to == self.place:
            return self
        return replace(self, key=self._rest(to, self.listens), place=to, dropped=self.dropped + (self.key == "down"))

    @staticmethod
    def _rest(place: Place, listens: bool) -> Key:
        """The key between turns at `place`."""
        return "listening" if listens and place == "desk" else "up"

    def hears(self, place: Place) -> bool:
        """Whether the pipeline hears the microphone at `place`: only hands' own, so one stream of frames reaches the clip owner."""
        return place == self.place

    @property
    def turn_open(self) -> bool:
        """The key has been held long enough to mean talk: what the microphone hears now is the user's turn."""
        return self.key == "down"

    def audible(self, audio: bytes) -> bytes:
        """The microphone bytes as the pipeline hears them: intact while the key is pressed or the desk listens, silence
        otherwise."""
        # [LAW:dataflow-not-control-flow] a frame of the same length always
        # goes out, so the clip owner sees an unbroken stream; the key
        # only decides its content.
        return audio if self.key in ("listening", "arming", "down") else bytes(len(audio))

    def framed(self, audio: bytes, captured: bytes, played: Played, sample_rate: int, num_channels: int, place: Place) -> KeyedAudio:
        """Audio captured at `place`, as the pipeline hears it, as the microphone `captured` it, and what the speaker
        `played` over it, tagged with this gate. What was played is not the microphone's, so the key mutes none of it."""
        return KeyedAudio(
            audio=self.audible(audio), captured=self.audible(captured), played=played, sample_rate=sample_rate, num_channels=num_channels,
            key=self.key, sent=self.sent, dropped=self.dropped, place=place, opened=self.opened,
        )


class PushToTalk:
    """The one owner of the key position and of where the user is: the talk key's edge writes, the microphone reads and
    tags every frame with it; and whether the user can see a screen there, which the place they talk from sets and they
    can switch by voice."""

    # [LAW:no-shared-mutable-globals] the event loop writes the key and the capture thread reads it,
    # so it lives here once with one writer.
    def __init__(self, clock: Clock, record: Record) -> None:
        self._clock = clock
        self._gate = Gate()
        self._modality: Modality = modality_at(self._gate.place)
        self._record = record

    def move(self, move: Move, by: Trigger) -> Move | None:
        """Report what `move`, made by `by`, did to the turn, and get back what the gate took it as, to cue and tell: the
        desk's trigger and the phone's button call this."""
        taken = self._gate.took(move, place_of(by))
        self._become(self._gate.after(move, by), "turn")
        return taken

    def go(self, to: Place) -> None:
        """Report that hands is at `to` now: the phone came, or went."""
        self._become(self._gate.moved(to), "call")

    def switch(self, to: Modality) -> None:
        """Report that the user asked to be taken as `to` from now on, until hands next moves between the desk and the phone."""
        self._modality = to

    def _become(self, gate: Gate, by: Literal["turn", "call"]) -> None:
        before, self._gate = self._gate, gate
        # [LAW:nothing-unseen] the one writer of the gate writes each move of the place as one `place.moved` event,
        # whichever trigger made it: from where to where, what made it, and whether a clip open at the place it left was
        # thrown away.
        if gate.place != before.place:
            # [LAW:one-source-of-truth] the way the user talks sets the modality, so a move sets it in the same write.
            self._modality = modality_at(gate.place)
            ended("place.moved", self._record, begun(self._clock), None, before=before.place, after=gate.place, by=by, dropped=gate.dropped != before.dropped)

    @property
    def gate(self) -> Gate:
        # Read from the audio callback thread too: one attribute load of a frozen value, whole either way.
        return self._gate

    @property
    def modality(self) -> Modality:
        return self._modality

    @property
    def opened(self) -> Trigger:
        """The trigger that opened the last turn, as it opened it: a trigger switched since does not rewrite it."""
        return self._gate.opened

