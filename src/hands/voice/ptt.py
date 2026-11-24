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

The key is pressed at a place: the desk, by the talk key, or the phone, by its page's button. The gate holds where
hands is: the pipeline hears that place's microphone and answers on its speaker. Only the user's hand opening a turn at
the other place moves it, or a call; the hand's move sends the clip open where hands was.

The gate is the one place that says who may open a turn, press over one and end one: its position names who has the key
pressed and who has a clip open, so a rule about whose a turn is is a transition of that position, and what follows a turn
is told by the gate as it opens and ends one.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from pipecat.frames.frames import InputAudioRawFrame

from hands.clock import Clock
from hands.core.place import Modality, Place, modality_at
from hands.sessions.audit import Record
from hands.sessions.wide import begun, ended
from hands.voice.clips import Fate, Played
from hands.voice.shortcut import Beginning, Ending, Instant, Move
from hands.voice.trigger import PhoneLevel, Trigger, by_hand, place_of
from hands.voice.voiceturn import TurnId

# [LAW:types-are-the-program] the key is up; listening, the desk heard between turns at the wake word or engaged, so a turn the voice
# opens keeps the words said before the detector was sure of them; arming, pressed but not yet meaning talk, which hears
# so the words said before it does are kept; or down, a clip. The edges decide every move, so the gate never sees one
# that is not a transition.
Key = Literal["up", "listening", "arming", "down"]

# Why a voice may not open a turn: hands is at the other place; the place it is at does not listen; or a turn is open or being
# opened there.
Refusal = Literal["elsewhere", "deaf", "busy"]

# What ended a turn: the trigger whose move it was, a call that came or went, or a place's listening going off, which no
# trigger is the turn of.
Cause = Trigger | Literal["call", "level"]


@dataclass(frozen=True)
class Resting:
    """No key pressed and no clip open. `last` is the trigger that opened the last turn."""

    last: Trigger


@dataclass(frozen=True)
class Pressing:
    """`by` has the key pressed, not yet meaning talk: arming. `last` is the trigger that opened the last turn."""

    by: Trigger
    last: Trigger


@dataclass(frozen=True)
class Open:
    """A clip is open, and `by` opened it: the one trigger that may end it."""

    by: Trigger


# [LAW:types-are-the-program] who has what: nobody, a press that has not yet meant talk, or a clip. A press over a clip, a
# clip with no opener and a press with no presser are none of them writable.
Position = Resting | Pressing | Open


@dataclass(frozen=True)
class GateOpened:
    """The gate opened turn `turn` for `by`, at `at`."""

    turn: TurnId
    by: Trigger
    at: Instant


def fate_of(move: Ending) -> Fate:
    """How a turn ends under `move`: thrown away by a drop, and sent as it stands by the others."""
    return "dropped" if move == "drop" else "sent"


@dataclass(frozen=True)
class GateEnded:
    """The gate ended turn `turn`, by `move`, because `by` moved it."""

    turn: TurnId
    move: Ending
    by: Cause


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
    # How every turn the gate has ended ended, in order, by this frame (`Gate.ended`).
    ended: tuple[Fate, ...]
    place: Place
    # The trigger that opened the last clip, as it opened it.
    opened: Trigger

    @property
    def turn(self) -> TurnId:
        """The gate's number for the turn this audio is of, or the next to open (`Gate.turn`)."""
        return TurnId(len(self.ended))


@dataclass(frozen=True)
class Gate:
    """The pure push-to-talk state: who has the key pressed or a clip open, the place hands is at, which places listen
    between turns, and how each turn it has ended ended."""

    position: Position = Resting("shortcut")
    place: Place = "desk"
    # [LAW:one-source-of-truth] the places that listen between turns: the desk at its wake word or engaged level, the phone
    # at its wake word level, which its page tells for the call and no longer. Each place's own, so the desk's level and the
    # phone's never move each other.
    listening: frozenset[Place] = frozenset()
    # [LAW:no-ambient-temporal-coupling] how each turn ended is recorded in order, not left in a key position: a frame is
    # captured every 20 ms and several turns can end between two of them, so the clip owner reads each turn's end, and which
    # it was, off this record as it grows.
    ended: tuple[Fate, ...] = ()

    @property
    def key(self) -> Key:
        """The key position the pipeline reads. At rest it follows what listens, so it cannot say a place listens that does not."""
        match self.position:
            case Resting():
                return "listening" if self.place in self.listening else "up"
            case Pressing():
                return "arming"
            case Open():
                return "down"

    @property
    def opened(self) -> Trigger:
        """The trigger that opened the last turn, as it opened it: what reads a clip's opening reads the trigger that opened
        it, whichever is in use by the time its words arrive."""
        match self.position:
            case Resting(last=last) | Pressing(last=last):
                return last
            case Open(by=by):
                return by

    def step(self, move: Move, by: Trigger) -> tuple[tuple[Move, ...], "Gate"]:
        """The moves the gate made as `by` made `move`, in order, and the gate after them; none, and itself, where it did
        nothing to the turn.

        [LAW:single-enforcer] the one place the rules of who may open a turn, press over one and end one are kept. Hands
        is moved only by the user's hand opening a turn at the other place, or by a call: a voice, which can be anyone's or
        nobody's, arms only where hands is, at a place that listens, with nothing pressed or open (`refusal`), starts only
        what it armed, and a Shift typed at the desk while the user talks on the phone arms nothing. A clip is ended by
        the trigger that opened it, once. A hand's arm over a clip sends it first, and so does its start over another trigger's,
        which over a press another has made lets it go, and so does a place going deaf over what its voice has open: every move
        the gate made is in the moves it returns, and nothing else of it is, for what follows each.
        """
        at = place_of(by)
        here = at == self.place
        hand = by_hand(by)
        match move, self.position:
            case "listen", _:
                return (move,), replace(self, listening=self.listening | {at})
            case "deafen", _:
                gone, gate = self._without_voice_at(at)
                return (*gone, move), replace(gate, listening=gate.listening - {at})
            case "arm", _ if here and (hand or self.refusal(by) is None):
                left, gate = self._vacated(by)
                return (*left, move), replace(gate, position=Pressing(by, gate.opened))
            case "disarm", Pressing(by=presser, last=last) if here and presser == by:
                return (move,), replace(self, position=Resting(last))
            case "start", Open(by=opener) if hand and here and opener == by:
                # The same trigger's press over its own clip goes on with it: it opens nothing, and the gate makes no move.
                return (), self
            case "start", _ if hand:
                # [LAW:dataflow-not-control-flow] hands goes where the turn opens: a clip open where it was is sent, and
                # where it already is, another trigger's is sent.
                left, there = self._moved(at)
                sent, gate = there._vacated(by)
                return (*left, *sent, move), replace(gate, position=Open(by))
            case "start", Pressing(by=presser) if here and presser == by:
                return (move,), replace(self, position=Open(by))
            case ("stop" | "drop" | "timeout") as ending, Open(by=opener) if here and opener == by:
                return (ending,), replace(self, position=Resting(by), ended=(*self.ended, fate_of(ending)))
            case _:
                return (), self

    def took(self, move: Move, by: Trigger) -> tuple[Move, ...]:
        """The moves the gate made as `by` made `move`."""
        return self.step(move, by)[0]

    def after(self, move: Move, by: Trigger) -> "Gate":
        """The gate once `by` made `move`."""
        return self.step(move, by)[1]

    def refusal(self, by: Trigger) -> Refusal | None:
        """Why a voice, `by`, may not arm a turn now, or None where it may."""
        match place_of(by) == self.place, self.position:
            case False, _:
                return "elsewhere"
            case True, Pressing() | Open():
                return "busy"
            case True, Resting():
                return None if self.place in self.listening else "deaf"

    def moved(self, to: Place) -> "Gate":
        """The gate once hands is at `to`: a clip open at the place it leaves is sent as it stands, and a press there is let go."""
        return self._moved(to)[1]

    def _moved(self, to: Place) -> tuple[tuple[Move, ...], "Gate"]:
        """The moves made, and the gate, once hands is at `to`."""
        if to == self.place:
            return (), self
        left, gate = self._vacated(None)
        return left, replace(gate, position=Resting(gate.opened), place=to)

    def called(self, to: Place) -> tuple[tuple[Move, ...], "Gate"]:
        """The moves made, and the gate, once a call has come, `to` the phone, or gone, `to` the desk.

        A page can call again by itself, so a call that comes never cuts off a press or a clip at the desk, whoever's:
        hands stays for it, and the phone's first press takes it there. One that goes takes the phone's audio with it,
        so its clip is sent as it stands. The phone's listening is the call's: a call that comes or goes leaves it off."""
        gate = replace(self, listening=self.listening - {"phone"})
        match to, self.position:
            case "phone", Pressing() | Open():
                return (), gate
            case _:
                return gate._moved(to)

    def _vacated(self, by: Trigger | None) -> tuple[tuple[Move, ...], "Gate"]:
        """The moves made, and the gate, once what is held is cleared for `by`: a clip open is sent as it stands, and a press
        that is not `by`'s is let go. `by` None clears a press whoever's."""
        match self.position:
            case Open(by=opener):
                return ("stop",), replace(self, position=Resting(opener), ended=(*self.ended, "sent"))
            case Pressing(by=presser, last=last) if presser != by:
                return ("disarm",), replace(self, position=Resting(last))
            case _:
                return (), self

    def _without_voice_at(self, place: Place) -> tuple[tuple[Move, ...], "Gate"]:
        """The moves made, and the gate, once what a voice holds at `place` is cleared, as the place stops listening."""
        match self.position:
            case Open(by=holder) | Pressing(by=holder) if place_of(holder) == place and not by_hand(holder):
                return self._vacated(None)
            case _:
                return (), self

    @property
    def turn(self) -> TurnId:
        """The gate's number for the turn open, or the next to open: how many turns it has ended. A press learns the turn
        it opens from it, and the clip owner numbers the clip the turn opens with it, so the two are bound exactly."""
        return TurnId(len(self.ended))

    def hears(self, place: Place) -> bool:
        """Whether the pipeline hears the microphone at `place`: only hands' own, so one stream of frames reaches the clip owner."""
        return place == self.place

    @property
    def turn_open(self) -> bool:
        """The key has been held long enough to mean talk: what the microphone hears now is the user's turn."""
        return isinstance(self.position, Open)

    @property
    def phone_level(self) -> PhoneLevel:
        """The level the phone's page has it listening at for this call."""
        return "wake word" if "phone" in self.listening else "off"

    @property
    def at_phone(self) -> PhoneLevel | None:
        """The phone's level where hands is at the phone; None at the desk, whose level is the desk's own."""
        return self.phone_level if self.place == "phone" else None

    def audible(self, audio: bytes) -> bytes:
        """The microphone bytes as the pipeline hears them: intact while the key is pressed or hands is at a place that listens,
        silence otherwise."""
        # [LAW:dataflow-not-control-flow] a frame of the same length always
        # goes out, so the clip owner sees an unbroken stream; the key
        # only decides its content.
        return audio if self.key in ("listening", "arming", "down") else bytes(len(audio))

    def framed(self, audio: bytes, captured: bytes, played: Played, sample_rate: int, num_channels: int, place: Place) -> KeyedAudio:
        """Audio captured at `place`, as the pipeline hears it, as the microphone `captured` it, and what the speaker
        `played` over it, tagged with this gate. What was played is not the microphone's, so the key mutes none of it."""
        return KeyedAudio(
            audio=self.audible(audio), captured=self.audible(captured), played=played, sample_rate=sample_rate, num_channels=num_channels,
            key=self.key, ended=self.ended, place=place, opened=self.opened,
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
        # Who is told as the gate opens and ends a turn, however it came to.
        self._watchers: list[Callable[[GateOpened | GateEnded], None]] = []

    def watch(self, told: Callable[[GateOpened | GateEnded], None]) -> None:
        """Have `told` called with each turn the gate opens and ends, as it does, in order."""
        self._watchers.append(told)

    def move(self, move: Beginning, by: Trigger) -> tuple[Move, ...]:
        """Report that `by` made `move`, and get back the moves the gate made, in order, to cue and tell: the desk's trigger
        and the phone's button and wake call this."""
        return self._take(move, by)

    def end(self, turn: TurnId, how: Ending, by: Trigger) -> tuple[Move, ...]:
        """Report that `by` ended turn `turn`, `how`, and get back the moves the gate made. A verdict, a limit or an ask is
        decided while the gate moves on, and one that comes after the turn is over ends nothing of the turn after it. What
        also ends a clip is the gate's own move: a hand's start over another's, a place going deaf, a call."""
        return self._take(how, by) if self._gate.turn == turn else ()

    def go(self, to: Place) -> None:
        """Report that a call has come, `to` the phone, or gone, `to` the desk. A clip it sends is told to the watchers; no
        tone follows it, for the call it would sound to is gone or not the one hands was at."""
        moves, gate = self._gate.called(to)
        self._become(moves, gate, "call")

    def switch(self, to: Modality) -> None:
        """Report that the user asked to be taken as `to` from now on, until hands next moves between the desk and the phone."""
        self._modality = to

    def _take(self, move: Move, by: Trigger) -> tuple[Move, ...]:
        moves, gate = self._gate.step(move, by)
        # A place's listening going off is no trigger's turn: what it ends, it ends as the level.
        self._become(moves, gate, "level" if move in ("listen", "deafen") else by)
        return moves

    def _become(self, moves: tuple[Move, ...], gate: Gate, by: Cause) -> None:
        before, self._gate = self._gate, gate
        # [LAW:one-source-of-truth] the moves are the gate's only account of what it did: whoever follows a turn is told
        # from them, in the order the gate made them, and keeps no copy of its own that could outlive the turn.
        number = len(before.ended)
        for move in moves:
            match move:
                case "stop" | "drop" | "timeout":
                    self._tell(GateEnded(TurnId(number), move, by))
                    number += 1
                case "start":
                    self._tell(GateOpened(TurnId(number), gate.opened, self._clock.now()))
                case "arm" | "disarm" | "listen" | "deafen":
                    pass
        # [LAW:nothing-unseen] the one writer of the gate writes each change of the phone's level as one `phone.leveled`
        # event, whichever made it: the page's word, or the call that came or went.
        if gate.phone_level != before.phone_level:
            ended("phone.leveled", self._record, begun(self._clock), None, was=before.phone_level, now=gate.phone_level, by=by)
        # [LAW:nothing-unseen] the one writer of the gate writes each move of the place as one `place.moved` event,
        # whichever trigger made it: from where to where, what made it, and whether a clip open at the place it left was
        # sent.
        if gate.place != before.place:
            # [LAW:one-source-of-truth] the way the user talks sets the modality, so a move sets it in the same write.
            self._modality = modality_at(gate.place)
            ended("place.moved", self._record, begun(self._clock), None, before=before.place, after=gate.place, by=by, sent="sent" in gate.ended[len(before.ended) :])

    def _tell(self, told: GateOpened | GateEnded) -> None:
        for watcher in self._watchers:
            watcher(told)

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

