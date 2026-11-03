"""The hold owner: it cuts the microphone's audio into holds where the gate cut them, numbers each, and gives every model
the same samples of it, in the order they were captured; it passes on what the models heard, in hold order.

The gate's key and counts travel with every frame (`hands.voice.ptt.KeyedAudio`), so a hold begins and ends where the
frames say the key moved, in capture order, and nothing else says so. As a hold opens, the owner says the user started
speaking (`TurnOpened`); as it ends, that they stopped, sent or thrown away (`HoldDiscarded`). Words the user typed are a
hold of their own, numbered with the key's and taken as they came. Each listener (`hands.voice.holds.Listener`) hears
every hold of the key's as it is spoken; the transcriber is given each hold the key sent, whole, once it ends, and its
words, or why it had none, go on in the order the holds ended, followed by the hold's `TurnResolved`.

[LAW:one-source-of-truth] the owner alone says where a hold is and what audio it is made of: no model keeps a hold's
bounds of its own, so no two can disagree about them.
"""

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, InputAudioRawFrame, StartFrame, VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language

from hands.clock import Clock
from hands.core.place import Place
from hands.sessions.audit import Record
from hands.sessions.wide import Begun, Fact, annotate, begun, continuing, ended, unit
from hands.voice.holds import LANGUAGE, RATE, Cleared, Fate, Hearing, Hearings, Hold, Listener, Recording, Sound, Transcriber, voices
from hands.voice.ptt import Key, KeyedAudio
from hands.voice.trigger import turn_start
from hands.voice.turnstop import Cut, HoldDiscarded, TurnOpened, TurnResolved, Typed, Words

# What every hold is said in, as Pipecat names it.
_LANGUAGE = Language(LANGUAGE)

# The least a hold's transcription is given before it fails. Whisper hears a hold of a few seconds in about a third of a
# second, so a minute is a transcription that is not coming back; see `_transcribing_seconds` for a long one.
TRANSCRIBING_SECONDS = 60.0

# The longest a listener's hearing may take, from when the listener returned it, so a hearing that waits behind a slow one
# is given what is left of its own. The streaming recognizer hears 20 ms of a hold open in about 8 ms, so a minute is a
# hearing that is not coming back.
HEARING_SECONDS = 60.0

# The longest a hold hands played over waits to close on its last identification, whose speakers decide whether it cuts
# hands off: a new timeout, the user's words held behind it. Scoring a speaker takes about 18 ms for each second scored,
# so the bound is for what the voiceprints' lane may have ahead of it, the model's load or a voice's enrolment; a hold that
# misses it cuts nothing, and its `hold` event says so.
IDENTIFIED_SECONDS = 2.0


def _transcribing_seconds(held: float) -> float:
    """The longest a hold `held` seconds long may take to transcribe before it fails: a minute, or the hold's own length.

    Ten minutes of speech (TURN_LIMIT_SECONDS) took 17 s on this Mac, so a model still running after as long as the hold
    took to say is not coming back; a bound that did not grow with the hold would fail a long one whole on a busier Mac,
    which is a turn lost to its length (promptctl-voice-je8.d20)."""
    return max(TRANSCRIBING_SECONDS, held)


@dataclass(frozen=True)
class _Spoken:
    """A hold of the key's as it ended: what the gate did with it, its audio, when it ended, and what its last
    identification comes to, settled from the voiceprints' lane."""

    hold: Hold
    fate: Fate
    sound: Sound
    began: Begun
    identified: "asyncio.Future[float | None]"


@dataclass(frozen=True)
class _Written:
    """A typed hold, its words, and when it was taken."""

    hold: Hold
    text: str
    began: Begun


@dataclass(frozen=True)
class _Lanes:
    """The tasks the owner runs off the audio path, made as the pipeline starts: what pushes, what resolves, and one
    hearing lane for each listener."""

    pushing: "asyncio.Task[None]"
    resolving: "asyncio.Task[None]"
    hearing: "tuple[asyncio.Task[None], ...]"


@dataclass(frozen=True)
class _Unstarted:
    """No lane runs: the pipeline has not started, or has stopped."""


def _settled(identified: "asyncio.Future[float | None]", first: float | None) -> None:
    """A hold's last identification, on the event loop: one the owner gave up waiting on goes unheard, as the hold's
    event already says."""
    match identified.cancelled():
        case True:
            pass
        case False:
            identified.set_result(first)


def _facts(ending: "_Spoken | _Written") -> dict[str, Fact]:
    """What the `hold` event says of a hold however it ends: which hold and what opened it, and of a spoken one what the
    gate did with it and how long and loud it was."""
    match ending:
        case _Written(hold=hold):
            return dict(hold=hold.number, opener=hold.opener)
        case _Spoken(hold=hold, fate=fate, sound=sound):
            return dict(hold=hold.number, opener=hold.opener, fate=fate, seconds=round(sound.seconds, 3), levels=sound.levels)


class HoldOwner(FrameProcessor):
    """The pipeline's voice activity detector, the owner of every hold's audio, and the one place the user's words enter
    the pipeline.

    [LAW:no-ambient-temporal-coupling] everything the owner says off the audio path goes out through one queue and one
    task that pushes it: a listener's words for the hold open are checked against it and queued in one step, with nothing
    awaited between, so they can never land behind that hold's `TurnResolved`, which is queued only after it ended.
    """

    def __init__(self, clock: Clock, transcriber: Transcriber, listeners: Sequence[Listener], record: Record) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self._clock = clock
        self._transcriber = transcriber
        self._listeners = tuple(listeners)
        self._record = record
        # The gate as the last frame was captured under it: the key, how many turns it had sent and thrown away, and
        # whose microphone it came from.
        self._key: Key = "up"
        self._sent = 0
        self._dropped = 0
        self._place: Place = "desk"
        self._recording = Recording()
        # How many holds have opened, the key's and those typed; and the last the key opened, hold 0 until one has. It is
        # open while the key is down.
        self._holds = 0
        self._held = Hold(0, "held key")
        # What is to be pushed, oldest first; the holds ended and not yet resolved; and each listener's hearings, with the
        # hold open as it returned them and when it did. None ends a queue, as the pipeline ends.
        self._out: asyncio.Queue[Frame | None] = asyncio.Queue()
        self._ended: asyncio.Queue[_Spoken | _Written | None] = asyncio.Queue()
        self._hearings: tuple[asyncio.Queue[tuple[Hold, Hearing, Begun]], ...] = tuple(asyncio.Queue() for _ in self._listeners)
        self._lanes: _Lanes | _Unstarted = _Unstarted()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case StartFrame():
                await self.push_frame(frame, direction)
                self._lanes = _Lanes(
                    self._task(self._push(), "the hold owner's pushes"),
                    self._task(self._resolve(), "the holds' resolution"),
                    tuple(self._task(self._hear(listener, hearings), f"{listener.name}'s hearings") for listener, hearings in zip(self._listeners, self._hearings, strict=True)),
                )
            case EndFrame():
                # As Pipecat's segmented transcription stops: every hold ended is resolved, and pushed, before the
                # pipeline ends.
                await self._drained()
                await self._stop()
                await self.push_frame(frame, direction)
            case CancelFrame():
                await self._stop()
                await self.push_frame(frame, direction)
            case Typed(text=text):
                # Taken here, not passed on: the hold it is goes on in its place.
                await self._typed(text)
            case InputAudioRawFrame():
                await self._captured(frame)
                await self.push_frame(frame, direction)
            case _:
                await self.push_frame(frame, direction)

    def _stamp(self) -> str:
        """The time a transcript is stamped with, in the form pipecat's own stamps take."""
        return self._clock.wall().isoformat(timespec="milliseconds")

    def _task(self, lane: Coroutine[Any, Any, None], name: str) -> "asyncio.Task[None]":
        """A lane, as a task of Pipecat's, which its task manager knows by `name`."""
        return cast("asyncio.Task[None]", self.create_task(lane, name))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)

    async def cleanup(self) -> None:
        await self._stop()
        await super().cleanup()

    async def _drained(self) -> None:
        """Every hold ended resolved, and everything said pushed."""
        match self._lanes:
            case _Lanes(pushing=pushing, resolving=resolving):
                await self._ended.put(None)
                await resolving
                await self._out.put(None)
                await pushing
            case _Unstarted():
                pass

    async def _stop(self) -> None:
        """End the lanes, which Pipecat leaves to the processor that made them. A hold still to be resolved, and a hearing
        still to run, end their events cancelled, as the hold and the hearings the lanes were on do."""
        match self._lanes:
            case _Lanes(pushing=pushing, resolving=resolving, hearing=hearing):
                self._lanes = _Unstarted()
                for lane in (resolving, *hearing, pushing):
                    await self.cancel_task(lane)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            case _Unstarted():
                pass
        while not self._ended.empty():
            match self._ended.get_nowait():
                case _Spoken() | _Written() as ending:
                    ended("hold", self._record, ending.began, asyncio.CancelledError(), **_facts(ending))
                case None:
                    pass
        for listener, hearings in zip(self._listeners, self._hearings, strict=True):
            while not hearings.empty():
                hold, _, began = hearings.get_nowait()
                ended("hearing", self._record, began, asyncio.CancelledError(), listener=listener.name, hold=hold.number)

    async def _typed(self, text: str) -> None:
        """A hold that opens and ends at once, and is resolved in line behind the holds ended ahead of it, as one the key
        sent would be: it joins a turn a hold of the key's has open, and leaves the key's hold and its audio as they were."""
        self._holds += 1
        hold = Hold(self._holds, "typed")
        await self.push_frame(TurnOpened(hold=hold))
        await self.push_frame(VADUserStoppedSpeakingFrame())
        await self._ended.put(_Written(hold, text, begun(self._clock)))

    async def _captured(self, frame: InputAudioRawFrame) -> None:
        # [LAW:parse-dont-validate] the microphone makes every frame this sees, and tags each one.
        if not isinstance(frame, KeyedAudio):
            raise TypeError(f"{type(frame).__name__} carries no key; the keyed microphone makes every frame a hold is cut from")
        if (frame.sample_rate, frame.num_channels) != (RATE, 1):
            raise ValueError(f"a frame came as {frame.num_channels} channel(s) at {frame.sample_rate} Hz; every hold is mono at {RATE} Hz")
        # [LAW:no-ambient-temporal-coupling] a turn ends as the gate's counts move, not as its key is next seen at rest: a
        # turn can end and the next arm between two frames, with no frame captured at rest between them. A move between
        # places drops the hold open in the same frame, which ends it over its own audio before the move clears it.
        match self._key, frame.sent != self._sent, frame.dropped != self._dropped:
            case "down", True, _:
                await self._end("sent")
            case "down", _, True:
                # Thrown away, as typing, or a call came or went.
                await self._end("dropped")
            case _:
                pass
        self._sent, self._dropped = frame.sent, frame.dropped
        if frame.place != self._place:
            # A turn is heard at one place: what the other's microphone heard before it is no part of it.
            self._recording.clear()
            self._place = frame.place
        match self._key, frame.key:
            case ("up", "arming") | ("arming", "up"):
                # A hold's audio begins at its press, and a press that was Shift after all heard no part of any turn.
                self._recording.clear()
            case "up" | "listening" | "arming", "down":
                self._holds += 1
                self._held = Hold(self._holds, frame.opened)
                await self.push_frame(TurnOpened(hold=self._held))
                # What the press heard while arming, or the second the desk heard while listening, is the hold's start.
                before, cleared = self._recording.sound, self._clearing(self._held)
                self._tell(self._held, lambda listener: listener.opened(self._held, before, cleared))
            case "down", "up" | "listening" | "arming":
                # [LAW:no-silent-failure] the gate counts every end of a turn, so a key leaving down uncounted is a gate
                # that lies.
                raise RuntimeError(f"the key went from down to {frame.key} with no turn sent or thrown away")
            case _:
                # From listening to arming, or back, the second the desk heard is kept: the turn's first words.
                pass
        self._key = frame.key
        sound = Sound(frame.audio, frame.captured, frame.played)
        # [LAW:no-ambient-temporal-coupling] a press is heard as speech from the start: nothing it hears is trimmed while
        # HOLD_SECONDS runs, however long.
        self._recording.add(sound, gathering=frame.key in ("arming", "down"))
        match frame.key:
            case "down":
                self._tell(self._held, lambda listener: listener.heard(sound))
            case _:
                pass

    def _tell(self, hold: Hold, told: Callable[[Listener], Hearings]) -> None:
        """Tell every listener of `hold`, in turn: each hearing it returns goes on its lane, beginning now."""
        for listener, hearings in zip(self._listeners, self._hearings, strict=True):
            # A hearing is a trace of its own, part of no unit the audio path is in, so its lane can always run it.
            with continuing(None):
                for hearing in self._told(listener, hold, told):
                    hearings.put_nowait((hold, hearing, begun(self._clock)))

    def _told(self, listener: Listener, hold: Hold, told: Callable[[Listener], Hearings]) -> Hearings:
        """The hearings `listener` returned as it was told of `hold`.

        [LAW:no-silent-failure] a listener that raises is a failed `hearing` event of its own, with what raised, and no
        hearings: it breaks only itself, never the hold, which every other listener and the transcriber still hear."""
        try:
            return told(listener)
        except Exception as error:
            ended("hearing", self._record, begun(self._clock), error, listener=listener.name, hold=hold.number)
            return ()

    async def _end(self, fate: Fate) -> None:
        """The hold open ends: every listener is told, and it goes in line to be resolved, transcribed where it was sent."""
        hold, sound = self._held, self._recording.sound
        # [LAW:no-ambient-temporal-coupling] closed before anything is awaited: words a hearing returns while the hold's
        # end is pushed find it closed, and cut nothing.
        self._key = "up"
        self._recording.clear()
        loop = asyncio.get_running_loop()
        identified: asyncio.Future[float | None] = loop.create_future()

        def settle(first: float | None) -> None:
            loop.call_soon_threadsafe(_settled, identified, first)

        self._tell(hold, lambda listener: listener.ended(hold, fate, sound, settle))
        match fate:
            case "sent":
                await self.push_frame(VADUserStoppedSpeakingFrame())
            case "dropped":
                await self.push_frame(HoldDiscarded())
        await self._ended.put(_Spoken(hold, fate, sound, begun(self._clock), identified))

    def _open(self, hold: Hold) -> bool:
        """Whether `hold` is the hold open now."""
        return self._key == "down" and self._held == hold

    def _clearing(self, hold: Hold) -> Cleared:
        """What a listener tells of each speaker it clears in `hold` to cut hands off, from whichever thread: the person is
        taken on the event loop, which owns whether the hold is open."""
        loop = asyncio.get_running_loop()

        def cleared(first: float) -> None:
            loop.call_soon_threadsafe(self._person, hold, first)

        return cleared

    def _person(self, hold: Hold, first: float) -> None:
        """[LAW:single-enforcer] a speaker cleared in `hold` cuts hands off while it is still open, checked and queued in
        one step, as a hearing's are: one cleared once it closed cuts nothing, so a late identification never cuts the
        turn after it, and the hold's last identification says what its close came to."""
        match self._open(hold):
            case True:
                self._out.put_nowait(Cut(hold=hold, first=first))
            case False:
                pass

    async def _push(self) -> None:
        """Push what the owner said off the audio path, in the order it said it, until told the pipeline ends."""
        while (frame := await self._out.get()) is not None:
            match frame:
                case ErrorFrame():
                    await self.push_error_frame(frame)
                case _:
                    await self.push_frame(frame)

    async def _resolve(self) -> None:
        """Resolve each hold in the order it ended, until told the pipeline ends."""
        while (ending := await self._ended.get()) is not None:
            await self._resolved(ending)

    async def _resolved(self, ending: _Spoken | _Written) -> None:
        """Queue what a hold came to and its `TurnResolved`, and tell it as the `hold` event, however it ends."""
        facts = _facts(ending)
        try:
            match ending:
                case _Written(text=text):
                    self._out.put_nowait(Words(text, "", self._stamp(), _LANGUAGE))
                    facts["transcript"] = "words"
                case _Spoken(hold=hold, fate=fate, sound=sound, identified=identified):
                    match fate:
                        case "dropped":
                            pass
                        case "sent":
                            words = await self._transcribed(hold, sound, facts)
                            # The cut lands ahead of the words it came with, as one while the hold was open would have.
                            match await self._closing(hold, sound, identified, words, facts):
                                case None:
                                    pass
                                case first:
                                    self._out.put_nowait(Cut(hold=hold, first=first))
                            for each in words:
                                self._out.put_nowait(each)
            # [LAW:dataflow-not-control-flow] heard, heard nothing, failed, or thrown away, the owner is done with the hold.
            self._out.put_nowait(TurnResolved(hold=ending.hold))
        except asyncio.CancelledError as cancelled:
            ended("hold", self._record, ending.began, cancelled, **facts)
            raise
        ended("hold", self._record, ending.began, None, **facts)

    async def _closing(self, hold: Hold, sound: Sound, identified: "asyncio.Future[float | None]", words: tuple[Words, ...], facts: dict[str, Fact]) -> float | None:
        """When the speaker who cuts hands off as `hold` closes was first heard, None for no cut: a turn the voice opened
        is cut here where nothing cut it while the hold was open.

        A hold hands played nothing over has no echo of hands in it, so its words cut, their speaker heard from its
        opening, and nothing waits on a voiceprint. One it played over cuts on its last identification, which judges a
        speaker heard under a second too (`hands.voice.voiceprints.CLOSING`): a wake phrase said alone over hands, then
        silence. A hold the user's hand opened cut as it opened."""
        match turn_start(hold.opener), voices(sound), words:
            case "on the hold", _, _:
                return None
            case "on a person", (), (Words(), *_):
                return 0.0
            case "on a person", (), ():
                return None
            case "on a person", _, _:
                return await self._identified(identified, facts)

    async def _identified(self, identified: "asyncio.Future[float | None]", facts: dict[str, Fact]) -> float | None:
        """What a hold's last identification came to, waited on for IDENTIFIED_SECONDS at most.

        [LAW:no-silent-failure] one that misses the bound cuts nothing, which the hold's event says, as it says what one
        in time came to."""
        bound = self._clock.timeout(IDENTIFIED_SECONDS)
        try:
            async with bound:
                first = await identified
        except TimeoutError:
            facts["identified"] = f"nothing after {IDENTIFIED_SECONDS:g} s"
            return None
        facts["identified"] = "nobody" if first is None else "a person"
        return first

    async def _transcribed(self, hold: Hold, sound: Sound, facts: dict[str, Fact]) -> tuple[Words, ...]:
        """What a sent hold's transcription came to: its words, to be pushed, and on its event, why it had none, an error
        queued where it failed.

        [LAW:no-silent-failure] a transcription that never returns would hold its turn open for ever, since nothing but the
        owner resolving its holds ends one: it fails instead. A model that is running cannot be stopped, so the next
        hold's transcription waits behind it on the model's own thread, inside its own bound."""
        seconds = _transcribing_seconds(sound.seconds)
        bound = self._clock.timeout(seconds)
        try:
            async with bound:
                said = await self._transcriber.transcribed(hold, sound.heard)
        except Exception as error:
            # [LAW:no-silent-failure] a failed transcription is heard: the pipeline says an ErrorFrame from here aloud.
            why = f"nothing after {seconds:g} s" if bound.expired() else f"{type(error).__name__}: {error}"
            self._out.put_nowait(ErrorFrame(error=f"Hold {hold.number} could not be transcribed: {why}", exception=error))
            facts["transcript"] = "failed"
            return ()
        match said:
            case None:
                # Not said: Brandon does not need to hear it (2026-09-27).
                facts["transcript"] = "none"
                return ()
            case words:
                facts["transcript"] = "words"
                return (Words(words, "", self._stamp(), _LANGUAGE),)

    async def _hear(self, listener: Listener, hearings: "asyncio.Queue[tuple[Hold, Hearing, Begun]]") -> None:
        """Run one listener's hearings in the order it returned them, each as one `hearing` event."""
        while True:
            hold, hearing, began = await hearings.get()
            try:
                await self._heard(listener, hold, hearing, began)
            except Exception:
                # [LAW:no-silent-failure] said already, on its `hearing` event, failed with what raised: the lane hears on.
                pass

    async def _heard(self, listener: Listener, hold: Hold, hearing: Hearing, began: Begun) -> None:
        """One hearing, bounded, and what came of it: a speaker its words clear, in the hold still open, cuts hands off."""
        # [LAW:nothing-unseen] the listener puts what its model made of the audio on this event, from inside the hearing;
        # its duration runs from when the listener returned it, so one that waited behind another says how long it waited.
        with continuing(None), unit("hearing", self._record, began):
            annotate(listener=listener.name, hold=hold.number)
            async with self._clock.timeout(HEARING_SECONDS - (self._clock.now() - began.began)):
                heard = await hearing()
            annotate(heard="nothing" if heard.words is None else "words")
            # [LAW:single-enforcer] a speaker a listener's words cleared cuts while their hold is still open, as one its
            # identification cleared does (`_person`). Heard once it has closed, they cut nothing, and the hold's close
            # says what it held. The check and the queueing are one step, with nothing awaited between.
            match heard.cuts, self._open(hold):
                case float() as first, True:
                    self._out.put_nowait(Cut(hold=hold, first=first))
                    annotate(cut=True)
                case _:
                    annotate(cut=False)
