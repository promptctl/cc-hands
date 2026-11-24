"""The phone: one WebRTC call from the page hands serves, carrying the phone's microphone in, hands' speech out, and
the page's talk button.

The call is where the phone is. A call arriving moves the gate there, unless the talk key is pressed or a clip is open
at the desk, which the phone's first press then takes over; the call ending moves it back to the desk, in the same step
that ends the call, so nothing is ever played to a phone that has gone. A newer call replaces the one before it: the page last opened is the one in the user's hand.

[LAW:nothing-unseen] a call is one unit of work, `phone.call`, from its page's offer to its end: refused at the page,
let go before it connected, or left after hands was at it, and why; its duration is how long it lasted.

The page sends its microphone as plain 16-bit audio over the call's data channel, in order with its button, and only
inside a turn: outside one it sends, on the same 20 ms, a `quiet` that counts the samples it held back, which hands hears as
that much silence. Its detector's word, `wake`, is a press that the gate may refuse: the phone's level, `level`, `off` or
`wake word` for the call, makes the key rest at `listening`, so the second the page sends ahead of `wake` is kept as the
desk's is, and a wake opens a turn the phone's driver (`hands.voice.atphone`) hears from the wake on and ends by the rules of
the desk's wake word turn. Hands plays to the page over an audio track. Over the same channel hands tells the page each milestone of the turn it is waiting on
(`hands.voice.milestone`): the clip each of its presses opened, by the press's number, or that it opened none and why; the clip taken, or thrown away; its words heard or none found, a stage failing, the reply begun,
and the first sound, which the page shows with how long each took; and each line of the transcript
(`hands.voice.transcript`), which the page shows when asked to. Earbuds keep hands' voice out of the phone's microphone, and the page asks the browser
for its echo cancellation as well; so the phone's audio is gated by its button alone, with none of the desk
microphone's wait for the room to go quiet.

A page whose call drops calls again on its own, and says so in its offer: it resumes the phone, where a press of its
Connect takes it. A resume is declined while another page has the call or an offer in, so two pages left open never take
the phone back and forth between them; only a press takes it from another page.
"""

import asyncio
import fractions
import json
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal, NewType

import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler
from loguru import logger

from hands.clock import Clock
from hands.sessions.audit import Record
from hands.sessions.payload import Payload, Rejected
from hands.sessions.wide import Begun, Fact, begun, continuing, ended, since
from hands.voice.milestone import ClipDiscarded, ClipOpened, ClipReleased, Milestone, Told, TurnUnopened
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.voice.shortcut import Instant, Move
from hands.voice.transcript import Line
from hands.voice.trigger import PhoneLevel
from hands.voice.voiceturn import TurnId
from hands.voice.wakeword import THRESHOLD

# The length of each frame sent to the phone: Opus's own 20 ms.
FRAME_SECS = 0.02
# How long the page may send nothing before its call is taken to be over. The page sends a message every 20 ms, audio or
# a `quiet`, for as long as it is open, so a channel this quiet is a page closed, a phone put to sleep, or a network gone: measured
# 2026-10-03, a browser closed outright leaves aiortc's connection "connected" for good.
QUIET_SECS = 3.0


# Why a call ended: a newer call took its place, the page hung up, the connection failed, the page stopped sending
# (closed outright, or the phone asleep), or hands stopped.
PhoneGone = Literal["replaced", "hung up", "failed", "went quiet", "stopped"]

# What an offer asks of the phone: to take it from whatever page has it, as a press of the page's Connect asks, or to
# resume it, as a page whose call dropped asks on its own, which no other page's call or offer gives way to.
Claim = Literal["take", "resume"]


@dataclass(frozen=True)
class Asked:
    """Which page an offer came from, by the name it made itself as it opened, and what it asked of the phone."""

    page: str
    claim: Claim


@dataclass(frozen=True)
class CallRefused:
    """The page's offer was not taken: it came without the phone's key, or was no offer."""

    why: str


@dataclass(frozen=True)
class CallUnreached:
    """The call was answered and let go before it connected: hands never moved to it."""

    reason: PhoneGone


@dataclass(frozen=True)
class CallLeft:
    """The call was up, hands at the phone from `arrived_ms` after its offer, and it ended, its page `told` that many
    milestones of its turns and that many `lines` of the transcript. The page held back `held_back` samples of its
    microphone as `quiet` messages, and sent `rejected` messages hands could not read."""

    reason: PhoneGone
    arrived_ms: float
    told: int
    lines: int
    held_back: int
    rejected: int


@dataclass(frozen=True)
class CallDeclined:
    """A resume offered while another page had the call or an offer in: not answered, and that page keeps the phone."""


# How a call ended; or what raised as hands answered it.
CallEnd = CallRefused | CallDeclined | CallUnreached | CallLeft | BaseException


def call_ended(record: Record, began: Begun, remote: str, asked: Asked | None, end: CallEnd) -> None:
    """[LAW:single-enforcer] the one place a call's event is written, as it ends, however it ends: failed where it was
    refused, where its connection failed, or where answering it raised. `asked` is None for an offer refused before it
    was read."""
    facts: dict[str, Fact] = {"remote": remote} if asked is None else {"remote": remote, "asked": asked}
    match end:
        case BaseException():
            ended("phone.call", record, began, end, **facts)
        case CallRefused(why=why):
            ended("phone.call", record, began, None, why, **facts, ended=end)
        case CallDeclined():
            ended("phone.call", record, began, None, **facts, ended=end)
        case CallUnreached(reason=reason) | CallLeft(reason=reason):
            failure = "the call's connection failed" if reason == "failed" else None
            ended("phone.call", record, began, None, failure, **facts, ended=end)


def kind(line: type[Line]) -> str:
    """The word the page knows a line of the transcript by: its class's name."""
    return line.__name__.lower()


@dataclass(frozen=True)
class Opened:
    """The clip a press of the page's opened: the page's number for the press, counted in the call, and hands' for the
    clip."""

    press: int
    clip: int


@dataclass(frozen=True)
class Unopened:
    """A press of the page's that opened no clip, and why: the page's number for it, which hands answers once, here or as
    `Opened`."""

    press: int
    why: str


def answer(press: int, told: ClipOpened | TurnUnopened) -> Opened | Unopened:
    """What the page is told of its press `press`, from what hands told of the turn the press opened."""
    match told:
        case ClipOpened(clip=clip):
            return Opened(press, clip.number)
        case TurnUnopened(why=why):
            return Unopened(press, why)


def wire(told: Opened | Unopened | ClipReleased | ClipDiscarded | Milestone | Line) -> str:
    """[LAW:single-enforcer] what the page is told, as it reads it: one JSON object, its `kind` saying which."""
    match told:
        case Opened(press=press, clip=clip):
            return json.dumps({"kind": "opened", "press": press, "clip": clip})
        case Unopened(press=press, why=why):
            return json.dumps({"kind": "unopened", "press": press, "why": why})
        case ClipReleased(clip=clip):
            return json.dumps({"kind": "milestone", "milestone": "released", "clip": clip.number})
        case ClipDiscarded(clip=clip):
            return json.dumps({"kind": "milestone", "milestone": "discarded", "clip": clip.number})
        case str():
            return json.dumps({"kind": "milestone", "milestone": told})
        case line:
            return json.dumps({"kind": kind(type(line)), **asdict(line)})


class Outbound(MediaStreamTrack):
    """Hands' speech as the phone hears it: what was given, in order, in real time, and silence between.

    Each piece given is a future resolved once it has been sent, so the speaker that gives it waits as long as the
    audio lasts, as it waits on a device's write; a call that ends resolves every future it still holds.
    """

    kind = "audio"

    def __init__(self, clock: Clock, sample_rate: int) -> None:
        super().__init__()
        self._clock = clock
        self._sample_rate = sample_rate
        self._samples = round(sample_rate * FRAME_SECS)
        self._pending: deque[tuple[bytes, asyncio.Future[None] | None]] = deque()
        self._sent = 0
        self._start: float | None = None

    def give(self, audio: bytes) -> asyncio.Future[None]:
        """Queue 16-bit mono audio to be sent; the future resolves when its last frame has gone."""
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        size = self._samples * 2
        # The last frame is padded with silence: the phone hears a frame's worth at a time.
        padded = audio + bytes(-len(audio) % size)
        frames = [padded[i : i + size] for i in range(0, len(padded), size)]
        match self.readyState:
            case "ended":
                done.set_result(None)
            case _:
                for index, frame in enumerate(frames):
                    self._pending.append((frame, done if index == len(frames) - 1 else None))
                if not frames:
                    done.set_result(None)
        return done

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        # [LAW:no-ambient-temporal-coupling] frames are dated by the count sent, against the clock at the first: a
        # late frame is followed at once by the next, so the call never drifts behind real time.
        now = self._clock.now()
        start = self._start = now if self._start is None else self._start
        await self._clock.sleep(max(0.0, start + self._sent * FRAME_SECS - now))
        frame, done = self._pending.popleft() if self._pending else (bytes(self._samples * 2), None)
        if done is not None and not done.done():
            done.set_result(None)
        out = AudioFrame.from_ndarray(np.frombuffer(frame, dtype=np.int16)[None, :], format="s16", layout="mono")
        out.sample_rate = self._sample_rate
        out.pts = self._sent * self._samples
        out.time_base = fractions.Fraction(1, self._sample_rate)
        self._sent += 1
        return out

    def stop(self) -> None:
        super().stop()
        while self._pending:
            _, done = self._pending.popleft()
            if done is not None and not done.done():
                done.set_result(None)


@dataclass(frozen=True)
class Offer:
    """A page's session description, as its browser made it, every candidate gathered; the rate its microphone's audio
    comes at, which is the browser's own; and which page it is, asking what."""

    sdp: str
    type: Literal["offer"]
    rate: int
    asked: Asked


# The sample rates hands answers a page at: the range Chrome runs an AudioContext within. The top of it bounds what a
# bad offer can make hands build from one message, by `largest_message`.
PAGE_RATES = range(3000, 768001)
# The longest one message of the page's may be, audio or count: five of the 20 ms it sends. hands' own limit, wide
# enough that a page cutting its messages another way is still read, and the one place a message that holds or claims
# more is refused before hands builds anything from it.
MESSAGE_MS = 100


def largest_message(rate: int) -> int:
    """The most samples one message of the page's may hold, or count in a `quiet`, at `rate`."""
    return rate * MESSAGE_MS // 1000


# The page's microphone as the pipeline is fed it: 16-bit mono samples at the call's rate, whole, and no more than one
# message may hold. Made only by `parse_said`, or for the silence a `Quiet` counts.
Pcm = NewType("Pcm", bytes)


@dataclass(frozen=True)
class Quiet:
    """The samples of its microphone, at the call's rate, that the page held back because no turn was open: heard as that
    much silence, so the page's messages stay the pipeline's clock. At least one, and no more than one message may hold,
    which `parse_said` has checked against the call's rate."""

    samples: int

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise Rejected(f"the phone's page held back {self.samples} samples, which is not a count of at least 1")


def quiet_message(samples: int) -> str:
    """A `quiet` as the Python players of the page write it, the smoke and the tests. The page writes its own in phone.html;
    `parse_said` decides whether either is read, and the Chrome test in tests/test_pages.py holds the page's to it."""
    return json.dumps({"kind": "quiet", "samples": samples})


@dataclass(frozen=True)
class Wake:
    """The page's detector heard the wake word, at `score`, and no turn of the page's is open: the page sent the second it kept
    from before, then this, then its microphone as it comes."""

    score: float


@dataclass(frozen=True)
class Listening:
    """The page's level, as it stands now, and each time it changes."""

    level: PhoneLevel


@dataclass(frozen=True)
class DetectorLoaded:
    """The page's detector loaded, in the browser the page said it is in."""

    browser: str


@dataclass(frozen=True)
class DetectorFailed:
    """The page's detector could not load, and why, in the browser the page said it is in."""

    browser: str
    why: str


def wake_message(score: float) -> str:
    """A `wake` as the Python players of the page write it; the page writes its own in phone.html."""
    return json.dumps({"kind": "wake", "score": score})


def level_message(level: PhoneLevel) -> str:
    """A `level` as the Python players of the page write it; the page writes its own in phone.html."""
    return json.dumps({"kind": "level", "level": level})


def detector_message(browser: str, why: str | None) -> str:
    """A `detector` as the Python players of the page write it: loaded, or failed for `why`."""
    return json.dumps({"kind": "detector", "state": "loaded", "browser": browser} if why is None else {"kind": "detector", "state": "failed", "browser": browser, "why": why})


# What the page sends over the call's data channel: its microphone, the samples it held back, its button, and what its
# detector heard and what level it listens at.
Said = Pcm | Quiet | Move | Wake | Listening | DetectorLoaded | DetectorFailed


def parse_said(message: object, rate: int) -> Said:
    """One message of the page's: 16-bit mono audio at the offer's rate `rate`, a `quiet`, the button pressed or released, or
    what its detector heard and its level; raises Rejected for anything else, and for audio or a count of more than one
    message may hold."""
    most = largest_message(rate)
    match message:
        case bytes():
            if not message or len(message) % 2 or len(message) > 2 * most:
                raise Rejected(f"the phone's page sent audio of {len(message)} bytes, which is not from 1 to {most} whole 16-bit samples")
            return Pcm(message)
        case "press":
            return "start"
        case "release":
            return "stop"
        case str():
            try:
                said = Payload.parse(message.encode())
                match said.text("kind"):
                    case "quiet":
                        samples = said.integer("samples")
                        if samples > most:
                            raise Rejected(f"it counts {samples} samples, and one message may count {most} at most")
                        return Quiet(samples)
                    case "wake":
                        return Wake(said.number("score"))
                    case "level":
                        match said.text("level"):
                            case "off" | "wake word" as level:
                                return Listening(level)
                            case other:
                                raise Rejected(f"its level is {other!r}, which is neither off nor wake word")
                    case "detector":
                        browser = said.text("browser")
                        match said.text("state"):
                            case "loaded":
                                return DetectorLoaded(browser)
                            case "failed":
                                return DetectorFailed(browser, said.text("why"))
                            case other:
                                raise Rejected(f"its state is {other!r}, which is neither loaded nor failed")
                    case other:
                        raise Rejected(f"its kind is {other!r}, which is none hands reads")
            except Rejected as error:
                raise Rejected(f"the phone's page sent {message[:80]!r}, which hands cannot read: {error}") from error
        case _:
            raise Rejected(f"the phone's page sent {message!r}, which is neither audio, a quiet, a press, a release, a wake, a level, nor a detector")


@dataclass(frozen=True)
class WakeAudio:
    """A frame of the phone's microphone, 16-bit mono at the pipeline's rate, captured while a turn a wake opened was open:
    `turn` is the gate's number for that turn, so what the frame says is for it and for no turn after; `at` is when the
    phone read it off the channel, which for a turn's first frame is when the wake opened it."""

    turn: TurnId
    audio: bytes
    at: Instant


@dataclass
class _Call:
    peer: RTCPeerConnection
    outbound: Outbound
    resampler: AudioResampler
    rate: int
    remote: str
    asked: Asked
    began: Begun
    # Set by every message the page sends that hands reads; the watch hangs up a call that goes QUIET_SECS without one.
    heard: asyncio.Event
    # Set as it arrives: its channel, its watch, and how long after its offer it came.
    channel: RTCDataChannel | None = None
    watch: asyncio.Task[None] | None = None
    arrived_ms: float = 0.0
    # How many presses the page has sent, as it numbers them, and the gate's turn each opened, until hands answers the press:
    # with the clip it opened, or that it opened none.
    # [LAW:one-source-of-truth] the channel is ordered and reliable, so the call counts every press the page counts.
    presses: int = 0
    pressing: dict[int, int] = field(default_factory=dict[int, int])
    # How many milestones of its turns, and lines of the transcript, the page has been told.
    told: int = 0
    lines: int = 0
    # The samples its page has held back as `quiet` messages, and the messages it sent that hands could not read.
    held_back: int = 0
    rejected: int = 0


class Phone:
    """The one call to the phone, and what it hears, plays, and presses.

    A call is offered once its page's offer is answered, and takes hands once its channel opens, the page's first
    word, unless a press or a clip at the desk keeps it there for the phone's first press: hands never moves to a call
    that cannot yet hear it, and an offer that never connects leaves the call
    that is up as it was. The newest offer is the page in the user's hand, so it lets go of any older one not yet up;
    but a resume is a page calling again on its own, not the user's hand, so it gives way to any other page's call or
    offer.

    [LAW:no-ambient-temporal-coupling] the microphone and the button come over one ordered channel, and each message is
    acted on as it arrives: a press moves the gate before the audio sent after it is keyed, and a release after the
    audio sent before it, so a turn holds exactly what was said while the button was down, whatever the network did.
    Every handler acts only for the call it was made for while that call is still the one it was: a word from a call
    already gone moves nothing.
    """

    def __init__(self, clock: Clock, key: PushToTalk, heard_rate: int, played_rate: int, speaking: Callable[[], bool], record: Record) -> None:
        self._clock = clock
        self._key = key
        self._heard_rate = heard_rate
        self._played_rate = played_rate
        # [LAW:one-source-of-truth] whether hands is speaking, read from the speaker that plays to the phone, as the desk's
        # wake word's event reads it.
        self._speaking = speaking
        self._record = record
        # The call hands is at, and the newest offer answered and not yet up.
        self._call: _Call | None = None
        self._offered: _Call | None = None
        # The phone's microphone as the pipeline hears it: keyed frames at the pipeline's rate, while hands is at the phone.
        self.heard: asyncio.Queue[KeyedAudio] = asyncio.Queue()
        # The audio that is of a turn a wake opened, in order: what the phone's driver hears the turn with, from the wake on
        # and no earlier (`hands.voice.atphone`).
        self.wakes: asyncio.Queue[WakeAudio] = asyncio.Queue()
        # The moves the page's button has made, as the gate took them, for what follows a move: its tone and its words.
        self.moves: asyncio.Queue[Move] = asyncio.Queue()

    @property
    def heard_rate(self) -> int:
        """The rate the phone's audio is heard at, which the models that hear it are loaded for."""
        return self._heard_rate

    async def answer(self, offer: Offer, remote: str, began: Begun) -> RTCSessionDescription | CallDeclined:
        """Answer the call a page offers, `began` as it was offered, with every candidate hands has; it is taken once it
        connects. A resume while another page has the call or an offer in is declined, and nothing changes."""
        # [LAW:one-source-of-truth] no ICE server: the phone reaches hands at an address it already has, on the LAN or
        # the tailnet, so hands' own addresses are every candidate there is.
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        outbound = Outbound(self._clock, self._played_rate)
        resampler = AudioResampler(format="s16", layout="mono", rate=self._heard_rate)
        call = _Call(peer, outbound, resampler, offer.rate, remote, offer.asked, began, asyncio.Event())
        peer.addTrack(outbound)

        @peer.on("datachannel")
        async def opened(channel: RTCDataChannel) -> None:  # pyright: ignore[reportUnusedFunction]
            # The page closes its channel as it goes: the plainest word that the user hung up.
            @channel.on("close")
            async def closed() -> None:  # pyright: ignore[reportUnusedFunction]
                if self._call is call:
                    await self.hang_up("hung up")

            @channel.on("message")
            def said(message: object) -> None:  # pyright: ignore[reportUnusedFunction]
                if self._call is not call:
                    # Queued before its call was let go: the call it spoke for is gone, and hands is elsewhere.
                    return
                try:
                    said = parse_said(message, call.rate)
                except Rejected as error:
                    # [LAW:no-silent-failure] the page and hands disagree about what it sends: said, not guessed at. It is
                    # no sign the page is alive: a page that sends only what hands cannot read is hung up as one that
                    # sends nothing.
                    call.rejected += 1
                    logger.error(str(error))
                    return
                call.heard.set()
                self._hear(call, said)

            if self._offered is call:
                await self._arrive(call, channel)

        @peer.on("connectionstatechange")
        async def changed() -> None:  # pyright: ignore[reportUnusedFunction]
            if peer.connectionState == "failed":
                if self._call is call:
                    await self.hang_up("failed")
                if self._offered is call:
                    self._offered = None
                    await self._let_go(call, "failed")

        try:
            await peer.setRemoteDescription(RTCSessionDescription(offer.sdp, offer.type))
            await peer.setLocalDescription(await peer.createAnswer())
        except BaseException as error:
            # Never offered, so nothing else ends its call or holds the peer to close it.
            call_ended(self._record, began, remote, offer.asked, error)
            await peer.close()
            raise
        # [LAW:single-enforcer] checked where the offer is put in, after every await of its answering: a press put in
        # while a resume was being answered is the page in the user's hand.
        if offer.asked.claim == "resume" and any(held.asked.page != offer.asked.page for held in (self._call, self._offered) if held is not None):
            call_ended(self._record, began, remote, offer.asked, CallDeclined())
            await peer.close()
            return CallDeclined()
        older, self._offered = self._offered, call
        await self._let_go(older, "replaced")
        return peer.localDescription

    async def _arrive(self, call: _Call, channel: RTCDataChannel) -> None:
        # [LAW:no-ambient-temporal-coupling] the call that was up goes, and this one comes, in one step before any
        # await: a speaker reading the gate between them never finds the phone without a call.
        self._offered = None
        gone = self._leave("replaced")
        self._call, call.channel, call.arrived_ms = call, channel, since(call.began.clock, call.began.began)
        # [LAW:nothing-unseen] the move a call makes is part of that call, in its trace, wherever the call's end is noticed.
        with continuing(call.began.span):
            self._key.go("phone")
        # Watched from the channel's opening: the page sends nothing before it.
        call.watch = asyncio.create_task(self._watch(call), name="the phone's call watch")
        match gone:
            case None:
                pass
            case _Call():
                await gone.peer.close()

    async def _let_go(self, offered: _Call | None, reason: PhoneGone) -> None:
        # An offer is let go of once it is no longer the one offered, so no other handler of it acts again.
        match offered:
            case None:
                pass
            case _Call():
                call_ended(self._record, offered.began, offered.remote, offered.asked, CallUnreached(reason))
                await offered.peer.close()

    async def _watch(self, call: _Call) -> None:
        while True:
            try:
                async with self._clock.timeout(QUIET_SECS):
                    await call.heard.wait()
            except TimeoutError:
                if self._call is call:
                    await self.hang_up("went quiet")
                return
            call.heard.clear()

    def _hear(self, call: _Call, said: Said) -> None:
        match said:
            case bytes():
                self._heard(call, said)
            case Quiet(samples=samples):
                call.held_back += samples
                # [LAW:dataflow-not-control-flow] silence is the audio it stands for: it takes the path audio takes.
                self._heard(call, Pcm(bytes(2 * samples)))
            case Wake(score=score):
                self._woke(call, score)
            case Listening(level=level):
                self._listening(level)
            case DetectorLoaded() | DetectorFailed():
                self._detected(said)
            case move:
                # [LAW:single-enforcer] a press over a turn a wake opened sends that turn first, as the talk key does at the
                # desk, so the two are two clips; the gate ends it by the number it opened it under.
                self._took(self._key.end(self._key.gate.wake_turn if move == "start" else None, "stop", "phone wake word"))
                taken = self._key.move(move, "phone button")
                match move:
                    case "start":
                        # [LAW:no-ambient-temporal-coupling] the press is bound to the turn it opened as it opens it, so
                        # the page learns its clip exactly, however late either reaches it.
                        call.presses += 1
                        call.pressing[self._key.gate.turn] = call.presses
                    case _:
                        pass
                self._took(taken)

    def _took(self, taken: Move | None) -> None:
        """What the gate took a move as, once the phone made it or its driver did, goes to the tone and the words that
        follow it; a move it took as nothing does not."""
        match taken:
            case None:
                pass
            case taken:
                self.moves.put_nowait(taken)

    def _woke(self, call: _Call, score: float) -> None:
        """The page's wake: a press, numbered as one, that the gate may refuse. The page is told which it was, `opened` by the
        clip its turn is recorded in or `unopened` and why, once."""
        call.presses += 1
        gate = self._key.gate
        began = begun(self._clock)
        speaking = self._speaking()
        # What the page is told when no turn opened, and the page's mistake if it was one: each way the wake went says both.
        why: str | None
        mistake: str | None
        match gate.place, gate.key:
            case "phone", "listening":
                self._took(self._key.move("arm", "phone wake word"))
                self._took(self._key.move("start", "phone wake word"))
                # The wake is bound to the turn it opened as it opens it, as a press is.
                call.pressing[self._key.gate.turn] = call.presses
                why = mistake = None
            case "phone", "arming" | "down":
                why = "a turn is open"
                mistake = "the phone's page sent a wake with a turn open, which a correct page does not"
            case "phone", "up":
                why = "the phone is not listening"
                mistake = "the phone's page sent a wake with its level off, which a correct page does not"
            case "desk", _:
                # Nothing is opened or moved: a voice opens turns only where hands is, and the user's hand takes it elsewhere.
                why = "hands is at the desk"
                mistake = None
        # [LAW:no-silent-failure] the page's own turn state covers hands', and `level` rides the same ordered channel, so a
        # mistake of the page's is said, and counted on the wake's event as a failure.
        match mistake:
            case None:
                pass
            case mistake:
                logger.error(mistake)
        match why:
            case None:
                pass
            case why:
                self._answered(call, Unopened(call.presses, why))
        ended("phone.wake", self._record, began, None, mistake, score=score, threshold=THRESHOLD, hands_speaking=speaking, opened=why is None)

    def _answered(self, call: _Call, answer: Opened | Unopened) -> None:
        """Tell the page the answer to its press or wake, over the call's channel, when it has one open."""
        match call.channel:
            case RTCDataChannel(readyState="open") as channel:
                channel.send(wire(answer))
                call.told += 1
            case _:
                pass

    def _listening(self, level: PhoneLevel) -> None:
        """The page's level: the gate's listening at the phone follows it. Off, a turn a wake opened is sent first, as the
        desk sends its voice's turn as its level goes off; a turn the button opened is left to end as it ends."""
        match level:
            case "off":
                self._took(self._key.end(self._key.gate.wake_turn, "stop", "phone wake word"))
                self._took(self._key.move("deafen", "phone wake word"))
            case "wake word":
                self._took(self._key.move("listen", "phone wake word"))

    def _detected(self, said: DetectorLoaded | DetectorFailed) -> None:
        """The page's detector loaded, or failed, as one event naming the browser the page is in."""
        match said:
            case DetectorLoaded(browser=browser):
                ended("phone.detector", self._record, begun(self._clock), None, loaded=True, browser=browser)
            case DetectorFailed(browser=browser, why=why):
                ended("phone.detector", self._record, begun(self._clock), None, loaded=False, browser=browser, why=why)

    def _heard(self, call: _Call, audio: Pcm) -> None:
        frame = AudioFrame.from_ndarray(np.frombuffer(audio, dtype=np.int16)[None, :], format="s16", layout="mono")
        frame.sample_rate = call.rate
        for out in call.resampler.resample(frame):
            # One read of the gate per frame, as the desk's capture thread reads it.
            gate = self._key.gate
            if gate.hears("phone"):
                heard = out.to_ndarray().astype(np.int16).tobytes()
                keyed = gate.framed(heard, heard, None, self._heard_rate, 1, "phone")
                self.heard.put_nowait(keyed)
                # The driver hears a wake's turn from the wake on, and nothing the gate held other than that.
                match keyed.wake_turn:
                    case None:
                        pass
                    case turn:
                        self.wakes.put_nowait(WakeAudio(TurnId(turn), keyed.audio, self._clock.now()))

    def _leave(self, reason: PhoneGone) -> _Call | None:
        # [LAW:no-ambient-temporal-coupling] the place moves in the same step the call goes, before any await: a
        # speaker reading the gate after this never finds the phone without a call.
        call, self._call = self._call, None
        match call:
            case None:
                return None
            case _Call():
                with continuing(call.began.span):
                    self._key.go("desk")
                call.outbound.stop()
                if call.watch is not None and call.watch is not asyncio.current_task():
                    call.watch.cancel()
                call_ended(self._record, call.began, call.remote, call.asked, CallLeft(reason, call.arrived_ms, call.told, call.lines, call.held_back, call.rejected))
                return call

    async def hang_up(self, reason: PhoneGone) -> None:
        """End the call, if there is one, and be at the desk again."""
        match self._leave(reason):
            case None:
                pass
            case call:
                await call.peer.close()

    async def stop(self) -> None:
        """End the call and let go of any offer, as hands stops."""
        offered, self._offered = self._offered, None
        await self._let_go(offered, "stopped")
        await self.hang_up("stopped")

    def tell(self, told: Told | Line) -> None:
        """Tell the page a milestone of the turn it is waiting on, or a line of the transcript, over the call's channel, as the
        page sends its button. With no call up, or its channel on its way shut, there is no page to tell."""
        match self._call:
            case _Call(channel=RTCDataChannel(readyState="open") as channel) as call:
                match told:
                    case ClipOpened(turn=turn) | TurnUnopened(turn=turn):
                        # [LAW:single-enforcer] a press leaves `pressing` as it is answered, so it is answered once. A turn no
                        # press of the page's made, the desk's, is nothing of the page's.
                        match call.pressing.pop(turn, None):
                            case None:
                                pass
                            case press:
                                channel.send(wire(answer(press, told)))
                                call.told += 1
                    case ClipReleased() | ClipDiscarded() | str():
                        channel.send(wire(told))
                        call.told += 1
                    case line:
                        channel.send(wire(line))
                        call.lines += 1
            case _:
                pass

    def play(self, audio: bytes) -> asyncio.Future[None]:
        """Send 16-bit mono audio at the played rate to the phone; resolved once it has gone, or the call has ended."""
        match self._call:
            case None:
                raise RuntimeError("hands was told to play to the phone with no call up")
            case call:
                return call.outbound.give(audio)
