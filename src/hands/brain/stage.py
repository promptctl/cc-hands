"""The brain as the pipeline's LLM stage: the user's turn goes to the brain's stdin, and its words come off the wire.

What reaches this stage is the words of each turn the user aggregator ends (`hands.voice.conversation`), and what leaves
it are the frames an LLM service pushes, which TTS already consumes. What the brain says is read from its requests on the
wire as the proxy hears them, never from its stdout [the design's rule: primary facts from the wire]: text deltas of a
main turn under the brain's own session, and of no other request.

Claude Code sends every tool result back for another response, so a turn that called stay_silent has its next request
answered by hands with a line that records why: nothing more is said, and the model is not asked to go on. Not so for
the user's turn that acted and has said nothing since: it is asked on once, told to say what it did, and a turn that still
closes on nothing is said as failed, never left to silence the user takes for nothing done. A barge-in,
the user cutting in while something the brain said or asked is still to be heard, stops the turn being said, and nothing
of the brain's: its turn runs to its own end unheard, and what the user did not hear of it is told the brain with their
words, so the brain judges whether to go back to it. Words said while the brain works in silence stop nothing: they are
the next turn, and hands answers them in the turn's own voice.
"""

import asyncio
import json
import time
from functools import partial
from itertools import cycle
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol, cast

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory, classify_http_status_code

from hands.brain.mcp import SERVER_NAME, CallSpans
from hands.brain.process import NOBODY, SPOKEN_OVER, Asked, BrainAnswered
from hands.core.effects import Deny
from hands.core.beside import beside
from hands.core.place import Modality
from hands.core.playback import Bookmark
from hands.core.permissions import heard
from hands.core.front import InFront
from hands.core.session import Permission, SessionId
from hands.core.trace import Span
from hands.core.wire import (
    Answering,
    BlockStarted,
    BlockStopped,
    Exchanged,
    Heard,
    Hold,
    MainTurn,
    Observed,
    Route,
    Seconds,
    Send,
    Sent,
    Tail,
    TextDelta,
    ToolAnswer,
    Unreached,
    streams,
    tool_answers,
)
from hands.sessions.model_facts import ModelActedUntold, ModelFact, ModelFailed, ModelFault, ModelReplyEmpty, ModelUnreachable
from hands.sessions.audit import Record
from hands.sessions.wide import annotate, child, continuing, count, fail, here, root, unit, within
from hands.voice.conversation import UserWords
from hands.voice.player import Mark
from hands.voice.trigger import Edge
from hands.voice.turnstop import HoldDiscarded
from hands.voice.speech import Aloud, Narrated, brain_asks, brain_refused
from hands.voice.utterance import Resumed, Utterance, Uttered, Uttering, uttering
from hands.voice.tool import Result, Tool, silent, whole


class Asking(Protocol):
    """What the stage needs of the brain: the session its requests carry, and a turn asked, told of each permission it
    holds for the user."""

    @property
    def session(self) -> SessionId: ...

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainAnswered: ...


@dataclass(frozen=True)
class UserAsked:
    """The user's words: whether they could see a screen, the edge that opened their turn, and what was in front on the
    Mac's as they were submitted, and how long reading it took, in milliseconds."""

    modality: Modality
    opened: Edge
    front: InFront
    read_ms: float


@dataclass(frozen=True)
class HandsAsked:
    """What hands handed the brain to tell."""


# Whose turn the brain answered.
Asker = UserAsked | HandsAsked

# What the brain is recorded as saying for a request hands held. Never spoken: a held request joins no turn.
SILENT = "(stayed silent)"
# What the next turn tells the brain of a turn the user spoke over that then failed, beside what they did not hear of it:
# the spoken-over note says what was heard, so this says only the failure.
FAILED_AFTER = "[hands] Your last turn then failed. Mention this only if the user asks."
# What hands adds to the request after a stay_silent that would end a turn which acted with nothing heard after what it did.
OWED = "[hands] You acted on the user's words this turn, so they were said to you, and the user has heard nothing since you did. Say in a sentence what you did."

# How long the tool work of the user's turn runs, from its first call, with nothing of the turn heard yet, before hands
# acknowledges the turn: past a call that comes straight back with its answer said soon after, so only long work is.
ACKNOWLEDGE_SECONDS = 2.0
# What hands acknowledges with, one after another, so no two turns running acknowledge alike.
ACKNOWLEDGEMENTS = ("One moment.", "On it.", "Okay, working on it.", "Give me a second.")
# What hands answers words said behind a turn with, one after another: each is true whenever it is said, since the words
# are the next turn, taken once this one ends.
ANSWERS = ("Got it. I'll come to that when this is done.", "Heard you. That's next, once this finishes.", "Okay, that's next.", "Noted. I'll get to it after this.")


@dataclass(frozen=True)
class _Aside:
    """A line of hands' own said in a turn, and when the user's wait for it began: the turn taken, while its tools run,
    or words said behind it heard. Never the brain's, so never a point of it."""

    line: str
    since: Seconds


@dataclass(frozen=True)
class _Close:
    """The end of one of the turn's text blocks: every word of it is handed on."""


# What the brain said or asked: each is counted until the speaker has played it, and only these make a cut a barge-in.
_Point = _Close | Asked


def wire_name(tool: Tool) -> str:
    """The name a hands tool has on the wire, where the brain calls it through hands' MCP server."""
    return f"mcp__{SERVER_NAME}__{tool.name}"


def has_words(text: str) -> bool:
    """Whether text the brain streamed says anything: whitespace alone is heard as nothing."""
    # [LAW:one-source-of-truth] what counts as a word, for whether the turn replied, its first word, an acknowledgement,
    # and whether it closes on nothing.
    return bool(text.strip())


_UNNAMED = ModelFailed(ErrorCategory.UNKNOWN)

# What each turn's event counts: the model's round trips on the wire, and the calls its replies made.
COUNTS = ("round_trips", "tools", "refused", "cut_in", "silence_refused")


@dataclass
class _Call:
    """A call one of a turn's replies made: its tool, when its block was whole and Claude Code ran it, by the stage's clock
    and the wall's, its span in the turn's trace, and when the request carrying its result left, and whether that result
    was an error."""

    tool: str
    ran: Seconds
    at: datetime
    span: Span
    answered: Seconds | None = None
    is_error: bool = False


@dataclass(frozen=True)
class Broken:
    """A reply the API broke before any block of it finished, which Claude Code asked again whole (2.1.289): the request
    that asked it again, what the user heard of it, and whether the reply asked again opened with that, so the user heard
    it once and the brain's history holds it."""

    retry: str
    heard: str
    repeated: bool = False


@dataclass
class _Turn:
    """One question to the brain, from its write to stdin to its result line."""

    # Words heard on the wire, the ends of their text blocks, permissions the turn asks the user for, and hands' own lines
    # in it, not yet handed to the speaker; None once the brain has said the turn is over.
    said: asyncio.Queue[str | _Point | _Aside | None]
    # Hands the words on to TTS until the turn is over or the user barges in.
    speaking: asyncio.Task[None] = field(init=False)
    spoken: list[str]
    # Who asked the turn: only the user is waiting on it, so only their turn is acknowledged.
    asker: Asker
    # What the turn says of the sessions, where hands asked it to tell them; none for the user's own turn.
    utterances: tuple[Utterance, ...]
    # The turn's span: each of its requests is a span inside it, which the proxy's record of that request carries.
    span: Span
    # When the user's wait for the turn began: the end of their words where they spoke them, and when hands handed it
    # over where not.
    since: Seconds
    # The requests on the wire that are this turn's own, in the order they left.
    exchanges: list[str] = field(default_factory=list[str])
    # Each call the turn's replies made, by id, timed as heard.
    tools: dict[str, _Call] = field(default_factory=dict[str, _Call])
    # When the first word of the turn's reply said to the user was heard on the wire; None while none has been. A word the
    # floor holds while the user speaks counts from here: the time they spend speaking is theirs, not a wait for hands.
    first_word: Seconds | None = None
    # A reply of the turn's own carried words or a call: a turn none of whose replies did said nothing to the user.
    replied: bool = False
    # What a text block opens with: nothing until the turn has said a word, then a paragraph break, as Claude Code shows
    # them. Joined bare, a block that ended on a closing fence ran the next one's words onto that fence, which no longer
    # closed it, and every word to the end of the turn was held as code (hands-narration-2mc.1zu).
    between: str = ""
    # What goes ahead of the next words heard: the `between` of the text block they open.
    ahead: str = ""
    # What the reply streaming now has said, as handed to the speaker: what the user has heard of it if it breaks.
    reply: str = ""
    # Each of the turn's replies that broke and was asked again whole, oldest first.
    broken: list[Broken] = field(default_factory=list[Broken])
    # The call blocks the reply streaming now has opened and not yet closed, by index: a call is not run until it is whole.
    opening: dict[int, tuple[str, str]] = field(default_factory=dict[int, tuple[str, str]])
    # The calls the turn's last reply made whole, by id, which run until its next request leaves.
    calls: dict[str, str] = field(default_factory=dict[str, str])
    # What hands says for the turn's calls once its own words are said: what each handed hands to say.
    readbacks: list[str] = field(default_factory=list[str])
    # Where the user barged in on the turn, which is said to nobody from then on: the reading on the speaker and how much
    # of it had played, once the speaker has been cut off. None while the user has not barged in.
    cut: asyncio.Future[Bookmark] | None = None
    # How many times the user cut in on the turn and stopped nothing of it: nothing it said or asked was still to be
    # heard, or what they said answers the question they heard.
    cut_in: int = 0
    # A reply of the turn's is under way for whatever hears it: started, and not yet ended.
    replying: bool = False
    # The permissions the turn has put to the user, oldest first, which are asked one at a time.
    asked: list[Asked] = field(default_factory=list[Asked])
    # The permission whose question the user has heard to its end: what they say next answers it while it is open. Only a
    # question heard can be answered, so a yes said over the brain's words, or over a question cut off, allows nothing.
    heard: Asked | None = None
    # The permissions the user heard asked and that were refused with no word of the turn's said since, each once, which
    # hands says at its end: a refusal met with silence would pass for the work done.
    unacknowledged: list[Permission] = field(default_factory=list[Permission])
    # How many permissions the user heard asked were refused, whoever spoke to them after.
    refused: int = 0
    # How many times the turn closed on a stay_silent it owed a word in and was sent on, told to say what it did: at most
    # once, since a second ends it failed.
    silence_refused: int = 0
    # What the turn failed of, if it fails, as its latest request's answer told it: nothing named until that answer says.
    failure: ModelFact = _UNNAMED
    # Waits out the turn's tool work to acknowledge it, which the turn's end cancels.
    acknowledging: asyncio.Task[None] = field(init=False)
    # Set at the turn's first call: its tool work has begun, and the wait to acknowledge it with it.
    working: asyncio.Event = field(default_factory=asyncio.Event)
    # Each line hands said in the turn, and how long after its `since` it was handed to the speaker, in milliseconds.
    asides: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    # [LAW:one-source-of-truth] a text block of the turn's is streaming. The turn's, not the reply's: a reply broken in
    # the middle of one is asked again, and the text the retry goes on with is the same block. Blocks stream one at a
    # time, so at most one is: here where it last started, the request and its index in that request's reply, so only its
    # own stop ends it, never a thinking block's of a retry opening ahead of the text it goes on with. None while none is.
    voicing: tuple[str, int] | None = None
    # How many points the turn has put to the user, and how many of them the speaker has played: its cursor through
    # what the brain said and asked.
    marked: int = 0
    reached: int = 0
    # Hands' lines given while a text block streams, which follow it once it ends, never through the middle of it.
    behind: list[_Aside] = field(default_factory=list[_Aside])

    @property
    def interrupted(self) -> bool:
        return self.cut is not None

    @property
    def to_hear(self) -> bool:
        """Something the brain said or asked is still to be heard: a text block still streaming, or a point the speaker
        has not played. It plays them in order, so once the last has played, every one before it has."""
        return self.voicing is not None or self.reached < self.marked

    def point(self, point: _Point) -> None:
        """Puts what the brain said or asked to the user, counted until its Mark has played."""
        # [LAW:single-enforcer] the one place a point is counted: `_speak` pushes its one Mark.
        self.marked += 1
        self.said.put_nowait(point)

    def reach(self, point: _Point) -> None:
        """The speaker has played everything up to `point`: a question so played is heard to its end."""
        self.reached += 1
        match point:
            case Asked() as asked:
                self.heard = asked
            case _Close():
                pass

    def close(self) -> None:
        """Ends the text block streaming now, if one is: its words are all handed on, and the lines waiting behind it follow."""
        if self.voicing is not None:
            self.voicing = None
            self.point(_Close())
            self._let_through()

    def aside(self, aside: _Aside) -> None:
        """Gives the turn a line of hands' own, said once no text block streams."""
        self.behind.append(aside)
        self._let_through()

    def over(self) -> None:
        """Ends the turn on `said`. A text block left open, by a reply broken past its retry or a brain that failed, ends
        with it, unmarked, since nothing cuts in on a turn that is over, and the lines waiting behind it are said first."""
        self.voicing = None
        self._let_through()
        self.said.put_nowait(None)

    def _let_through(self) -> None:
        # [LAW:no-ambient-temporal-coupling] where hands' line lands is the turn's state, not when it was given.
        while self.behind and self.voicing is None:
            self.said.put_nowait(self.behind.pop(0))

    def empty(self) -> bool:
        """The model answered the turn, and nothing it answered said or did anything."""
        return bool(self.exchanges) and not self.replied

    def unheard(self) -> bool:
        """The user asked the turn and has heard nothing of it: no word, no question, and they have not spoken in it, over
        it or not. Their words in it are answered instead."""
        return isinstance(self.asker, UserAsked) and self.first_word is None and not self.asked and not self.interrupted and not self.cut_in

    def owes_word(self, tools: Mapping[str, Tool]) -> bool:
        """The user is owed a word of what the turn did: they asked it and did not speak over it, it acted, and its latest
        reply closes it on nothing, no words and no call but stay_silent. That reply's words would follow every act before
        it, since a reply's words come ahead of its calls' runs."""
        # A call to no tool of hands' own, Bash or Write, is an act; a whole reply hands says something for is not, and
        # stay_silent says it did nothing. [LAW:one-source-of-truth] each read off the tool's own `then`.
        acted = any((tool := tools.get(call.tool)) is None or tool.then == "reply" for call in self.tools.values())
        closes_on_nothing = not has_words(self.reply) and all((tool := tools.get(name)) is not None and tool.then == "nothing" for name in self.calls.values())
        return isinstance(self.asker, UserAsked) and not self.interrupted and acted and closes_on_nothing

    def asking(self) -> Asked | None:
        """The permission the user's next words answer: the one they heard asked, while it is still open."""
        return self.heard if self.heard is not None and self.heard.open else None


class BrainStage(FrameProcessor):
    """The LLM stage under the brain: the user's words in, the brain's words out as LLM text frames, and a barge-in passed on."""

    def __init__(
        self,
        brain: Asking,
        tools: Sequence[Tool],
        tail: Callable[[], str],
        refocus: Callable[[SessionId], Awaitable[None]],
        front: Callable[[], Awaitable[InFront]],
        modality: Callable[[], Modality],
        opened: Callable[[], Edge],
        record: Record,
        spans: CallSpans,
        cut_off: Callable[[], asyncio.Future[Bookmark]],
        clock: Callable[[], Seconds] = time.monotonic,
        pause: Callable[[Seconds], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._brain = brain
        # What is in front on the Mac's screen, read as the user's words arrive.
        self._front = front
        # Whether the user can see a screen, read as their words arrive.
        self._modality = modality
        # The edge that opened the gate's last turn.
        self._opened = opened
        # Moves the focus to a session whose telling the brain takes.
        self._refocus = refocus
        # What hands appends to each request of a turn, composed as that request leaves.
        self._tail = tail
        self._record = record
        # Each call's span, where hands' MCP server finds it as the call's run reaches it.
        self._spans = spans
        # [LAW:one-source-of-truth] what the speaker was cut off in by the barge-in it is next to reach: the player's to
        # say, which alone sees what played.
        self._cut_off = cut_off
        self._now = clock
        # Waits out a call's time before the turn is acknowledged.
        self._pause = pause
        self._acknowledgements = cycle(ACKNOWLEDGEMENTS)
        self._answers = cycle(ANSWERS)
        self._tools = {wire_name(tool): tool for tool in tools}
        # [LAW:no-ambient-temporal-coupling] the turn is the brain's, from its write to its result line, not the pipeline's:
        # it runs beside the frames passing through, a barge-in ends what is said of it but not it, and the next is written only
        # once the brain has ended it. What waits is in two lanes, the user's and hands', and the user's goes first.
        # Each thing waiting is kept with when it arrived, and the user's words with when they let go of the key on them,
        # if they did, and the read of the screen begun as they arrived. Hands' lane holds what it says as written beside
        # what it hands the brain, so a session's story is heard in the order it happened.
        self._words: deque[tuple[str, Seconds, Seconds | None, asyncio.Task[UserAsked]]] = deque()
        self._hands: deque[tuple[Narrated | Aloud, Seconds]] = deque()
        # When the user last let go of the key on words not yet handed to the brain: what their wait is timed from. One, not
        # one per hold: a hold let go of while an earlier one is still transcribed joins that hold's turn (KeyTurnStop), and
        # the turn's words follow the last release of the holds it took in. Kept with the edge that opened that
        # hold, read from the gate as it is let go of, before any later hold can open.
        self._released: tuple[Seconds, Edge] | None = None
        # [LAW:one-source-of-truth] the question the user's next words answer, read once at the cut they open with: one
        # asked while they are still being transcribed is not what they were answering. Every turn that carries words is
        # cut first (`hands.voice.turnstart`).
        self._answering: Asked | None = None
        # [LAW:no-ambient-temporal-coupling] set while hands has the floor: taken by every cut, as it says the user started
        # speaking, and given back as the user's turn stops, which Pipecat says of every turn, words or none: the pair
        # Pipecat's turn brackets, so no interruption of any other kind holds it. Nothing of a turn is handed on while the user
        # speaks: what it says meanwhile waits for them, as what hands tells of the sessions does (`hands.voice.floor`).
        self._floor = asyncio.Event()
        self._floor.set()
        self._waiting = asyncio.Event()
        self._turn: _Turn | None = None
        # What the user did not hear of the last turn, told to the brain with its next turn: what the API broke off, which
        # Claude Code keeps out of the brain's history (2.1.285), or what the user barged in on, which only hands knows
        # was never said.
        self._broken_off = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case UserWords(words=words) if (asked := self._answering) is not None and asked.open:
                # What the user says while the brain asks them is their answer, not a turn of their own: the brain is held
                # mid-turn, and hears it as its tool's run or refusal.
                self._released, self._answering = None, None
                asked.settle(heard(words))
            case UserWords(words=words):
                # [LAW:no-ambient-temporal-coupling] the screen is read from as the words arrive, not as their turn is
                # taken: the user may look elsewhere, or switch to audio-only, while it waits.
                released, self._released, self._answering = self._released, None, None
                arrived = self._now()
                match released:
                    case None:
                        let_go, opened = None, self._opened()
                    case (let_go, opened):
                        pass
                self._words.append((words, arrived, let_go, asyncio.ensure_future(self._read_front(self._modality(), opened))))
                self._waiting.set()
                match self._turn:
                    case _Turn(asker=UserAsked()) as turn if not turn.interrupted:
                        # Said behind the user's turn, and heard: they hear so in its own voice.
                        turn.aside(_Aside(next(self._answers), _since(arrived, let_go)))
                    case _:
                        # No turn of the user's to answer in, or one spoken over, whose voice is stopped: the words are
                        # simply the next turn. A telling is not answered in, as it is not acknowledged: what it says is
                        # the session's, and its words are all heard of it.
                        pass
            case HoldDiscarded():
                await self.push_frame(frame, direction)
            case UserStartedSpeakingFrame():
                # Broadcast by the cut alone, just ahead of its interruption (`hands.voice.turnstart.interrupting`).
                self._floor.clear()
                await self.push_frame(frame, direction)
            case UserStoppedSpeakingFrame():
                self._floor.set()
                await self.push_frame(frame, direction)
            case VADUserStoppedSpeakingFrame():
                # The key let go on words to be transcribed: the end of the user's words, and the start of their wait.
                self._released = (self._now(), self._opened())
                await self.push_frame(frame, direction)
            case Narrated() | Aloud():
                self._hands.append((frame, self._now()))
                self._waiting.set()
            case InterruptionFrame():
                # [LAW:no-ambient-temporal-coupling] the turn stops being spoken before the pipeline is told, so no word of
                # it follows the barge-in down. The brain is told nothing: its turn runs to its own end.
                turn = self._turn
                self._answering = None if turn is None else turn.asking()
                self._barge_in(self._answering)
                await self.push_frame(frame, direction)
                if turn is not None and not turn.interrupted:
                    if turn.replying:
                        # A barge-in ends the reply under way for everything behind this stage, and the turn goes on
                        # through it: what it says next is a reply started again, written whole at its end.
                        await self.push_frame(LLMFullResponseStartFrame())
                    # It cut off none of what the turn is still to say of the sessions.
                    await self.push_frame(Resumed(turn.utterances))
            case LLMContextFrame():
                # [LAW:no-silent-failure] a call to run inference over a message list hands keeps none of: some Pipecat path
                # ended a turn without `UserTurns`, whose words would otherwise pass on unasked.
                raise TypeError(f"{frame} reached the brain's stage: a turn ended somewhere other than UserTurns, and its words were lost")
            case _:
                await self.push_frame(frame, direction)

    async def ask_each(self) -> None:
        """Asks the brain each turn the pipeline hands this stage, one at a time, for as long as it runs; returns only by
        raising what stopped it, since a stage that can no longer ask leaves every question unanswered."""
        while True:
            waiting, arrived, released = await self._upcoming()
            # Taken from its lane: how long it waited there, which reading the screen is not.
            taken = self._now()
            match waiting:
                case (str() as text, reading):
                    asker = await reading
                    await self._ask(f"{text}\n\n{beside(asker.front, asker.modality)}", asker, (), (), arrived, released, taken)
                case Narrated(text=text, unsaid=unsaid, session=session, utterances=utterances):
                    # [LAW:no-ambient-temporal-coupling] moved as the telling is taken, with the user's last turn ended and
                    # none waiting, since they go first, and before the brain is asked, so its request reads the new focus.
                    await self._refocus(session)
                    # The turn is what says them, sent with what tells what of them was heard, and a part of the first
                    # of them, in its trace.
                    await self.push_frame(Uttering(utterances))
                    with continuing(utterances[0].begun.span if utterances else None):
                        failure = await self._ask(text, HandsAsked(), (unsaid,), utterances, arrived, None, taken)
                    match failure:
                        case str():
                            for utterance in utterances:
                                # [LAW:nothing-unseen] what was heard of a telling the brain failed is that it could not be told.
                                utterance.fail(failure)
                        case None:
                            pass
                    await self.push_frame(Uttered(utterances))
                case Aloud(spoken=spoken, utterances=utterances):
                    # In hands' lane no turn of the brain's is under way.
                    for frame in uttering(utterances, (spoken,)):
                        await self.push_frame(frame)

    async def _read_front(self, modality: Modality, opened: Edge) -> UserAsked:
        began = self._now()
        front = await self._front()
        return UserAsked(modality, opened, front, (self._now() - began) * 1000)

    async def _upcoming(self) -> tuple[tuple[str, asyncio.Task[UserAsked]] | Narrated | Aloud, Seconds, Seconds | None]:
        """What is next: the user's words while any wait, since what they said goes ahead of what hands has to tell, all
        that waits of them asked as one turn, with when the last of them arrived and when the user let go of the key on it,
        since the user waits from the last words they said."""
        while not (self._words or self._hands):
            self._waiting.clear()
            await self._waiting.wait()
        if not self._words:
            told, arrived = self._hands.popleft()
            return told, arrived, None
        *earlier, (_, arrived, released, reading) = self._words
        # Asked as one turn, read against the screen as the last of them arrived.
        for _, _, _, superseded in earlier:
            superseded.cancel()
        news = "\n\n".join(text for text, _, _, _ in self._words)
        self._words.clear()
        return (news, reading), arrived, released

    async def _ask(
        self, text: str, asker: Asker, unsaid: Sequence[str], utterances: tuple[Utterance, ...], arrived: Seconds, released: Seconds | None, taken: Seconds
    ) -> str | None:
        """One turn of the brain's, and its one event; `unsaid` is what hands says as written if the brain cannot take it,
        and `utterances` what the turn says of the sessions. `released` is when the user let go of the key on the words
        the turn asks, None for a turn not asked aloud, and `taken` when the turn left its lane. Returns why the turn
        failed, None where it did not."""
        # [LAW:nothing-unseen] every turn passes through here, whoever asked it and however it ends.
        with unit("voice.turn", self._record, COUNTS):
            return await self._turn_of(text, asker, unsaid, utterances, arrived, released, taken)

    async def _turn_of(
        self, text: str, asker: Asker, unsaid: Sequence[str], utterances: tuple[Utterance, ...], arrived: Seconds, released: Seconds | None, taken: Seconds
    ) -> str | None:
        note, self._broken_off = self._broken_off, ""
        # After the words, so a turn of the user's never opens with [hands], which marks a message as hands' own.
        text = _notes(text, note)
        # What is typed into the brain for the turn, as it is typed.
        annotate(asked=text)
        said: asyncio.Queue[str | _Point | _Aside | None] = asyncio.Queue()
        spoken: list[str] = []
        turn = self._turn = _Turn(said, spoken, asker, utterances, here(), _since(arrived, released))
        turn.speaking = asyncio.create_task(self._speak(turn), name="the brain's words")
        turn.acknowledging = asyncio.create_task(self._acknowledge(turn), name="the turn's acknowledgement")
        # When the brain ended the turn, or it was stopped: set before anything reads it, however the turn ends.
        ended = taken
        # What the next turn tells the brain of a barge-in on this one; nothing when there was none.
        over = ""
        try:
            try:
                asked = asyncio.ensure_future(self._brain.ask(text, lambda permission: self._put(turn, permission)))
                await asyncio.wait({asked})
            finally:
                self._turn = None
                ended = self._now()
                # [LAW:no-ambient-temporal-coupling] nothing acknowledges a turn once it is over.
                turn.acknowledging.cancel()
                turn.over()
            await asyncio.wait({turn.speaking})
            # [LAW:single-enforcer] a refusal the user heard asked is never left to silence, whatever the model did after it.
            refusals = tuple(brain_refused(permission) for permission in turn.unacknowledged)
            annotate(refusals=refusals)
            await self._unsaid(refusals)
            if turn.cut is None:
                for readback in turn.readbacks:
                    # Said by hands, as each call handed it to say.
                    await self._say(TTSSpeakFrame(readback))
            else:
                # Said to nobody: the turn's words past what the speaker played, and the readbacks. The brain judges whether
                # to go back to any of it.
                over = _spoken_over(await turn.cut, "".join((*spoken, *_left(said))), turn.readbacks)
            annotate(spoken_over=over)
        finally:
            # [LAW:nothing-unseen] what the turn did is on its event however it ended, a turn cancelled mid-way included.
            self._account(turn, asker, arrived, released, taken, ended)
        # A brain that failed the turn itself has no answer to read.
        failure = None if asked.exception() is not None else _failed(turn, asked.result().error, self._tools)
        annotate(failed=None if failure is None else failure.fact)
        if (error := asked.exception()) is not None:
            failed = f"the brain failed a turn: {error}"
            # [LAW:no-silent-failure] said as the turn's failure whatever failed it: a brain that is gone also stops the run
            # from its own watch, but one that never took the turn, or could not be typed into, is still running.
            fail(failed)
            # A turn the brain never took did not tell it what the user heard: the turn after it does. One it spoke in took
            # that, and was cut where the user stopped hearing it.
            # [LAW:one-source-of-truth] what the user heard of a turn they spoke over is the speaker's to say, not the words it was handed.
            self._broken_off = _notes(over, FAILED_AFTER) if over else (_broken_off("".join(spoken)) if spoken else note)
            await self._unsaid(unsaid)
            await self.push_error(failed)  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            return failed
        elif failure is not None:
            # [LAW:no-silent-failure] said as an error from the model's stage, which the system channel words. No category:
            # Pipecat takes an invalid request or a refused login as permanent and stops the stage, and the brain goes on.
            self._broken_off = _notes(over, FAILED_AFTER) if over else failure.note
            fail(failure.error)
            await self._unsaid(unsaid)
            await self.push_error(failure.error, exception=ModelFault(failure.fact))  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            return failure.error
        # The brain's history holds the replies asked again, never the broken ones: what the user heard of one is told it
        # with its next turn, unless the reply asked again opened with it.
        self._broken_off = _notes(*(_asked_again(broken.heard) for broken in turn.broken if broken.heard and not broken.repeated), over)
        return None

    def _account(self, turn: _Turn, asker: Asker, arrived: Seconds, released: Seconds | None, taken: Seconds, ended: Seconds) -> None:
        """The turn's event: how long the user waited for its first word and where the time went, what it said, and each
        call its replies made, as a part of it. Its round trips to the model are the proxy's records of its requests."""
        annotate(
            asker=asker,
            exchanges=tuple(turn.exchanges),
            text="".join(turn.spoken),
            readbacks=tuple(turn.readbacks),
            broken=tuple(turn.broken),
            interrupted=turn.interrupted,
            # The wait, then where it went: transcribing what was said, waiting behind the turn before it, and the rest,
            # from the turn leaving its lane to its first word: what was left of reading the screen (the asker's read_ms
            # is the whole read, begun as the words arrived), then the model's and its tools', as its parts show.
            waited_ms=None if turn.first_word is None else _ms(turn.first_word - turn.since),
            # Each line hands said in the turn, its acknowledgement and its answers to words said behind it, and how long
            # the user had waited for it.
            asides=tuple(turn.asides),
            transcribed_ms=None if released is None else _ms(arrived - released),
            queued_ms=_ms(taken - arrived),
        )
        count(round_trips=len(turn.exchanges), tools=len(turn.tools), refused=turn.refused, cut_in=turn.cut_in, silence_refused=turn.silence_refused)
        self._spans.ended(turn.tools)
        for call, ran in turn.tools.items():
            # A call whose result never left ran until the turn ended without it.
            outcome = "cancelled" if ran.answered is None else "failed" if ran.is_error else "ok"
            child("tool.call", ran.span, ran.at, _ms((ended if ran.answered is None else ran.answered) - ran.ran), outcome, call=call, tool=ran.tool)

    async def _unsaid(self, unsaid: Sequence[str]) -> None:
        """What hands had for the brain to tell, said as written since the brain did not: a system fact, kept out of the replies the audit log records."""
        for line in unsaid:
            await self._say(TTSSpeakFrame(line, append_to_context=False))

    async def _say(self, line: TTSSpeakFrame) -> None:
        """Says a line of hands' own after a turn, once hands has the floor: never over the user. A cut after it is
        handed on flushes it with everything else on its way to the speaker."""
        await self._floor.wait()
        await self.push_frame(line)

    def _put(self, turn: _Turn, asked: Asked) -> None:
        """Puts a permission the brain holds to the user, after what the turn has said so far."""
        if turn is not self._turn:
            # The stage stopped asking this turn, so nothing would put it to the user or hear their answer.
            asked.settle(Deny(NOBODY))
            return
        if turn.interrupted:
            # Nothing more of the turn is said, so they would never hear it asked.
            asked.settle(Deny(SPOKEN_OVER))
            return
        turn.asked.append(asked)
        turn.point(asked)

    async def _reply(self, turn: _Turn, under_way: bool) -> None:
        """Starts or ends the turn's reply for whatever hears it."""
        # [LAW:no-ambient-temporal-coupling] noted ahead of the frame that says so, with nothing awaited between: a
        # barge-in the turn goes on through starts again only a reply that has not ended.
        turn.replying = under_way
        await self.push_frame(LLMFullResponseStartFrame() if under_way else LLMFullResponseEndFrame())

    async def _speak(self, turn: _Turn) -> None:
        await self._reply(turn, True)
        while (words := await self._handed(turn)) is not None:
            match words:
                case str():
                    turn.spoken.append(words)
                    # The brain's own words after a refusal are its acknowledgement of it.
                    if has_words(words):
                        turn.unacknowledged.clear()
                    await self.push_frame(LLMTextFrame(words))
                case _Aside(line=line, since=since):
                    turn.asides.append((line, _ms(self._now() - since)))
                    # Kept out of the replies the audit log records: the brain never said it.
                    await self._aside(turn, TTSSpeakFrame(line, append_to_context=False))
                    await self._reply(turn, True)
                case _Close() | Asked() as point:
                    await self._point(turn, point)
        # Not on a barge-in, which cancels this: an end would have TTS say the sentence the user spoke over.
        await self._reply(turn, False)

    async def _handed(self, turn: _Turn) -> str | _Point | _Aside | None:
        """What the turn hands on next, once hands has the floor: nothing of it is said while the user speaks."""
        words = await turn.said.get()
        await self._floor.wait()
        return words

    async def _point(self, turn: _Turn, point: _Point) -> None:
        """Says a point of the brain's: the end of a text block, or the question it asks, while that is still open, then
        the point's Mark, behind which a question waits for its answer."""
        # Read once. One settled before its turn came to be said, by its hook's day ending, the turn's end, or an answer,
        # asks nothing, and its Mark plays behind what is ahead of it.
        question = point if isinstance(point, Asked) and point.open else None
        # The reply so far ends first, so what TTS holds of it is said ahead of the Mark: a text block's last sentence as
        # the block closes, never held back for words that come after a call.
        match question:
            case Asked(permission=permission):
                await self._aside(turn, TTSSpeakFrame(brain_asks(permission)))
            case None:
                await self._reply(turn, False)
        # [LAW:single-enforcer] each point's one Mark. [LAW:no-ambient-temporal-coupling] a question is answerable once
        # the speaker has played it to its end, which the mark is told of, and never if a barge-in cut it off.
        await self.push_frame(Mark(partial(turn.reach, point)))
        match question:
            case Asked(permission=permission) as asked:
                # One question at a time, so what the user answers is the question they heard last.
                await asyncio.wait({asked.decision})
                # Owed only for a question the user heard to its end: one cut off or never reached they cannot take
                # for the work done. Said once, however often the brain asked it again in silence.
                if turn.heard is asked and isinstance(asked.decision.result(), Deny):
                    turn.refused += 1
                    if permission not in turn.unacknowledged:
                        turn.unacknowledged.append(permission)
            case None:
                pass
        await self._reply(turn, True)

    async def _aside(self, turn: _Turn, line: TTSSpeakFrame) -> None:
        """Says a line of hands' own in the turn, as a sentence of its own once the brain's words before it are: the response
        so far ends first, so what TTS holds of it is said ahead of the line."""
        await self._reply(turn, False)
        await self.push_frame(line)

    def _barge_in(self, answering: Asked | None) -> None:
        """Stops the turn in flight being spoken, if the user cut in while something the brain said or asked was still to
        be heard, and nothing of the brain's: its turn runs on to its own end, said to nobody."""
        turn = self._turn
        # A turn with nothing of the brain's still to be heard is not spoken over: the user's words follow it as the next
        # turn, as they do one none of whose requests has left. One that asks the user a permission they heard is waiting
        # on what they say: their words answer it, and stop nothing.
        if turn is None or turn.interrupted:
            return
        if not turn.to_hear or answering is not None:
            turn.cut_in += 1
            return
        # Asked of the speaker before the barge-in goes down to it: it says how far it played once it gets there.
        turn.cut = self._cut_off()
        # Cancelled before anything else runs, so no word of the turn follows the barge-in down the pipeline.
        turn.speaking.cancel()
        # None of them was heard to its end, or the user's words would have answered it: now none can be.
        for asked in turn.asked:
            asked.settle(Deny(SPOKEN_OVER))
        # The user is speaking: what they say is the next turn, which the brain answers knowing what was refused.
        turn.unacknowledged.clear()

    def route(self, sent: Sent) -> Route:
        """Where a request on the wire goes: each of a turn's own requests with hands' tail on it, and the next one after
        stay_silent held."""
        # Another session's request is made for no unit of the stage's: the root of a trace of its own.
        if sent.session != self._brain.session:
            return Send(span=root())
        # [LAW:single-enforcer] every request of the brain's is final, whatever its kind and whether a turn asked it: asked
        # again, Claude Code would keep the user waiting minutes on its retries, and a spent limit it would wait out to
        # continue the task on its own at the reset, hours on, with nobody asking (hands-wire-zi2).
        turn = self._turn
        # A request the brain makes while a turn is asked of it, a subagent's or a fork's among them, is that turn's part.
        if not isinstance(sent.kind, MainTurn):
            return Send(refusal="final", span=root() if turn is None else within(turn.span))
        if turn is None:
            # [LAW:no-silent-failure] the brain asked the model something with no turn written to it: heard, never spoken.
            logger.warning(f"the brain sent a main turn (exchange {sent.exchange}) with no turn asked of it; nothing it says will be spoken")
            return Send(refusal="final", span=root())
        now = self._now()
        carried = tool_answers(sent.body)
        # A call's result leaving is the end of its run, whether the request carrying it goes on or is held.
        for answer in carried:
            if (ran := turn.tools.get(answer.call)) is not None and ran.answered is None:
                ran.answered, ran.is_error = now, answer.is_error
        # Only the calls this turn's last reply opened: a request carries every result of the brain's history.
        answers = [(self._tools.get(turn.calls[answer.call]), _result(answer)) for answer in carried if answer.call in turn.calls]
        # Said by hands once the brain's own words are, whatever the model does next: what a call hands hands to say.
        turn.readbacks.extend(says for _, result in answers if result is not None and isinstance(says := result.get("says"), str))
        held = whole([tool is not None and result is not None and silent(tool, result) for tool, result in answers])
        # [LAW:single-enforcer] a turn that acted is not let close on stay_silent with nothing heard after what it did: the
        # model is asked on, told to say what it did. Once: a second time, it is held, and the turn's end fails it.
        refused = held and turn.owes_word(self._tools) and not turn.silence_refused
        if not held or refused:
            turn.silence_refused += refused
            turn.exchanges.append(sent.exchange)
            # [LAW:no-ambient-temporal-coupling] the reply it asks again for is all heard by now: the proxy tells each byte
            # before Claude Code reads it, and Claude Code asks again only once it has read the break.
            if not streams(sent.body):
                # Heard by nobody once the user spoke over the turn: nothing of it is said from there on.
                turn.broken.append(Broken(sent.exchange, "" if turn.interrupted else turn.reply))
            turn.opening, turn.calls, turn.failure, turn.reply = {}, {}, _UNNAMED, ""
            # Refused once is the turn's failure, said at once rather than after Claude Code's retries. The proxy's
            # record of the request is the turn's round trip to the model: a span inside the turn's.
            return Send((*[Tail(OWED)] * refused, Tail(self._tail())), refusal="final", span=within(turn.span))
        return Hold(SILENT, within(turn.span))

    async def _acknowledge(self, turn: _Turn) -> None:
        """Acknowledges the turn once its tool work has run its time, however many calls it took, if the user has heard
        nothing of it: they hear it was taken, rather than silence until its answer."""
        await turn.working.wait()
        await self._pause(ACKNOWLEDGE_SECONDS)
        if turn.unheard():
            turn.aside(_Aside(next(self._acknowledgements), turn.since))

    def hear(self, observed: Observed) -> None:
        turn = self._turn
        if turn is None:
            return
        now = self._now()
        match observed:
            # Said as it arrives. A reply the API breaks once a block of it is finished is not asked for again; the turn
            # ends in StopFailure, with the broken reply kept out of the brain's history (2.1.285, hands-wire-6ic.6dz).
            # One broken before then is asked again without a stream (2.1.289): the reply asked again goes on from where
            # the user stopped hearing the broken one when it opens as that did, and is said whole when it does not.
            case Heard(exchange=exchange, event=BlockStarted(index=index, block={"type": "text"})) if exchange in turn.exchanges:
                turn.ahead = turn.between
                # Opened, or gone on with, where a retry carries on with the block a broken reply left streaming.
                turn.voicing = (exchange, index)
            case Heard(exchange=exchange, event=TextDelta(text=text)) if exchange in turn.exchanges:
                again = next((at for at, broken in enumerate(turn.broken) if not turn.reply and broken.retry == exchange and broken.heard and text.startswith(broken.heard)), None)
                match again:
                    case int():
                        turn.broken[again] = replace(turn.broken[again], repeated=True)
                        turn.said.put_nowait(text.removeprefix(turn.broken[again].heard))
                    case None:
                        turn.said.put_nowait(turn.ahead + text)
                turn.reply += text
                turn.ahead, turn.between = "", "\n\n"
                turn.replied = turn.replied or has_words(text)
                # The first word the user waited for: none heard after they spoke over the turn is said to them.
                if has_words(text) and turn.first_word is None and not turn.interrupted:
                    turn.first_word = now
            case Heard(exchange=exchange, event=BlockStarted(index=index, block={"type": "tool_use", "id": str() as call, "name": str() as name})) if exchange in turn.exchanges:
                # A block that is not text ends the one streaming, which only a retry opening with a call leaves open.
                turn.close()
                turn.opening[index] = (call, name)
                turn.replied = True
            case Heard(exchange=exchange, event=BlockStarted(block={"type": "thinking" | "redacted_thinking"})) if exchange in turn.exchanges:
                # Thinking opens a reply, ahead of what it says: a retry's text after it goes on with the block left open.
                pass
            case Heard(exchange=exchange, event=BlockStarted()) if exchange in turn.exchanges:
                turn.close()
            # Claude Code runs a call once its block is whole, before the reply's last byte, and hands hears each byte
            # before Claude Code does: the call is running from here on.
            case Heard(exchange=exchange, event=BlockStopped(index=index)) if exchange in turn.exchanges and index in turn.opening:
                call, name = turn.opening.pop(index)
                turn.calls[call] = name
                ran = turn.tools[call] = _Call(name, now, datetime.now(UTC), within(turn.span))
                self._spans.opened(call, ran.span)
                turn.working.set()
            case Heard(exchange=exchange, event=BlockStopped(index=index)) if turn.voicing == (exchange, index):
                turn.close()
            # [LAW:one-source-of-truth] why a turn failed is the wire's: the head of its latest answer, heard before Claude Code reads any of it, or an API the proxy could not reach,
            # told before its 502. Its own requests are held final, so Claude Code asks once; a 401 it asks again after
            # refreshing its login: the latest request's is the turn's.
            case Answering(exchange=exchange, status=status, limit=limit) if exchange in turn.exchanges:
                turn.failure = limit or ModelFailed(classify_http_status_code(status))
            case Exchanged(exchange=exchange, reply=Unreached()) if exchange in turn.exchanges:
                turn.failure = ModelUnreachable()
            case _:
                pass


def _since(arrived: Seconds, released: Seconds | None) -> Seconds:
    """When the user's wait began: the end of their words where they spoke them, and when hands handed them over where not."""
    return arrived if released is None else released


def _ms(seconds: Seconds) -> float:
    return round(seconds * 1000, 3)


@dataclass(frozen=True)
class _Failure:
    """What a turn the brain finished failed of: the stage's error, the fact it is, and the note the brain's next turn carries."""

    error: str
    fact: ModelFact
    note: str


def _failed(turn: _Turn, error: str | None, tools: Mapping[str, Tool]) -> _Failure | None:
    """What a turn the brain finished failed of; None for one that did not."""
    if error is not None:
        return _Failure(f"the brain's turn ended in error: {error}", turn.failure, _broken_off("".join(turn.spoken)))
    if turn.empty():
        # An empty reply is a whole one, kept in the brain's history: there is nothing broken off to tell it of.
        return _Failure("the brain's reply was empty", ModelReplyEmpty(), "")
    if turn.owes_word(tools):
        # [LAW:no-silent-failure] a turn that acted and closed on nothing, by an empty reply or a stay_silent it was already
        # refused once, ends heard as failed rather than as silence the user takes for nothing done. Whole in the brain's
        # history, as an empty reply is.
        return _Failure("the brain acted and said nothing of it", ModelActedUntold(), "")
    return None


def _broken_off(spoken: str) -> str:
    """The note that tells the brain what the user heard of a turn broken off before its end; none when nothing of it was said."""
    return f'[hands] Your last turn was broken off. The user heard you say "{spoken}", then that it failed. Mention this only if the user asks.' if spoken else ""


def _spoken_over(cut: Bookmark, words: str, readbacks: Sequence[str]) -> str:
    """The note that tells the brain the user spoke over its last turn: the last sentence they heard, if the reading cut
    off holds one, and what they never heard, which is the rest of that reading, the turn's words the speaker was never
    handed, and the readbacks of its calls."""
    # [LAW:one-source-of-truth] the turn's words are all it said; the speaker only marks how far into them was heard. Each
    # sentence of them it was handed is a piece of them, in order, so one still on its way to it is unheard with the rest.
    rest_of_reading: list[str] = []
    into = 0
    for at, sentence in enumerate(cut.sentences):
        if (found := words.find(sentence.strip(), into)) >= 0:
            into = found + len(sentence.strip())
        if at >= cut.at:
            rest_of_reading.append(sentence)
    unheard = " ".join(part for part in (*rest_of_reading, " ".join(words[into:].split()), *readbacks) if part)
    last = f' They last heard "{cut.sentences[cut.at - 1]}".' if cut.at else ""
    rest = f' They did not hear: "{unheard}".' if unheard else ""
    return f"[hands] The user spoke over you, and nothing more was said aloud.{last}{rest}"


def _left(said: asyncio.Queue[str | _Point | _Aside | None]) -> list[str]:
    """The turn's words still waiting for a speaker that stopped taking them."""
    left: list[str] = []
    while not said.empty():
        match said.get_nowait():
            case str() as words:
                left.append(words)
            case _:
                pass
    return left


def _notes(*notes: str) -> str:
    """What hands tells the brain with a turn, ahead of it: each note there is, a paragraph apart."""
    return "\n\n".join(note for note in notes if note)


def _asked_again(heard: str) -> str:
    """The note that tells the brain what the user heard of a reply that broke and was asked again, which its history holds
    only as the reply asked again: true however the turn went on from there."""
    return f'[hands] A reply of yours last turn broke off and was asked for again; your history holds only the reply asked again. Before it broke, the user heard you say "{heard}". Mention this only if the user asks.'


def _result(answer: ToolAnswer) -> Result | None:
    """What the tool handed back, or None for the MCP server's own failure: a line of text naming the tool and what went wrong."""
    try:
        result: object = json.loads(answer.text)
    except ValueError:
        return None
    return cast(Result, result) if isinstance(result, dict) else None

