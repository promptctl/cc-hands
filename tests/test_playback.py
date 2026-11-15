"""Playback: where the speaker is, where readings were cut off, and what going back, skipping, and repeating say."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import reduce

import pytest
from pipecat.frames.frames import AggregatedTextFrame, Frame, InterruptionFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import AggregationType

from conftest import Running, new_sessions, running
from hands.clock import SYSTEM
from hands.core.playback import (
    BOOKMARKS,
    Bookmark,
    LastOne,
    NothingBargedIn,
    NothingSaid,
    Playback,
    Played,
    Replay,
    barge_in,
    finished,
    handed,
    queued,
    repeat,
    resume,
    skip,
)
from hands.sessions.audit import BargeIn, Entry
from hands.sessions.wide import WideEvent
from hands.core.pending import News
from hands.core.session import PromptId, SessionId
from hands.virtualclock import VirtualClock
from hands.voice.narrator import Recounts
from hands.voice.player import Player, said
from hands.voice.spoken import sentences
from hands.voice.tools import audited, playback_tools, read_reply_aloud_tool

# What happens at the speaker, as the taps report it: a sentence handed to it, one played to its end, the user cutting in.
Happening = str


def played(*happenings: Happening) -> Playback:
    """Playback after the happenings in order: a sentence's text is that sentence handed, "done" the next one finished,
    and "cut" a barge-in."""

    def step(playback: Playback, happening: Happening) -> Playback:
        match happening:
            case "done":
                return finished(playback)
            case "cut":
                return barge_in(playback)
            case sentence:
                return handed(playback, sentence)

    return reduce(step, happenings, Playback())


@pytest.mark.parametrize(
    ("happenings", "act", "replayed", "left"),
    [
        # Cut in on the second sentence: going back says it again from its start, and what followed it.
        (("A.", "B.", "C.", "done", "cut"), resume, Replay(("B.", "C.")), ()),
        # Cut in on the first: it is said again whole.
        (("A.", "B.", "cut"), resume, Replay(("A.", "B.")), ()),
        # Skipping goes on from the sentence after the one cut off.
        (("A.", "B.", "C.", "done", "cut"), skip, Replay(("C.",)), ()),
        (("A.", "B.", "done", "cut"), skip, LastOne(), ()),
        # Saying it again says the whole reading, cut off or not, and one said whole no longer waits to be gone back to.
        (("A.", "B.", "done", "cut"), repeat, Replay(("A.", "B.")), ()),
        (("A.", "B.", "done", "done", "cut"), repeat, Replay(("A.", "B.")), ()),
        # The model said a word of its own before asking: what is said again is what the user cut in on, not that word.
        (("A.", "done", "cut", "Sure."), repeat, Replay(("A.",)), ()),
        # A reading played to its end was not cut off, whenever the user speaks next.
        (("A.", "done", "cut"), resume, NothingBargedIn(), ()),
        ((), resume, NothingBargedIn(), ()),
        ((), skip, NothingBargedIn(), ()),
        ((), repeat, NothingSaid(), ()),
        # Cut in on the answer to a side question to say "go back": it goes back past that answer, which is left, to
        # the reading cut off before it.
        (("A.", "B.", "cut", "C.", "D.", "done", "cut"), resume, Replay(("A.", "B.")), ()),
        # The answer had played to its end: going back is to the reading cut off before it all the same.
        (("A.", "B.", "cut", "C.", "done", "cut"), resume, Replay(("A.", "B.")), ()),
        # Skipping skips what was cut in on.
        (("A.", "B.", "cut", "C.", "D.", "E.", "done", "cut"), skip, Replay(("E.",)), (Bookmark(("A.", "B."), 0),)),
        # Readings cut off one after another, each played to its end since, are gone back to latest first.
        (("A.", "cut", "B.", "cut", "C.", "done", "cut"), resume, Replay(("B.",)), (Bookmark(("A.",), 0),)),
        # A report of a sentence finishing after its reading was cut changes nothing.
        (("A.", "B.", "cut", "done"), resume, Replay(("A.", "B.")), ()),
    ],
)
def test_each_act_says_what_the_speaker_was_at(
    happenings: tuple[Happening, ...], act: Callable[[Playback], tuple[Playback, Played]], replayed: Played, left: tuple[Bookmark, ...]
) -> None:
    after, said = act(played(*happenings))
    assert said == replayed
    assert after.bookmarks == left


def test_a_sentence_handed_after_a_reading_ended_starts_a_new_one_and_one_handed_before_continues_it() -> None:
    assert played("A.", "done", "B.").reading == ("B.",)
    assert played("A.", "B.", "done", "C.").reading == ("A.", "B.", "C.")
    # A reading cut off is over: what is said next is another.
    assert played("A.", "cut", "B.").reading == ("B.",)


def test_what_is_said_again_is_itself_a_reading_that_can_be_cut_and_gone_back_to() -> None:
    playback, _ = resume(played("A.", "B.", "C.", "cut"))
    again = reduce(handed, ("A.", "B.", "C."), queued(playback, ("A.", "B.", "C.")))
    after, said = resume(barge_in(finished(again)))
    assert said == Replay(("B.", "C."))
    assert after.bookmarks == ()


def test_lines_said_again_while_a_reading_still_plays_go_on_from_it() -> None:
    # The model said a few words of its own before asking to go back: those still playing, the lines follow them, and
    # the words finishing count as the words, not as the first line.
    playback = finished(handed(queued(handed(Playback(), "Sure."), ("A.", "B.")), "A."))
    assert playback.reading == ("Sure.", "A.") and playback.played == 1
    assert resume(barge_in(playback))[1] == Replay(("A.", "B."))


def test_a_word_handed_before_the_lines_queued_takes_its_own_place_and_loses_none_of_them() -> None:
    # The lines were queued while the model's word was still on its way to the speaker.
    playback = barge_in(handed(handed(queued(Playback(), ("A.", "B.", "C.")), "Sure."), "A."))
    assert playback.stopped == Bookmark(("Sure.", "A.", "B.", "C."), 0)


def test_a_reading_said_again_and_cut_before_all_of_it_reached_the_speaker_is_gone_back_to_whole() -> None:
    # The barge-in drops what had not yet reached the speaker; the reading knew it from the start, and keeps it.
    said_again = handed(queued(Playback(), ("A.", "B.", "C.")), "A.")
    assert said_again.reading == ("A.",)
    assert resume(barge_in(said_again))[1] == Replay(("A.", "B.", "C."))
    assert repeat(barge_in(said_again))[1] == Replay(("A.", "B.", "C."))


def test_the_stack_keeps_only_the_latest_readings_cut_off() -> None:
    playback = reduce(lambda playback, n: barge_in(handed(playback, f"{n}.")), range(BOOKMARKS + 3), Playback())
    assert len(playback.interrupted) == BOOKMARKS
    assert playback.interrupted[0] == Bookmark(("2.",), 0)


def spoken(text: str) -> AggregatedTextFrame:
    """A sentence as Pipecat's TTS hands it on ahead of its audio."""
    frame = AggregatedTextFrame(text, AggregationType.SENTENCE)
    frame.will_be_spoken = True
    return frame


def ended(text: str) -> TTSTextFrame:
    """A sentence as the output transport lets it go once its audio is written."""
    frame = TTSTextFrame(text, AggregationType.SENTENCE)
    frame.will_be_spoken = True
    return frame


class Heard(FrameProcessor):
    """What reaches the end of the pipeline."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self.frames: list[Frame] = []
        self.arrived = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        self.arrived.set()
        await self.push_frame(frame, direction)

    async def until(self, count: int, kind: type[Frame]) -> list[Frame]:
        while len(seen := [frame for frame in self.frames if isinstance(frame, kind)]) < count:
            self.arrived.clear()
            await asyncio.wait_for(self.arrived.wait(), 2.0)
        return seen


@asynccontextmanager
async def stood(player: Player) -> AsyncGenerator[tuple[Running, Heard]]:
    """The player as the daemon stands it: its lines ahead of the speaker, and its observer on the speaker and the output
    transport, each reduced to a processor that lets frames through in order."""
    speaker, output, heard = IdentityFilter(), IdentityFilter(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(speaker, output)]) as run:
        yield run, heard


async def test_the_player_reads_the_speaker_off_the_pipeline_and_says_the_sentence_cut_off_from_its_start() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    async with stood(player) as (run, heard):
        for frame in (spoken("The parser is fixed."), spoken("Its tests pass."), spoken("It pushed."), ended("The parser is fixed.")):
            await run.worker.queue_frame(frame)
        await heard.until(1, TTSTextFrame)
        await run.worker.queue_frame(InterruptionFrame())
        await heard.until(1, InterruptionFrame)
        assert recorded == [BargeIn("Its tests pass.", 1)]

        assert await player.act(resume) == Replay(("Its tests pass.", "It pushed."))
        said = await heard.until(2, TTSSpeakFrame)
        assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == ["Its tests pass.", "It pushed."]


@pytest.mark.parametrize(
    ("act", "line"),
    [
        (resume, "Nothing was cut off to go back to."),
        (skip, "Nothing was cut off to go back to."),
        (repeat, "I haven't said anything yet."),
    ],
)
async def test_an_act_with_nothing_to_say_again_says_why(act: Callable[[Playback], tuple[Playback, Played]], line: str) -> None:
    player = Player(lambda _: None)
    async with stood(player) as (_, heard):
        assert said(await player.act(act)) == (line,)
        frames = await heard.until(1, TTSSpeakFrame)
        assert [frame.text for frame in frames if isinstance(frame, TTSSpeakFrame)] == [line]


async def test_each_playback_tool_answers_with_how_much_was_said_never_what_and_how_many_readings_still_wait() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    tools = {tool.name: audited(SYSTEM, tool, recorded.append) for tool in playback_tools(player)}
    async with stood(player) as (run, heard):
        # A barge-in overtakes frames still queued, so each waits for what is ahead of it to be heard.
        for frame, seen in ((spoken("A."), 1), (spoken("B."), 2), (InterruptionFrame(), 1), (spoken("C."), 3), (InterruptionFrame(), 2)):
            await run.worker.queue_frame(frame)
            await heard.until(seen, type(frame))
        recorded.clear()
        assert await tools["resume"].body() == {"said": "two sentences said again", "waiting": 0}
        assert await tools["skip"].body() == {"said": "Nothing was cut off to go back to.", "waiting": 0}
        assert [(type(entry), entry.event) for entry in recorded if isinstance(entry, WideEvent)] == [(WideEvent, "tool.run")] * 2 and len(recorded) == 2


def test_skipping_the_last_sentence_says_so() -> None:
    assert said(skip(played("A.", "B.", "done", "cut"))[1]) == ("That was the last of it.",)


async def test_a_barge_in_on_a_quiet_speaker_is_a_line_that_cut_nothing_off() -> None:
    recorded: list[Entry] = []
    async with stood(Player(recorded.append)) as (run, heard):
        await run.worker.queue_frame(InterruptionFrame())
        await heard.until(1, InterruptionFrame)
    assert recorded == [BargeIn(None, 0)]


async def test_the_next_barge_in_asked_for_ahead_of_it_says_how_far_the_reading_played_when_it_reached_the_speaker() -> None:
    player = Player(lambda _: None)
    async with stood(player) as (run, heard):
        for frame in (spoken("The parser is fixed."), spoken("Its tests pass."), ended("The parser is fixed.")):
            await run.worker.queue_frame(frame)
        await heard.until(1, TTSTextFrame)
        reading = player.next_barge_in()
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(reading, 2.0) == Bookmark(("The parser is fixed.", "Its tests pass."), 1)
        # Each asked for is settled by the next barge-in alone: one played to its end cut nothing off.
        for frame in (spoken("Done."), ended("Done.")):
            await run.worker.queue_frame(frame)
        await heard.until(2, TTSTextFrame)
        reading = player.next_barge_in()
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(reading, 2.0) == Bookmark(("Done.",), 1)


SID = SessionId("s1")
REPLY = "It fixed the parser. The tests pass now.\n\n```py\nx = 1. y = 2\n```\nThen it pushed."


def telling(reply: str | None, turn: str = "p1") -> News:
    return News(PromptId(turn), reply, "", "", (), frozenset())


def holding(*replies: str | None) -> Recounts:
    recounts = Recounts()
    for reply in replies:
        recounts.put(SID, PromptId("p1"), telling(reply))
    return recounts


async def test_a_reply_is_cut_a_sentence_at_a_time_with_a_fenced_block_whole() -> None:
    assert await sentences(REPLY) == ("It fixed the parser.", "The tests pass now.", "```py\nx = 1. y = 2\n```\n", "Then it pushed.")


async def test_a_reading_is_a_frame_a_sentence_and_goes_back_to_from_the_sentence_a_barge_in_stopped_on() -> None:
    player = Player(lambda _: None)
    async with stood(player) as (run, heard):
        await player.read(("One.", "Two.", "Three."))
        said_ = await heard.until(3, TTSSpeakFrame)
        assert [frame.text for frame in said_ if isinstance(frame, TTSSpeakFrame)] == ["One.", "Two.", "Three."]
        # The speaker made all three, and the user cut in as the first one played.
        for frame in (spoken("One."), spoken("Two."), spoken("Three."), ended("One.")):
            await run.worker.queue_frame(frame)
        await heard.until(1, TTSTextFrame)
        await run.worker.queue_frame(InterruptionFrame())
        await heard.until(1, InterruptionFrame)
        assert await player.act(resume) == Replay(("Two.", "Three."))


async def test_read_reply_aloud_says_the_reply_to_the_speaker_and_hands_the_model_only_that_it_was_started() -> None:
    player = Player(lambda _: None)
    tool = read_reply_aloud_tool(new_sessions(clock=VirtualClock()), holding("a first reply", REPLY), player)
    async with stood(player) as (_, heard):
        result = await tool.body(session=SID)
        said_ = await heard.until(4, TTSSpeakFrame)
    assert [frame.text for frame in said_ if isinstance(frame, TTSSpeakFrame)] == list(await sentences(REPLY))
    # [LAW:verifiable-goals] the whole result, so no word of the reply is in it under any key.
    assert result == {"reading": "s1's last reply, 19 words, is being read to the user now"}
    assert tool.then == "silence"


async def test_read_reply_aloud_reads_the_latest_reply_a_turn_that_went_on_without_words_left_standing() -> None:
    player = Player(lambda _: None)
    tool = read_reply_aloud_tool(new_sessions(clock=VirtualClock()), holding("Its words.", None), player)
    async with stood(player) as (_, heard):
        assert await tool.body(session=SID) == {"reading": "s1's last reply, two words, is being read to the user now"}
        said_ = await heard.until(1, TTSSpeakFrame)
    assert [frame.text for frame in said_ if isinstance(frame, TTSSpeakFrame)] == ["Its words."]


async def test_read_reply_aloud_says_when_there_is_nothing_to_read_and_says_nothing_to_the_speaker() -> None:
    player = Player(lambda _: None)
    sessions = new_sessions(clock=VirtualClock())
    async with stood(player) as (_, heard):
        assert await read_reply_aloud_tool(sessions, Recounts(), player).body(session=SID) == {
            "error": "no turn of s1 has finished since hands started; read_session reads what it did before"
        }
        assert await read_reply_aloud_tool(sessions, holding(None), player).body(session=SID) == {
            "error": "the last turn of s1 said nothing, so there is nothing to read; tell_turn says what it did"
        }
        assert not [frame for frame in heard.frames if isinstance(frame, TTSSpeakFrame)]


async def test_read_reply_aloud_says_the_last_turn_could_not_be_read_rather_than_read_an_older_reply() -> None:
    player = Player(lambda _: None)
    recounts = holding("An older reply.")
    recounts.unread(SID, PromptId("p1"))
    async with stood(player) as (_, heard):
        assert await read_reply_aloud_tool(new_sessions(clock=VirtualClock()), recounts, player).body(session=SID) == {
            "error": "hands could not read all of the last turn of s1, so what it said last is not known; tell_turn says what hands could read of it"
        }
        assert not [frame for frame in heard.frames if isinstance(frame, TTSSpeakFrame)]
