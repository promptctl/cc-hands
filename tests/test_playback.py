"""Playback: where the speaker is, where readings were cut off, and what going back, skipping, and repeating say."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import reduce

import pytest
from pipecat.frames.frames import AggregatedTextFrame, Frame, InterruptionFrame, TTSSpeakFrame, TTSTextFrame
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameCallback, FrameDirection, FrameProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import AggregationType

from conftest import Running, running
from hands.clock import SYSTEM
from hands.core.playback import (
    BOOKMARKS,
    Bookmark,
    Labelled,
    LastOne,
    Line,
    NothingBargedIn,
    NothingSaid,
    Own,
    Playback,
    Played,
    Replay,
    as_told,
    barge_in,
    dropped,
    finished,
    handed,
    hands_says,
    queued,
    repeat,
    resume,
    skip,
)
from hands.sessions.audit import BargeIn, Entry
from hands.sessions.wide import WideEvent
from hands.voice.player import LineFrame, LineText, Player, Unsaid, line_frame, said
from hands.voice.spoken import SpokenForm
from hands.voice.tools import audited, playback_tools
from hands.voice.voices import SaidInTTS

# What happens at the speaker, as the taps report it: a line handed to it (a sentence's text is a sentence of the brain's),
# one played to its end, the user cutting in.
Happening = str | Labelled

# A draft read back, which the brain is told of by its label alone.
DRAFT = Labelled("Staged for api: run the tests.", "the draft for api read back")


def own(*texts: str) -> tuple[Own, ...]:
    return tuple(Own(text) for text in texts)


def played(*happenings: Happening) -> Playback:
    """Playback after the happenings in order: a line handed, "done" the next one finished, and "cut" a barge-in."""

    def step(playback: Playback, happening: Happening) -> Playback:
        match happening:
            case "done":
                return finished(playback)
            case "cut":
                return barge_in(playback)
            case Labelled() as line:
                return handed(playback, line)
            case sentence:
                return handed(playback, Own(sentence))

    return reduce(step, happenings, Playback())


def all_queued(playback: Playback, lines: tuple[Line, ...]) -> Playback:
    return reduce(queued, lines, playback)


@pytest.mark.parametrize(
    ("happenings", "act", "replayed", "left"),
    [
        # Cut in on the second sentence: going back says it again from its start, and what followed it.
        (("A.", "B.", "C.", "done", "cut"), resume, Replay(own("B.", "C.")), ()),
        # Cut in on the first: it is said again whole.
        (("A.", "B.", "cut"), resume, Replay(own("A.", "B.")), ()),
        # Skipping goes on from the sentence after the one cut off.
        (("A.", "B.", "C.", "done", "cut"), skip, Replay(own("C.")), ()),
        (("A.", "B.", "done", "cut"), skip, LastOne(), ()),
        # Saying it again says the whole reading, cut off or not, and one said whole no longer waits to be gone back to.
        (("A.", "B.", "done", "cut"), repeat, Replay(own("A.", "B.")), ()),
        (("A.", "B.", "done", "done", "cut"), repeat, Replay(own("A.", "B.")), ()),
        # The model said a word of its own before asking: what is said again is what the user cut in on, not that word.
        (("A.", "done", "cut", "Sure."), repeat, Replay(own("A.")), ()),
        # A reading played to its end was not cut off, whenever the user speaks next.
        (("A.", "done", "cut"), resume, NothingBargedIn(), ()),
        ((), resume, NothingBargedIn(), ()),
        ((), skip, NothingBargedIn(), ()),
        ((), repeat, NothingSaid(), ()),
        # Cut in on the answer to a side question to say "go back": it goes back past that answer, which is left, to
        # the reading cut off before it.
        (("A.", "B.", "cut", "C.", "D.", "done", "cut"), resume, Replay(own("A.", "B.")), ()),
        # The answer had played to its end: going back is to the reading cut off before it all the same.
        (("A.", "B.", "cut", "C.", "done", "cut"), resume, Replay(own("A.", "B.")), ()),
        # Skipping skips what was cut in on.
        (("A.", "B.", "cut", "C.", "D.", "E.", "done", "cut"), skip, Replay(own("E.")), (Bookmark(own("A.", "B."), 0),)),
        # Readings cut off one after another, each played to its end since, are gone back to latest first.
        (("A.", "cut", "B.", "cut", "C.", "done", "cut"), resume, Replay(own("B.")), (Bookmark(own("A."), 0),)),
        # A report of a sentence finishing after its reading was cut changes nothing.
        (("A.", "B.", "cut", "done"), resume, Replay(own("A.", "B.")), ()),
        # A line hands said as written is gone back to as itself, among the brain's words.
        (("A.", DRAFT, "B.", "done", "cut"), resume, Replay((DRAFT, Own("B."))), ()),
        (("A.", DRAFT, "B.", "done", "cut"), repeat, Replay((Own("A."), DRAFT, Own("B."))), ()),
    ],
)
def test_each_act_says_what_the_speaker_was_at(
    happenings: tuple[Happening, ...], act: Callable[[Playback], tuple[Playback, Played]], replayed: Played, left: tuple[Bookmark, ...]
) -> None:
    after, said = act(played(*happenings))
    assert said == replayed
    assert after.bookmarks == left


def test_a_sentence_handed_after_a_reading_ended_starts_a_new_one_and_one_handed_before_continues_it() -> None:
    assert played("A.", "done", "B.").reading == own("B.")
    assert played("A.", "B.", "done", "C.").reading == own("A.", "B.", "C.")
    # A reading cut off is over: what is said next is another.
    assert played("A.", "cut", "B.").reading == own("B.")


def test_what_is_said_again_is_itself_a_reading_that_can_be_cut_and_gone_back_to() -> None:
    playback, _ = resume(played("A.", DRAFT, "C.", "cut"))
    lines = (Own("A."), DRAFT, Own("C."))
    again = reduce(handed, lines, all_queued(playback, lines))
    after, said = resume(barge_in(finished(again)))
    # Each line is still whose it was, said again and cut again.
    assert said == Replay((DRAFT, Own("C.")))
    assert after.bookmarks == ()


def test_lines_said_again_while_a_reading_still_plays_go_on_from_it() -> None:
    # The model said a few words of its own before asking to go back: those still playing, the lines follow them, and
    # the words finishing count as the words, not as the first line.
    playback = finished(handed(all_queued(handed(Playback(), Own("Sure.")), own("A.", "B.")), Own("A.")))
    assert playback.reading == own("Sure.", "A.") and playback.played == 1
    assert resume(barge_in(playback))[1] == Replay(own("A.", "B."))


def test_a_word_handed_before_the_lines_queued_takes_its_own_place_and_loses_none_of_them() -> None:
    # The lines were queued while the model's word was still on its way to the speaker.
    playback = barge_in(handed(handed(all_queued(Playback(), (DRAFT, Own("B."), Own("C."))), Own("Sure.")), DRAFT))
    assert playback.stopped == Bookmark((Own("Sure."), DRAFT, Own("B."), Own("C.")), 0)


def test_a_reading_said_again_and_cut_before_all_of_it_reached_the_speaker_is_gone_back_to_whole() -> None:
    # The barge-in drops what had not yet reached the speaker; the reading knew it from the start, and keeps it.
    said_again = handed(all_queued(Playback(), own("A.", "B.", "C.")), Own("A."))
    assert said_again.reading == own("A.")
    assert resume(barge_in(said_again))[1] == Replay(own("A.", "B.", "C."))
    assert repeat(barge_in(said_again))[1] == Replay(own("A.", "B.", "C."))


def test_a_line_the_speaker_dropped_is_no_longer_coming_so_its_reading_ends() -> None:
    playback = finished(handed(all_queued(Playback(), (Own("A."), hands_says(" "))), Own("A.")))
    assert not playback.over
    after = dropped(playback, hands_says(" "))
    assert after.over and barge_in(after).stopped is None


def test_a_line_handed_is_the_first_of_its_equal_still_coming() -> None:
    # The same line twice: the first handed is the first queued, and the second still comes.
    playback = handed(all_queued(Playback(), (DRAFT, Own("A."), DRAFT)), DRAFT)
    assert playback.coming == (Own("A."), DRAFT)


def test_the_stack_keeps_only_the_latest_readings_cut_off() -> None:
    playback = reduce(lambda playback, n: barge_in(handed(playback, Own(f"{n}."))), range(BOOKMARKS + 3), Playback())
    assert len(playback.interrupted) == BOOKMARKS
    assert playback.interrupted[0] == Bookmark(own("2."), 0)


def test_lines_are_told_the_brain_with_its_own_words_quoted_and_every_other_line_by_its_label() -> None:
    told = as_told((Own("First, it rained."), Own(" Then it\nsnowed."), DRAFT, Own("Done.")))
    assert told == '"First, it rained. Then it snowed.", then the draft for api read back, then "Done."'
    assert DRAFT.text not in told
    # Words that say nothing are told as nothing.
    assert as_told((Own(" "), DRAFT)) == "the draft for api read back"
    assert as_told(()) == ""
    # A line in hands' own words is told by what it says.
    assert as_told((hands_says("That was the last of it."),)) == 'hands saying "That was the last of it."'


def spoken(text: str) -> AggregatedTextFrame:
    """A sentence of the brain's as Pipecat's TTS hands it on ahead of its audio."""
    frame = AggregatedTextFrame(text, AggregationType.SENTENCE)
    frame.will_be_spoken = True
    return frame


def spoken_line(line: Line) -> LineText:
    """A line of hands' as its TTS hands it on ahead of its audio, still whose it is."""
    frame = LineText(line.text, AggregationType.SENTENCE, line=line)
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

        assert await player.act(resume) == own("Its tests pass.", "It pushed.")
        said = await heard.until(2, LineFrame)
        assert [frame.line for frame in said if isinstance(frame, LineFrame)] == list(own("Its tests pass.", "It pushed."))


async def test_a_line_that_crosses_the_lines_is_coming_so_a_barge_in_before_it_reaches_the_speaker_keeps_it() -> None:
    player = Player(lambda _: None)
    async with stood(player) as (run, heard):
        await run.worker.queue_frame(spoken("Here it is."))
        await run.worker.queue_frame(line_frame(DRAFT))
        await heard.until(1, LineFrame)
        cut = player.next_barge_in()
        await run.worker.queue_frame(InterruptionFrame())
        # The speaker here passes the frame on as it is, so the line never reached it: it is unheard with the rest.
        assert await asyncio.wait_for(cut, 2.0) == Bookmark((Own("Here it is."), DRAFT), 0)


class CutAtFirstLine(FrameProcessor):
    """Stood right behind the player's lines: the moment the first line is pushed to it, a barge-in reaches the output,
    with the rest still to be pushed."""

    def __init__(self, player: Player) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._player = player
        self._cut = False

    async def queue_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM, callback: FrameCallback | None = None) -> None:
        if isinstance(frame, LineFrame) and not self._cut:
            self._cut = True
            self._player.played(InterruptionFrame())
        await super().queue_frame(frame, direction, callback)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def test_a_reading_said_again_and_cut_while_it_is_still_being_pushed_is_gone_back_to_whole() -> None:
    player = Player(lambda _: None)
    # A reading cut off in its first sentence, by the player's own reads of the speaker.
    for frame in (spoken("A."), spoken_line(DRAFT)):
        player.handed(frame)
    player.played(InterruptionFrame())
    heard = Heard()
    async with running([player.lines, CutAtFirstLine(player), heard]):
        await player.act(resume)
        await heard.until(2, LineFrame)
        # Cut as the first of it was pushed: the whole of it was coming, so the whole of it waits to be gone back to.
        assert player.waiting == 1
        assert (await player.act(resume)) == (Own("A."), DRAFT)


async def test_going_back_to_a_line_hands_said_as_written_tells_the_brain_its_label_never_its_text() -> None:
    player = Player(lambda _: None)
    tools = {tool.name: tool for tool in playback_tools(player)}
    async with stood(player) as (run, heard):
        for frame in (spoken("Here it is."), spoken_line(DRAFT), InterruptionFrame()):
            await run.worker.queue_frame(frame)
        await heard.until(1, InterruptionFrame)
        result = await tools["resume"].body()
        await heard.until(2, LineFrame)
    assert result == {"said": '"Here it is.", then the draft for api read back', "waiting": 0}
    assert DRAFT.text not in str(result)


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
        assert await player.act(act) == (hands_says(line),)
        said = await heard.until(1, LineFrame)
        assert [frame.text for frame in said if isinstance(frame, LineFrame)] == [line]


async def test_each_playback_tool_answers_with_what_was_said_and_how_many_readings_still_wait() -> None:
    recorded: list[Entry] = []
    player = Player(recorded.append)
    tools = {tool.name: audited(SYSTEM, tool, recorded.append) for tool in playback_tools(player)}
    async with stood(player) as (run, heard):
        # A barge-in overtakes frames still queued, so each waits for what is ahead of it to be heard.
        for frame, seen in ((spoken("A."), 1), (spoken("B."), 2), (InterruptionFrame(), 1), (spoken("C."), 3), (InterruptionFrame(), 2)):
            await run.worker.queue_frame(frame)
            await heard.until(seen, type(frame))
        recorded.clear()
        assert await tools["resume"].body() == {"said": '"A. B."', "waiting": 0}
        assert await tools["skip"].body() == {"said": 'hands saying "Nothing was cut off to go back to."', "waiting": 0}
        assert [(type(entry), entry.event) for entry in recorded if isinstance(entry, WideEvent)] == [(WideEvent, "tool.run")] * 2 and len(recorded) == 2


def test_skipping_the_last_sentence_says_so() -> None:
    assert said(skip(played("A.", "B.", "done", "cut"))[1]) == (hands_says("That was the last of it."),)


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
        assert await asyncio.wait_for(reading, 2.0) == Bookmark(own("The parser is fixed.", "Its tests pass."), 1)
        # Each asked for is settled by the next barge-in alone: one played to its end cut nothing off.
        for frame in (spoken("Done."), ended("Done.")):
            await run.worker.queue_frame(frame)
        await heard.until(2, TTSTextFrame)
        reading = player.next_barge_in()
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(reading, 2.0) == Bookmark(own("Done."), 1)


class Stubbed(SaidInTTS):
    """hands' TTS on Pipecat's own text path, with the synthesis taken out: no model is loaded, and each line is said as
    nothing at all."""

    def __init__(self) -> None:
        TTSService.__init__(self, settings=PocketTTSSettings(model=None, voice="alba", language=Language.EN), text_filters=[SpokenForm()])  # pyright: ignore[reportUnknownMemberType]

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        return
        yield


async def test_hands_tts_hands_each_line_on_still_whose_it_is_and_says_which_it_dropped() -> None:
    tts, heard = Stubbed(), Heard()
    async with running([tts, heard]):
        await tts.queue_frame(line_frame(DRAFT))
        # All space, which Pipecat says as nothing.
        await tts.queue_frame(line_frame(hands_says("  ")))
        [handed] = await heard.until(1, LineText)
        [unsaid] = await heard.until(1, Unsaid)
    assert isinstance(handed, LineText) and handed.line == DRAFT and handed.will_be_spoken
    assert isinstance(unsaid, Unsaid) and unsaid.line == hands_says("  ")


async def test_hands_tts_refuses_a_line_that_says_not_whose_it_is() -> None:
    with pytest.raises(TypeError, match="no LineFrame"):
        await Stubbed().process_frame(TTSSpeakFrame("Nobody's line."), FrameDirection.DOWNSTREAM)
