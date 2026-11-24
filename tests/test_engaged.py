"""The desk's driver, engaged: a triple tap engages, the user's voice opens each turn, end-of-turn detection closes it,
and another triple tap disengages."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import pytest
from conftest import ScriptedEars, hearing_with

from hands.clock import Clock
from hands.virtualclock import VirtualClock
from hands.voice import engaged
from hands.sessions.wide import WideEvent
from hands.fanout import Fanout
from hands.voice.desk import ANSWER_WINDOW_SECONDS, AnswerWindowOpened, AnswerWindowOver, Leveled, Made, Served, Woken
from hands.voice.handsvoice import HandsQuiet, HandsSpeaking, HandsVoice
from hands.voice.listening import Level, Levels
from hands.voice.clips import Trigger
from hands.voice.engaged import Hears, drive
from hands.voice.turnlife import Ears
from hands.voice import voiceturn
from hands.voice.shortcut import KeyEvent, Move, Pressed, Released, Ripe

# Right Shift's triple tap: what engages, and what disengages.
TAP: tuple[KeyEvent, ...] = tuple(e for i in range(3) for e in (Pressed(1.0 + i / 10), Released()))
# What the gate is handed for the taps themselves: each press arms the microphone and its release disarms it.
TAPPED: list[Move] = ["arm", "disarm", "arm", "disarm", "arm"]


class Rig:
    """The desk's driver with a fake talk key and a fake microphone."""

    def __init__(self, verdicts: Sequence[bool], level: Level = "off") -> None:
        self.ears = ScriptedEars(verdicts)
        self.keys: list[Callable[[KeyEvent], None]] = []
        self.audio: asyncio.Queue[bytes] = asyncio.Queue()
        self.made: list[Move] = []
        self.made_by: list[Made] = []
        self.levels: list[Leveled] = []
        # The one owner of the level, which the run starts at `level`.
        self.owner = Levels(level)
        self.events: list[WideEvent] = []
        self.clock = VirtualClock()
        # Told each time the driver makes a move or takes a buffer, so a test waits on what it did, not on time passing.
        self.changed = asyncio.Condition()
        # Set as the driver begins to load the models held, and by a test to have them loaded.
        self.loading, self.loaded = asyncio.Event(), asyncio.Event()
        # Hands' voice, as the speaker tells it, and set once the driver listens to it.
        self.speech: Fanout[HandsVoice] = Fanout()
        self.telling = asyncio.Event()
        self.windows: list[AnswerWindowOpened] = []

    async def _told(self) -> None:
        async with self.changed:
            self.changed.notify_all()

    async def until(self, done: Callable[[], bool]) -> None:
        async with self.changed:
            await self.changed.wait_for(done)

    @asynccontextmanager
    async def tapped(self, into: Callable[[KeyEvent], None]) -> AsyncGenerator[None]:
        self.keys.append(into)
        yield

    @asynccontextmanager
    async def overheard(self) -> AsyncGenerator[AsyncIterator[bytes]]:
        async def buffers() -> AsyncIterator[bytes]:
            while True:
                each = await self.audio.get()
                await self._told()
                yield each

        yield buffers()

    async def served(self, act: Served) -> None:
        match act:
            case Made():
                self.made_by.append(act)
                self.made.append(act.move)
            case Leveled():
                self.levels.append(act)
            case AnswerWindowOpened():
                self.windows.append(act)
        await self._told()

    @asynccontextmanager
    async def spoken(self) -> AsyncGenerator[AsyncIterator[HandsVoice]]:
        async with self.speech.subscribed() as edges:
            self.telling.set()
            yield edges

    async def say(self, edge: HandsVoice) -> None:
        """Tell the driver of hands' voice, once it listens: the speaker tells only those listening as it speaks."""
        await self.telling.wait()
        self.speech.publish(edge)

    @asynccontextmanager
    async def held(self) -> AsyncGenerator[tuple[Ears, Hears]]:
        """The models, loaded once the test sets `loaded`."""
        self.loading.set()
        await self.loaded.wait()
        yield self.ears, unwoken

    def start(self, ears: Ears | None = None) -> asyncio.Task[None]:
        return asyncio.create_task(drive(self.clock, self.tapped, self.overheard, hearing_with(ears or self.ears, unwoken), self.spoken, self.owner, self.served, self.events.append))

    async def start_loading(self) -> asyncio.Task[None]:
        """The driver, its models loading until the test sets `loaded`, once it has stepped the start's level."""
        driving = asyncio.create_task(drive(self.clock, self.tapped, self.overheard, self.held, self.spoken, self.owner, self.served, self.events.append))
        # The start's level is stepped before the driver first waits, and so before the models begin to load.
        await self.loading.wait()
        return driving

    async def tap(self) -> None:
        while not self.keys:
            await asyncio.sleep(0)
        for event in TAP:
            self.keys[0](event)

    async def choose(self, level: Level) -> Level:
        """Choose `level` as the tool does, and get back the level the driver stepped it from."""
        return await self.owner.choose(level, "tool")

    async def hear(self, *audio: bytes) -> None:
        for each in audio:
            self.audio.put_nowait(each)

    async def settle(self, made: int) -> None:
        await self.until(lambda: len(self.made) >= made)
        for _ in range(20):
            await asyncio.sleep(0)

    async def drained(self) -> None:
        await self.until(lambda: not self.audio.qsize())


async def unwoken(audio: bytes, begun: bool) -> Woken | None:
    return None


async def stopped(driving: asyncio.Task[None]) -> None:
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving


async def test_three_turns_with_no_key_between_them_then_a_disengage() -> None:
    rig = Rig(verdicts=[False, True, True, False])
    driving = rig.start()
    await rig.tap()
    # A pause mid-thought held open, then the thought finished; a second turn; a third held open by its verdict, then
    # ended by the silence after it running on.
    await rig.hear(b"s", b"S", b"q", b"s", b"S", b"q")
    await rig.settle(10)
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(13)
    await rig.hear(b"s", b"S", b"q", b"x")
    await rig.settle(16)
    await rig.tap()
    await rig.hear(b"s", b"S", b"q")
    for _ in range(20):
        await asyncio.sleep(0)
    await stopped(driving)
    assert rig.made == [*TAPPED, "disarm", "listen", *["arm", "start", "stop"] * 3, *TAPPED, "disarm", "deafen"]
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", "engaged", "tap"), ("engaged", "off", "tap")]
    # One event for each stint at a level, emitted as the level changes: the stint engaged counts its turns.
    stints = [event for event in rig.events if event.event == "desk"]
    assert [(event.facts["level"], event.outcome) for event in stints] == [("off", "ok"), ("engaged", "ok"), ("off", "cancelled")]
    desk = stints[1]
    counts = dict(desk.counts)
    assert {name: counts[name] for name in ("start", "stop", "held_open", "ended_on_silence", "by_engaged_conversation", "to_engaged", "to_off")} == {
        "start": 3, "stop": 3, "held_open": 2, "ended_on_silence": 1, "by_engaged_conversation": 10, "to_engaged": 0, "to_off": 1,
    }
    judged = [event for event in rig.events if event.event == "desk.judged"]
    assert [event.facts["complete"] for event in judged] == [False, True, True, False]
    assert all(event.parent_id == desk.span_id for event in judged)
    leveled = [event.facts for event in rig.events if event.event == "desk.leveled"]
    assert [(facts["was"], facts["now"], facts["by"]) for facts in leveled] == [("off", "engaged", "tap"), ("engaged", "off", "tap")]


async def test_off_the_room_opens_nothing_and_the_held_key_still_talks() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    await rig.hear(b"s", b"S", b"q", b"x")
    while not rig.keys:
        await asyncio.sleep(0)
    for event in (Pressed(1.0), Ripe(1.0), Released()):
        rig.keys[0](event)
    await rig.settle(3)
    await stopped(driving)
    assert rig.made_by == [Made("arm", "shortcut"), Made("start", "shortcut"), Made("stop", "shortcut")]


async def test_a_level_the_tool_chose_is_stepped_as_a_tap_is() -> None:
    rig = Rig(verdicts=[True])
    driving = rig.start()
    was = [await rig.choose("engaged")]
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(4)
    was += [await rig.choose("engaged"), await rig.choose("off")]
    await rig.settle(5)
    await stopped(driving)
    assert rig.made == ["listen", "arm", "start", "stop", "deafen"]
    # The level each choice found, as the driver stepped it; one chosen again changes nothing.
    assert was == ["off", "engaged", "engaged"]
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", "engaged", "tool"), ("engaged", "off", "tool")]


@pytest.mark.parametrize(("level", "voice"), [("engaged", "engaged conversation"), ("wake word", "wake word")])
async def test_a_restart_listens_at_the_level_it_carried_as_a_tap_to_it_would_once_its_models_are_loaded(level: Level, voice: Trigger) -> None:
    rig = Rig(verdicts=[], level=level)
    driving = await rig.start_loading()
    # While the models load nothing hears the room: the desk listens at off, holding the level the restart carried, which
    # a restart now would carry on, and makes nothing of it, no listen move and no cue.
    assert (rig.made_by, rig.levels) == ([], [])
    assert (rig.owner.level, rig.owner.last_chosen) == ("off", level)
    rig.loaded.set()
    # Loaded, the gate is told the desk listens, and the change is said, as for a tap or the tool's choice.
    await rig.until(lambda: bool(rig.made))
    await stopped(driving)
    assert rig.made_by == [Made("listen", voice)]
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", level, "start")]
    assert (rig.owner.level, rig.owner.last_chosen) == (level, level)


async def test_a_triple_tap_while_the_models_load_toggles_the_level_chosen_not_the_off_the_desk_listens_at() -> None:
    rig = Rig(verdicts=[], level="engaged")
    driving = await rig.start_loading()
    # Right Shift's triple tap from engaged, the level the restart carried: off, though the desk has listened at nothing.
    await rig.tap()
    for _ in range(20):
        await asyncio.sleep(0)
    assert (rig.owner.level, rig.owner.last_chosen) == ("off", "off")
    rig.loaded.set()
    # Loaded at off, as chosen: nothing to listen for, and no change to say.
    assert await rig.choose("off") == "off"
    await stopped(driving)
    assert (rig.made, rig.levels) == ([*TAPPED, "disarm"], [])


async def test_a_start_at_off_moves_nothing_and_says_no_change() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    assert await rig.choose("off") == "off"
    await stopped(driving)
    assert (rig.made_by, rig.levels) == ([], [])
    assert (rig.owner.level, rig.owner.last_chosen) == ("off", "off")


async def test_a_level_a_page_chose_is_said_to_be_the_pages() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    assert await rig.owner.choose("wake word", "page") == "off"
    await stopped(driving)
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", "wake word", "page")]


async def test_a_drive_cancelled_while_a_move_is_handed_on_leaves_the_owner_at_the_level_the_desk_moved_to() -> None:
    rig = Rig(verdicts=[])
    handed = asyncio.Event()

    async def held(_act: Served) -> None:
        # The listen move's line, said through a speaker that never finishes: the change of level is handed on after it.
        handed.set()
        await asyncio.Event().wait()

    driving = asyncio.create_task(drive(rig.clock, rig.tapped, rig.overheard, hearing_with(rig.ears, unwoken), rig.spoken, rig.owner, held, rig.events.append))
    # The tool's choice, whose one move is the listen ahead of the change of level.
    assert await rig.choose("engaged") == "off"
    await handed.wait()
    # A restart lands here, cancelling the drive inside the move: the level the desk moved to is the one it carries.
    await stopped(driving)
    assert rig.levels == []
    assert (rig.owner.level, rig.owner.last_chosen) == ("engaged", "engaged")


async def test_the_key_talks_while_the_models_load_and_a_level_chosen_then_is_listened_at_once_they_are() -> None:
    rig = Rig(verdicts=[])
    driving = await rig.start_loading()
    while not rig.keys:
        await asyncio.sleep(0)
    for event in (Pressed(1.0), Ripe(1.0), Released()):
        rig.keys[0](event)
    await rig.settle(3)
    # Chosen while nothing hears the room: the desk listens at off, and its listen move and cue wait for the models.
    assert await rig.choose("engaged") == "off"
    assert (rig.owner.level, rig.owner.last_chosen) == ("off", "engaged")
    assert rig.made_by == [Made("arm", "shortcut"), Made("start", "shortcut"), Made("stop", "shortcut")]
    rig.loaded.set()
    await rig.settle(4)
    await stopped(driving)
    assert rig.made_by[3:] == [Made("listen", "engaged conversation")]
    assert [(leveled.was, leveled.now, leveled.by) for leveled in rig.levels] == [("off", "engaged", "tool")]


async def test_a_model_that_cannot_be_loaded_fails_the_drive() -> None:
    rig = Rig(verdicts=[])

    @asynccontextmanager
    async def unloadable() -> AsyncGenerator[tuple[Ears, Hears]]:
        raise OSError("no Smart Turn in the hub's cache, and no network")
        yield rig.ears, unwoken

    with pytest.raises(ExceptionGroup) as failed:
        await drive(rig.clock, rig.tapped, rig.overheard, unloadable, rig.spoken, rig.owner, rig.served, rig.events.append)
    assert failed.group_contains(OSError, match="no network")


async def test_a_choice_whose_tool_call_was_cancelled_is_stepped_and_the_desk_goes_on() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    # The tool call asks, and is cancelled before the driver takes its choice: the user cutting the brain off.
    asking = asyncio.create_task(rig.choose("engaged"))
    await asyncio.sleep(0)
    asking.cancel()
    assert await rig.choose("off") == "engaged"
    await stopped(driving)
    assert rig.made == ["listen", "deafen"]


async def test_each_level_hears_from_where_it_started() -> None:
    rig = Rig(verdicts=[])
    begun: list[bool] = []

    async def woken(audio: bytes, afresh: bool) -> Woken | None:
        begun.append(afresh)
        return None

    driving = asyncio.create_task(drive(rig.clock, rig.tapped, rig.overheard, hearing_with(rig.ears, woken), rig.spoken, rig.owner, rig.served, rig.events.append))
    # Off, nothing hears the room; the wake word's model hears from its level's first buffer, afresh each time it starts.
    await rig.hear(b"q")
    for level in ("wake word", "off", "wake word"):
        await rig.choose(level)
        await rig.hear(b"q", b"q")
        await rig.drained()
        for _ in range(20):
            await asyncio.sleep(0)
    await stopped(driving)
    assert begun == [True, False, True, False]


async def test_speech_begun_under_a_tap_of_shift_opens_a_turn_once_the_key_is_up() -> None:
    rig = Rig(verdicts=[True])
    driving = rig.start()
    await rig.choose("engaged")
    rig.keys[0](Pressed(1.0))
    await rig.hear(b"s", b"S")
    await rig.drained()
    rig.keys[0](Released())
    await rig.hear(b"S", b"q")
    await rig.settle(6)
    await stopped(driving)
    assert rig.made == ["listen", "arm", "disarm", "arm", "start", "stop"]


async def test_a_verdict_ends_the_turn_it_judged_however_long_smart_turn_thinks() -> None:
    rig = Rig(verdicts=[True, True])
    rig.ears.thinking = asyncio.Event()
    driving = rig.start()
    await rig.choose("engaged")
    # A pause is judged; while Smart Turn thinks, the user speaks again, and the silence after that runs on.
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(3)
    await rig.hear(b"s", b"S", b"q", b"x")
    for _ in range(20):
        await asyncio.sleep(0)
    rig.ears.thinking.set()
    await rig.settle(6)
    await stopped(driving)
    # The first verdict sends the first turn; the speech heard while it was given opens the next, which is judged too.
    assert rig.made[:7] == ["listen", "arm", "start", "stop", "arm", "start", "stop"]


async def test_a_wake_nothing_is_asked_after_is_dropped_when_the_ask_runs_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # The ask window is moved in its one place, and the desk's drop moves with it.
    monkeypatch.setattr("hands.voice.voiceturn.ASK_SECONDS", 3.0)
    rig = Rig(verdicts=[])
    spoke = [False]

    async def woken(audio: bytes, begun: bool) -> Woken | None:
        said, spoke[0] = not spoke[0], True
        return Woken(rig.clock.now(), over_hands=False, built_in=False) if said else None

    driving = asyncio.create_task(drive(rig.clock, rig.tapped, rig.overheard, hearing_with(rig.ears, woken), rig.spoken, rig.owner, rig.served, rig.events.append))
    await rig.choose("wake word")
    await rig.hear(b"q")
    # The wake word's deadlines are set as it opens the turn, from the instant it was heard.
    await rig.clock.scheduled(voiceturn.ASK_SECONDS)
    rig.clock.advance_to(voiceturn.ASK_SECONDS)
    await rig.settle(4)
    await stopped(driving)
    assert rig.made_by == [Made("listen", "wake word"), Made("arm", "wake word"), Made("start", "wake word"), Made("drop", "wake word")]
    # Smart Turn starts the next turn with none of this one's audio in it.
    assert rig.ears.cleared == 1


async def test_an_engaged_turn_open_to_the_limit_is_sent_as_it_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hands.voice.shortcut.TURN_LIMIT_SECONDS", 3.0)
    rig = Rig(verdicts=[])
    driving = rig.start()
    await rig.choose("engaged")
    await rig.hear(b"s", b"S")
    await rig.clock.scheduled(3.0)
    rig.clock.advance_to(3.0)
    await rig.settle(4)
    await stopped(driving)
    assert rig.made_by == [Made("listen", "engaged conversation"), Made("arm", "engaged conversation"), Made("start", "engaged conversation"), Made("timeout", "engaged conversation")]
    assert rig.ears.cleared == 1


WINDOWED = [Made(move, "answer window") for move in ("listen", "arm", "start", "stop", "deafen")]
QUIET = HandsQuiet(0.0, "desk")


async def test_off_the_room_is_heard_only_in_the_window_after_hands_spoke_from_its_start_and_speech_there_is_a_turn() -> None:
    rig = Rig(verdicts=[True])
    driving = rig.start()
    # Off, what may be speech opens nothing, and is no part of what the window hears.
    await rig.hear(b"s")
    await rig.drained()
    await rig.say(QUIET)
    await rig.until(lambda: bool(rig.windows))
    await rig.hear(b"s", b"S", b"q")
    await rig.settle(len(WINDOWED))
    await stopped(driving)
    assert rig.windows == [AnswerWindowOpened(AnswerWindowOver(ANSWER_WINDOW_SECONDS))] and rig.made_by == WINDOWED
    # Heard from the window's start, with nothing of the room before it.
    assert rig.ears.afreshed == 1
    (desk,) = (event for event in rig.events if event.event == "desk")
    assert {name: desk.counts[name] for name in ("hands_quiet", "windows", "closed_answered", "by_answer_window", "heard")} == {"hands_quiet": 1, "windows": 1, "closed_answered": 1, "by_answer_window": 5, "heard": 3}


async def test_under_the_wake_word_an_answer_begun_over_hands_last_words_opens_the_turn_as_the_window_opens() -> None:
    rig = Rig(verdicts=[True], level="wake word")
    driving = rig.start()
    # The user starts to answer before hands has fallen quiet: the desk, between turns, opens nothing on it.
    await rig.hear(b"s", b"S")
    await rig.drained()
    await rig.say(QUIET)
    await rig.until(lambda: bool(rig.windows))
    await rig.hear(b"S", b"q")
    await rig.settle(4)
    await stopped(driving)
    assert rig.made_by == [Made("listen", "wake word"), *(Made(move, "answer window") for move in ("arm", "start", "stop"))]


async def test_a_window_nothing_was_said_in_closes_when_it_is_due_and_is_counted_as_run_out() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    await rig.say(QUIET)
    await rig.clock.scheduled(ANSWER_WINDOW_SECONDS)
    rig.clock.advance_to(ANSWER_WINDOW_SECONDS)
    await rig.settle(2)
    await stopped(driving)
    assert rig.made_by == [Made("listen", "answer window"), Made("deafen", "answer window")]
    (desk,) = (event for event in rig.events if event.event == "desk")
    assert {name: desk.counts[name] for name in ("windows", "closed_time", "closed_hands", "closed_answered")} == {"windows": 1, "closed_time": 1, "closed_hands": 0, "closed_answered": 0}



async def test_off_a_window_closed_and_opened_again_between_two_buffers_is_heard_afresh_and_each_quiet_is_accounted_for() -> None:
    rig = Rig(verdicts=[])
    driving = rig.start()
    await rig.say(QUIET)
    await rig.until(lambda: len(rig.windows) == 1)
    await rig.hear(b"q")
    await rig.drained()
    # Hands speaks again and falls quiet once more, with no buffer heard between.
    await rig.say(HandsSpeaking())
    await rig.say(HandsQuiet(1.0, "phone"))
    await rig.say(QUIET)
    await rig.until(lambda: len(rig.windows) == 2)
    # What may be speech, armed for once its buffer is stepped: listen, deafen, listen, arm.
    await rig.hear(b"s")
    await rig.settle(4)
    await stopped(driving)
    # Heard afresh from where the desk began to hear the room again.
    assert rig.ears.afreshed == 2
    (desk,) = (event for event in rig.events if event.event == "desk")
    assert {name: desk.counts[name] for name in ("hands_quiet", "windows", "no_window_phone", "closed_hands")} == {"hands_quiet": 3, "windows": 2, "no_window_phone": 1, "closed_hands": 1}

async def test_ears_that_cannot_be_loaded_fail_as_the_unit_they_were_loaded_for_whoever_hears_with_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """The phone's ears are loaded as the daemon starts and before its page is answered, so a load that fails is the start's failure,
    named `phone.loaded`; the desk's is `desk.loaded`."""

    class Unloadable:
        def __init__(self, clock: Clock, sample_rate: int) -> None:
            raise OSError("no Smart Turn in the hub's cache, and no network")

    monkeypatch.setattr(engaged, "Models", Unloadable)
    for name in ("phone.loaded", "desk.loaded"):
        events: list[WideEvent] = []
        with pytest.raises(OSError, match="no Smart Turn"):
            async with engaged.loaded(VirtualClock(), 16000, events.append, name):
                pytest.fail("ears that could not be loaded were given to what hears with them")
        [failed] = events
        assert (failed.event, failed.outcome) == (name, "failed")
