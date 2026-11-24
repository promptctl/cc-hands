"""The phone listening for its wake word: the page's `wake`, `level` and `detector` messages as the phone reads them off a call,
the gate they move, the clips they open, and the driver that ends a turn a wake opened, on a clock the test moves. The
call's channel is a recording one: aiortc's own carries these messages no differently from the ones in test_phone.py."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from aiortc import RTCDataChannel
from av.audio.resampler import AudioResampler
from conftest import errors as errors, events, running  # noqa: F401  (the fixture is used by name)
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.metrics.metrics import TurnMetricsData

from hands.sessions.audit import Entry
from hands.sessions.wide import begun
from hands.virtualclock import VirtualClock
from hands.voice import shortcut, voiceturn
from hands.voice.atphone import drive
from hands.voice.clipowner import ClipOwner
from hands.voice.latency import LatencyObserver
from hands.sessions.payload import Rejected
from hands.voice.phone import Asked, Phone, UnreadableWake, _Call, parse_said, quiet_message  # pyright: ignore[reportPrivateUsage]
from hands.voice.ptt import PushToTalk
from hands.voice.trigger import PhoneLevel
from hands.voice.wakeword import THRESHOLD
from test_clipowner import Scripted

pytestmark = pytest.mark.usefixtures("errors")

RATE = 16000
# Twenty milliseconds of the page's microphone as one message at `value`, which is how loud the detector below hears it.
FRAME = RATE // 50
SECOND = 50


def wake_message(score: float) -> str:
    """A `wake` as the page writes it (phone.html)."""
    return json.dumps({"kind": "wake", "score": score})


def level_message(level: PhoneLevel) -> str:
    """A `level` as the page writes it (phone.html)."""
    return json.dumps({"kind": "level", "level": level})


def detector_message(browser: str, why: str | None) -> str:
    """A `detector` as the page writes it (phone.html): loaded, or failed for `why`."""
    return json.dumps({"kind": "detector", "state": "loaded", "browser": browser} if why is None else {"kind": "detector", "state": "failed", "browser": browser, "why": why})


def audio(value: int) -> bytes:
    return np.full(FRAME, value, dtype=np.int16).tobytes()


# The second the page kept from before its wake, a person's speech, and the quiet between.
BEFORE, SPEECH, QUIET = audio(5), audio(8000), audio(0)


class LoudEars:
    """The models, deciding by how loud a buffer is: speech where it is loud, and Smart Turn calls what ended complete."""

    def __init__(self) -> None:
        self.cleared = 0
        self.afreshed = 0

    async def detect(self, audio: bytes) -> VADState:
        return VADState.SPEAKING if np.frombuffer(audio, dtype=np.int16).max() > 1000 else VADState.QUIET

    def heard(self, audio: bytes, speech: bool) -> bool:
        return False

    async def judge(self) -> tuple[bool, TurnMetricsData | None]:
        return True, None

    def clear(self) -> None:
        self.cleared += 1

    def afresh(self) -> None:
        self.afreshed += 1


@dataclass
class Up:
    """A call up with hands at the phone, a page that has heard nothing yet, and the phone's own driver and clip owner."""

    clock: VirtualClock
    phone: Phone
    key: PushToTalk
    call: _Call
    recorded: list[Entry]
    told: list[dict[str, object]]
    transcribed: Scripted
    speaking: list[bool]
    moves: list[str] = field(default_factory=list[str])

    def send(self, *messages: bytes | str) -> None:
        """What the page sends, read in order as the channel gives it, each as the call acts on it."""
        for message in messages:
            said = parse_said(message, RATE)
            # As the channel's handler does, which a page that sends is never left without.
            self.call.heard.set()
            self.phone._hear(self.call, said)  # pyright: ignore[reportPrivateUsage]

    async def until(self, done: Callable[[], bool]) -> None:
        """Let the loop run until `done`; nothing here waits on time."""
        for _ in range(500):
            while not self.phone.moves.empty():
                self.moves.append(self.phone.moves.get_nowait())
            if done():
                return
            await asyncio.sleep(0)
        pytest.fail("what the test waited on never came")

    async def told_count(self, count: int) -> list[dict[str, object]]:
        await self.until(lambda: len(self.told) >= count)
        return self.told

    async def due(self, when: float, sending: bytes = QUIET) -> None:
        """The clock moved to `when` a second at a time, the page sending `sending` through each as a page open does, so the
        call is never taken for a closed page."""
        while self.clock.now() < when:
            self.clock.advance_to(min(when, self.clock.now() + 1.0))
            self.send(sending)
            for _ in range(20):
                await asyncio.sleep(0)

    def at_the_desk(self) -> None:
        """The desk's microphone, one frame of it: what carries the gate's moves to the clip owner while hands is there."""
        self.phone.heard.put_nowait(self.key.gate.framed(QUIET, QUIET, None, RATE, 1, "desk"))


@asynccontextmanager
async def up(clips: int = 3) -> AsyncGenerator[Up]:
    clock = VirtualClock()
    key, recorded, speaking = PushToTalk(clock, lambda _entry: None), list[Entry](), [False]
    phone = Phone(clock, key, heard_rate=RATE, played_rate=24000, speaking=lambda: speaking[0], record=recorded.append)
    channel, told = Mock(spec=RTCDataChannel), list[dict[str, object]]()
    channel.readyState = "open"
    def sent(message: str) -> None:
        told.append(json.loads(message))

    channel.send.side_effect = sent
    call = _Call(AsyncMock(), Mock(), AudioResampler(format="s16", layout="mono", rate=RATE), RATE, "192.168.7.20", Asked("the page", "take"), begun(clock), asyncio.Event())
    await phone._arrive(call, channel)  # pyright: ignore[reportPrivateUsage]
    transcribed = Scripted([None] * clips)
    owner = ClipOwner(clock, transcribed, (), recorded.append)
    ears = LoudEars()
    this = Up(clock, phone, key, call, recorded, told, transcribed, speaking)
    async with running([owner], [LatencyObserver(clock, owner, phone.tell)]) as run:

        async def feed() -> None:
            while True:
                await run.worker.queue_frame(await phone.heard.get())

        feeding = asyncio.create_task(feed())
        driving = asyncio.create_task(drive(clock, key, phone.turns, ears, phone.took, recorded.append))
        try:
            yield this
        finally:
            feeding.cancel()
            driving.cancel()
            await asyncio.gather(feeding, driving, return_exceptions=True)
            await phone.hang_up("stopped")


async def test_a_wake_then_speech_then_quiet_opens_a_turn_at_the_phone_that_starts_a_second_before_the_wake_and_ends_as_a_wake_turn_does() -> None:
    async with up() as page:
        page.send(level_message("wake word"), *[BEFORE] * (SECOND + 20), wake_message(0.9), *[QUIET] * 3, *[SPEECH] * 10, *[QUIET] * 3)
        await page.until(lambda: page.moves[-3:] == ["arm", "start", "stop"])
        # A frame after the end is what carries it to the clip owner, as at the desk.
        page.send(QUIET)
        assert await page.told_count(2) == [{"kind": "opened", "press": 1, "clip": 1}, {"kind": "milestone", "milestone": "released", "clip": 1}]
        [sound] = page.transcribed.given
        # The clip starts with the second the page kept before the wake, and the wake's turn was opened by no hand.
        assert sound.heard[: 2 * RATE] == BEFORE * SECOND and sound.seconds > 1.0
        [woke] = events(page.recorded, "phone.wake")
        assert (woke.facts["score"], woke.facts["threshold"], woke.facts["while_speaking"], woke.facts["opened"]) == (0.9, THRESHOLD, False, True)
        assert (page.key.gate.key, page.key.gate.opened, page.key.gate.ended) == ("listening", "phone wake word", ("sent",))
        [judged] = events(page.recorded, "phone.judged")
        assert judged.outcome == "ok"


async def test_a_wake_nothing_is_asked_after_is_discarded_and_the_page_is_told() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET)
        assert (await page.told_count(1))[0] == {"kind": "opened", "press": 1, "clip": 1}
        await page.due(voiceturn.ASK_SECONDS)
        await page.until(lambda: page.moves[-3:] == ["arm", "start", "drop"])
        page.send(QUIET)
        assert (await page.told_count(2))[1] == {"kind": "milestone", "milestone": "discarded", "clip": 1}
        assert page.key.gate.ended == ("dropped",)


async def test_the_turn_limit_sends_a_wakes_turn() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 5)
        await page.told_count(1)
        await page.due(shortcut.TURN_LIMIT_SECONDS, SPEECH)
        await page.until(lambda: page.moves[-3:] == ["arm", "start", "timeout"])
        page.send(QUIET)
        assert (await page.told_count(2))[1] == {"kind": "milestone", "milestone": "released", "clip": 1}
        assert page.key.gate.ended == ("sent",)


async def test_a_wake_while_hands_is_at_the_desk_opens_nothing_moves_hands_nowhere_and_is_answered_in_the_step_that_reads_it() -> None:
    async with up() as page:
        page.send(level_message("wake word"))
        page.key.go("desk")
        gate, queued = page.key.gate, page.phone.moves.qsize()
        page.send(wake_message(0.8))
        # Answered as it was read, with nothing else having run; no turn, and none for the driver to follow.
        assert page.told == [{"kind": "unopened", "press": 1, "why": "hands is at the desk"}]
        assert (page.key.gate, page.phone.moves.qsize(), events(page.recorded, "phone.turn")) == (gate, queued, [])
        [woke] = events(page.recorded, "phone.wake")
        assert (woke.facts["opened"], woke.outcome) == (False, "ok")


async def test_a_wake_with_a_turn_open_or_the_level_off_opens_nothing_is_answered_and_is_the_pages_mistake() -> None:
    async with up() as page:
        # Level off: the page has not told hands it listens.
        page.send(wake_message(0.8))
        assert page.told == [{"kind": "unopened", "press": 1, "why": "the phone is not listening"}]
        # A turn open: the button's own.
        page.send(level_message("wake word"), "press", SPEECH, wake_message(0.8))
        assert page.told[-1] == {"kind": "unopened", "press": 3, "why": "a turn is open"}
        [first, second] = events(page.recorded, "phone.wake")
        assert (first.outcome, second.outcome) == ("failed", "failed")
        assert first.error is not None and "level off" in first.error
        assert second.error is not None and "a turn open" in second.error
        # The next wake, once nothing is in the way, is no mistake and opens a turn: the button's turn goes first.
        page.send("release", QUIET)
        await page.until(lambda: page.key.gate.key == "listening")
        page.send(wake_message(0.8))
        assert page.key.gate.wake_turn is not None
        assert events(page.recorded, "phone.wake")[-1].outcome == "ok"


async def test_a_press_during_a_wakes_turn_sends_it_and_opens_the_buttons_own_as_two_clips() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 5, "press", *[SPEECH] * 5, "release", QUIET)
        done = await page.told_count(4)
        assert done[:4] == [
            {"kind": "opened", "press": 1, "clip": 1},
            {"kind": "milestone", "milestone": "released", "clip": 1},
            {"kind": "opened", "press": 2, "clip": 2},
            {"kind": "milestone", "milestone": "released", "clip": 2},
        ]
        assert page.key.gate.ended == ("sent", "sent")


async def test_level_off_during_a_wakes_turn_sends_it_and_the_key_rests_at_up() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 5, level_message("off"), QUIET)
        done = await page.told_count(2)
        assert done[1] == {"kind": "milestone", "milestone": "released", "clip": 1}
        assert (page.key.gate.key, page.key.gate.ended, page.key.gate.phone_level) == ("up", ("sent",), "off")


async def test_a_hand_at_the_desk_taking_hands_there_sends_a_wakes_turn_and_the_page_is_told_released() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 5)
        await page.told_count(1)
        # The desk's key pressed: hands goes to the desk and the turn open at the phone is sent.
        page.key.move("arm", "shortcut")
        page.key.move("start", "shortcut")
        assert (page.key.gate.place, page.key.gate.ended) == ("desk", ("sent",))
        page.at_the_desk()
        assert (await page.told_count(2))[1] == {"kind": "milestone", "milestone": "released", "clip": 1}


async def test_the_call_ending_during_a_wakes_turn_sends_it_and_the_phones_level_is_gone_with_the_call() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 5)
        await page.told_count(1)
        await page.phone.hang_up("hung up")
        assert (page.key.gate.place, page.key.gate.turn_open, page.key.gate.listening, page.key.gate.ended) == ("desk", False, frozenset(), ("sent",))


async def test_the_desks_level_and_the_phones_do_not_move_each_other() -> None:
    async with up() as page:
        page.key.move("listen", "wake word")
        page.send(level_message("wake word"))
        assert page.key.gate.listening == {"desk", "phone"}
        page.send(level_message("off"))
        assert page.key.gate.listening == {"desk"}
        page.key.move("deafen", "wake word")
        assert page.key.gate.listening == frozenset()


async def test_a_verdict_after_a_press_took_over_ends_nothing_of_the_buttons_turn() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 5, "press", *[SPEECH] * 5)
        await page.told_count(3)
        # Frames of the button's turn, none of them the wake's, are heard by the driver no more: it let the wake's go as the
        # gate ended it, and ended nothing of the button's.
        page.send(QUIET, QUIET)
        await page.until(lambda: page.moves[-2:] == ["stop", "start"] and page.key.gate.turn_open)
        assert (page.key.gate.opened, page.key.gate.ended) == ("phone button", ("sent",))
        [followed] = events(page.recorded, "phone.turn")
        assert (followed.facts["end"], followed.facts["by"]) == ("gate", "phone wake word")


async def test_after_a_refused_wake_or_a_desk_takeover_the_next_wake_opens_a_turn_and_is_no_mistake() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 3)
        await page.told_count(1)
        # A hand at the desk takes hands there and lets go, and a wake the phone hears there is refused.
        for move in ("arm", "start", "stop"):
            page.key.move(move, "shortcut")
        page.send(wake_message(0.9))
        assert page.told[-1] == {"kind": "unopened", "press": 2, "why": "hands is at the desk"}
        # The user presses the phone's button, which takes hands back, and lets go.
        page.send("press", SPEECH, "release")
        await page.until(lambda: page.key.gate.key == "listening")
        page.send(wake_message(0.9), QUIET, *[SPEECH] * 3, *[QUIET] * 3)
        await page.until(lambda: page.moves[-1] == "stop" and page.key.gate.key == "listening" and len(page.key.gate.ended) == 4)
        assert [(event.facts["opened"], event.outcome) for event in events(page.recorded, "phone.wake")] == [(True, "ok"), (False, "ok"), (True, "ok")]


async def test_the_detector_loading_or_failing_is_one_event_naming_the_browser() -> None:
    async with up() as page:
        page.send(detector_message("Safari 18", None), detector_message("Firefox 130", "no WebAssembly SIMD"))
        loaded, failed = events(page.recorded, "phone.detector")
        assert (loaded.facts["loaded"], loaded.facts["browser"], loaded.outcome) == (True, "Safari 18", "ok")
        assert (failed.facts["loaded"], failed.facts["browser"], failed.outcome, failed.error) == (False, "Firefox 130", "failed", "no WebAssembly SIMD")


async def test_a_page_sending_none_of_the_new_messages_gets_what_it_gets_today() -> None:
    async with up() as page:
        page.send(quiet_message(320), "press", SPEECH, SPEECH, "release", QUIET)
        done = await page.told_count(2)
        assert done[:2] == [{"kind": "opened", "press": 1, "clip": 1}, {"kind": "milestone", "milestone": "released", "clip": 1}]
        assert (events(page.recorded, "phone.wake"), events(page.recorded, "phone.turn")) == ([], [])


async def test_a_release_over_a_wakes_turn_ends_nothing_of_it() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 3)
        await page.told_count(1)
        # The page's button was never pressed: a release it sends is the page's, and the wake's turn is its wake's to end.
        page.send("release")
        assert (page.key.gate.turn_open, page.key.gate.opened, page.key.gate.ended) == (True, "phone wake word", ())


@pytest.mark.parametrize("score", [-0.1, 1.5, float("nan")])
def test_a_wake_whose_score_is_no_probability_is_read_as_a_wake_it_cannot_use(score: float) -> None:
    assert parse_said(wake_message(score), RATE) == UnreadableWake(f"its score is {score}, which is not a probability")


async def test_a_wake_with_no_score_is_counted_answered_and_a_failed_event_and_the_next_press_is_numbered_as_the_page_does() -> None:
    async with up() as page:
        page.send(level_message("wake word"), json.dumps({"kind": "wake", "score": None}), wake_message(0.9), QUIET, QUIET)
        assert (await page.told_count(2))[:2] == [{"kind": "unopened", "press": 1, "why": "its score was unreadable"}, {"kind": "opened", "press": 2, "clip": 1}]
        [bad, good] = events(page.recorded, "phone.wake")
        assert (bad.outcome, bad.facts["opened"], good.outcome) == ("failed", False, "ok")


async def test_hanging_up_over_a_wakes_turn_ends_what_the_driver_followed_with_no_frame_to_say_so() -> None:
    async with up() as page:
        page.send(level_message("wake word"), wake_message(0.9), QUIET, *[SPEECH] * 3)
        await page.told_count(1)
        await page.phone.hang_up("hung up")
        await page.until(lambda: bool(events(page.recorded, "phone.turn")))
        [followed] = events(page.recorded, "phone.turn")
        assert (followed.facts["end"], followed.facts["by"]) == ("gate", "call")
        assert page.key.gate.ended == ("sent",)


@pytest.mark.parametrize(
    ("message", "unreadable"),
    [
        ({"kind": "level", "level": "engaged"}, "neither off nor wake word"),
        ({"kind": "detector", "state": "loading", "browser": "Safari"}, "neither loaded nor failed"),
        ({"kind": "shout"}, "none hands reads"),
    ],
)
def test_a_level_a_detector_state_or_a_kind_hands_does_not_know_is_refused_as_unreadable(message: dict[str, str], unreadable: str) -> None:
    with pytest.raises(Rejected, match=unreadable):
        parse_said(json.dumps(message), RATE)


async def test_each_change_of_the_phones_level_is_one_event_and_a_level_it_already_has_is_none() -> None:
    async with up() as page:
        page.send(level_message("wake word"), level_message("wake word"), level_message("off"))
        assert [(change.facts["was"], change.facts["now"]) for change in events(page.recorded, "phone.leveled")] == [("off", "wake word"), ("wake word", "off")]
