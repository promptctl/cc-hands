"""The phone listening for its wake word: what a page sends of its detector and its level (`wake`, `level`, `detector`) moves the
gate in the step that reads it, and every press and wake of the page's is answered once, `opened` or `unopened`, and every
clip ended once. The calls are real ones made by aiortc over loopback, as in test_phone.py; the page's messages are those the
next ticket's page will send."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import pytest

from conftest import Running, events, running
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent
from hands.clock import SYSTEM
from hands.voice.clips import Clip, Sound
from hands.voice.clipowner import ClipOwner
from hands.voice.latency import LatencyObserver
from hands.voice.milestone import ClipOpened
from hands.voice.phone import DetectorFailed, DetectorLoaded, Level, Unreadable, Wake, parse_said
from hands.voice.ptt import KeyedAudio, PushToTalk
from hands.voice.trigger import PHONE_WAKE
from test_clipowner import Heeding, Scripted
from test_phone import LOUD, Call, calling, quiet_message

# The quiet the page sends outside a turn, and the second it keeps ahead of a wake.
QUIET = quiet_message(960)
KEPT = [LOUD] * 5


def wake_message(score: object = 0.93) -> str:
    return json.dumps({"kind": "wake", "score": score})


def level_message(level: str) -> str:
    return json.dumps({"kind": "level", "level": level})


@pytest.mark.parametrize(
    ("said", "read"),
    [
        (wake_message(0.93), Wake(0.93)),
        (wake_message(0), Wake(0.0)),
        (wake_message(1), Wake(1.0)),
        (level_message("wake word"), Level("wake word")),
        (level_message("off"), Level("off")),
        (json.dumps({"kind": "detector", "loaded": True, "browser": "Safari 26"}), DetectorLoaded("Safari 26")),
        (json.dumps({"kind": "detector", "loaded": False, "browser": "Safari 26", "why": "out of memory"}), DetectorFailed("Safari 26", "out of memory")),
        (json.dumps({"kind": "detector", "loaded": False, "browser": "Safari 26"}), DetectorFailed("Safari 26", "no reason given")),
    ],
)
def test_what_the_page_says_of_its_detector_and_its_level_is_read(said: str, read: object) -> None:
    assert parse_said(said, 48000) == read


@pytest.mark.parametrize("score", [-0.1, 1.5, "0.9", None, True, [0.9], float("nan"), float("inf")])
def test_a_wake_whose_score_is_no_probability_is_still_a_wake_the_page_numbered(score: object) -> None:
    # A wake the page counted is a number hands must answer, so it is read, as unreadable, and never refused outright.
    assert isinstance(parse_said(wake_message(score), 48000), Unreadable)


def test_a_wake_with_no_score_is_still_a_wake() -> None:
    assert isinstance(parse_said(json.dumps({"kind": "wake"}), 48000), Unreadable)


@pytest.mark.parametrize(
    "said",
    [level_message("engaged"), level_message(""), json.dumps({"kind": "level"}), json.dumps({"kind": "detector", "browser": "Safari"}), json.dumps({"kind": "detector", "loaded": True}), json.dumps({"kind": "detector", "loaded": "yes", "browser": "x"}), json.dumps({"kind": "hello"})],
)
def test_a_level_or_a_detector_hands_cannot_read_is_not_read(said: str) -> None:
    with pytest.raises(Rejected):
        parse_said(said, 48000)


@dataclass
class Heard:
    """The call's microphone fed to a clip owner whose observer tells the page, with a listener that keeps what each clip opened with."""

    frames: list[KeyedAudio]
    heeding: Heeding
    run: Running

    async def at_the_desk(self, key: PushToTalk) -> None:
        """A frame of the desk's microphone under the gate as it stands, which is what the pipeline hears once hands is at the desk."""
        await self.run.worker.queue_frame(key.gate.framed(bytes(640), bytes(640), None, 16000, 1, "desk"))


@asynccontextmanager
async def hearing(call: Call) -> AsyncGenerator[Heard]:
    heeding = Heeding("kept")
    owner = ClipOwner(SYSTEM, Scripted([None, None, None]), (heeding,), call.recorded.append)
    given: list[KeyedAudio] = []
    async with running([owner], [LatencyObserver(SYSTEM, owner, call.phone.tell)]) as run:

        async def feed() -> None:
            while True:
                given.append(await call.phone.heard.get())
                await run.worker.queue_frame(given[-1])

        feeding = asyncio.create_task(feed())
        try:
            yield Heard(given, heeding, run)
        finally:
            feeding.cancel()


def told(call: Call) -> list[dict[str, object]]:
    return [json.loads(cast(str, each)) for each in call.page.told]


def wakes(call: Call) -> list[WideEvent]:
    return events(call.recorded, "phone.wake")


@pytest.mark.outside
async def test_a_wake_opens_a_turn_at_the_phone_made_by_no_hand_that_starts_with_the_second_sent_before_it() -> None:
    async with calling() as call, hearing(call) as heard:
        for said in [level_message("wake word"), QUIET, *KEPT, QUIET]:
            call.page.send(said)
        await call.until(lambda: call.key.gate.key == "listening")
        for said in [wake_message(), LOUD, LOUD]:
            call.page.send(said)
        await call.until(lambda: len(call.page.told) == 1)
        assert told(call) == [{"kind": "opened", "press": 1, "clip": 1}]
        assert (call.key.gate.key, call.key.gate.opened, call.key.gate.place) == ("down", PHONE_WAKE, "phone")
        # The tones follow the moves the gate made, as the button's do: the press arms, and the turn starts.
        assert [call.phone.moves.get_nowait() for _ in range(3)] == ["listen", "arm", "start"]
        # The clip was opened with what the phone sent while it listened: the kept second, before the wake.
        match heard.heeding.given[0]:
            case ("opened", (Clip() as clip, Sound() as before)):
                assert clip == Clip(1, PHONE_WAKE) and any(before.heard)
            case other:
                pytest.fail(f"the listener was not first told of an opened clip: {other}")
        [event] = wakes(call)
        assert (event.outcome, event.facts) == ("ok", {"score": 0.93, "threshold": 0.5, "while_speaking": False, "opened": True, "press": 1})


@pytest.mark.outside
async def test_a_wake_while_hands_is_at_the_desk_opens_nothing_and_the_next_one_at_the_phone_does() -> None:
    async with calling() as call:
        call.page.send(level_message("wake word"))
        await call.until(lambda: call.key.gate.phone_level == "wake word")
        # A hand at the desk takes hands there: the call is up, and hands is not at it.
        call.key.move("start", "shortcut")
        call.key.end(call.key.gate.turn, "stop", "shortcut")
        call.page.send(wake_message())
        await call.until(lambda: len(call.page.told) == 1)
        assert told(call) == [{"kind": "unopened", "press": 1, "why": "hands is at the desk"}]
        assert (call.key.gate.place, call.key.gate.turn_open, call.key.gate.ended) == ("desk", False, ("sent",))
        # Hands being at the desk is no mistake of the page's, for the page cannot know it.
        [refused] = wakes(call)
        assert (refused.outcome, refused.facts["opened"]) == ("ok", False)
        # The button brings hands back, and the next wake is a turn and no mistake.
        for said in ["press", "release"]:
            call.page.send(said)
        await call.until(lambda: call.key.gate.place == "phone" and call.key.gate.turn == 2)
        call.page.send(wake_message())
        await call.until(lambda: call.key.gate.turn_open)
        assert (call.key.gate.opened, [event.outcome for event in wakes(call)]) == (PHONE_WAKE, ["ok", "ok"])


@pytest.mark.outside
async def test_a_wake_at_off_or_over_an_open_turn_opens_nothing_and_is_the_pages_mistake() -> None:
    async with calling() as call:
        call.page.send(wake_message())
        await call.until(lambda: len(call.page.told) == 1)
        assert told(call) == [{"kind": "unopened", "press": 1, "why": "the phone is not listening for the wake word"}]
        call.page.send(level_message("wake word"))
        call.page.send("press")
        await call.until(lambda: call.key.gate.key == "down")
        call.page.send(wake_message())
        await call.until(lambda: len(wakes(call)) == 2)
        assert call.key.gate.opened == "phone button" and call.key.gate.ended == ()
        assert [(event.outcome, event.facts["opened"]) for event in wakes(call)] == [("failed", False), ("failed", False)]
        assert [event.error for event in wakes(call)] == [
            "the page's wake opened nothing: the phone is not listening for the wake word",
            "the page's wake opened nothing: a turn is open",
        ]


@pytest.mark.outside
async def test_a_wake_whose_score_is_unreadable_is_counted_answered_unopened_and_failed_and_the_next_press_keeps_its_number() -> None:
    async with calling() as call:
        call.page.send(level_message("wake word"))
        call.page.send(wake_message("loud"))
        call.page.send("press")
        await call.until(lambda: call.key.gate.key == "down")
        [event] = wakes(call)
        assert (event.outcome, event.facts["score"], event.facts["opened"]) == ("failed", None, False)
        assert call.phone.moves.get_nowait() == "listen"
        # The press is the page's second: it is told under that number.
        call.phone.tell(ClipOpened(Clip(1, "phone button"), call.key.gate.turn))
        await call.until(lambda: len(call.page.told) == 2)
        assert told(call)[0]["press"] == 1 and told(call)[1] == {"kind": "opened", "press": 2, "clip": 1}


@pytest.mark.outside
async def test_the_phone_listens_where_its_page_says_and_each_change_is_an_event_and_a_call_starts_at_off() -> None:
    async with calling() as call:
        assert call.key.gate.at_phone == "off"
        for said in [level_message("wake word"), level_message("wake word"), level_message("off"), level_message("wake word")]:
            call.page.send(said)
        await call.until(lambda: len(events(call.recorded, "phone.leveled")) == 3)
        assert (call.key.gate.key, call.key.gate.at_phone) == ("listening", "wake word")
        assert [event.facts for event in events(call.recorded, "phone.leveled")] == [{"was": "off", "now": "wake word", "by": "level"}, {"was": "wake word", "now": "off", "by": "level"}, {"was": "off", "now": "wake word", "by": "level"}]
        # The call ending takes the listening with it, and a call that comes starts at off.
        await call.phone.hang_up("hung up")
        assert call.key.gate.phone_level == "off"


@pytest.mark.outside
async def test_the_desks_level_and_the_phones_do_not_move_each_other() -> None:
    async with calling() as call:
        # The desk at its wake word, the phone at off: the key rests at up while hands is at the phone.
        call.key.move("listen", "wake word")
        assert (call.key.gate.key, call.key.gate.phone_level) == ("up", "off")
        call.page.send(level_message("wake word"))
        await call.until(lambda: call.key.gate.key == "listening")
        call.page.send(level_message("off"))
        await call.until(lambda: call.key.gate.key == "up")
        # Hands is back at the desk and the desk is listening again.
        await call.phone.hang_up("hung up")
        assert call.key.gate.key == "listening"


@pytest.mark.outside
async def test_the_level_going_off_during_a_wakes_turn_sends_it_and_the_page_is_told_released() -> None:
    async with calling() as call, hearing(call):
        call.page.send(level_message("wake word"))
        for said in [QUIET, wake_message(), LOUD, LOUD]:
            call.page.send(said)
        await call.until(lambda: call.key.gate.key == "down")
        call.page.send(level_message("off"))
        call.page.send(QUIET)
        await call.until(lambda: len(call.page.told) == 2)
        assert told(call) == [{"kind": "opened", "press": 1, "clip": 1}, {"kind": "milestone", "milestone": "released", "clip": 1}]
        assert (call.key.gate.key, call.key.gate.ended) == ("up", ("sent",))


@pytest.mark.outside
async def test_a_press_over_a_wakes_turn_sends_it_and_opens_the_buttons_own_each_answered_and_ended_under_its_own_number() -> None:
    async with calling() as call, hearing(call):
        call.page.send(level_message("wake word"))
        for said in [QUIET, wake_message(), LOUD, "press", LOUD, "release", QUIET]:
            call.page.send(said)
        await call.until(lambda: len(call.page.told) == 4)
        assert told(call) == [
            {"kind": "opened", "press": 1, "clip": 1},
            {"kind": "milestone", "milestone": "released", "clip": 1},
            {"kind": "opened", "press": 2, "clip": 2},
            {"kind": "milestone", "milestone": "released", "clip": 2},
        ]
        assert call.key.gate.ended == ("sent", "sent")


@pytest.mark.outside
async def test_a_hand_at_the_desk_taking_hands_during_a_wakes_turn_sends_it_and_the_page_is_told_released() -> None:
    async with calling() as call, hearing(call) as heard:
        call.page.send(level_message("wake word"))
        for said in [QUIET, wake_message(), LOUD, LOUD]:
            call.page.send(said)
        await call.until(lambda: call.key.gate.key == "down")
        call.key.move("start", "shortcut")
        await heard.at_the_desk(call.key)
        await call.until(lambda: len(call.page.told) == 2)
        assert told(call)[1] == {"kind": "milestone", "milestone": "released", "clip": 1}
        assert call.key.gate.place == "desk"


@pytest.mark.outside
async def test_the_call_ending_during_a_wakes_turn_sends_it_and_tells_the_page_nothing() -> None:
    async with calling() as call:
        call.page.send(level_message("wake word"))
        for said in [QUIET, wake_message(), LOUD]:
            call.page.send(said)
        await call.until(lambda: call.key.gate.key == "down")
        await call.phone.hang_up("hung up")
        assert (call.key.gate.place, call.key.gate.ended) == ("desk", ("sent",))
        assert told(call) == []


@pytest.mark.outside
async def test_a_detector_that_loaded_or_failed_is_an_event_with_the_browser_the_page_is_in() -> None:
    async with calling() as call:
        call.page.send(json.dumps({"kind": "detector", "loaded": True, "browser": "Safari 26"}))
        call.page.send(json.dumps({"kind": "detector", "loaded": False, "browser": "Safari 26", "why": "out of memory"}))
        await call.until(lambda: len(events(call.recorded, "phone.detector")) == 2)
        loaded, failed = events(call.recorded, "phone.detector")
        assert (loaded.outcome, loaded.facts) == ("ok", {"browser": "Safari 26", "loaded": True})
        assert (failed.outcome, failed.error, failed.facts) == ("failed", "out of memory", {"browser": "Safari 26", "loaded": False})
