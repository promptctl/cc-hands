"""The push-to-talk decisions, with no pipeline and no audio device."""

import pytest

from conftest import events
from hands.virtualclock import VirtualClock
from hands.voice.shortcut import Move
from hands.sessions.audit import Entry
from hands.voice.ptt import Gate, Key, PushToTalk
from hands.voice.trigger import Trigger
from hands.voice.tools import modality_tool

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160


def test_starts_up_and_silent() -> None:
    gate = Gate()
    assert gate.key == "up"
    assert gate.audible(LOUD) == QUIET


def test_a_started_turn_is_heard() -> None:
    gate = Gate().after("start", "shortcut")
    assert gate.audible(LOUD) == LOUD


@pytest.mark.parametrize("ended", ["stop", "drop", "timeout"])
def test_an_ended_turn_is_silence_of_the_same_length(ended: Move) -> None:
    gate = Gate().after("start", "shortcut").after(ended, "shortcut")
    assert gate.audible(LOUD) == QUIET


def test_a_press_is_heard_before_it_means_talk_and_silence_once_it_is_shift() -> None:
    assert Gate().after("arm", "shortcut").audible(LOUD) == LOUD
    assert Gate().after("arm", "shortcut").after("disarm", "shortcut").audible(LOUD) == QUIET


def test_every_turn_is_counted_sent_or_thrown_away_and_the_key_rests_after_either() -> None:
    assert Gate().after("start", "shortcut").after("stop", "shortcut") == Gate("up", "desk", sent=1)
    assert Gate().after("start", "shortcut").after("drop", "shortcut") == Gate("up", "desk", dropped=1)
    assert Gate().after("start", "shortcut").after("timeout", "shortcut") == Gate("up", "desk", sent=1)  # sent, as it stood
    # A turn ended and the next opened before a frame is captured: the count says the first was sent.
    assert Gate().after("start", "shortcut").after("stop", "shortcut").after("arm", "shortcut").after("start", "shortcut") == Gate("down", "desk", sent=1)


def test_an_engaged_desk_is_heard_between_turns_and_each_turn_ends_back_to_listening() -> None:
    engaged = Gate().after("listen", "engaged conversation")
    assert (engaged.key, engaged.listens, engaged.turn_open) == ("listening", True, False)
    assert engaged.audible(LOUD) == LOUD
    turn = engaged.after("arm", "engaged conversation").after("start", "engaged conversation")
    assert turn.turn_open
    assert turn.after("stop", "engaged conversation").key == "listening"
    assert engaged.after("arm", "engaged conversation").after("disarm", "engaged conversation").key == "listening"
    # And listens on; the turn the limit sent was opened by the voice.
    assert turn.after("timeout", "engaged conversation") == Gate("listening", "desk", listens=True, sent=1, opened="engaged conversation")


@pytest.mark.parametrize(("before", "after"), [("listening", "up"), ("arming", "arming"), ("down", "down")])
def test_disengaging_stops_the_desk_listening_and_leaves_a_press_or_a_turn_to_end_as_it_ends(before: Key, after: Key) -> None:
    gate = Gate(before, "desk", listens=True).after("deafen", "engaged conversation")
    assert (gate.key, gate.listens) == (after, False)


def test_the_phone_never_listens_between_turns_and_the_desk_does_again_once_hands_is_back() -> None:
    at_the_phone = Gate().after("listen", "engaged conversation").moved("phone")
    assert (at_the_phone.key, at_the_phone.audible(LOUD)) == ("up", QUIET)
    assert at_the_phone.after("start", "phone button").after("stop", "phone button").key == "up"
    assert at_the_phone.moved("desk").key == "listening"
    # Engaged while hands is at the phone: the desk listens once hands is back, and the call hears nothing meanwhile.
    engaged_away = Gate().moved("phone").after("listen", "engaged conversation")
    assert (engaged_away.key, engaged_away.place, engaged_away.listens) == ("up", "phone", True)
    assert engaged_away.moved("desk").key == "listening"


def test_a_turn_opened_at_the_other_place_moves_the_gate_there() -> None:
    gate = Gate().after("start", "phone button")
    assert (gate.key, gate.place) == ("down", "phone")
    assert gate.hears("phone") and not gate.hears("desk")


@pytest.mark.parametrize("move", ["arm", "disarm", "stop", "drop", "timeout"])
def test_the_other_place_cannot_touch_a_turn_but_by_opening_one(move: Move) -> None:
    # A Shift typed at the desk while the user talks on the phone.
    on_the_phone = Gate().after("start", "phone button")
    assert on_the_phone.after(move, "shortcut") == on_the_phone


def test_leaving_a_place_mid_clip_sends_the_clip() -> None:
    assert Gate().after("start", "phone button").moved("desk") == Gate("up", "desk", sent=1, opened="phone button")
    assert Gate().after("arm", "shortcut").moved("phone") == Gate("up", "phone")  # no turn open, so none sent
    assert Gate().moved("phone") == Gate("up", "phone")
    assert Gate("down", "phone").moved("phone") == Gate("down", "phone")


def test_every_move_of_the_place_is_recorded_once_by_what_made_it() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    key.move("arm", "shortcut")
    key.move("start", "shortcut")
    key.go("phone")  # a call arrives while the user talks at the desk
    key.move("start", "phone button")
    key.move("stop", "phone button")
    key.move("start", "shortcut")
    key.move("stop", "shortcut")
    assert [move.facts for move in events(recorded, "place.moved")] == [
        {"before": "desk", "after": "phone", "by": "call", "sent": True},
        {"before": "phone", "after": "desk", "by": "turn", "sent": False},
    ]
    assert all(move.outcome == "ok" for move in events(recorded, "place.moved")) and len(recorded) == 2


def test_a_press_at_one_place_while_the_other_is_still_arming_opens_the_turn_and_sends_nothing() -> None:
    # Right Shift down at the desk, not yet meaning talk, and the phone's button pressed: the press may be Shift.
    gate = Gate("arming", "desk")
    assert gate.took("start", "phone button") == "start"
    assert gate.after("start", "phone button") == Gate("down", "phone", opened="phone button")


def test_a_hand_that_opens_a_turn_while_a_clip_the_voice_opened_is_open_at_the_other_place_sends_that_clip_and_opens_its_own() -> None:
    # Engaged conversation heard the user at the desk as they pressed the phone's button: both are the user's.
    gate = Gate("down", "desk", opened="engaged conversation")
    assert gate.took("start", "phone button") == "start"
    assert gate.after("start", "phone button") == Gate("down", "phone", sent=1, opened="phone button")
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    key.move("start", "wake word")
    assert key.move("start", "phone button") == "start"
    assert events(recorded, "place.moved")[-1].facts == {"before": "desk", "after": "phone", "by": "turn", "sent": True}


@pytest.mark.parametrize("voice", ["engaged conversation", "wake word"])
def test_a_voice_the_desk_hears_never_ends_a_clip_open_at_the_phone(voice: Trigger) -> None:
    # The user talks into the phone near the Mac, or someone in the room does: the phone's clip goes on, and hands stays.
    held = Gate("down", "phone", opened="phone button")
    assert held.took("arm", voice) is None
    assert held.took("start", voice) is None
    assert held.after("start", voice) == held


@pytest.mark.parametrize(("held", "pressed"), [(Gate("down", "phone", opened="phone button"), "shortcut"), (Gate("down", "desk", opened="shortcut"), "phone button")])
def test_a_clip_a_hand_holds_open_is_ended_by_no_hand_at_the_other_place(held: Gate, pressed: Trigger) -> None:
    assert held.took("start", pressed) is None
    assert held.after("start", pressed) == held


def test_a_turn_opened_where_a_clip_is_open_goes_on_as_the_same_clip() -> None:
    # The page sent its press twice before its release: one clip, never split in two.
    held = Gate("down", "phone", opened="phone button")
    assert held.took("start", "phone button") == "start"
    assert held.after("start", "phone button") == held


@pytest.mark.parametrize("ending", ["stop", "drop", "timeout"])
def test_the_end_of_a_clip_the_gate_already_sent_ends_nothing_more(ending: Move) -> None:
    # The desk's turn was sent as a call came, and the key now comes up: nothing is sent twice, nothing cued twice.
    sent = Gate("up", "desk", sent=1)
    assert sent.took(ending, "shortcut") is None
    assert sent.after(ending, "shortcut") == sent
    assert sent.took("arm", "shortcut") == "arm"


def test_the_other_place_moves_nothing_and_the_gate_says_so() -> None:
    on_the_phone = Gate("down", "phone")
    assert on_the_phone.took("stop", "shortcut") is None
    assert on_the_phone.took("stop", "phone button") == "stop"


def test_the_way_the_user_talks_sets_whether_they_can_see_a_screen() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    assert key.modality == "screen"
    key.go("phone")  # a call arrives
    assert key.modality == "audio-only"
    key.move("start", "shortcut")  # a turn opened at the desk while the call is up
    assert key.modality == "screen"


def test_a_turn_is_known_by_the_trigger_that_opened_it_until_another_opens() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    key.go("phone")  # a call arrives: it opens no turn
    assert key.opened == "shortcut"
    key.move("start", "phone button")
    key.move("stop", "phone button")
    assert key.opened == "phone button"
    key.go("desk")
    key.move("arm", "shortcut")  # a press still arming opens nothing
    key.move("disarm", "shortcut")
    assert key.opened == "phone button"
    key.move("start", "shortcut")
    assert key.opened == "shortcut"


def test_a_switch_by_voice_holds_until_hands_moves_between_the_desk_and_the_phone() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    key.switch("audio-only")
    # Turns at the place it was switched at keep it.
    key.move("start", "shortcut")
    key.move("stop", "shortcut")
    assert key.modality == "audio-only"
    key.go("phone")
    key.switch("screen")
    key.move("start", "phone button")
    assert key.modality == "screen"
    key.go("desk")
    assert key.modality == "screen"
    key.go("phone")
    assert key.modality == "audio-only"


async def test_set_modality_switches_it_says_so_and_refuses_what_is_neither() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    switch = modality_tool(key.switch)
    assert await switch.body(modality="audio-only") == {"modality": "audio-only", "readback": "Okay, audio only."}
    assert key.modality == "audio-only"
    assert await switch.body(modality="screen") == {"modality": "screen", "readback": "Okay, you can see a screen."}
    assert key.modality == "screen"
    assert "error" in await switch.body(modality="video")
    assert key.modality == "screen"
