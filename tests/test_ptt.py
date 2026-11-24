"""The push-to-talk decisions, with no pipeline and no audio device."""

import pytest

from conftest import events
from hands.virtualclock import VirtualClock
from hands.core.place import Place
from hands.voice.shortcut import Move
from hands.sessions.audit import Entry
from hands.voice.ptt import Gate, Key, PushToTalk, TurnEnded
from hands.voice.tools import modality_tool
from hands.voice.trigger import Trigger, place_of
from hands.voice.voiceturn import TurnId

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160
DESK: frozenset[Place] = frozenset({"desk"})
PHONE: frozenset[Place] = frozenset({"phone"})


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
    assert Gate().after("start", "shortcut").after("stop", "shortcut") == Gate("up", "desk", ended=("sent",))
    assert Gate().after("start", "shortcut").after("drop", "shortcut") == Gate("up", "desk", ended=("dropped",))
    assert Gate().after("start", "shortcut").after("timeout", "shortcut") == Gate("up", "desk", ended=("sent",))  # sent, as it stood
    # A turn ended and the next opened before a frame is captured: the count says the first was sent.
    assert Gate().after("start", "shortcut").after("stop", "shortcut").after("arm", "shortcut").after("start", "shortcut") == Gate("down", "desk", ended=("sent",))


def test_an_engaged_desk_is_heard_between_turns_and_each_turn_ends_back_to_listening() -> None:
    engaged = Gate().after("listen", "engaged conversation")
    assert (engaged.key, engaged.listening, engaged.turn_open) == ("listening", DESK, False)
    assert engaged.audible(LOUD) == LOUD
    turn = engaged.after("arm", "engaged conversation").after("start", "engaged conversation")
    assert turn.turn_open
    assert turn.after("stop", "engaged conversation").key == "listening"
    assert engaged.after("arm", "engaged conversation").after("disarm", "engaged conversation").key == "listening"
    # And listens on; the turn the limit sent was opened by the voice.
    assert turn.after("timeout", "engaged conversation") == Gate("listening", "desk", listening=DESK, ended=("sent",), opened="engaged conversation")


@pytest.mark.parametrize(("before", "after"), [("listening", "up"), ("arming", "arming"), ("down", "down")])
def test_disengaging_stops_the_desk_listening_and_leaves_a_press_or_a_turn_to_end_as_it_ends(before: Key, after: Key) -> None:
    gate = Gate(before, "desk", listening=DESK).after("deafen", "engaged conversation")
    assert (gate.key, gate.listening) == (after, frozenset())


def test_the_phone_never_listens_between_turns_and_the_desk_does_again_once_hands_is_back() -> None:
    at_the_phone = Gate().after("listen", "engaged conversation").moved("phone")
    assert (at_the_phone.key, at_the_phone.audible(LOUD)) == ("up", QUIET)
    assert at_the_phone.after("start", "phone button").after("stop", "phone button").key == "up"
    assert at_the_phone.moved("desk").key == "listening"
    # Engaged while hands is at the phone: the desk listens once hands is back, and the call hears nothing meanwhile.
    engaged_away = Gate().moved("phone").after("listen", "engaged conversation")
    assert (engaged_away.key, engaged_away.place, engaged_away.listening) == ("up", "phone", DESK)
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


def test_leaving_a_place_mid_clip_sends_the_clip_as_it_stands() -> None:
    assert Gate().after("start", "phone button").moved("desk") == Gate("up", "desk", ended=("sent",), opened="phone button")
    assert Gate().after("arm", "shortcut").moved("phone") == Gate("up", "phone")  # no turn open, so none sent
    assert Gate().moved("phone") == Gate("up", "phone")
    assert Gate("down", "phone").moved("phone") == Gate("down", "phone")


@pytest.mark.parametrize("held", [Gate("arming", "desk"), Gate().after("start", "shortcut"), Gate("listening", listening=DESK).after("arm", "wake word").after("start", "wake word")])
def test_a_call_that_comes_never_cuts_off_a_press_or_a_clip_at_the_desk(held: Gate) -> None:
    # The page called again by itself while the user talks at the desk: hands stays, and the phone's first press takes it.
    assert held.called("phone") == held
    assert held.called("phone").after("start", "phone button").place == "phone"


def test_a_call_that_goes_sends_its_clip_and_one_that_comes_to_a_quiet_desk_takes_hands() -> None:
    assert Gate().after("start", "phone button").called("desk") == Gate("up", "desk", ended=("sent",), opened="phone button")
    assert Gate(listening=DESK).after("listen", "wake word").called("phone") == Gate("up", "phone", listening=DESK)


def test_every_move_of_the_place_is_recorded_once_by_what_made_it() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    key.move("arm", "shortcut")
    key.move("start", "shortcut")
    key.go("phone")  # a call arrives while the user talks at the desk: hands stays for the clip
    key.move("start", "phone button")
    key.move("stop", "phone button")
    key.go("desk")
    assert [move.facts for move in events(recorded, "place.moved")] == [
        {"before": "desk", "after": "phone", "by": "turn", "sent": True},
        {"before": "phone", "after": "desk", "by": "call", "sent": False},
    ]
    assert all(move.outcome == "ok" for move in events(recorded, "place.moved")) and len(recorded) == 2


@pytest.mark.parametrize("held", ["arming", "down"])
def test_a_hand_at_the_other_place_takes_hands_there_and_sends_the_clip_open_where_it_was(held: Key) -> None:
    # Right Shift held at the desk, and the phone's button pressed: the user's hand is at the phone now.
    gate = Gate(held, "desk")
    assert gate.took("start", "phone button") == "start"
    assert gate.after("start", "phone button") == Gate("down", "phone", ended=("sent",) if held == "down" else (), opened="phone button")
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    key.move("start", "phone button")
    assert key.move("start", "shortcut") == "start"
    assert events(recorded, "place.moved")[-1].facts == {"before": "phone", "after": "desk", "by": "turn", "sent": True}


@pytest.mark.parametrize("voice", ["engaged conversation", "wake word"])
def test_a_voice_the_desk_hears_never_takes_hands_from_the_phone(voice: Trigger) -> None:
    # Anyone's or nobody's: the TV, or a wake word said in the room while the user talks on the phone.
    for on_the_phone in (Gate("down", "phone", opened="phone button"), Gate("up", "phone")):
        assert on_the_phone.took("start", voice) is None
        assert on_the_phone.after("start", voice) == on_the_phone


def test_a_start_where_hands_is_keeps_the_clip_open() -> None:
    # The talk key pressed during a wake word's turn: one clip, never split, and the hand's from here.
    assert Gate().after("start", "wake word").after("start", "shortcut") == Gate("down", "desk", opened="shortcut")


@pytest.mark.parametrize("ending", ["stop", "timeout"])
def test_the_end_of_a_clip_the_gate_already_sent_ends_nothing_more(ending: Move) -> None:
    # The phone's clip was sent as the talk key took hands to the desk, and the phone's button now comes up.
    taken = Gate("down", "phone").after("start", "shortcut")
    assert taken.took(ending, "phone button") is None
    assert taken.after(ending, "phone button") == taken
    # And at the desk, a clip sent ends nothing more as the key is let go.
    sent = Gate("up", "desk", ended=("sent",))
    assert sent.took(ending, "shortcut") is None
    assert sent.took("arm", "shortcut") == "arm"


def test_the_other_place_moves_nothing_and_the_gate_says_so() -> None:
    on_the_phone = Gate("down", "phone")
    assert on_the_phone.took("stop", "shortcut") is None
    assert on_the_phone.took("start", "phone button") == "start"


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


def listening_at_the_phone() -> Gate:
    return Gate().moved("phone").after("listen", "phone wake word")


def test_the_phone_listens_between_turns_at_its_own_level_and_a_wake_opens_a_turn_that_ends_back_to_listening() -> None:
    waiting = listening_at_the_phone()
    assert (waiting.key, waiting.listening, waiting.audible(LOUD)) == ("listening", PHONE, LOUD)
    turn = waiting.after("arm", "phone wake word").after("start", "phone wake word")
    assert (turn.turn_open, turn.place, turn.opened, turn.wake_turn) == (True, "phone", "phone wake word", 0)
    assert turn.after("stop", "phone wake word") == Gate("listening", "phone", listening=PHONE, ended=("sent",), opened="phone wake word")
    assert waiting.after("deafen", "phone wake word").key == "up"


def test_the_desks_level_and_the_phones_never_move_each_other() -> None:
    at_the_desk = Gate(listening=DESK).after("listen", "wake word")
    assert at_the_desk.after("deafen", "phone wake word").key == "listening"
    assert at_the_desk.moved("phone").key == "up"
    assert at_the_desk.moved("phone").after("listen", "phone wake word").key == "listening"
    # The phone at its wake word, hands at the desk: the desk's key rests as the desk's level says, and the phone's is waiting.
    away = listening_at_the_phone().moved("desk")
    assert (away.key, away.listening) == ("up", PHONE)
    assert away.moved("phone").key == "listening"
    assert away.after("listen", "wake word").after("deafen", "wake word").listening == PHONE


@pytest.mark.parametrize("to", ["phone", "desk"])
def test_a_call_that_comes_or_goes_leaves_the_phones_level_off(to: Place) -> None:
    held = Gate(listening=DESK | PHONE).moved("phone")
    assert held.called(to).listening == DESK


def test_a_wake_heard_on_the_phone_opens_nothing_while_hands_is_at_the_desk_and_moves_hands_nowhere() -> None:
    for at_the_desk in (Gate(), Gate(listening=DESK), Gate("down", "desk", opened="shortcut")):
        assert at_the_desk.took("arm", "phone wake word") is None
        assert at_the_desk.took("start", "phone wake word") is None
        assert at_the_desk.after("start", "phone wake word") == at_the_desk


def test_the_turn_a_wake_opened_is_ended_by_its_own_number_and_by_nothing_stale() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    key.go("phone")
    key.move("listen", "phone wake word")
    key.move("arm", "phone wake word")
    key.move("start", "phone wake word")
    assert key.gate.wake_turn == 0
    # A turn that is not the one the gate is at, and a trigger that did not open this one, are none of it to end.
    assert key.end(TurnId(1), "stop", "phone wake word") is None
    assert key.end(TurnId(0), "stop", "phone button") is None and key.gate.wake_turn == 0
    assert key.end(TurnId(0), "stop", "phone wake word") == "stop"
    # The same verdict again, now that the turn is over, ends nothing; and nor does it end the button's turn that took its place.
    assert key.end(TurnId(0), "stop", "phone wake word") is None
    key.move("start", "phone button")
    assert (key.end(TurnId(0), "stop", "phone wake word"), key.gate.turn_open, key.gate.wake_turn) == (None, True, None)
    assert key.end(TurnId(1), "stop", "phone button") == "stop"


def test_the_gate_names_the_turn_a_wake_has_open_and_no_other() -> None:
    waiting = listening_at_the_phone()
    turn = waiting.after("arm", "phone wake word").after("start", "phone wake word")
    assert [gate.wake_turn for gate in (waiting, turn, turn.after("stop", "phone wake word"))] == [None, 0, None]
    assert Gate().moved("phone").after("start", "phone button").wake_turn is None


@pytest.mark.parametrize(
    ("gate", "refusal"),
    [
        (Gate("listening", listening=DESK), None),
        (Gate("up"), "deaf"),
        (Gate("arming"), "busy"),
        (Gate("down", opened="shortcut"), "busy"),
        (Gate("listening", "desk", listening=DESK | PHONE), None),
    ],
)
def test_a_voice_may_open_a_turn_only_at_the_place_hands_is_at_and_listens(gate: Gate, refusal: str | None) -> None:
    assert gate.refusal("wake word") == refusal
    # Where hands is at the other place, the voice of this one opens nothing; the gate says so by saying where.
    assert gate.refusal("phone wake word") == "elsewhere"


@pytest.mark.parametrize("voice", ["wake word", "engaged conversation", "phone wake word"])
def test_a_voice_arms_a_turn_only_from_where_it_listens(voice: Trigger) -> None:
    place = place_of(voice)
    listening = Gate("listening", place, listening=frozenset({place}))
    armed = listening.after("arm", voice)
    assert (armed.key, armed.after("start", voice).key) == ("arming", "down")
    # Deaf, or with a press or a turn open, the voice arms nothing; and it starts only what is armed, so it opens no turn
    # where it was refused, and takes none from a hand that has one open.
    for refused in (Gate("up", place), Gate("arming", place), Gate("down", place, opened="shortcut")):
        assert refused.took("arm", voice) is None
    for unarmed in (Gate("up", place), Gate("listening", place), Gate("down", place, opened="shortcut")):
        assert unarmed.took("start", voice) is None and unarmed.after("start", voice) == unarmed
    # A hand opens a turn as it always has.
    assert Gate("up", "desk").took("start", "shortcut") == "start"


def test_a_clip_is_ended_by_the_trigger_that_opened_it() -> None:
    # Each pair is of one place, so the place is no part of the refusal.
    cases: tuple[tuple[Trigger, Trigger], ...] = (("phone wake word", "phone button"), ("phone button", "phone wake word"), ("wake word", "shortcut"), ("shortcut", "engaged conversation"))
    ends: tuple[Move, ...] = ("stop", "drop", "timeout")
    for opened, ender in cases:
        down = Gate("down", place_of(opened), opened=opened)
        for end in ends:
            assert (down.took(end, ender), down.took(end, opened)) == (None, end)


def test_the_gate_tells_those_who_watch_each_turn_it_ends_in_order_and_what_ended_it() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    told: list[TurnEnded] = []
    key.watch(told.append)
    key.move("start", "shortcut")
    key.move("stop", "shortcut")
    key.move("start", "shortcut")
    key.move("drop", "shortcut")
    key.move("start", "shortcut")
    # A hand at the other place takes hands there and sends the clip open, and a call that goes sends the one it has.
    key.move("start", "phone button")
    key.go("desk")
    assert told == [
        TurnEnded(TurnId(0), "sent", "shortcut"),
        TurnEnded(TurnId(1), "dropped", "shortcut"),
        TurnEnded(TurnId(2), "sent", "phone button"),
        TurnEnded(TurnId(3), "sent", "call"),
    ]
    # A move that ends nothing tells no one.
    key.move("stop", "shortcut")
    assert len(told) == 4
