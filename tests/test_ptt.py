"""The push-to-talk decisions, with no pipeline and no audio device."""

import itertools
from typing import cast, get_args

import pytest

from conftest import events
from hands.core.place import Place
from hands.virtualclock import VirtualClock
from hands.voice.clips import Fate, Trigger as TriggerType
from hands.voice.shortcut import Beginning, Ending, Move
from hands.sessions.audit import Entry
from hands.voice.ptt import Gate, GateEnded, GateOpened, Open, Position, Pressing, PushToTalk, Resting
from hands.voice.tools import modality_tool
from hands.voice.trigger import Trigger, by_hand, place_of
from hands.voice.voiceturn import TurnId

LOUD = b"\x7f\x7f" * 160
QUIET = b"\x00\x00" * 160

DESK: frozenset[Place] = frozenset({"desk"})
PHONE: frozenset[Place] = frozenset({"phone"})
TRIGGERS = cast(tuple[Trigger, ...], get_args(TriggerType))
MOVES = cast(tuple[Move, ...], (*get_args(Beginning), *get_args(Ending)))


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
    assert Gate().after("start", "shortcut").after("stop", "shortcut") == Gate(Resting("shortcut"), "desk", ended=("sent",))
    assert Gate().after("start", "shortcut").after("drop", "shortcut") == Gate(Resting("shortcut"), "desk", ended=("dropped",))
    assert Gate().after("start", "shortcut").after("timeout", "shortcut") == Gate(Resting("shortcut"), "desk", ended=("sent",))  # sent, as it stood
    # A turn ended and the next opened before a frame is captured: the count says the first was sent.
    assert Gate().after("start", "shortcut").after("stop", "shortcut").after("arm", "shortcut").after("start", "shortcut") == Gate(Open("shortcut"), "desk", ended=("sent",))


def test_an_engaged_desk_is_heard_between_turns_and_each_turn_ends_back_to_listening() -> None:
    engaged = Gate().after("listen", "engaged conversation")
    assert (engaged.key, engaged.listening, engaged.turn_open) == ("listening", DESK, False)
    assert engaged.audible(LOUD) == LOUD
    turn = engaged.after("arm", "engaged conversation").after("start", "engaged conversation")
    assert turn.turn_open
    assert turn.after("stop", "engaged conversation").key == "listening"
    assert engaged.after("arm", "engaged conversation").after("disarm", "engaged conversation").key == "listening"
    # And listens on; the turn the limit sent was opened by the voice.
    assert turn.after("timeout", "engaged conversation") == Gate(Resting("engaged conversation"), "desk", DESK, ended=("sent",))


@pytest.mark.parametrize(
    ("position", "after"),
    [(Resting("shortcut"), "up"), (Pressing("shortcut", "shortcut"), "arming"), (Open("shortcut"), "down")],
)
def test_disengaging_stops_the_desk_listening_and_leaves_a_hands_press_or_turn_to_end_as_it_ends(position: Position, after: str) -> None:
    gate = Gate(position, "desk", DESK).after("deafen", "engaged conversation")
    assert (gate.key, gate.listening, gate.position) == (after, frozenset(), position)


@pytest.mark.parametrize(("voice", "place"), [("wake word", "desk"), ("engaged conversation", "desk"), ("phone wake word", "phone")])
def test_a_place_going_deaf_sends_the_clip_its_voice_has_open_and_lets_go_of_the_press_it_armed(voice: Trigger, place: Place) -> None:
    listening = Gate(Resting("shortcut"), place, frozenset({place}))
    turn = listening.after("arm", voice).after("start", voice)
    assert turn.took("deafen", voice) == ("stop", "deafen")
    assert turn.after("deafen", voice) == Gate(Resting(voice), place, frozenset(), ended=("sent",))
    armed = listening.after("arm", voice)
    assert armed.took("deafen", voice) == ("disarm", "deafen")
    assert armed.after("deafen", voice) == Gate(Resting("shortcut"), place, frozenset())


def test_each_place_listens_by_its_own_level_and_neither_moves_the_other() -> None:
    desk_listens = Gate().after("listen", "wake word")
    # The desk at its wake word and the phone at off: the key rests at up while hands is at the phone, at listening back at the desk.
    at_the_phone = desk_listens.moved("phone")
    assert (at_the_phone.key, at_the_phone.audible(LOUD), at_the_phone.listening) == ("up", QUIET, DESK)
    assert at_the_phone.after("start", "phone button").after("stop", "phone button").key == "up"
    assert at_the_phone.moved("desk").key == "listening"
    # And the reverse: the phone listening, the desk not.
    phone_listens = Gate().moved("phone").after("listen", "phone wake word")
    assert (phone_listens.key, phone_listens.audible(LOUD), phone_listens.phone_level) == ("listening", LOUD, "wake word")
    assert phone_listens.moved("desk").key == "up"
    # Either listens on its own, wherever hands is.
    assert Gate().moved("phone").after("listen", "wake word").key == "up"
    assert Gate().after("listen", "phone wake word").key == "up"
    assert Gate().after("listen", "phone wake word").moved("phone").key == "listening"


def test_a_call_that_comes_or_goes_leaves_the_phone_not_listening_and_the_key_follows() -> None:
    gate = Gate().moved("phone").after("listen", "phone wake word")
    assert (gate.key, gate.at_phone) == ("listening", "wake word")
    # A call coming where hands already is: the level is the call's, so it starts off, and the key at rest says so.
    moves, again = gate.called("phone")
    assert (moves, again.key, again.listening, again.refusal("phone wake word"), again.at_phone) == ((), "up", frozenset(), "deaf", "off")
    # Going: hands is at the desk, and the desk's own listening is untouched.
    assert Gate(Resting("shortcut"), "phone", DESK | PHONE).called("desk") == ((), Gate(Resting("shortcut"), "desk", DESK))


def test_a_turn_opened_at_the_other_place_moves_the_gate_there() -> None:
    gate = Gate().after("start", "phone button")
    assert (gate.key, gate.place) == ("down", "phone")
    assert gate.hears("phone") and not gate.hears("desk")


@pytest.mark.parametrize("move", ["arm", "disarm", "stop", "drop", "timeout"])
def test_the_other_place_cannot_touch_a_turn_but_by_opening_one(move: Move) -> None:
    # A Shift typed at the desk while the user talks on the phone.
    on_the_phone = Gate().after("start", "phone button")
    assert on_the_phone.after(move, "shortcut") == on_the_phone
    assert on_the_phone.took(move, "shortcut") == ()


def test_leaving_a_place_mid_clip_sends_the_clip_as_it_stands() -> None:
    assert Gate().after("start", "phone button").moved("desk") == Gate(Resting("phone button"), "desk", ended=("sent",))
    assert Gate().after("arm", "shortcut").moved("phone") == Gate(Resting("shortcut"), "phone")  # no turn open, so none sent
    assert Gate().moved("phone") == Gate(Resting("shortcut"), "phone")
    assert Gate(Open("phone button"), "phone").moved("phone") == Gate(Open("phone button"), "phone")


@pytest.mark.parametrize("held", [Gate(Pressing("shortcut", "shortcut"), "desk"), Gate().after("start", "shortcut"), Gate(Resting("shortcut"), "desk", DESK).after("arm", "wake word").after("start", "wake word")])
def test_a_call_that_comes_never_cuts_off_a_press_or_a_clip_at_the_desk(held: Gate) -> None:
    # The page called again by itself while the user talks at the desk: hands stays, and the phone's first press takes it.
    assert held.called("phone") == ((), held)
    assert held.called("phone")[1].after("start", "phone button").place == "phone"


def test_a_call_that_goes_sends_its_clip_and_one_that_comes_to_a_quiet_desk_takes_hands() -> None:
    assert Gate().after("start", "phone button").called("desk") == (("stop",), Gate(Resting("phone button"), "desk", ended=("sent",)))
    assert Gate().after("listen", "wake word").called("phone") == ((), Gate(Resting("shortcut"), "phone", DESK))
    # A press at the place a call leaves is let go, and the gate says so.
    assert Gate(Pressing("phone wake word", "shortcut"), "phone", PHONE).called("desk") == (("disarm",), Gate(Resting("shortcut"), "desk"))


def test_every_move_of_the_place_is_recorded_once_by_what_made_it() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    key.move("arm", "shortcut")
    key.move("start", "shortcut")
    key.go("phone")  # a call arrives while the user talks at the desk: hands stays for the clip
    key.move("start", "phone button")
    key.end(key.gate.turn, "stop", "phone button")
    key.go("desk")
    assert [move.facts for move in events(recorded, "place.moved")] == [
        {"before": "desk", "after": "phone", "by": "phone button", "sent": True},
        {"before": "phone", "after": "desk", "by": "call", "sent": False},
    ]
    assert all(move.outcome == "ok" for move in events(recorded, "place.moved")) and len(recorded) == 2


@pytest.mark.parametrize("held", [Pressing("shortcut", "shortcut"), Open("shortcut")])
def test_a_hand_at_the_other_place_takes_hands_there_and_sends_the_clip_open_where_it_was(held: Position) -> None:
    # Right Shift held at the desk, and the phone's button pressed: the user's hand is at the phone now.
    gate = Gate(held, "desk")
    sent = isinstance(held, Open)
    assert gate.took("start", "phone button") == (("stop", "start") if sent else ("disarm", "start"))
    assert gate.after("start", "phone button") == Gate(Open("phone button"), "phone", ended=("sent",) if sent else ())
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    key.move("start", "phone button")
    assert key.move("start", "shortcut") == ("stop", "start")
    assert events(recorded, "place.moved")[-1].facts == {"before": "phone", "after": "desk", "by": "shortcut", "sent": True}


@pytest.mark.parametrize("voice", ["engaged conversation", "wake word", "phone wake word"])
def test_a_voice_never_takes_hands_to_its_place(voice: Trigger) -> None:
    # Anyone's or nobody's: the TV, or a wake word said in the room while the user talks on the other place.
    other: Place = "phone" if place_of(voice) == "desk" else "desk"
    hand: Trigger = "phone button" if other == "phone" else "shortcut"
    for there in (Gate(Open(hand), other), Gate(Resting(hand), other)):
        assert there.took("start", voice) == ()
        assert there.after("start", voice) == there


@pytest.mark.parametrize(("voice", "hand"), [("wake word", "shortcut"), ("phone wake word", "phone button")])
def test_a_voice_arms_only_where_hands_is_and_listens_with_nothing_pressed_or_open(voice: Trigger, hand: Trigger) -> None:
    place = place_of(voice)
    listening = Gate(Resting("shortcut"), place, frozenset({place}))
    armed = listening.after("arm", voice)
    assert (listening.refusal(voice), armed.key, armed.after("start", voice).key) == (None, "arming", "down")
    # Deaf, or with a press or a turn open, the voice arms nothing, and the gate says why.
    for refused, why in ((Gate(Resting("shortcut"), place), "deaf"), (armed, "busy"), (Gate(Pressing(hand, "shortcut"), place), "busy"), (Gate(Open(hand), place), "busy")):
        assert (refused.refusal(voice), refused.took("arm", voice), refused.after("arm", voice)) == (why, (), refused)
    other: Place = "phone" if place == "desk" else "desk"
    assert Gate(Resting("shortcut"), other, frozenset({place})).refusal(voice) == "elsewhere"


@pytest.mark.parametrize(("voice", "hand"), [("wake word", "shortcut"), ("phone wake word", "phone button")])
def test_a_voice_starts_only_what_it_armed(voice: Trigger, hand: Trigger) -> None:
    place = place_of(voice)
    listening = Gate(Resting("shortcut"), place, frozenset({place}))
    # What nobody armed, what a hand armed, and a hand's open clip: none is the voice's to start.
    for unarmed in (listening, listening.after("arm", hand), Gate(Resting("shortcut"), place), Gate(Open(hand), place)):
        assert unarmed.took("start", voice) == () and unarmed.after("start", voice) == unarmed
    if place == "desk":
        assert listening.after("arm", "engaged conversation").took("start", voice) == ()


@pytest.mark.parametrize(("voice", "hand"), [("wake word", "shortcut"), ("phone wake word", "phone button")])
def test_a_hands_arm_or_start_sends_the_clip_another_trigger_has_open_and_a_hands_own_goes_on(voice: Trigger, hand: Trigger) -> None:
    place = place_of(voice)
    wake = Gate(Resting("shortcut"), place, frozenset({place})).after("arm", voice).after("start", voice)
    taken = wake.after("start", hand)
    assert wake.took("start", hand) == ("stop", "start")
    assert (taken.position, taken.ended, taken.turn) == (Open(hand), ("sent",), wake.turn + 1)
    # The voice's own end now ends nothing, and the hand's ends its clip.
    assert (taken.took("stop", voice), taken.took("stop", hand)) == ((), ("stop",))
    # The same trigger's press over its own clip goes on with it.
    assert taken.took("start", hand) == () and taken.after("start", hand) == taken
    # A press over the voice's clip sends it, then arms.
    pressed = wake.after("arm", hand)
    assert (wake.took("arm", hand), pressed.position, pressed.ended) == (("stop", "arm"), Pressing(hand, voice), ("sent",))
    # A hand's press takes the arming a voice made, which the voice then cannot start.
    armed = Gate(Resting("shortcut"), place, frozenset({place})).after("arm", voice).after("arm", hand)
    assert (armed.position, armed.took("start", voice)) == (Pressing(hand, "shortcut"), ())
    # The gate says it let go of the voice's press, and a hand over its own press lets go of nothing.
    voice_armed = Gate(Resting("shortcut"), place, frozenset({place})).after("arm", voice)
    assert (voice_armed.took("arm", hand), voice_armed.took("start", hand)) == (("disarm", "arm"), ("disarm", "start"))
    assert Gate().after("arm", "shortcut").took("start", "shortcut") == ("start",)


def test_a_disarm_lets_go_only_of_the_press_it_belongs_to() -> None:
    armed = Gate().after("arm", "shortcut")
    assert armed.took("disarm", "shortcut") == ("disarm",)
    assert Gate(Pressing("phone button", "shortcut"), "desk").took("disarm", "shortcut") == ()
    assert Gate(Open("shortcut"), "desk").took("disarm", "shortcut") == ()
    assert Gate().took("disarm", "shortcut") == ()


def test_a_clip_is_ended_by_the_trigger_that_opened_it_once() -> None:
    wake = Gate(Resting("shortcut"), "phone", PHONE).after("arm", "phone wake word").after("start", "phone wake word")
    for ender in ("phone button", "shortcut", "wake word"):
        for how in get_args(Ending):
            assert wake.after(how, ender) == wake
    sent = wake.after("stop", "phone wake word")
    assert sent.turn == wake.turn + 1 and sent.took("stop", "phone wake word") == ()


@pytest.mark.parametrize("ending", ["stop", "timeout"])
def test_the_end_of_a_clip_the_gate_already_sent_ends_nothing_more(ending: Move) -> None:
    # The phone's clip was sent as the talk key took hands to the desk, and the phone's button now comes up.
    taken = Gate(Open("phone button"), "phone").after("start", "shortcut")
    assert taken.took(ending, "phone button") == ()
    assert taken.after(ending, "phone button") == taken
    # And at the desk, a clip sent ends nothing more as the key is let go.
    sent = Gate(Resting("shortcut"), "desk", ended=("sent",))
    assert sent.took(ending, "shortcut") == ()
    assert sent.took("arm", "shortcut") == ("arm",)


def test_the_gate_says_what_it_made_of_a_move_in_the_order_it_made_it() -> None:
    wake = Gate(Resting("shortcut"), "desk", DESK).after("arm", "wake word").after("start", "wake word")
    assert wake.took("arm", "shortcut") == ("stop", "arm")
    assert wake.took("start", "phone button") == ("stop", "start")
    assert Gate(Open("shortcut"), "desk").took("start", "phone button") == ("stop", "start")
    assert Gate().took("start", "shortcut") == ("start",)
    assert Gate().took("stop", "shortcut") == ()


def _reachable() -> list[Gate]:
    """Every gate the moves can make: each position at each place, every set of places that listens, and a turn or none ended."""
    out: list[Gate] = []
    places: tuple[Place, ...] = ("desk", "phone")
    listenings: tuple[frozenset[Place], ...] = (frozenset(), DESK, PHONE, DESK | PHONE)
    endeds: tuple[tuple[Fate, ...], ...] = ((), ("sent",))
    for place, listening, ended in itertools.product(places, listenings, endeds):
        here: list[Trigger] = [trigger for trigger in TRIGGERS if place_of(trigger) == place]
        positions: list[Position] = [Resting("shortcut"), Resting("phone button")]
        positions += [Pressing(by, "shortcut") for by in here] + [Open(by) for by in here]
        out += [Gate(position, place, listening, ended) for position in positions]
    return out


@pytest.mark.parametrize("gate", _reachable())
def test_no_move_by_any_trigger_loses_a_turn_makes_a_clip_two_triggers_or_lets_a_voice_take_a_hands(gate: Gate) -> None:
    for move, by in itertools.product(MOVES, TRIGGERS):
        moves, after = gate.step(move, by)
        context = (gate, move, by, after)
        grew = len(after.ended) - len(gate.ended)
        # The record only grows, by the turns the moves ended, and each ending move is one of them.
        assert after.ended[: len(gate.ended)] == gate.ended and grew in (0, 1), context
        assert grew == sum(1 for made in moves if made in get_args(Ending)), context
        # A turn open before is open after only as the same turn of the same trigger; otherwise it was ended, once.
        match gate.position, after.position:
            case Open(by=was), Open(by=now) if after.turn == gate.turn:
                assert was == now and grew == 0, context
            case Open(), _:
                assert grew == 1, context
            case _:
                pass
        # A press is gone only by becoming its presser's clip, or by a disarm the moves say.
        match gate.position:
            case Pressing(by=presser) if after.position != gate.position and after.position != Open(presser):
                assert "disarm" in moves, context
            case _:
                pass
        # A clip is open only at the place of the trigger that opened it, and a press only at its presser's.
        match after.position:
            case Open(by=opener) | Pressing(by=opener):
                assert place_of(opener) == after.place, context
            case Resting():
                pass
        # Moves name what was made: a start makes the trigger's clip, an arm its press.
        if "start" in moves:
            assert after.position == Open(by), context
        if "arm" in moves:
            assert isinstance(after.position, Pressing) and after.position.by == by, context
        # A voice moves neither the hand's press nor the hand's clip, and starts only what it armed.
        if not by_hand(by):
            match gate.position:
                case Pressing(by=presser) | Open(by=presser) if by_hand(presser):
                    assert after.position == gate.position, context
                case _:
                    pass
            if "start" in moves:
                assert gate.position == Pressing(by, gate.opened), context
        # The key at rest follows what listens; it says nothing a place does not.
        assert (after.key == "listening") == (isinstance(after.position, Resting) and after.place in after.listening), context


def test_the_gate_tells_those_who_watch_each_turn_it_opens_and_ends_in_order_and_what_ended_it() -> None:
    clock = VirtualClock()
    key = PushToTalk(clock, lambda _: None)
    told: list[GateOpened | GateEnded] = []
    key.watch(told.append)
    key.move("start", "shortcut")
    key.move("start", "shortcut")  # the same trigger's press goes on: one turn
    key.end(key.gate.turn, "stop", "shortcut")
    key.move("start", "shortcut")
    key.end(key.gate.turn, "drop", "shortcut")
    key.move("start", "shortcut")
    # A hand at the other place takes hands there and sends the clip open, and a call that goes sends the one it has.
    key.move("start", "phone button")
    key.go("desk")
    assert told == [
        GateOpened(TurnId(0), "shortcut", 0.0),
        GateEnded(TurnId(0), "stop", "shortcut"),
        GateOpened(TurnId(1), "shortcut", 0.0),
        GateEnded(TurnId(1), "drop", "shortcut"),
        GateOpened(TurnId(2), "shortcut", 0.0),
        GateEnded(TurnId(2), "stop", "phone button"),
        GateOpened(TurnId(3), "phone button", 0.0),
        GateEnded(TurnId(3), "stop", "call"),
    ]
    # A move that opens and ends nothing tells no one.
    key.end(key.gate.turn, "stop", "shortcut")
    assert len(told) == 8


def test_a_place_going_deaf_ends_the_turn_its_voice_has_open_as_the_level_and_the_level_is_recorded_whoever_changed_it() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(VirtualClock(), recorded.append)
    told: list[GateOpened | GateEnded] = []
    key.watch(told.append)
    key.go("phone")
    key.move("listen", "phone wake word")
    key.move("arm", "phone wake word")
    key.move("start", "phone wake word")
    key.move("deafen", "phone wake word")
    assert told[-1] == GateEnded(TurnId(0), "stop", "level")
    key.move("listen", "phone wake word")
    key.go("desk")  # the call goes: the level is the call's, and goes off with it
    assert [level.facts for level in events(recorded, "phone.leveled")] == [
        {"was": "off", "now": "wake word", "by": "level"},
        {"was": "wake word", "now": "off", "by": "level"},
        {"was": "off", "now": "wake word", "by": "level"},
        {"was": "wake word", "now": "off", "by": "call"},
    ]


def test_every_turn_the_gate_opens_is_ended_once_before_the_next_opens_whatever_order_things_come_in() -> None:
    # Every sequence of three moves by any trigger, with a call coming or going, from a gate that listens at both places.
    steps: list[tuple[str, Trigger | Place]] = [*itertools.product(MOVES, TRIGGERS), ("go", "phone"), ("go", "desk")]
    for sequence in itertools.product(steps, repeat=3):
        key = PushToTalk(VirtualClock(), lambda _: None)
        key.move("listen", "wake word")
        key.move("listen", "phone wake word")
        told: list[GateOpened | GateEnded] = []
        key.watch(told.append)
        for move, by in sequence:
            if move == "go":
                key.go(by)  # type: ignore[arg-type]
            elif move in get_args(Ending):
                key.end(key.gate.turn, move, by)  # type: ignore[arg-type]
            else:
                key.move(move, by)  # type: ignore[arg-type]
        # Openings and endings alternate, numbered as the gate numbers them, and the open one is the last told.
        expected = 0
        open_now = False
        for each in told:
            match each:
                case GateOpened(turn=turn):
                    assert not open_now and turn == expected, (sequence, told)
                    open_now = True
                case GateEnded(turn=turn):
                    assert open_now and turn == expected, (sequence, told)
                    open_now, expected = False, expected + 1
        assert open_now == key.gate.turn_open and len(key.gate.ended) == expected, (sequence, told)


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
    key.end(key.gate.turn, "stop", "phone button")
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
    key.end(key.gate.turn, "stop", "shortcut")
    assert key.modality == "audio-only"
    key.go("phone")
    key.switch("screen")
    key.move("start", "phone button")
    assert key.modality == "screen"
    key.go("desk")
    assert key.modality == "screen"
    key.go("phone")
    assert key.modality == "audio-only"


def test_an_end_names_the_turn_it_is_for_and_one_for_a_turn_the_gate_has_moved_past_ends_nothing() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    key.move("start", "shortcut")
    first = key.gate.turn
    assert key.end(first, "stop", "shortcut") == ("stop",)
    key.move("start", "shortcut")
    # A verdict for the first turn, decided while the gate moved on: it ends nothing of the second.
    assert (key.end(first, "stop", "shortcut"), key.gate.turn_open) == ((), True)
    assert key.end(key.gate.turn, "stop", "shortcut") == ("stop",)


async def test_set_modality_switches_it_says_so_and_refuses_what_is_neither() -> None:
    key = PushToTalk(VirtualClock(), lambda _: None)
    switch = modality_tool(key.switch)
    assert await switch.body(modality="audio-only") == {"modality": "audio-only", "readback": "Okay, audio only."}
    assert key.modality == "audio-only"
    assert await switch.body(modality="screen") == {"modality": "screen", "readback": "Okay, you can see a screen."}
    assert key.modality == "screen"
    assert "error" in await switch.body(modality="video")
    assert key.modality == "screen"
