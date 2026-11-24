"""The desk: the talk key opens turns at every level, triple taps set the level, and the voice opens turns under two."""

from collections.abc import Sequence

import pytest

from hands.voice.cues import level_cue
from hands.voice.desk import Act, Chosen, Desk, Event, Leveled, Loaded, Made, ModelsLoaded, Resting, Woken, step
from hands.voice.listening import Level
from hands.voice.shortcut import CommandPressed, CommandReleased, Pressed, Released, Ripe, Typed
from hands.voice.voiceturn import ASK_SECONDS, Judging, NothingAsked, Speaking, SpeechStarted, SpeechStarting, SpeechStopped, TurnEnded, TurnTooLong, TurnId, Waiting, deadlines


T1, T2 = TurnId(1), TurnId(2)
# The desk once the models it hears with are loaded, as every test but the load's starts.
LOADED = Desk(Loaded(Resting("off")))


def run(events: Sequence[Event], desk: Desk = LOADED) -> tuple[Desk, list[Act]]:
    made: list[Act] = []
    for event in events:
        desk, acts = step(desk, event)
        made.extend(acts)
    return desk, made


def at(level: Level) -> Desk:
    return run([Chosen(level, "tool")])[0]


def held(t: float) -> list[Event]:
    return [Pressed(t), Ripe(t), Released()]


def taps(t: float, command: bool = False) -> list[Event]:
    return [e for i in range(3) for e in ([CommandPressed(t + i / 10), CommandReleased()] if command else [Pressed(t + i / 10), Released()])]


KEY_TURN = [Made("arm", "shortcut"), Made("start", "shortcut"), Made("stop", "shortcut")]
TAP_SHIFT = [Made("arm", "shortcut"), Made("disarm", "shortcut")] * 2 + [Made("arm", "shortcut")]


def moves(acts: list[Act]) -> list[Act]:
    return [act for act in acts if isinstance(act, Made)]


def test_the_desk_starts_off_and_deaf_and_the_held_key_talks() -> None:
    assert Desk.starting("off").level == "off"
    assert run(held(1.0), Desk.starting("off"))[1] == KEY_TURN


def test_a_level_chosen_while_the_models_load_is_listened_at_once_they_are_and_not_before() -> None:
    loading, made = run(held(1.0), Desk.starting("engaged"))
    # Nothing hears the room yet: the desk listens at off, holding the level chosen, and only the key moves the gate.
    assert (loading.level, loading.chosen, made) == ("off", "engaged", KEY_TURN)
    loaded, made = run([ModelsLoaded()], loading)
    assert (loaded.level, made) == ("engaged", [Made("listen", "engaged conversation"), Leveled("off", "engaged", "start")])
    # The models load once a run: told so again, the desk was stepped wrongly, and says so.
    with pytest.raises(RuntimeError, match="already were"):
        run([ModelsLoaded()], loaded)


def test_the_last_level_chosen_while_the_models_load_is_the_one_listened_at_and_said_by_its_chooser() -> None:
    # Right Shift's triple tap from the engaged the start chose is off, and the tool then chose the wake word.
    loading, made = run([*taps(1.0), Chosen("wake word", "tool")], Desk.starting("engaged"))
    assert (loading.level, loading.chosen, made) == ("off", "wake word", [*TAP_SHIFT, Made("disarm", "shortcut")])
    assert run([ModelsLoaded()], loading)[1] == [Made("listen", "wake word"), Leveled("off", "wake word", "tool")]
    # Loaded at off, as chosen, there is nothing to listen for and no change to say.
    assert run([*taps(1.0), ModelsLoaded()], Desk.starting("engaged"))[1] == [*TAP_SHIFT, Made("disarm", "shortcut")]


@pytest.mark.parametrize("level", ["off", "wake word", "engaged"])
def test_the_held_key_opens_a_turn_at_every_level(level: Level) -> None:
    desk, made = run(held(1.0), at(level))
    assert made == KEY_TURN and desk.level == level


@pytest.mark.parametrize(
    ("start", "events", "level", "made"),
    [
        # Right Command's triple tap toggles the wake word, from off and back.
        ("off", taps(1.0, command=True), "wake word", [Made("listen", "wake word"), Leveled("off", "wake word", "tap")]),
        ("wake word", taps(1.0, command=True), "off", [Made("deafen", "wake word"), Leveled("wake word", "off", "tap")]),
        # Right Shift's toggles engaged as its third press is let go, known then not to mean talk.
        ("off", taps(1.0), "engaged", [*TAP_SHIFT, Made("disarm", "shortcut"), Made("listen", "engaged conversation"), Leveled("off", "engaged", "tap")]),
        # Either switches straight from the other, with the desk listening throughout.
        ("engaged", taps(1.0, command=True), "wake word", [Leveled("engaged", "wake word", "tap")]),
        # Two taps are nothing, and a key between starts the count again.
        ("off", [CommandPressed(1.0), CommandReleased(), Typed(), CommandPressed(1.1), CommandReleased(), CommandPressed(1.2)], "off", []),
    ],
)
def test_a_triple_tap_sets_the_level(start: Level, events: list[Event], level: Level, made: list[Act]) -> None:
    desk, acts = run(events, at(start))
    assert desk.level == level and acts == made


def test_a_press_held_to_a_turn_is_no_tap() -> None:
    desk, _ = run([Pressed(1.0), Released(), Pressed(1.1), Ripe(1.1), Released(), Pressed(1.3), Released()])
    assert desk.level == "off"


def test_a_third_press_held_to_talk_is_a_turn_and_no_tap() -> None:
    desk, made = run([Pressed(1.0), Released(), Pressed(1.1), Released(), *held(1.2)])
    assert desk.level == "off" and made == [*TAP_SHIFT[:4], *KEY_TURN]


def test_under_the_wake_word_the_word_opens_a_turn_and_end_of_turn_closes_it() -> None:
    desk, made = run([Woken(1.0, over_hands=False, built_in=False), SpeechStopped(), SpeechStarted(1.5), SpeechStopped(), TurnEnded("verdict", T1)], at("wake word"))
    assert made == [Made("arm", "wake word"), Made("start", "wake word"), Judging(Speaking(T1, 1.0)), Made("stop", "wake word")]
    assert desk.level == "wake word"


@pytest.mark.parametrize("level", ["off", "engaged"])
def test_the_wake_word_opens_nothing_but_under_its_own_level(level: Level) -> None:
    assert run([Woken(1.0, over_hands=False, built_in=False)], at(level))[1] == []


def test_the_wake_word_over_hands_on_the_macs_own_pair_opens_nothing_and_said_after_it_opens_a_turn() -> None:
    desk, made = run([Woken(1.0, over_hands=True, built_in=True), Woken(2.0, over_hands=False, built_in=True)], at("wake word"))
    assert made == [Made("arm", "wake word"), Made("start", "wake word")]
    assert desk.turns == 1


def test_the_wake_word_over_hands_on_any_other_pair_opens_a_turn() -> None:
    assert run([Woken(1.0, over_hands=True, built_in=False)], at("wake word"))[1] == [Made("arm", "wake word"), Made("start", "wake word")]


@pytest.mark.parametrize("level", ["off", "wake word"])
def test_speech_alone_opens_nothing_but_engaged(level: Level) -> None:
    assert run([SpeechStarting(), SpeechStarted(1.0), SpeechStopped()], at(level))[1] == []


def test_engaged_speech_opens_a_turn_and_a_noise_disarms() -> None:
    _, made = run([SpeechStarting(), SpeechStopped(), SpeechStarted(2.0), SpeechStopped(), TurnEnded("silence", T1)], at("engaged"))
    assert made == [Made("arm", "engaged conversation"), Made("disarm", "engaged conversation"), Made("arm", "engaged conversation"), Made("start", "engaged conversation"), Judging(Speaking(T1, 2.0)), Made("stop", "engaged conversation")]


@pytest.mark.parametrize(("level", "opening"), [("wake word", [Woken(1.0, over_hands=False, built_in=False)]), ("engaged", [SpeechStarted(1.0)]), ("engaged", [SpeechStarting()])])
def test_a_shortcut_over_what_the_voice_has_open_talks_and_leaves_the_gate_to_send_or_take_what_the_voice_had(level: Level, opening: list[Event]) -> None:
    # [LAW:single-enforcer] the desk ends nothing of the voice's here: the gate sends its turn, or takes its arming, as the shortcut arms.
    _, made = run([*opening, Pressed(2.0), Ripe(2.0), Released()], at(level))
    assert moves(made)[-3:] == [Made("arm", "shortcut"), Made("start", "shortcut"), Made("stop", "shortcut")]


@pytest.mark.parametrize(("level", "opening"), [("wake word", [Woken(1.0, over_hands=False, built_in=False)]), ("engaged", [SpeechStarted(1.0)])])
def test_shift_typed_over_a_voice_turn_leaves_it_open(level: Level, opening: list[Event]) -> None:
    trigger = "wake word" if level == "wake word" else "engaged conversation"
    desk, made = run([*opening, SpeechStarted(1.5), Pressed(2.0), Typed(), Released(), SpeechStopped(), TurnEnded("verdict", T1)], at(level))
    assert moves(made)[-1] == Made("stop", trigger) and Made("arm", "shortcut") not in made and desk.level == level


def test_a_shortcut_begun_under_a_voice_turn_that_ends_opens_its_own() -> None:
    _, made = run([SpeechStarted(1.0), Pressed(2.0), SpeechStopped(), TurnEnded("verdict", T1), Ripe(2.0), Released()], at("engaged"))
    assert moves(made)[-4:] == [Made("stop", "engaged conversation"), *KEY_TURN]


def test_shift_typing_under_a_voice_turn_that_ends_opens_nothing() -> None:
    _, made = run([SpeechStarted(1.0), Pressed(2.0), Typed(), SpeechStopped(), TurnEnded("verdict", T1), Released()], at("engaged"))
    assert moves(made)[-1] == Made("stop", "engaged conversation")


@pytest.mark.parametrize("level", ["wake word", "engaged"])
def test_the_voice_opens_nothing_while_the_key_is_down(level: Level) -> None:
    _, made = run([Pressed(1.0), Ripe(1.0), Woken(1.2, over_hands=False, built_in=False), SpeechStarted(1.2), SpeechStopped(), TurnEnded("verdict", T1), Released()], at(level))
    assert made == KEY_TURN


@pytest.mark.parametrize(("level", "opening"), [("wake word", [Woken(1.0, over_hands=False, built_in=False)]), ("engaged", [SpeechStarted(1.0)])])
def test_a_level_left_with_a_voice_turn_open_sends_it(level: Level, opening: list[Event]) -> None:
    desk, made = run([*opening, Chosen("off", "tool")], at(level))
    trigger = "wake word" if level == "wake word" else "engaged conversation"
    assert made[-3:] == [Made("stop", trigger), Made("deafen", trigger), Leveled(level, "off", "tool")]
    assert desk.level == "off"


def test_a_key_turn_goes_on_across_a_change_of_level() -> None:
    _, made = run([Pressed(1.0), Ripe(1.0), Chosen("engaged", "tool"), Released()])
    assert made == [Made("arm", "shortcut"), Made("start", "shortcut"), Made("listen", "engaged conversation"), Leveled("off", "engaged", "tool"), Made("stop", "shortcut")]


def test_a_level_chosen_again_changes_nothing() -> None:
    assert run([Chosen("off", "tool")])[1] == []


def test_a_voice_turn_open_past_the_limit_is_sent_and_a_stale_limit_is_nothing() -> None:
    _, made = run([Woken(1.0, over_hands=False, built_in=False), TurnTooLong(T2), TurnTooLong(T1)], at("wake word"))
    assert moves(made) == [Made("arm", "wake word"), Made("start", "wake word"), Made("timeout", "wake word")]


@pytest.mark.parametrize(
    "events",
    [
        # Before what is asked has started, a stop or silence ends nothing: it is the end of the wake word.
        [Woken(1.0, over_hands=False, built_in=False), SpeechStopped(), TurnEnded("silence", T1)],
        # The wake word said again inside a turn is part of the turn.
        [Woken(1.0, over_hands=False, built_in=False), SpeechStarted(1.5), Woken(2.0, over_hands=False, built_in=False)],
    ],
)
def test_the_wake_words_own_pause_and_its_saying_again_end_nothing(events: list[Event]) -> None:
    assert moves(run(events, at("wake word"))[1]) == [Made("arm", "wake word"), Made("start", "wake word")]


def test_each_level_has_its_own_cue() -> None:
    assert len({level_cue(Leveled("off", now, "tap")) for now in ("wake word", "engaged")} | {level_cue(Leveled("engaged", "off", "tap"))}) == 3


def test_a_wake_with_nothing_asked_after_it_is_thrown_away_and_a_stale_one_is_nothing() -> None:
    assert (ASK_SECONDS + 1.0, NothingAsked(T1)) in deadlines(Waiting(T1, 1.0))
    desk, made = run([Woken(1.0, over_hands=False, built_in=False), NothingAsked(T2), NothingAsked(T1)], at("wake word"))
    assert moves(made) == [Made("arm", "wake word"), Made("start", "wake word"), Made("drop", "wake word")] and desk.level == "wake word"
    _, asked = run([Woken(1.0, over_hands=False, built_in=False), SpeechStarted(1.5), NothingAsked(T1)], at("wake word"))
    assert Made("drop", "wake word") not in asked


def test_each_turn_the_voice_opens_is_named_apart_from_the_last_though_it_opens_at_the_same_instant() -> None:
    # The first turn ends and the second opens at the very instant the first did: a verdict for the first, late, is none
    # of the second's, and the second ends by its own.
    _, made = run([SpeechStarted(1.0), TurnEnded("verdict", T1), SpeechStarted(1.0), TurnEnded("verdict", T1), SpeechStopped(), TurnEnded("verdict", T2)], at("engaged"))
    assert moves(made) == [
        Made("arm", "engaged conversation"), Made("start", "engaged conversation"), Made("stop", "engaged conversation"),
        Made("arm", "engaged conversation"), Made("start", "engaged conversation"), Made("stop", "engaged conversation"),
    ]
    assert made.count(Judging(Speaking(T2, 1.0))) == 1
