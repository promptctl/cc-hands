"""The level the desk listens at, as the brain's tools say it and choose it; and a wake word model that cannot be heard
with, refused as the desk loads it."""

import asyncio
from pathlib import Path
from typing import get_args

import pytest

from hands.voice.clips import Trigger
from hands.voice.desk import Desk, Loaded, Resting
from hands.voice.tools import listening_tools
from hands.voice.listening import Level, Levels
from hands.voice.trigger import PHONE_WAKE_BUTTON, PhoneLevel, by_hand, place_of, turn_start
from hands.voice.wake import Unheard, WakeWord
from hands.voice.wakemodels import EMBEDDING, MELSPECTROGRAM, PRETRAINED, Files
from hands.voice.wakeword import Pretrained, Trained

# A model of two words, as openWakeWord scores them, which no wake word is: tests/fixtures/two_words.onnx scores both 0.5.
TWO_WORDS = Path(__file__).parent / "fixtures" / "two_words.onnx"

def listening_at(level: Level) -> Desk:
    """A desk resting at `level`, its models loaded, as its driver reports it."""
    return Desk(Loaded(Resting(level)))


OFF = "The microphone is off between turns. Hold Right Shift to talk, and let go to send. Triple tap Right Command for the wake word, or Right Shift for engaged conversation."


async def test_the_level_is_said_and_starts_off() -> None:
    level, _ = listening_tools(Levels("off"), lambda: None, Pretrained())
    assert await level.body() == {"level": "off", "readback": OFF}


async def test_a_restarts_level_is_not_said_until_the_desk_listens_at_it() -> None:
    levels = Levels("engaged")
    level, _ = listening_tools(levels, lambda: None, Pretrained())
    # The run carried engaged, and its desk is yet to step it: the microphone is closed, and the tool says so.
    assert await level.body() == {"level": "off", "readback": OFF}
    levels.stepped(listening_at("engaged"))
    assert (await level.body())["level"] == "engaged"


async def test_a_level_not_built_is_refused() -> None:
    _, choose = listening_tools(Levels("off"), lambda: None, Pretrained())
    assert "error" in await choose.body(level="clap")


async def test_a_level_chosen_goes_to_the_desk_and_is_said_once_the_desk_is_at_it() -> None:
    levels = Levels("off")
    level, choose = listening_tools(levels, lambda: None, Pretrained())
    asked: list[str] = []

    async def driver() -> None:
        # As the desk's driver does: answers each choice with the level it found, then reports the level it went to.
        while True:
            choice = await levels.taken()
            asked.append(f"{choice.level} by {choice.by}")
            levels.answer(choice)
            levels.stepped(listening_at(choice.level))

    driving = asyncio.create_task(driver())
    chosen = await choose.body(level="wake word")
    assert (chosen["level"], chosen["was"]) == ("wake word", "off")
    assert str(chosen["readback"]).startswith("Okay. The wake word: say Hey Jarvis, then what you want;")
    assert (await level.body())["level"] == "wake word"
    assert str((await choose.body(level="wake word"))["readback"]).startswith("Already. The wake word")
    assert asked == ["wake word by tool", "wake word by tool"]
    driving.cancel()


async def test_the_wake_word_the_settings_name_is_the_one_said(tmp_path: Path) -> None:
    for word, phrase in ((Pretrained("Hey Mycroft"), "Hey Mycroft"), (Trained("Hey Computer", tmp_path / "hey_computer.onnx"), "Hey Computer")):
        levels = Levels("off")
        levels.stepped(listening_at("wake word"))
        level, _ = listening_tools(levels, lambda: None, word)
        assert str((await level.body())["readback"]).startswith(f"The wake word: say {phrase}, then what you want;")


@pytest.mark.outside
async def test_a_trained_wake_word_is_heard_with_its_own_model(tmp_path: Path, models: Path) -> None:
    # openWakeWord's Hey Mycroft model stands in for one the user trained, under a name of their own.
    trained = tmp_path / "hey_computer.onnx"
    trained.write_bytes((models / PRETRAINED["Hey Mycroft"].name).read_bytes())
    WakeWord(Files(models / MELSPECTROGRAM.name, models / EMBEDDING.name, trained))


@pytest.mark.parametrize(
    ("model", "said"),
    [
        (b"not a model", "the wake word model at {model} cannot be loaded: "),
        pytest.param(TWO_WORDS.read_bytes(), "the wake word model at {model} scores 2 words, and a wake word is one", marks=pytest.mark.outside),
    ],
)
def test_a_trained_wake_word_it_cannot_hear_with_is_refused_naming_the_model(tmp_path: Path, models: Path, model: bytes, said: str) -> None:
    trained = tmp_path / "hey_computer.onnx"
    trained.write_bytes(model)
    with pytest.raises(Unheard) as refused:
        WakeWord(Files(models / MELSPECTROGRAM.name, models / EMBEDDING.name, trained))
    assert str(refused.value).startswith(said.format(model=trained))


def test_a_wake_heard_on_the_phone_is_a_trigger_of_the_phone_made_by_no_hand_that_cuts_hands_off_once_a_person_is_heard() -> None:
    assert (place_of("phone wake word"), by_hand("phone wake word"), turn_start("phone wake word")) == ("phone", False, "on a person")
    # Beside the button, whose turn cuts hands off at once, and the desk's own triggers.
    assert [(trigger, place_of(trigger), by_hand(trigger), turn_start(trigger)) for trigger in ("phone button", "wake word")] == [
        ("phone button", "phone", True, "on the clip"),
        ("wake word", "desk", False, "on a person"),
    ]
    triggers: tuple[Trigger, ...] = get_args(Trigger)
    assert {trigger for trigger in triggers if place_of(trigger) == "phone"} == {"phone button", "phone wake word"}


@pytest.mark.parametrize(("level", "listens"), [("off", False), ("wake word", True)])
async def test_where_hands_is_at_the_phone_the_tools_say_the_phones_level_and_the_button_that_sets_it_and_change_neither_level(level: PhoneLevel, listens: bool) -> None:
    levels = Levels("off")
    said, choose = listening_tools(levels, lambda: level, Pretrained())
    readback = str((await said.body())["readback"])
    assert (await said.body())["level"] == level and PHONE_WAKE_BUTTON in readback and readback.count("Right") == 0
    assert ("is not listening between turns" in readback) == (not listens) and ("listens for the wake word" in readback) == listens
    # A choice made at the phone is the page's to make: it says so, and the desk's level and the phone's stay where they were.
    chosen = await choose.body(level="engaged")
    assert (chosen["level"], PHONE_WAKE_BUTTON in str(chosen["readback"])) == (level, True)
    assert (levels.level, levels.unanswered) == ("off", 0)
