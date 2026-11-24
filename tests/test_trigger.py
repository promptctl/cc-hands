"""The level the desk listens at, as the brain's tools say it and choose it; and a wake word model that cannot be heard
with, refused as the desk loads it."""

import asyncio
from pathlib import Path

import pytest

from hands.voice.desk import Leveled
from hands.voice.ptt import Gate
from hands.voice.tools import listening_tools
from hands.voice.trigger import Levels, by_hand, place_of, turn_start
from hands.voice.wake import Unheard, WakeWord
from hands.voice.wakemodels import EMBEDDING, MELSPECTROGRAM, PRETRAINED, Files
from hands.voice.wakeword import Pretrained, Trained

# A model of two words, as openWakeWord scores them, which no wake word is: tests/fixtures/two_words.onnx scores both 0.5.
TWO_WORDS = Path(__file__).parent / "fixtures" / "two_words.onnx"

OFF = "The microphone is off between turns. Hold Right Shift to talk, and let go to send. Triple tap Right Command for the wake word, or Right Shift for engaged conversation."


async def test_the_level_is_said_and_starts_off() -> None:
    level, _ = listening_tools(Levels(), Pretrained(), lambda: None)
    assert await level.body() == {"level": "off", "readback": OFF}


async def test_a_level_not_built_is_refused() -> None:
    _, choose = listening_tools(Levels(), Pretrained(), lambda: None)
    assert "error" in await choose.body(level="clap")


async def test_a_level_chosen_goes_to_the_desk_and_is_said_once_the_desk_is_at_it() -> None:
    levels = Levels()
    level, choose = listening_tools(levels, Pretrained(), lambda: None)
    asked: list[str] = []

    async def driver() -> None:
        # As the desk's driver does: answers each choice with the level it found, then reports the level it went to.
        while True:
            choice = await levels.chosen.get()
            asked.append(choice.level)
            choice.was.set_result(levels.level)
            if choice.level != levels.level:
                levels.leveled(Leveled(levels.level, choice.level, "tool"))

    driving = asyncio.create_task(driver())
    chosen = await choose.body(level="wake word")
    assert (chosen["level"], chosen["was"]) == ("wake word", "off")
    assert str(chosen["readback"]).startswith("Okay. The wake word: say Hey Jarvis, then what you want;")
    assert (await level.body())["level"] == "wake word"
    assert str((await choose.body(level="wake word"))["readback"]).startswith("Already. The wake word")
    assert asked == ["wake word", "wake word"]
    driving.cancel()


async def test_the_wake_word_the_settings_name_is_the_one_said(tmp_path: Path) -> None:
    for word, phrase in ((Pretrained("Hey Mycroft"), "Hey Mycroft"), (Trained("Hey Computer", tmp_path / "hey_computer.onnx"), "Hey Computer")):
        levels = Levels()
        levels.leveled(Leveled("off", "wake word", "tap"))
        level, _ = listening_tools(levels, word, lambda: None)
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


def at_the_phone(listening: bool) -> Gate:
    """The gate with hands at the phone, its page at its wake word or not."""
    gate = Gate().moved("phone")
    return gate.after("listen", "phone wake word") if listening else gate


async def test_at_the_phone_the_phones_level_is_said_and_the_pages_button_is_what_sets_it() -> None:
    levels = Levels()
    levels.leveled(Leveled("off", "engaged", "tap"))
    for listening, level, said in ((True, "wake word", "The phone listens for the wake word: say Hey Jarvis, then what you want;"), (False, "off", "The phone's wake word is off.")):
        read, _ = listening_tools(levels, Pretrained(), lambda listening=listening: at_the_phone(listening).at_phone)
        # The desk's level is no part of what is said at the phone.
        assert (await read.body())["level"] == level
        assert str((await read.body())["readback"]).startswith(said)
        assert "Phone wake word button" in str((await read.body())["readback"])


async def test_choosing_a_level_at_the_phone_changes_nothing_and_says_the_page_sets_it() -> None:
    levels = Levels()
    _, choose = listening_tools(levels, Pretrained(), lambda: at_the_phone(True).at_phone)
    chosen = await choose.body(level="engaged")
    assert chosen["level"] == "wake word"
    assert str(chosen["readback"]).startswith("Nothing changed.")
    # Nothing was asked of the desk's driver, so the desk's level is where it was.
    assert (levels.chosen.empty(), levels.level) == (True, "off")


def test_a_turn_a_wake_opened_at_the_phone_cuts_hands_off_once_a_person_is_heard_and_is_no_hands() -> None:
    assert (turn_start("phone wake word"), turn_start("phone button")) == ("on a person", "on the clip")
    assert (place_of("phone wake word"), by_hand("phone wake word")) == ("phone", False)
