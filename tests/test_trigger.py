"""The level the desk listens at, as the brain's tools say it and choose it; and a wake word model that cannot be heard
with, refused as the desk loads it."""

from pathlib import Path

import pytest

from hands.voice.desk import Chosen, Leveled
from hands.voice.tools import listening_tools
from hands.voice.trigger import Levels
from hands.voice.wake import Unheard, WakeWord
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained

# A model of two words, as openWakeWord scores them, which no wake word is: tests/fixtures/two_words.onnx scores both 0.5.
TWO_WORDS = Path(__file__).parent / "fixtures" / "two_words.onnx"

OFF = "The microphone is off between turns. Hold Right Shift to talk, and let go to send. Triple tap Right Command for the wake word, or Right Shift for engaged conversation."


async def test_the_level_is_said_and_starts_off() -> None:
    level, _ = listening_tools(Levels(), Pretrained())
    assert await level.body() == {"level": "off", "readback": OFF}


async def test_a_level_not_built_is_refused() -> None:
    _, choose = listening_tools(Levels(), Pretrained())
    assert "error" in await choose.body(level="clap")


async def test_a_level_chosen_goes_to_the_desk_and_is_said_once_the_desk_is_at_it() -> None:
    levels = Levels()
    level, choose = listening_tools(levels, Pretrained())
    chosen = await choose.body(level="wake word")
    assert (chosen["level"], chosen["was"]) == ("wake word", "off")
    assert str(chosen["readback"]).startswith("Okay. The wake word: say Hey Jarvis, then what you want;")
    assert levels.chosen.get_nowait() == Chosen("wake word")
    # The level said is the desk's, which the driver reports once it has stepped the choice.
    assert (await level.body())["level"] == "off"
    levels.leveled(Leveled("off", "wake word", "tool"))
    assert (await level.body())["level"] == "wake word"
    assert str((await choose.body(level="wake word"))["readback"]).startswith("Already. The wake word")


async def test_the_wake_word_the_settings_name_is_the_one_said(tmp_path: Path) -> None:
    for word, phrase in ((Pretrained("Hey Mycroft"), "Hey Mycroft"), (Trained("Hey Computer", tmp_path / "hey_computer.onnx"), "Hey Computer")):
        levels = Levels()
        levels.leveled(Leveled("off", "wake word", "tap"))
        level, _ = listening_tools(levels, word)
        assert str((await level.body())["readback"]).startswith(f"The wake word: say {phrase}, then what you want;")


async def test_a_trained_wake_word_is_heard_with_its_own_model(tmp_path: Path, models: Path) -> None:
    # openWakeWord's Hey Mycroft model stands in for one the user trained, under a name of their own.
    trained = tmp_path / "hey_computer.onnx"
    trained.write_bytes((models / PRETRAINED["Hey Mycroft"]).read_bytes())
    WakeWord(models, Trained("Hey Computer", trained))


@pytest.mark.parametrize(
    ("model", "said"),
    [
        (None, "no wake word model at {model}"),
        (b"not a model", "the wake word model at {model} cannot be loaded: "),
        (TWO_WORDS.read_bytes(), "the wake word model at {model} scores 2 words, and a wake word is one"),
    ],
)
def test_a_trained_wake_word_it_cannot_hear_with_is_refused_naming_the_model(tmp_path: Path, models: Path, model: bytes | None, said: str) -> None:
    trained = tmp_path / "hey_computer.onnx"
    if model is not None:
        trained.write_bytes(model)
    with pytest.raises(Unheard) as refused:
        WakeWord(models, Trained("Hey Computer", trained))
    assert str(refused.value).startswith(said.format(model=trained))
