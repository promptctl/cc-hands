"""scripts/record-double-talk.py, which is run by hand in a quiet room and so is never run by a test: what it renders and
what `make check` can hold of it without the room, so it is imported here (a name it imports that has gone from hands fails
type checking, which covers scripts/ as it does tests/)."""

import importlib.util
from pathlib import Path
from types import ModuleType

from echoscore import TAKES as SCORED

from hands.voice.spoken import SpokenForm
from hands.voice.voicechoice import VOICES, parse_voice

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record-double-talk.py"


def recorder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("record_double_talk", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_the_script_that_records_takes_can_record_each_take_the_scoring_scores() -> None:
    assert set(recorder().TAKES) == {take.name for take in SCORED}


async def test_every_line_a_take_plays_is_spoken_as_it_is_written() -> None:
    # [LAW:no-silent-failure] hands' TTS filters what it says, and a take whose audio is not its script's words is scored
    # against words nobody said: the render refuses it, and this finds the line before a person is in the room.
    for script, _ in recorder().TAKES.values():
        lines = [reply.text for reply in script.hands] + [said.text for said in script.person]
        assert [await SpokenForm().filter(line) for line in lines] == lines


def test_every_voice_a_take_names_is_one_hands_speaks_in() -> None:
    script = recorder()
    named = {reply.voice for take, _ in script.TAKES.values() for reply in take.hands} | {script.PERSON}
    assert {parse_voice(name) for name in named} <= set(VOICES)
