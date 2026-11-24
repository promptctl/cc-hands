"""scripts/record-double-talk.py, which is run by hand in a quiet room and so is never run by a test: a name it imports that has
gone from hands waits for that run to be found, so it is imported here."""

import importlib.util
from pathlib import Path

from echoscore import TAKES as SCORED

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record-double-talk.py"


def test_the_script_that_records_takes_can_record_each_take_the_scoring_scores() -> None:
    spec = importlib.util.spec_from_file_location("record_double_talk", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert set(script.TAKES) == {take.name for take in SCORED}
