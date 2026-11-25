"""The weights hands loads (src/hands/voice/pins.py): CI keys a cache of each group on its pins, from a runner with nothing installed,
and a run fetches the groups its tests load."""

import ast
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fetches import LOADS, Fetches
from hands.voice import engaged, pins, streaming, transcription, voiceprints

PINS = Path(pins.__file__)


def test_ci_reads_each_groups_key_and_directories() -> None:
    lines = pins.github_output().splitlines()
    for group, held in pins.GROUPS.items():
        assert f"{group}_key={pins.key(group)}" in lines
        start = lines.index(f"{group}_paths<<EOF")
        assert lines[start + 1 : start + 1 + len(held)] == [f"~/.cache/huggingface/hub/{pin.directory}" for pin in held]
        assert lines[start + 1 + len(held)] == "EOF"


def test_the_pins_import_nothing_but_the_standard_library_since_ci_reads_them_with_nothing_installed() -> None:
    imported = {alias.name.split(".")[0] for node in ast.walk(ast.parse(PINS.read_text())) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(ast.parse(PINS.read_text())) if isinstance(node, ast.ImportFrom)}
    assert imported <= set(sys.stdlib_module_names)


def test_a_group_has_the_key_of_what_it_fetches_and_no_other_groups_does() -> None:
    assert len({pins.key(group) for group in pins.GROUPS}) == len(pins.GROUPS)


@pytest.mark.parametrize("moved", [replace(pins.WHISPER, revision="0" * 40), replace(pins.WHISPER, repo="someone/else"), replace(pins.WHISPER, file="model.bin")])
def test_a_weight_moved_to_another_revision_repository_or_file_is_a_new_key(moved: pins.Pin) -> None:
    assert pins.fingerprint((moved,)) != pins.key("whisper")


def test_every_pin_of_every_group_has_the_call_that_fetches_it_and_no_other_does() -> None:
    assert set(LOADS) == {pin for held in pins.GROUPS.values() for pin in held}


def test_the_voice_modules_load_what_the_pins_name() -> None:
    assert (streaming.MODEL, streaming.REVISION) == (pins.STREAMING.repo, pins.STREAMING.revision)
    assert (streaming.DIARIZATION, streaming.DIARIZATION_REVISION) == (pins.DIARIZATION.repo, pins.DIARIZATION.revision)
    assert (engaged.SMART_TURN, engaged.SMART_TURN_REVISION) == (pins.SMART_TURN.repo, pins.SMART_TURN.revision)
    assert (transcription.MODEL, transcription.REVISION) == (pins.WHISPER.repo, pins.WHISPER.revision)
    assert (voiceprints.MODEL, voiceprints.REVISION, voiceprints.FILE) == (pins.VOICEPRINT.repo, pins.VOICEPRINT.revision, pins.VOICEPRINT.file)


def fetches(unit: bool, weights: tuple[str, ...]) -> Fetches:
    return Fetches(Path("/n"), Path("/w"), Path("/c"), Path("/m"), Path("/d"), Path("/r"), {}, unit, weights)


def test_a_unit_run_that_fetches_weights_is_not_a_run() -> None:
    with pytest.raises(ValueError, match="a unit run fetches no weights"):
        fetches(True, ("whisper",))
    assert Fetches.read(fetches(False, ("whisper",)).written()).weights == ("whisper",)
