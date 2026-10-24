import json
from datetime import UTC, datetime
from pathlib import Path

from wake_training.evaluate import THRESHOLD, evaluation
from wake_training.stage import Stage, line, record
from wake_training.word import word

AT = datetime(2026, 10, 23, 19, 0, tzinfo=UTC)


def test_a_stage_is_one_line_of_how_long_it_took_and_what_it_made(tmp_path: Path) -> None:
    log = tmp_path / "runs" / "hey_hands.jsonl"
    record(log, word("Hey Hands"), Stage("generate_clips", 612.04, {"positive_train": 20000}), AT)
    record(log, word("Hey Hands"), Stage("install", 0.01, {"model": "/w/hey_hands.onnx"}), AT)
    lines = [json.loads(text) for text in log.read_text().splitlines()]
    assert lines == [
        {"at": "2026-10-23T19:00:00+00:00", "word": "Hey Hands", "stage": "generate_clips", "seconds": 612.0, "positive_train": 20000},
        {"at": "2026-10-23T19:00:00+00:00", "word": "Hey Hands", "stage": "install", "seconds": 0.0, "model": "/w/hey_hands.onnx"},
    ]
    assert json.loads(line(word("Hey Hands"), Stage("recipe", 0, {}), AT)) == {"at": "2026-10-23T19:00:00+00:00", "word": "Hey Hands", "stage": "recipe", "seconds": 0}


def test_an_evaluation_counts_wakes_at_hands_threshold() -> None:
    assert evaluation([0.9, THRESHOLD, 0.1, 0.99], [0.2, 0.7]) == {"threshold": THRESHOLD, "positives": 4, "recall": 0.75, "negatives": 2, "false_accepts": 0.5}
