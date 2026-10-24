"""A run's record: one JSON line per stage, in the word's run log, saying how long the stage took and what it made.
Those lines are how long training a word takes, measured rather than guessed."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from wake_training.word import Word

type Fact = int | float | str


@dataclass(frozen=True)
class Stage:
    """One stage done: its `name`, its `seconds`, and the `facts` about what it made (counts, sizes, scores)."""

    name: str
    seconds: float
    facts: dict[str, Fact]


def line(word: Word, stage: Stage, at: datetime) -> str:
    """The stage as its run log line."""
    return json.dumps({"at": at.isoformat(), "word": word.phrase, "stage": stage.name, "seconds": round(stage.seconds, 1), **stage.facts})


def record(log: Path, word: Word, stage: Stage, at: datetime) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as lines:
        lines.write(line(word, stage, at) + "\n")
