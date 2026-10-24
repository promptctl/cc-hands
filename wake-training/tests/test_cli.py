import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

from wake_training import cli, recipe
from wake_training.corpus import Corpus
from wake_training.stage import Fact
from wake_training.word import word

TRAINER_FAILED = subprocess.CalledProcessError(1, "trainer")


def _trainer(monkeypatch: pytest.MonkeyPatch, outputs: list[Path], fails: bool = False) -> list[list[str]]:
    """The trainer, as a stand-in that finds `outputs` as it starts, then makes them whole: what each step it ran saw."""
    seen: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        seen.append(sorted(path.name for path in outputs if path.exists()))
        if fails:
            raise TRAINER_FAILED
        for output in outputs:
            output.mkdir(parents=True, exist_ok=True)
            (output / "whole.wav").touch()

    monkeypatch.setattr(cli.subprocess, "run", run)
    return seen


def test_a_step_cut_short_starts_over_and_a_finished_one_keeps_what_it_made(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [tmp_path / "positive_train", tmp_path / "negative_train"]
    done = tmp_path / "generate_clips.done"

    def cut_short() -> None:
        clips[0].mkdir(exist_ok=True)
        (clips[0] / "cut_short.wav").touch()

    cut_short()
    seen = _trainer(monkeypatch, clips, fails=True)
    with pytest.raises(subprocess.CalledProcessError):
        cli._train(tmp_path, "generate_clips", clips, dict)  # pyright: ignore[reportPrivateUsage]
    assert seen == [[]] and not done.exists()

    cut_short()
    seen = _trainer(monkeypatch, clips)
    assert cli._train(tmp_path, "generate_clips", clips, lambda: cli._clips(clips)) == {"positive_train": 1, "negative_train": 1}  # pyright: ignore[reportPrivateUsage]
    assert seen == [[]] and done.exists()

    seen = _trainer(monkeypatch, clips)
    cli._train(tmp_path, "generate_clips", clips, dict)  # pyright: ignore[reportPrivateUsage]
    assert seen == [["negative_train", "positive_train"]]


def test_a_feature_file_is_counted_by_its_rows(tmp_path: Path) -> None:
    features = tmp_path / "positive_features_train.npy"
    np.save(features, np.zeros((7, 16, 96), dtype=np.float32))
    assert cli._features([features]) == {"positive_features_train": 7}  # pyright: ignore[reportPrivateUsage]


def test_the_recipe_stage_writes_the_trainer_config_and_says_its_sizes(tmp_path: Path) -> None:
    hey = word("Hey Hands")
    corpus = Corpus(tmp_path / "corpus")
    run = tmp_path / "words" / recipe.made(hey, corpus)

    facts = cli._recipe(hey, corpus, run)  # pyright: ignore[reportPrivateUsage]

    assert facts == {"samples": recipe.SAMPLES, "validation_samples": recipe.VALIDATION_SAMPLES, "steps": recipe.STEPS}
    assert yaml.safe_load((run / "training.yaml").read_text()) == recipe.config(hey, corpus, run)
    assert (run / "piper" / "generate_samples.py").read_text() == recipe.adapter(corpus)


def test_the_install_stage_puts_the_model_where_it_says(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "hey_hands.onnx").write_bytes(b"model")
    model = tmp_path / "wake" / "hey_hands.onnx"

    assert cli._install(word("Hey Hands"), run, model) == {"model": str(model)}  # pyright: ignore[reportPrivateUsage]
    assert model.read_bytes() == b"model"


def test_a_relative_destination_is_printed_as_hands_can_find_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    stages: list[str] = []

    def stood_in(name: str) -> Callable[..., dict[str, Fact]]:
        def stage(*_: object) -> dict[str, Fact]:
            stages.append(name)
            return {}

        return stage

    def train(_run: Path, step: str, *_: object) -> dict[str, Fact]:
        return stood_in(step)()

    for name in ("_corpus", "_recipe", "_evaluate", "_install"):
        monkeypatch.setattr(cli, name, stood_in(name))
    monkeypatch.setattr(cli, "_train", train)

    assert cli.main(["Hey Hands", "--work", "train", "--to", "models"]) == 0

    assert stages == ["_corpus", "_recipe", "generate_clips", "augment_clips", "train_model", "_evaluate", "_install"]
    assert f'wake_word_model = "{tmp_path.resolve() / "models" / "hey_hands.onnx"}"' in capsys.readouterr().out
    logged = [json.loads(line)["stage"] for line in (tmp_path / "train" / "runs" / "hey_hands.jsonl").read_text().splitlines()]
    assert logged == ["corpus", "recipe", "generate_clips", "augment_clips", "train_model", "evaluate", "install"]
