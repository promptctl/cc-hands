import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

from wake_training import cli, recipe
from wake_training.corpus import ASSETS, Asset, Corpus, Download
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
        cli._finished(tmp_path, "generate_clips", clips, dict)  # pyright: ignore[reportPrivateUsage]
    assert seen == [[]] and not done.exists()

    cut_short()
    seen = _trainer(monkeypatch, clips)
    assert cli._finished(tmp_path, "generate_clips", clips, lambda: cli._clips(clips)) == {"positive_train": 1, "negative_train": 1}  # pyright: ignore[reportPrivateUsage]
    assert seen == [[]] and done.exists()

    seen = _trainer(monkeypatch, clips)
    assert cli._finished(tmp_path, "generate_clips", clips, lambda: cli._clips(clips)) == {"positive_train": 1, "negative_train": 1}  # pyright: ignore[reportPrivateUsage]
    assert seen == []


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
    assert model.read_bytes() == b"model" and [path.name for path in model.parent.iterdir()] == ["hey_hands.onnx"]


def test_a_relative_destination_is_printed_as_hands_can_find_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    stages: list[str] = []

    def stood_in(name: str) -> Callable[..., dict[str, Fact]]:
        def stage(*_: object) -> dict[str, Fact]:
            stages.append(name)
            return {}

        return stage

    def finished(_run: Path, step: str, *_: object) -> dict[str, Fact]:
        return stood_in(step)()

    for name in ("_fetch", "_feature_models", "_recipe", "_trained", "_evaluate", "_install"):
        monkeypatch.setattr(cli, name, stood_in(name))
    monkeypatch.setattr(cli, "_finished", finished)

    assert cli.main(["Hey Hands", "--work", "train", "--to", "models"]) == 0

    fetches = ["_fetch"] * len(ASSETS)
    assert stages == [*fetches, "_feature_models", "_recipe", "generate_clips", "augment_clips", "_trained", "_evaluate", "_install"]
    printed = capsys.readouterr()
    assert f'wake_word_model = "{tmp_path.resolve() / "models" / "hey_hands.onnx"}"' in printed.out
    logged = (tmp_path / "train" / "runs" / "hey_hands.jsonl").read_text().splitlines()
    assert [json.loads(line)["stage"] for line in logged] == [
        *["corpus"] * len(ASSETS), "feature_models", "recipe", "generate_clips", "augment_clips", "train_model", "evaluate", "install"
    ]
    # What each stage made is said as it ends, as it is logged.
    assert all(line in printed.err.splitlines() for line in logged)


def test_an_asset_fetched_says_what_it_is_and_how_big(tmp_path: Path) -> None:
    served = tmp_path / "served.bin"
    served.write_bytes(b"features")
    corpus = Corpus(tmp_path / "corpus")
    asset = Asset("features.bin", Download(served.as_uri()))
    assert cli._fetch(corpus, asset) == {"asset": "features.bin", "fetched": 1, "bytes": 8}  # pyright: ignore[reportPrivateUsage]
    assert cli._fetch(corpus, asset) == {"asset": "features.bin", "fetched": 0, "bytes": 8}  # pyright: ignore[reportPrivateUsage]
