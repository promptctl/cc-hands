"""`train-wake-word "Hey Hands"`: train a wake word, from the corpus to a model in place for hands.

Every stage runs on every run, in order. A run cut short resumes: the corpus keeps what it fetched (wake_training.
fetch), and openWakeWord's trainer keeps the clips it generated and the features it computed; training itself, the
model's own stage, runs again.
"""

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import openwakeword.utils
import yaml

from wake_training import recipe
from wake_training.corpus import ASSETS, VALIDATION_FEATURES, Corpus
from wake_training.evaluate import FRAME_SECONDS, evaluation, peaks, scores, wakes
from wake_training.fetch import fetch
from wake_training.stage import Fact, Stage, record
from wake_training.word import Rejected, Word, word

CLIPS = ("positive_train", "positive_test", "negative_train", "negative_test")
FEATURES = ("positive_features_train", "positive_features_test", "negative_features_train", "negative_features_test")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train-wake-word", description="Train a wake word for hands to listen for.")
    parser.add_argument("phrase", help='what is said to wake hands, such as "Hey Hands"')
    parser.add_argument("--work", type=Path, default=Path.home() / "wake" / "train", help="where the corpus, each word's clips and features, and the run logs live (default: ~/wake/train)")
    parser.add_argument("--to", type=Path, default=Path.home() / "wake", help="where the trained model goes (default: ~/wake)")
    args = parser.parse_args(argv)
    try:
        chosen = word(args.phrase)
    except Rejected as rejected:
        parser.error(str(rejected))
    work: Path = args.work.expanduser()
    to: Path = args.to.expanduser()
    corpus = Corpus(work / "corpus")
    run = work / "words" / recipe.made(chosen, corpus)
    log = work / "runs" / f"{chosen.name}.jsonl"

    def stage(name: str, done: Callable[[], dict[str, Fact]]) -> None:
        print(f"train-wake-word: {chosen.phrase}: {name}", file=sys.stderr, flush=True)
        began = time.monotonic()
        facts = done()
        record(log, chosen, Stage(name, time.monotonic() - began, facts), datetime.now(UTC))

    stage("corpus", lambda: _corpus(corpus))
    stage("recipe", lambda: _recipe(chosen, corpus, run))
    # openWakeWord's trainer (wake_training.trainer), one stage per step, each step by its own flag and what it made.
    made = run / chosen.name
    stage("generate_clips", lambda: _train(run, "generate_clips", lambda: {clips: sum(1 for _ in (made / clips).glob("*.wav")) for clips in CLIPS}))
    stage("augment_clips", lambda: _train(run, "augment_clips", lambda: {features: int(np.load(made / f"{features}.npy", mmap_mode="r").shape[0]) for features in FEATURES}))
    stage("train_model", lambda: _train(run, "train_model", lambda: {"model_bytes": (run / f"{chosen.name}.onnx").stat().st_size}))
    stage("evaluate", lambda: _evaluate(chosen, corpus, run))
    model = to / f"{chosen.name}.onnx"
    stage("install", lambda: _install(chosen, run, model))
    print(f'[talk]\nwake_word = "{chosen.phrase}"\nwake_word_model = "{model}"')
    return 0


def _corpus(corpus: Corpus) -> dict[str, Fact]:
    fetched = sum(fetch(corpus, asset) for asset in ASSETS)
    # The trainer computes features with the melspectrogram and embedding models installed beside openwakeword.
    openwakeword.utils.download_models()  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
    return {"assets": len(ASSETS), "fetched": fetched}


def _recipe(chosen: Word, corpus: Corpus, run: Path) -> dict[str, Fact]:
    (run / "piper").mkdir(parents=True, exist_ok=True)
    (run / "training.yaml").write_text(yaml.safe_dump(recipe.config(chosen, corpus, run)))
    (run / "piper" / "generate_samples.py").write_text(recipe.adapter(corpus))
    return {"samples": recipe.SAMPLES, "validation_samples": recipe.VALIDATION_SAMPLES, "steps": recipe.STEPS}


def _train(run: Path, step: str, made: Callable[[], dict[str, Fact]]) -> dict[str, Fact]:
    subprocess.run([sys.executable, "-m", "wake_training.trainer", "--training_config", str(run / "training.yaml"), f"--{step}"], check=True)
    return made()


def _evaluate(chosen: Word, corpus: Corpus, run: Path) -> dict[str, Fact]:
    model = run / f"{chosen.name}.onnx"
    made = run / chosen.name
    validation = np.load(corpus.at(VALIDATION_FEATURES))
    said, frames = scores(model, validation)
    return evaluation(
        peaks(model, sorted((made / "positive_test").glob("*.wav"))),
        peaks(model, sorted((made / "negative_test").glob("*.wav"))),
        wakes(said, frames),
        len(validation) * FRAME_SECONDS / 3600,
    )


def _install(chosen: Word, run: Path, model: Path) -> dict[str, Fact]:
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / f"{chosen.name}.onnx", model)
    return {"model": str(model)}


if __name__ == "__main__":
    sys.exit(main())
