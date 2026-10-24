"""`train-wake-word "Hey Hands"`: train a wake word, from the corpus to a model in place for hands.

Every stage runs on every run, in order, and says what it made as it ends. A run cut short resumes: the corpus keeps
what it fetched (wake_training.fetch), and a trainer step that finished keeps its clips or features; a step cut short
starts over, and training itself, the model's own stage, runs again.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import openwakeword.utils
import yaml

from wake_training import recipe
from wake_training.corpus import ASSETS, VALIDATION_FEATURES, Asset, Corpus
from wake_training.evaluate import FRAME_SECONDS, evaluation, peaks, scores, wakes
from wake_training.fetch import fetch
from wake_training.stage import Fact, Stage, line, record
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
    # Absolute, as the printed model path must be for hands, and the corpus's path in the run's digest is.
    work: Path = args.work.expanduser().resolve()
    to: Path = args.to.expanduser().resolve()
    corpus = Corpus(work / "corpus")
    run = work / "words" / recipe.made(chosen, corpus)
    log = work / "runs" / f"{chosen.name}.jsonl"

    def stage(name: str, done: Callable[[], dict[str, Fact]]) -> None:
        print(f"train-wake-word: {chosen.phrase}: {name}", file=sys.stderr, flush=True)
        began = time.monotonic()
        facts = done()
        ended = Stage(name, time.monotonic() - began, facts)
        at = datetime.now(UTC)
        record(log, chosen, ended, at)
        print(line(chosen, ended, at), file=sys.stderr, flush=True)

    for asset in ASSETS:
        stage("corpus", partial(_fetch, corpus, asset))
    stage("feature_models", _feature_models)
    stage("recipe", lambda: _recipe(chosen, corpus, run))
    # openWakeWord's trainer (wake_training.trainer), one stage per step, each step by its own flag and what it made.
    made = run / chosen.name
    clips = [made / name for name in CLIPS]
    features = [made / f"{name}.npy" for name in FEATURES]
    stage("generate_clips", lambda: _finished(run, "generate_clips", clips, lambda: _clips(clips)))
    stage("augment_clips", lambda: _finished(run, "augment_clips", features, lambda: _features(features)))
    stage("train_model", lambda: _trained(run, chosen))
    stage("evaluate", lambda: _evaluate(chosen, corpus, run))
    model = to / f"{chosen.name}.onnx"
    stage("install", lambda: _install(chosen, run, model))
    # A JSON string is a TOML basic string, escaped as TOML escapes.
    print(f"[talk]\nwake_word = {json.dumps(chosen.phrase)}\nwake_word_model = {json.dumps(str(model))}")
    return 0


def _fetch(corpus: Corpus, asset: Asset) -> dict[str, Fact]:
    fetched = fetch(corpus, asset)
    at = corpus.at(asset)
    return {"asset": asset.path, "fetched": int(fetched), "bytes": sum(path.stat().st_size for path in (at, *at.rglob("*")) if path.is_file())}


def _feature_models() -> dict[str, Fact]:
    # The trainer computes features with the melspectrogram and embedding models installed beside openwakeword.
    openwakeword.utils.download_models()  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
    return {}


def _recipe(chosen: Word, corpus: Corpus, run: Path) -> dict[str, Fact]:
    (run / "piper").mkdir(parents=True, exist_ok=True)
    (run / "training.yaml").write_text(yaml.safe_dump(recipe.config(chosen, corpus, run)))
    (run / "piper" / "generate_samples.py").write_text(recipe.adapter(corpus))
    return {"samples": recipe.SAMPLES, "validation_samples": recipe.VALIDATION_SAMPLES, "steps": recipe.STEPS}


def _finished(run: Path, step: str, outputs: Sequence[Path], made: Callable[[], dict[str, Fact]]) -> dict[str, Fact]:
    """The trainer's `step` finished, its clips or features `outputs`. The trainer would keep outputs it finds, finished
    or cut short, so [LAW:one-source-of-truth] what says they are finished is the step's mark, made once it exits
    cleanly: a step with one is not run again, and a step without one starts over from none."""
    done = run / f"{step}.done"
    if not done.exists():
        for output in outputs:
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink(missing_ok=True)
        _train(run, step)
        done.touch()
    return made()


def _trained(run: Path, chosen: Word) -> dict[str, Fact]:
    _train(run, "train_model")
    return {"model_bytes": (run / f"{chosen.name}.onnx").stat().st_size}


def _train(run: Path, step: str) -> None:
    subprocess.run([sys.executable, "-m", "wake_training.trainer", "--training_config", str(run / "training.yaml"), f"--{step}"], check=True)


def _clips(made: Sequence[Path]) -> dict[str, Fact]:
    return {clips.name: sum(1 for _ in clips.glob("*.wav")) for clips in made}


def _features(made: Sequence[Path]) -> dict[str, Fact]:
    return {features.stem: int(np.load(features, mmap_mode="r").shape[0]) for features in made}


def _evaluate(chosen: Word, corpus: Corpus, run: Path) -> dict[str, Fact]:
    model = run / f"{chosen.name}.onnx"
    made = run / chosen.name
    validation = np.load(corpus.at(VALIDATION_FEATURES))
    return evaluation(
        peaks(model, sorted((made / "positive_test").glob("*.wav"))),
        peaks(model, sorted((made / "negative_test").glob("*.wav"))),
        wakes(scores(model, validation)),
        len(validation) * FRAME_SECONDS / 3600,
    )


def _install(chosen: Word, run: Path, model: Path) -> dict[str, Fact]:
    # Copied whole, then renamed into place: hands never loads a model half copied.
    partial = model.with_name(model.name + ".partial")
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / f"{chosen.name}.onnx", partial)
    partial.replace(model)
    return {"model": str(model)}


if __name__ == "__main__":
    sys.exit(main())
