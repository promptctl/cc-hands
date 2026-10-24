"""What openWakeWord's trainer, `openwakeword.train`, is told to do for one word: its YAML config, and the adapter
that lets it drive the piper-sample-generator this corpus carries. Both are text, made here and written by the run.
"""

import hashlib
import json
from pathlib import Path

from wake_training.corpus import BACKGROUND, IMPULSES, PIPER, TRAINING_FEATURES, VALIDATION_FEATURES, Corpus
from wake_training.word import Word

# The sizes of a run, the same for every word. openWakeWord's custom_model.yml recommends at least 20,000 positive
# clips; the trainer generates as many adversarial negatives, and a tenth of each again to validate on.
SAMPLES = 20_000
VALIDATION_SAMPLES = 2_000
STEPS = 50_000
# How many of the negative clips are said as the word's near misses: a quarter, the rest being the trainer's own
# adversarial phrases. The trainer says the phrases it is given in turn, from the first, and takes as many as it wants
# clips: its held-out set, a tenth the size, is the list's first tenth. So the near misses are dealt in rounds, each
# round every one of them, and every share of the list holds them all in equal parts.
NEAR_MISSES = SAMPLES // 4


def clips(word: Word, corpus: Corpus) -> dict[str, object]:
    """What the trainer makes `word`'s clips and features from, as its config's keys."""
    near = word.near_misses
    return {
        "target_phrase": [word.spoken],
        "custom_negative_phrases": [miss for _ in range(NEAR_MISSES // len(near)) for miss in near],
        "n_samples": SAMPLES,
        "n_samples_val": VALIDATION_SAMPLES,
        "rir_paths": [str(corpus.at(IMPULSES))],
        "background_paths": [str(corpus.at(BACKGROUND))],
        "background_paths_duplication_rate": [1],
        "augmentation_rounds": 1,
    }


def made(word: Word, corpus: Corpus) -> str:
    """The name of the run that makes `word`'s clips and features: the word's, and a digest of what they are made
    from, the trainer's config for them and the generator that says them. [LAW:one-source-of-truth] the trainer keeps
    clips and features it finds and does not ask what they were made from, so a run's name is what says it: a change
    to any of it is a run of its own, made fresh."""
    made_from = {"clips": clips(word, corpus), "generator": adapter(corpus)}
    digest = hashlib.sha256(json.dumps(made_from, sort_keys=True).encode()).hexdigest()
    return f"{word.name}-{digest[:8]}"


def config(word: Word, corpus: Corpus, run: Path) -> dict[str, object]:
    """The trainer's config for `word`, its clips, features and model made under `run`. The keys are those of
    openWakeWord's examples/custom_model.yml; the values not about this word or this corpus are that file's own, but
    for the model's width and its weight on negatives, which are doubled: on "Hey Hands", that took false wakes from
    1.1 an hour to 0.4, under Hey Jarvis's 0.5, for recall 98% to 97% (docs/wake-word-training.md)."""
    return {
        **clips(word, corpus),
        "model_name": word.name,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(run / "piper"),
        "output_dir": str(run),
        "false_positive_validation_data_path": str(corpus.at(VALIDATION_FEATURES)),
        "feature_data_files": {"ACAV100M_sample": str(corpus.at(TRAINING_FEATURES))},
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
        "model_type": "dnn",
        "layer_size": 64,
        "steps": STEPS,
        "max_negative_weight": 3000,
        "target_false_positives_per_hour": 0.2,
    }


def adapter(corpus: Corpus) -> str:
    """The `generate_samples` module the trainer imports from `piper_sample_generator_path`. The trainer was written
    against an older piper-sample-generator: it passes no generator model, and it takes only 16 kHz clips, where
    the generator writes at its voices' own rate. The adapter binds the corpus's generator, and brings every clip
    it wrote to 16 kHz."""
    return f'''"""Written by wake_training.recipe.adapter; see there."""

import sys
from pathlib import Path

import soundfile as sf

from wake_training import audio

sys.path.insert(0, {str(corpus.at(PIPER))!r})
from piper_sample_generator.__main__ import generate_samples as _generate  # noqa: E402


def generate_samples(output_dir, **settings):
    _generate(output_dir=output_dir, model={str(corpus.generator)!r}, **settings)
    for clip in Path(output_dir).glob("*.wav"):
        audio.write(clip, *sf.read(clip))
'''
