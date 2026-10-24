"""What openWakeWord's trainer, `openwakeword.train`, is told to do for one word: its YAML config, and the adapter
that lets it drive the piper-sample-generator this corpus carries. Both are text, made here and written by the run.
"""

from pathlib import Path

from wake_training.corpus import BACKGROUND, IMPULSES, PIPER, TRAINING_FEATURES, VALIDATION_FEATURES, Corpus
from wake_training.word import Word

# The sizes of a run, the same for every word. openWakeWord's custom_model.yml recommends at least 20,000 positive
# clips; the trainer generates as many adversarial negatives, and a tenth of each again to validate on.
SAMPLES = 20_000
VALIDATION_SAMPLES = 2_000
STEPS = 50_000


def config(word: Word, corpus: Corpus, run: Path) -> dict[str, object]:
    """The trainer's config for `word`, its clips, features and model made under `run`. The keys are those of
    openWakeWord's examples/custom_model.yml; the values not about this word or this corpus are that file's own."""
    return {
        "model_name": word.name,
        "target_phrase": [word.spoken],
        "custom_negative_phrases": [],
        "n_samples": SAMPLES,
        "n_samples_val": VALIDATION_SAMPLES,
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(run / "piper"),
        "output_dir": str(run),
        "rir_paths": [str(corpus.at(IMPULSES))],
        "background_paths": [str(corpus.at(BACKGROUND))],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(corpus.at(VALIDATION_FEATURES)),
        "augmentation_rounds": 1,
        "feature_data_files": {"ACAV100M_sample": str(corpus.at(TRAINING_FEATURES))},
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
        "model_type": "dnn",
        "layer_size": 32,
        "steps": STEPS,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }


def adapter(corpus: Corpus) -> str:
    """The `generate_samples` module the trainer imports from `piper_sample_generator_path`. The trainer was written
    against an older piper-sample-generator: it passes no generator model, and it takes only 16 kHz clips, where
    the generator writes at its voices' own rate. The adapter binds the corpus's generator, and brings every clip
    it wrote to 16 kHz."""
    return f'''"""Written by wake_training.recipe.adapter; see there."""

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, {str(corpus.at(PIPER))!r})
from piper_sample_generator.__main__ import generate_samples as _generate  # noqa: E402

RATE = 16_000


def generate_samples(output_dir, **settings):
    _generate(output_dir=output_dir, model={str(corpus.generator)!r}, **settings)
    for clip in Path(output_dir).glob("*.wav"):
        if (rate := sf.info(clip).samplerate) != RATE:
            samples, _ = sf.read(clip, dtype="int16")
            common = gcd(RATE, rate)
            resampled = resample_poly(samples.astype(np.float32), RATE // common, rate // common)
            sf.write(clip, np.clip(resampled, -32768, 32767).astype(np.int16), RATE, subtype="PCM_16")
'''
