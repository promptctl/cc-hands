import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from wake_training import recipe
from wake_training.corpus import BACKGROUND, IMPULSES, PIPER, TRAINING_FEATURES, VALIDATION_FEATURES, Corpus
from wake_training.word import word


def test_the_trainer_is_told_the_word_and_where_the_corpus_is(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    run = tmp_path / "words" / "okay_hands"
    config = recipe.config(word("Okay Hands"), corpus, run)
    assert config["model_name"] == "okay_hands"
    assert config["target_phrase"] == ["okay hands"]
    assert config["output_dir"] == str(run)
    assert config["piper_sample_generator_path"] == str(run / "piper")
    assert config["rir_paths"] == [str(corpus.at(IMPULSES))]
    assert config["background_paths"] == [str(corpus.at(BACKGROUND))]
    assert config["false_positive_validation_data_path"] == str(corpus.at(VALIDATION_FEATURES))
    assert config["feature_data_files"] == {"ACAV100M_sample": str(corpus.at(TRAINING_FEATURES))}
    assert set(config["batch_n_per_class"]) == {"ACAV100M_sample", "adversarial_negative", "positive"}  # type: ignore[arg-type]


FAKE_GENERATOR = '''
import wave
from pathlib import Path

calls = []


def generate_samples(text, output_dir, model, max_samples=None, **settings):
    calls.append({"text": text, "model": model, "max_samples": max_samples, **settings})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for n in range(max_samples):
        with wave.open(str(Path(output_dir) / f"{n}.wav"), "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(22050)
            clip.writeframes(b"\\x00\\x10" * 22050)
'''


def test_the_adapter_binds_the_corpus_generator_and_brings_clips_to_16_khz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = Corpus(tmp_path / "corpus")
    package = corpus.at(PIPER) / "piper_sample_generator"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(FAKE_GENERATOR)
    adapter = tmp_path / "generate_samples.py"
    adapter.write_text(recipe.adapter(corpus))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [name for name in sys.modules if name.startswith("piper_sample_generator")]:
        monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("generate_samples", adapter)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    clips = tmp_path / "clips"
    module.generate_samples(text=["hey hands"], max_samples=2, output_dir=str(clips), batch_size=5, auto_reduce_batch_size=True)

    calls = sys.modules["piper_sample_generator.__main__"].calls
    assert calls == [{"text": ["hey hands"], "model": str(corpus.generator), "max_samples": 2, "batch_size": 5, "auto_reduce_batch_size": True}]
    for clip in sorted(clips.glob("*.wav")):
        with wave.open(str(clip)) as heard:
            assert (heard.getframerate(), heard.getnframes()) == (16_000, 16_000)
            samples = np.frombuffer(heard.readframes(heard.getnframes()), dtype=np.int16)
        assert np.median(samples) == 0x1000
