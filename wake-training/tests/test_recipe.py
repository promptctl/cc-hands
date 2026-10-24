import importlib.util
from collections import Counter
import sys
import wave
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from wake_training import corpus as corpus_module
from wake_training import recipe
from wake_training.corpus import BACKGROUND, IMPULSES, PIPER, TRAINING_FEATURES, VALIDATION_FEATURES, Corpus
from wake_training.word import word


def test_the_trainer_is_told_the_word_and_where_the_corpus_is(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    run = tmp_path / "words" / "okay_hands"
    config = recipe.config(word("Okay Hands"), corpus, run)
    assert config["model_name"] == "okay_hands"
    assert config["target_phrase"] == ["okay hands"]
    near = config["custom_negative_phrases"]
    assert isinstance(near, list) and set(near) == set(word("Okay Hands").near_misses) and len(near) <= recipe.NEAR_MISSES  # pyright: ignore[reportUnknownArgumentType]
    assert config["output_dir"] == str(run)
    assert config["piper_sample_generator_path"] == str(run / "piper")
    assert config["rir_paths"] == [str(corpus.at(IMPULSES))]
    assert config["background_paths"] == [str(corpus.at(BACKGROUND))]
    assert config["false_positive_validation_data_path"] == str(corpus.at(VALIDATION_FEATURES))
    assert config["feature_data_files"] == {"ACAV100M_sample": str(corpus.at(TRAINING_FEATURES))}
    assert set(config["batch_n_per_class"]) == {"ACAV100M_sample", "adversarial_negative", "positive"}  # type: ignore[arg-type]


def test_a_run_is_named_for_what_its_clips_are_made_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = Corpus(tmp_path / "corpus")
    hey = recipe.made(word("Hey Hands"), corpus)
    assert hey.startswith("hey_hands-") and hey == recipe.made(word("hey  hands"), corpus)
    assert recipe.made(word("Okay Hands"), corpus) != hey
    assert recipe.made(word("Hey Hands"), Corpus(tmp_path / "elsewhere")) != hey
    # How the model is trained is not what its clips are made from: the run, and its clips, stay.
    monkeypatch.setattr(recipe, "STEPS", recipe.STEPS * 2)
    assert recipe.made(word("Hey Hands"), corpus) == hey
    monkeypatch.setattr(recipe, "SAMPLES", recipe.SAMPLES * 2)
    assert recipe.made(word("Hey Hands"), corpus) != hey


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


def test_the_held_out_negatives_say_every_near_miss_alike(tmp_path: Path) -> None:
    # The trainer says the phrases from the first, and holds out the first tenth it says.
    hey = word("Hey Hands")
    said = cast(list[str], recipe.clips(hey, Corpus(tmp_path))["custom_negative_phrases"])
    held_out = Counter(said[: recipe.VALIDATION_SAMPLES])
    assert set(held_out) == set(hey.near_misses) and max(held_out.values()) - min(held_out.values()) <= 1


def test_a_run_made_with_another_generator_is_a_run_of_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = Corpus(tmp_path / "corpus")
    hey = recipe.made(word("Hey Hands"), corpus)
    monkeypatch.setattr(corpus_module, "GENERATOR", "another.pt")
    assert recipe.made(word("Hey Hands"), corpus) != hey
