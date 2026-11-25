"""The voiceprint model: its prints are of unit length, and a model that does not load fails loudly, naming its file."""

import numpy as np
import pytest

from hands.voice import voiceprints
from hands.voice.clips import RATE


@pytest.mark.outside
@pytest.mark.weights("voiceprint")
def test_the_model_loads_and_prints_are_of_unit_length() -> None:
    embed = voiceprints.load()
    assert np.linalg.norm(embed(np.random.default_rng(0).normal(0, 0.1, RATE).astype(np.float32))) == pytest.approx(1.0)


def test_a_model_that_does_not_load_fails_loudly_naming_its_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voiceprints, "FILE", "no_such_model.onnx")
    with pytest.raises(RuntimeError, match="the voiceprint model no_such_model.onnx from csukuangfj/speaker-embedding-models at 0743f30"):
        voiceprints.load()
