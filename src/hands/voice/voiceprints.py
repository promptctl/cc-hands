"""The voiceprint model: 3D-Speaker's ERes2Net speaker embedding run by sherpa-onnx on the CPU; nothing here loads Pipecat.

A voiceprint is the model's embedding of a stretch of speech, scaled to unit length, so two compare by their cosine
similarity. The model is kept for telling people apart; nothing uses it to tell whose words are hands': hands' voice is
taken out of the sound before anything hears it (`hands.voice.echo`), so the words left in it are a person's.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from hands.voice import pins
from hands.voice.clips import RATE, Samples

# The model, fetched from the Hugging Face hub into its cache, at the revision measured for it.
MODEL = pins.VOICEPRINT.repo
REVISION = pins.VOICEPRINT.revision
FILE = pins.VOICEPRINT_FILE

# A speaker's embedding, of unit length.
type Print = npt.NDArray[np.float32]

# The model: a stretch of mono speech at RATE, as its voiceprint.
type Embed = Callable[[Samples], Print]


def weights() -> str:
    """The path of the model's FILE at REVISION, in the Hugging Face cache, fetched there if it never was."""
    # The hub's download types one argument incompletely.
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

    return hf_hub_download(MODEL, FILE, revision=REVISION)


def load() -> Embed:
    """The model, from weights.

    [LAW:no-silent-failure] a model that does not load raises naming the file."""
    # sherpa-onnx ships no types.
    import sherpa_onnx  # pyright: ignore[reportMissingTypeStubs]

    try:
        path = weights()
        extractor = cast(Any, sherpa_onnx).SpeakerEmbeddingExtractor(cast(Any, sherpa_onnx).SpeakerEmbeddingExtractorConfig(model=path, num_threads=2))
    except Exception as error:
        raise RuntimeError(f"the voiceprint model {FILE} from {MODEL} at {REVISION} did not load: {type(error).__name__}: {error}") from error

    def embed(voice: Samples) -> Print:
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=RATE, waveform=voice)
        stream.input_finished()
        embedded = np.asarray(extractor.compute(stream), np.float32)
        return embedded / np.linalg.norm(embedded)

    return embed
