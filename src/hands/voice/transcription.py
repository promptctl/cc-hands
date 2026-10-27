"""NVIDIA Parakeet v3 on MLX, in hands' own process: the words a hold's audio is heard as; nothing here loads Pipecat.

The voice transcribes every hold through it, and `hands smoke` judges what hands said aloud with it, so what the smoke
test hears is what the voice would have heard. Parakeet decodes a 20 s hold in 0.15 s warm, and it hears nothing in
silence, a tone or most noise, where Whisper large-v3-turbo heard "Thank you." in every one of 120 clips of white noise
its scores kept (promptctl-voice-r7y.8rr, 2026-10-26). It takes no prompt and no language: it hears 25, English among
them.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from parakeet_mlx import from_pretrained
from parakeet_mlx.alignment import AlignedToken, merge_longest_common_subsequence, merge_longest_contiguous
from parakeet_mlx.audio import get_logmel

# The model every hold is transcribed with, fetched from the Hugging Face hub into its cache on the first load.
MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# The sample rate Parakeet hears at; the audio it is given is mono 16-bit at this rate.
RATE = 16_000

# The window a hold is decoded in, and how far each overlaps the one before, parakeet-mlx's own for long audio. Ten
# minutes of speech decoded whole peaked at 15 GB of GPU memory and took 9.3 s; in these windows, 3.2 GB and 5.1 s.
WINDOW_SECONDS = 120
OVERLAP_SECONDS = 15

Samples = npt.NDArray[np.float32]

# A second of silence: what the model is loaded by.
_SILENCE = np.zeros(RATE, dtype=np.float32)


@dataclass(frozen=True)
class Model:
    """Parakeet loaded: the tokens it hears in a window of samples at RATE, each timed from the window's start."""

    window: Callable[[Samples], list[AlignedToken]]


def load() -> Model:
    """Fetch the model, load it, and run it once: the first run after a load is slower than every later one, and it is
    paid here rather than by the first hold [LAW:no-ambient-temporal-coupling]."""
    parakeet = from_pretrained(MODEL)

    def window(samples: Samples) -> list[AlignedToken]:
        [heard] = parakeet.generate(get_logmel(mx.array(samples), parakeet.preprocessor_config))
        return heard.tokens

    model = Model(window)
    transcribed(model, _SILENCE)
    return model


def decoded(audio: bytes) -> Samples:
    """Mono 16-bit `audio` as the samples in [-1, 1) a model hears; the one decoding of a hold, for Parakeet and the
    diarizer alike."""
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / np.float32(32768)


def transcribed(model: Model, samples: Samples) -> str:
    """What `model` hears in `samples` at RATE, "" where it hears nothing. It blocks while the model runs.

    [LAW:dataflow-not-control-flow] every hold is decoded in windows, a short one in one: each window after the first
    starts `OVERLAP_SECONDS` before the one before it ends, and the tokens heard in the overlap are merged on the longest
    run the two windows agree on, or failing one, on their longest common subsequence, as parakeet-mlx merges them.
    """
    window, step = WINDOW_SECONDS * RATE, (WINDOW_SECONDS - OVERLAP_SECONDS) * RATE
    tokens: list[AlignedToken] = []
    for start in range(0, max(len(samples) - OVERLAP_SECONDS * RATE, 1), step):
        heard = [replace(token, start=token.start + start / RATE) for token in model.window(samples[start : start + window])]
        try:
            tokens = merge_longest_contiguous(tokens, heard, overlap_duration=OVERLAP_SECONDS)
        except RuntimeError:
            tokens = merge_longest_common_subsequence(tokens, heard, overlap_duration=OVERLAP_SECONDS)
    return "".join(token.text for token in tokens).strip()
