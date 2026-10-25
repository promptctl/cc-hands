"""Whisper on MLX, in hands' own process: the segments a hold's WAV is heard as, each with Whisper's two scores for
dropping it; nothing here loads Pipecat.

The voice transcribes every hold through it, and `hands smoke` judges what hands said aloud with it, so what the smoke
test hears is what the voice would have heard. LowTalker serves the same large-v3-turbo weights from the Neural Engine,
and hands measured it slower: key release to transcript 613 to 724 ms through LowTalker against 298 to 329 ms here, over
the same six spoken turns, by upload and by its Realtime socket alike (hands-dictation-2bs.kpm, 2026-10-04).
"""

from typing import Any, cast

import mlx_whisper
import numpy as np
import numpy.typing as npt

from hands.sessions.audit import Unsaid

# What every hold is said in.
LANGUAGE = "en"

# The model every hold is transcribed with, fetched from the Hugging Face hub into its cache on the first load.
MODEL = "mlx-community/whisper-large-v3-turbo"

# The sample rate Whisper hears at; the audio it is given is mono 16-bit at this rate.
RATE = 16_000

# A second of silence: what the model is loaded by.
_SILENCE = np.zeros(RATE, dtype=np.float32)


def load() -> None:
    """Fetch the model, load it, and run it once.

    MLX Whisper loads its model inside the first transcription it is asked for, downloading it first if it was never
    fetched: the first turn of the first run waited 28 s on a 1.6 GB download (2026-09-26). Once fetched, loading takes
    0.5 s and the first run after it 1.3 s against 0.3 s for every later one. The model is kept for the process, keyed
    on its name, so silence transcribed here through the call every hold makes leaves every hold finding it loaded and
    run [LAW:no-ambient-temporal-coupling].
    """
    _transcribe(_SILENCE, None)


def decoded(audio: bytes) -> npt.NDArray[np.float32]:
    """Mono 16-bit `audio` as the samples in [-1, 1) a model hears; the one decoding of a hold, for Whisper and the
    diarizer alike."""
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / np.float32(32768)


def segments(audio: bytes, prompt: str | None) -> list[Unsaid]:
    """The segments Whisper hears in mono 16-bit `audio` at RATE, primed with `prompt`. It blocks while the model runs."""
    return [
        Unsaid(str(segment["text"]).strip(), float(segment["compression_ratio"]), float(segment["avg_logprob"]))
        for segment in _transcribe(decoded(audio), prompt)
    ]


def _transcribe(audio: np.ndarray, prompt: str | None) -> list[dict[str, Any]]:
    """The segments MLX Whisper heard in `audio`, greedily decoded, primed with `prompt`; what the load and every hold run.

    Each 30 s window is decoded on its own, not conditioned on the text of the one before: conditioned, a long hold fell
    into a loop ("$.h. $.h. ...") that carried from window to window, and the compression filter dropped every segment
    after it, the back half of a 600 s hold of spoken prose; decoded on their own, 180 s and 600 s lost none
    (promptctl-voice-je8.d20). MLX Whisper then primes only the first window with `prompt`.
    """
    transcribed: dict[str, Any] = mlx_whisper.transcribe(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  (untyped in mlx_whisper)
        audio, path_or_hf_repo=MODEL, temperature=0.0, language=LANGUAGE, initial_prompt=prompt, condition_on_previous_text=False
    )
    return cast("list[dict[str, Any]]", transcribed["segments"])
