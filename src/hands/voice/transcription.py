"""Whisper on MLX, in hands' own process: the segments a clip's samples are heard as, each with the stretch of it it was heard
in and Whisper's two scores for dropping it; nothing here loads Pipecat.

The voice transcribes every clip through it, and `hands smoke` judges what hands said aloud with it, so the smoke test
hears with the model the voice hears with; what the voice drops of what it heard is the voice's (`hands.voice.whisper`). LowTalker serves the same large-v3-turbo weights from the Neural Engine,
and hands measured it slower: key release to transcript 613 to 724 ms through LowTalker against 298 to 329 ms here, over
the same six spoken turns, by upload and by its Realtime socket alike (hands-dictation-2bs.kpm, 2026-10-04).
"""

import wave
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any, cast

import mlx_whisper
import numpy as np
from huggingface_hub import is_offline_mode, snapshot_download  # pyright: ignore[reportUnknownVariableType]  (partly untyped in huggingface_hub)
from huggingface_hub.errors import LocalEntryNotFoundError

from hands.voice.clips import LANGUAGE, decoded
from hands.voice.mlxthread import OnMLX

# The model every clip is transcribed with, fetched from the Hugging Face hub into its cache on the first load, at the
# revision pinned: the weights a test measured are the weights hands runs, and CI keys its cache of them on it.
MODEL = "mlx-community/whisper-large-v3-turbo"
REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"

# hands' own TTS saying "Hands here.", a second of it, mono 16-bit at a clip's rate: what the model is loaded by.
_HANDS_HERE = files("hands.voice") / "hands_here.wav"


def load(on: OnMLX) -> None:
    """Fetch the model, load it, and run it once, on the MLX thread.

    MLX Whisper loads its model inside the first transcription it is asked for, downloading it first if it was never
    fetched: the first turn of the first run waited 28 s on a 1.6 GB download (2026-09-26). Once fetched, loading takes
    0.5 s and the first run after it 1.3 s against 0.3 s for every later one. The model is kept for the process, keyed on
    the directory `weights` names, so a sentence transcribed here leaves every clip finding it loaded and run, whichever
    way it is decoded [LAW:no-ambient-temporal-coupling].
    """
    with wave.open(str(_HANDS_HERE)) as said:
        _transcribe(on, decoded(said.readframes(said.getnframes())), None)


@dataclass(frozen=True)
class Segment:
    """What Whisper heard in one stretch of a clip, from `start` to `end` seconds into it, with its two scores for dropping
    it."""

    text: str
    start: float
    end: float
    compression_ratio: float
    avg_logprob: float


def segments(on: OnMLX, samples: np.ndarray, prompt: str | None) -> list[Segment]:
    """The segments Whisper hears in mono `samples` at RATE, primed with `prompt`, on the MLX thread."""
    return [
        Segment(
            str(segment["text"]).strip(),
            float(segment["start"]),
            float(segment["end"]),
            float(segment["compression_ratio"]),
            float(segment["avg_logprob"]),
        )
        for segment in _transcribe(on, samples, prompt)
    ]


@cache
def weights() -> str:
    """The directory MODEL's weights are in at REVISION, fetched into the hub's cache if they are not all there yet; the
    hub is not asked again once they are, since a commit names its files for good. Asked for a commit, the hub's client
    lists its files from the hub unless it kept the list, offline or not, so the cache is looked in first.

    The client keeps the list before it fetches a file, and the cache refuses a snapshot the list names a file it lacks
    with an IncompleteSnapshotError naming each one: one a fetch cut short is fetched whole here, and offline that error
    is what is raised, since the fetch could only fail on the first file it asked the hub for. A snapshot with no list
    kept, fetched by a client older than the list, is taken as whole [LAW:single-enforcer]."""
    try:
        return snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        if is_offline_mode():
            raise
        return snapshot_download(MODEL, revision=REVISION)


def _transcribe(on: OnMLX, audio: np.ndarray, prompt: str | None) -> list[dict[str, Any]]:
    """The segments MLX Whisper heard in `audio`, greedily decoded, primed with `prompt`; what the load and every clip run.

    Each 30 s window is decoded on its own, not conditioned on the text of the one before: conditioned, a long clip fell
    into a loop ("$.h. $.h. ...") that carried from window to window, and the compression filter dropped every segment
    after it, the back half of a 600 s clip of spoken prose; decoded on their own, 180 s and 600 s lost none
    (promptctl-voice-je8.d20). MLX Whisper then primes only the first window with `prompt`.
    """
    transcribed: dict[str, Any] = mlx_whisper.transcribe(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  (untyped in mlx_whisper)
        audio, path_or_hf_repo=weights(), temperature=0.0, language=LANGUAGE, initial_prompt=prompt, condition_on_previous_text=False
    )
    return cast("list[dict[str, Any]]", transcribed["segments"])
