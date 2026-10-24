"""How the trained model hears its word: the clips the trainer held out, run through openWakeWord's runtime, the
same `Model` hands listens with, at the same threshold."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime
from numpy.typing import NDArray
from openwakeword.model import Model

from wake_training.stage import Fact

# hands.voice.wake.THRESHOLD: the score at which hands takes the wake word as said. This package does not import hands.
THRESHOLD = 0.5
# openWakeWord makes one feature frame of each 80 ms of audio.
FRAME_SECONDS = 0.08


def peaks(model: Path, clips: Sequence[Path]) -> list[float]:
    """Each clip's highest score from the model, streamed through it as hands streams the microphone."""
    heard = Model(wakeword_models=[str(model)], inference_framework="onnx")
    scores: list[float] = []
    for clip in clips:
        heard.reset()
        frames = cast(list[dict[str, float]], heard.predict_clip(str(clip)))  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
        scores.append(max(score for frame in frames for score in frame.values()))
    return scores


def scores(model: Path, features: NDArray[np.float32]) -> tuple[list[float], int]:
    """The model's score on each window of `features` as the window slides on one frame at a time, as it slides over
    the microphone's; and how many frames a window is."""
    session = onnxruntime.InferenceSession(str(model))
    given = session.get_inputs()[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  (untyped in onnxruntime)
    name = cast(str, given.name)  # pyright: ignore[reportUnknownMemberType]
    frames = cast(int, given.shape[1])  # pyright: ignore[reportUnknownMemberType]
    run = cast(Callable[[None, dict[str, NDArray[np.float32]]], list[NDArray[np.float32]]], session.run)  # pyright: ignore[reportUnknownMemberType]  (untyped in onnxruntime)
    # The trainer exports the model for one window at a time.
    windows = np.lib.stride_tricks.sliding_window_view(features, frames, axis=0).transpose(0, 2, 1)
    said = [float(run(None, {name: windows[n][None]})[0].item()) for n in range(len(windows))]
    return said, frames


def wakes(said: Sequence[float], deaf: int) -> int:
    """How many times hands wakes on these scores: at each score at its threshold, after which it is deaf for `deaf`
    frames, as hands' reset of the model after a wake refills its window."""
    woken, frame = 0, 0
    while frame < len(said):
        heard = said[frame] >= THRESHOLD
        woken += heard
        frame += deaf if heard else 1
    return woken


def evaluation(positives: Sequence[float], negatives: Sequence[float], false_wakes: int, hours: float) -> dict[str, Fact]:
    """The share of the word's clips the model wakes on, and of the near-miss phrases' clips it wrongly wakes on; and
    how often it wakes on hours of speech, music and noise that never say it."""
    return {
        "threshold": THRESHOLD,
        "positives": len(positives),
        "recall": round(sum(score >= THRESHOLD for score in positives) / len(positives), 4),
        "negatives": len(negatives),
        "false_accepts": round(sum(score >= THRESHOLD for score in negatives) / len(negatives), 4),
        "hours": round(hours, 2),
        "false_wakes": false_wakes,
        "false_wakes_per_hour": round(false_wakes / hours, 3),
    }
