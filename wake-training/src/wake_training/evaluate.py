"""How the trained model hears its word: the clips the trainer held out, run through openWakeWord's runtime, the
same `Model` hands listens with, at the same threshold."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from openwakeword.model import Model

from wake_training.stage import Fact

# hands.voice.wake.THRESHOLD: the score at which hands takes the wake word as said. This package does not import hands.
THRESHOLD = 0.5


def peaks(model: Path, clips: Sequence[Path]) -> list[float]:
    """Each clip's highest score from the model, streamed through it as hands streams the microphone."""
    heard = Model(wakeword_models=[str(model)], inference_framework="onnx")
    scores: list[float] = []
    for clip in clips:
        heard.reset()
        frames = cast(list[dict[str, float]], heard.predict_clip(str(clip)))  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
        scores.append(max(score for frame in frames for score in frame.values()))
    return scores


def evaluation(positives: Sequence[float], negatives: Sequence[float]) -> dict[str, Fact]:
    """The share of the word's clips the model wakes on, and of the near-miss phrases' clips it wrongly wakes on."""
    return {
        "threshold": THRESHOLD,
        "positives": len(positives),
        "recall": round(sum(score >= THRESHOLD for score in positives) / len(positives), 4),
        "negatives": len(negatives),
        "false_accepts": round(sum(score >= THRESHOLD for score in negatives) / len(negatives), 4),
    }
