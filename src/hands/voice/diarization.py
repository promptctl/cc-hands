"""Speaker diarization on NVIDIA's Nemotron 3 Diarization, in hands' own process: which speakers spoke in each hold the
key sent or dropped, a speaker keeping its index from the first hold of the run to the last; nothing here loads Pipecat.

The model is a Streaming Sortformer: it scores up to eight speakers in every 10 ms frame, its speakers ordered by when
each was first heard, and the arrival-order speaker cache it hands back after each step carries who has been heard into
the next. hands keeps one stream for the run and feeds it only the holds, as they are spoken, so a speaker is the same
speaker in every hold, which one embedding per hold against a fixed threshold did not manage (promptctl-voice-r7y.dy1).

It runs on the CPU, at about nine times real time a step, which leaves the GPU to Whisper: a hold is scored while it is
spoken, so when it ends only its last step and the look-ahead after it are left to score. It is one listener on the hold
owner (`hands.voice.holdowner`), which gives it every hold of the key's as it is captured.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt


from hands.sessions.audit import Record
from hands.sessions.wide import annotate, fail, unit
from hands.threads import SerialThread
from hands.voice.holds import RATE, Fate, Heard, Hearings, Hold, Sound, decoded

# The model, fetched from the Hugging Face hub into its cache on the first load, at the revision measured for it.
MODEL = "nvidia/Nemotron-3-Diarization"
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"

# The model's streaming mode, the best-scoring of them: a step scores 0.72 s and looks 0.32 s past it, so a frame is
# known 1.04 s after it is spoken at the latest.
STREAMING_MODE = "low_latency"

# Samples to a 10 ms frame, the frame every speaker is scored in.
FRAME = 160

# The probability above which a frame is the speaker's, the model card's own.
ACTIVE = 0.5

type Samples = npt.NDArray[np.float32]
# Each speaker's probability in each frame a step scored: frames x 8.
type Probabilities = npt.NDArray[np.float32]
# One step of the model: a chunk of samples, whether it is the first of the run, and the speaker cache the step before
# handed back; the step's frames scored, and the cache to hand the next.
type Step = Callable[[Samples, bool, Any], tuple[Probabilities, Any]]

@dataclass(frozen=True)
class Chunking:
    """How the model's steps cut the run's samples, as its processor says: the first step's chunk, each later step's,
    the frames each step scores, and how far before its first frame a later chunk starts."""

    first: int
    later: int
    frames: int
    lead: int

    def start(self, step: int) -> int:
        """The first sample of `step`'s chunk."""
        return 0 if step == 0 else step * self.frames * FRAME - self.lead

    def end(self, step: int) -> int:
        """The sample after the last of `step`'s chunk: the samples it waits for."""
        return self.first if step == 0 else self.start(step) + self.later


@dataclass(frozen=True)
class Model:
    """The model's step, and how it cuts the samples it is given."""

    step: Step
    chunking: Chunking


@dataclass(frozen=True)
class Speech:
    """One speaker's speech in a hold: the speaker, by when the run first heard them, 0 the first; and for how many
    seconds of the hold they were heard speaking."""

    speaker: int
    seconds: float


@dataclass(frozen=True)
class Diarized:
    """A hold's speakers, longest first, none where nobody was heard speaking; and how many steps scored it."""

    speakers: tuple[Speech, ...]
    steps: int


@dataclass(frozen=True)
class Undiarized:
    """A hold whose speakers are not known, since a step scoring it failed or the model never loaded; and how many steps
    scored it."""

    error: str
    steps: int


def speakers(probabilities: Probabilities) -> tuple[Speech, ...]:
    """The seconds each speaker was active in a hold's frames, longest first; none where nobody was heard speaking."""
    seconds = (probabilities > ACTIVE).sum(axis=0) * FRAME / RATE
    return tuple(
        Speech(int(speaker), round(float(seconds[speaker]), 2)) for speaker in np.argsort(-seconds, kind="stable") if seconds[speaker] > 0
    )


class Stream:
    """The run's diarization: the samples not yet scored, the speaker cache, and the frames of the hold open.

    [LAW:one-source-of-truth] a frame's place in the run is its index; a hold is the frames from where it opened to where
    it ended, so every frame the model scores belongs to one hold or to the silence after one, never to two.
    """

    def __init__(self, model: Model) -> None:
        self._model = model
        # The run's samples from `_base` on: the next step's chunk starts at or after it.
        self._samples: Samples = np.zeros(0, np.float32)
        self._base = 0
        # The step to run next, and the cache the last that ran handed back, None before the first.
        self._step = 0
        self._cache: Any = None
        # The hold open: the frame it opened at, each speaker's probability in its frames scored so far, the steps that
        # scored them, and why it cannot be diarized, once a step in it has failed.
        self._opened = 0
        self._scored: list[Probabilities] = []
        self._steps = 0
        self._failed: str | None = None

    @property
    def _length(self) -> int:
        return self._base + len(self._samples)

    def heard(self, samples: Samples) -> None:
        """Samples of the hold open, in the order spoken: every step they complete is scored."""
        self._samples = np.concatenate([self._samples, samples])
        self._score()

    def ended(self) -> Diarized | Undiarized:
        """The speakers of the hold open, which ends here: it is followed by silence until every frame of it is scored,
        and the next hold opens after that silence."""
        chunking, frames = self._model.chunking, -(-self._length // FRAME)
        # The step that scores the hold's last frame, and every step before it, wait on no sample past the silence.
        last = -(-frames // chunking.frames) - 1
        waited = chunking.end(last) if frames else self._length
        self._samples = np.concatenate([self._samples, np.zeros(-(-waited // FRAME) * FRAME - self._length, np.float32)])
        self._score()
        scored = np.concatenate(self._scored) if self._scored else np.zeros((0, 8), np.float32)
        held = scored[: frames - self._opened]
        diarized = Diarized(speakers(held), self._steps) if self._failed is None else Undiarized(self._failed, self._steps)
        self._opened, self._scored, self._steps, self._failed = self._length // FRAME, [], 0, None
        return diarized

    def _score(self) -> None:
        """Run every step whose chunk the samples complete. [LAW:no-silent-failure] a step that raises is skipped, the
        cache as it was before it, and the hold it is in cannot be diarized: its frames were never scored."""
        chunking = self._model.chunking
        while chunking.end(self._step) <= self._length:
            start = chunking.start(self._step) - self._base
            chunk = self._samples[start : start + chunking.end(self._step) - chunking.start(self._step)]
            try:
                probabilities, self._cache = self._model.step(chunk, self._step == 0, self._cache)
            except Exception as error:
                probabilities = np.zeros((chunking.frames, 8), np.float32)
                self._failed = self._failed or f"step {self._step} failed: {type(error).__name__}: {error}"
            first = self._step * chunking.frames
            # Frames before the hold opened are the silence after the one before: no hold's.
            self._scored.append(probabilities[max(0, self._opened - first) :])
            self._steps += 1
            self._step += 1
            kept = chunking.start(self._step) - self._base
            self._samples, self._base = self._samples[kept:], self._base + kept


@dataclass(frozen=True)
class Unloaded:
    """The run's diarization where the model did not load: nothing scores what it hears, and each hold ends undiarized
    with why. [LAW:one-type-per-behavior] it hears and ends holds as a Stream does, so the Diarizer is the same either way."""

    error: str

    def heard(self, samples: Samples) -> None:
        """Samples no model scores."""

    def ended(self) -> Undiarized:
        return Undiarized(self.error, 0)


class Diarizer:
    """The run's stream on a thread of its own: a listener on the hold owner, given a hold's audio as the hold is spoken
    and told when it ends, which nothing the owner does waits on.

    Each hold the key sent or dropped settles as one hearing, which the owner tells as its `hearing` event, beside its
    `hold` event for the same hold: the two are joined by the hold's number, and this puts the hold's fate, its speakers
    and the steps that scored it on the hearing's event.
    """

    name = "diarization"

    # The run's stream, set by the thread's first work and read only by the work queued behind it, on the same thread.
    _stream: Stream | Unloaded

    def __init__(self, model: Callable[[], Model], record: Record) -> None:
        self._record = record
        # [LAW:no-ambient-temporal-coupling] one thread scores the run in the order it was spoken: a hold's end is queued
        # behind its last audio and ahead of the next hold's first. The model loads there first, so holds spoken while it
        # loads wait behind it, and nothing else does.
        self._thread = SerialThread("Diarization")
        self._thread.give(lambda: self._load(model))

    def _load(self, model: Callable[[], Model]) -> None:
        """The run's stream, on the model; or, where it fails to load, one that ends each hold undiarized with why.
        [LAW:nothing-unseen] the load is a unit of its own, as the transcriber's is."""
        try:
            with unit("diarization.loaded", self._record):
                annotate(model=MODEL, revision=REVISION)
                self._stream = Stream(model())
        except Exception as error:
            self._stream = Unloaded(f"the model did not load: {type(error).__name__}: {error}")

    def opened(self, hold: Hold, before: Sound) -> Hearings:
        """The hold opens: what it kept from before it opened is the first of its audio."""
        return self.heard(before)

    def heard(self, sound: Sound) -> Hearings:
        """Audio of the hold open, scored behind everything given before it; this returns at once, with nothing for the
        owner to run: the steps it completes are scored on the thread."""
        samples = decoded(sound.heard)
        self._thread.give(lambda: self._stream.heard(samples))
        return ()

    def ended(self, hold: Hold, fate: Fate) -> Hearings:
        """The hold open ends, sent or dropped: its end is queued on the thread now, behind its audio and ahead of the next
        hold's, and its speakers settle in the hearing returned, which the owner runs and times."""
        # Read on the thread, where the load has set it.
        settled = self._thread.settles(lambda: self._stream.ended())

        async def settle() -> Heard:
            annotate(fate=fate)
            match await settled:
                case Diarized(speakers=found, steps=steps):
                    annotate(speakers=found, steps=steps)
                case Undiarized(error=error, steps=steps):
                    annotate(steps=steps)
                    # [LAW:no-silent-failure] a hold whose speakers are not known is a failed hearing, with why.
                    fail(error)
            # The speakers of a hold are no words: nothing here cuts hands off.
            return Heard(None)

        return (settle,)


def load() -> Model:
    """The model's step on the CPU, fetched if it never was, and how its processor cuts the samples for it.

    Loading it takes about 8 s the first time in a process, most of it importing torch and transformers, which only this
    does: the suite never imports them unless a test asks for the real model.
    """
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = cast(Any, AutoProcessor.from_pretrained(MODEL, revision=REVISION))  # pyright: ignore[reportUnknownMemberType]  (untyped in transformers)
    processor.set_streaming_mode(STREAMING_MODE)
    network = cast(Any, AutoModelForAudioFrameClassification.from_pretrained(MODEL, revision=REVISION)).eval()  # pyright: ignore[reportUnknownMemberType]  (untyped in transformers)
    extractor = processor.feature_extractor

    def step(chunk: Samples, first: bool, cache: Any) -> tuple[Probabilities, Any]:
        inputs = processor(chunk, sampling_rate=RATE, is_streaming=True, is_first_audio_chunk=first)
        with torch.inference_mode():
            scored = network(**inputs, speaker_cache=cache)
        return torch.sigmoid(scored.logits[0]).numpy(), scored.speaker_cache

    chunking = Chunking(
        int(processor.num_samples_first_audio_chunk),
        int(processor.num_samples_per_audio_chunk),
        int(processor.num_mel_frames_per_step),
        int(extractor.n_fft) // 2,
    )
    # [LAW:parse-dont-validate] the frames are 10 ms at RATE, as every count here takes them to be.
    if (int(extractor.sampling_rate), int(extractor.hop_length)) != (RATE, FRAME):
        raise ValueError(f"{MODEL} hears {extractor.sampling_rate} Hz in frames of {extractor.hop_length}; hands diarizes {RATE} Hz in frames of {FRAME}")
    return Model(step, chunking)
