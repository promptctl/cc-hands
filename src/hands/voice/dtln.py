"""DTLN-aec, the desk's echo canceller: a dual-signal transformation LSTM network that takes the speaker's echo out of the
microphone given the reference, the sound the speaker was given (Westhausen and Meyer, "Acoustic Echo Cancellation with the
Dual-Signal Transformation LSTM Network", 2021; github.com/breizhn/DTLN-aec, MIT).

Two stateful parts run over 512-sample blocks shifted by 128, at 16 kHz, and their blocks overlap-add: the first masks
the microphone's magnitude spectrum given the reference's, the second refines the masked block in time given the reference
block. Held to the harness's twelve takes of a person and hands at the desk (scripts/measure-echo.py), through
`hands.voice.echo.EchoCanceller`: none of hands' words leaked and 1 of 101 of a person's lost, the one every chain loses,
"wait" in hands_no_echo_in_two_voices.flac; WebRTC's AEC3 leaked none and lost 7. The 128- and 256-unit models, at a fixed
lag, leaked 9 and 6 words and lost 4 and 2, where the 512 leaked none and lost 1: so the 512 is the one used.

The model is not told the room, only the reference, so how it is lined up with its echo is the one thing it depends on
(`_EARLY`), and what it learned over one alignment is wrong for another (`_SETTLED`).

Cost, one thread: 0.38 ms of inference per 8 ms hop. Latency: `Dtln.LATENCY`, 32 ms.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from hands.voice import pinnedfile
from hands.voice.pinnedfile import Pin

SAMPLE_RATE = 16000
BLOCK, SHIFT = 512, 128

# The reference's delay at most, in samples: 1 s, the stretch `hands.voice.echo` looks for the echo path over.
MAX_DELAY = SAMPLE_RATE

# How much sooner than its echo the reference is read, in samples: 15 ms. Read late, where the echo is already in the
# block before the reference that made it, DTLN-aec cancels none of it (34 of hands' words leaked over the twelve takes with
# the lag 12 ms late, where none did at 69 ms, the echo's own on one take); read early, it keeps finding it. Over those
# takes, with the echo path's lag read as `hands.voice.echo` reads it, 10 to 20 ms early leaked none of hands' words and
# 5 ms early leaked one.
_EARLY = 240

# How far a delay may move, in samples, with what the model learned of the room kept: 1 ms. A delay that moves further
# is another alignment of the reference with its echo, and the model's state, learned of the last, leaks hands' voice
# through it (13 words leaked and 13 of a person's were lost on the twelve takes where the state kept was of the delay
# before it was read, 1 and 4 where it started afresh).
_SETTLED = 16

# How much of the microphone just heard the model reads again, in hops of `SHIFT`, when it begins afresh: 128 ms, some 8 ms
# of the capture thread's time. Begun on the middle of a word with nothing of it learned, it took out 17 dB of hands'
# first word where, having read the 128 ms before, it took out 40 dB (hands_cancelled_and_another.flac, the 0.25 s after
# its echo path was read), and Whisper heard "hey" of the first 8 s of hands alone (tests/test_whisper_recorded.py).
_REPLAYED = 16

REPOSITORY = "breizhn/DTLN-aec"
REVISION = "9d24e128b4f409db18227b8babb343016625921f"
# The 512-unit model, the two parts of it, each by the digest of its file.
PARTS = (
    ("dtln_aec_512_1.tflite", "569f7c3cfac96b1e093229c3ca10b5d892f5b1906105b644e457f8245b4f7383"),
    ("dtln_aec_512_2.tflite", "fb423d867ab25d5f4716bd369c7126b6c84926175c019b832e71bf21de0e9907"),
)


# A part of the model: the signal it works on, the reference, and its state; what it gives, and its state after.
type Part = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Stages:
    """The model's two parts as functions, each with the state its LSTMs carry from one hop to the next.

    `mask` takes the microphone's and the reference's magnitude spectra, each BLOCK // 2 + 1 bins, and returns the mask for
    the microphone's spectrum; `refine` takes the masked block and the reference block, each BLOCK samples, and returns the
    block cleaned. Both take and return the part's state."""

    mask: Part
    refine: Part
    mask_state: tuple[int, ...]
    refine_state: tuple[int, ...]


# Where the model's files are kept. A test run points it at a directory of its own (tests/conftest.py), fetched once for
# the run, because each test has a HOME of its own that holds nothing.
CACHE = Path.home() / ".cache" / "hands" / "dtln-aec" / REVISION


@cache
def weights() -> tuple[Path, Path]:
    """The two parts' files in `CACHE`, fetched if they are not there."""
    return fetched(CACHE)


def fetched(directory: Path) -> tuple[Path, Path]:
    """The two parts' files, fetched from the model's repository at REVISION into `directory` unless they are there, each held to its digest.

    [LAW:no-silent-failure] a canceller that cancels with other weights would leak hands' voice or eat a person's with
    nothing to say why, so a file that is not the one pinned is refused by name (`hands.voice.pinnedfile`)."""
    directory.mkdir(parents=True, exist_ok=True)
    first, second = (
        pinnedfile.fetched(directory, Pin(name, f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/pretrained_models/{name}", digest, REVISION))
        for name, digest in PARTS
    )
    return first, second


def _part(path: Path) -> tuple[Part, tuple[int, ...]]:
    """One of the model's parts as TFLite runs it, on one thread, and the shape of the state its LSTMs carry. Both take
    the signal, the reference and the state, in the model's own order of inputs: the signal, the state, the reference."""
    # The interpreter's tensors and details are untyped.
    interpreter: Any = Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()

    def run(signal: np.ndarray, reference: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        interpreter.set_tensor(inputs[0]["index"], signal.reshape(1, 1, -1))
        interpreter.set_tensor(inputs[2]["index"], reference.reshape(1, 1, -1))
        interpreter.set_tensor(inputs[1]["index"], state)
        interpreter.invoke()
        return np.asarray(interpreter.get_tensor(outputs[0]["index"])), np.asarray(interpreter.get_tensor(outputs[1]["index"]))

    return run, tuple(int(size) for size in inputs[1]["shape"])


def tflite_stages(first: Path, second: Path) -> Stages:
    """The model's two parts as TFLite runs them."""
    (mask, mask_state), (refine, refine_state) = _part(first), _part(second)
    return Stages(mask, refine, mask_state, refine_state)


class Dtln:
    """DTLN-aec over a stream, in frames of any length: `cancel` takes the microphone's and the reference's, the same
    length, and returns the microphone's cleaned, the same length, `LATENCY` samples behind it.

    `delay` is how long after the speaker played a sound its echo reaches the microphone, in samples, and the reference is
    read that far back, `_EARLY` less, so that it lines up with the echo: DTLN-aec learns the room's response, not how far
    the speaker is from the microphone. The capture thread alone touches it."""

    # What the output trails the microphone by: the hop that keeps a frame whole whatever its length, and the three hops
    # each block spans.
    LATENCY = BLOCK

    def __init__(self, stages: Stages) -> None:
        self._stages = stages
        self._mask_state = np.zeros(stages.mask_state, np.float32)
        self._refine_state = np.zeros(stages.refine_state, np.float32)
        self._pending = np.zeros(0, np.float32)
        # The reference received, as much of its end as a delay reaches back, and the microphone's stream position it
        # begins at.
        self._reference = np.zeros(0, np.float32)
        self._received = 0
        self._position = 0
        self._delay: int | None = None
        self._microphone_block = np.zeros(BLOCK, np.float32)
        self._reference_block = np.zeros(BLOCK, np.float32)
        self._overlap = np.zeros(BLOCK, np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=_REPLAYED)
        self._cleaned = np.zeros(self.LATENCY - (BLOCK - SHIFT), np.float32)

    def cancel(self, microphone: np.ndarray, reference: np.ndarray, delay: int | None) -> np.ndarray:
        """The microphone's frame with the echo of the reference taken out, the same length. With no `delay` yet, where
        the echo path is not known, it is the frame as it was: there is no alignment for the model to learn the room from."""
        # [LAW:parse-dont-validate] a frame of one side with a different length from the other's is a stream opened wrong.
        if len(microphone) != len(reference):
            raise ValueError(f"{len(microphone)} samples of microphone against {len(reference)} of reference")
        if delay is not None and not 0 <= delay <= MAX_DELAY:
            raise ValueError(f"a reference delay of {delay} samples is not within 0 to {MAX_DELAY}")
        began = self._align(delay)
        self._reference = np.concatenate((self._reference, reference))[-(MAX_DELAY + (_REPLAYED + 1) * SHIFT + len(reference)) :]
        self._received += len(reference)
        if began and delay is not None:
            self._replay(delay)
        pending = np.concatenate((self._pending, microphone))
        hops = len(pending) // SHIFT
        self._pending = pending[hops * SHIFT :]
        cleaned = np.concatenate((self._cleaned, *(self._hop(pending[hop * SHIFT : (hop + 1) * SHIFT], delay) for hop in range(hops))))
        self._cleaned = cleaned[len(microphone) :]
        return cleaned[: len(microphone)]

    def _align(self, delay: int | None) -> bool:
        """Start the model afresh where the reference is lined up with its echo another way than it was, and say whether it
        did."""
        moved = delay is None or self._delay is None or abs(delay - self._delay) > _SETTLED
        if moved:
            self._mask_state = np.zeros_like(self._mask_state)
            self._refine_state = np.zeros_like(self._refine_state)
        self._delay = delay if moved else self._delay
        return moved

    def _replay(self, delay: int) -> None:
        """Read the last of the microphone again against the reference as `delay` lines it up, for what the model learns
        of it and the blocks it is overlapping: what it gives is what was already heard."""
        heard = list(self._history)
        self._position -= len(heard) * SHIFT
        # The blocks and the overlap are of the stream as it ran: read afresh, they begin on silence as the model's first
        # hop does, not on the end of the stream they would join the first of the replay to.
        self._microphone_block = np.zeros(BLOCK, np.float32)
        self._reference_block = np.zeros(BLOCK, np.float32)
        self._overlap = np.zeros(BLOCK, np.float32)
        for hop in heard:
            self._run(hop, delay)

    def _delayed(self, delay: int) -> np.ndarray:
        """The reference's hop `delay` samples before the microphone's next one; silence before it began."""
        # Index 0 of the padded reference is silence, so a sample outside what is held reads it.
        padded = np.concatenate((np.zeros(1, np.float32), self._reference))
        at = self._position - delay + np.arange(SHIFT) - (self._received - len(self._reference))
        return padded[np.where((at >= 0) & (at < len(self._reference)), at + 1, 0)]

    def _hop(self, microphone: np.ndarray, delay: int | None) -> np.ndarray:
        self._history.append(microphone)
        return self._run(microphone, delay)

    def _run(self, microphone: np.ndarray, delay: int | None) -> np.ndarray:
        reference = np.zeros(SHIFT, np.float32) if delay is None else self._delayed(max(0, delay - _EARLY))
        self._position += SHIFT
        self._microphone_block = np.concatenate((self._microphone_block[SHIFT:], microphone))
        self._reference_block = np.concatenate((self._reference_block[SHIFT:], reference))
        if delay is None:
            return self._microphone_block[:SHIFT].copy()
        spectrum = np.fft.rfft(self._microphone_block).astype("complex64")
        reference_magnitude = np.abs(np.fft.rfft(self._reference_block)).astype("float32")
        mask, self._mask_state = self._stages.mask(np.abs(spectrum).astype("float32"), reference_magnitude, self._mask_state)
        masked = np.fft.irfft(spectrum * mask.reshape(-1)).astype("float32")
        block, self._refine_state = self._stages.refine(masked, self._reference_block, self._refine_state)
        self._overlap = np.concatenate((self._overlap[SHIFT:], np.zeros(SHIFT, np.float32))) + block.reshape(-1)
        return self._overlap[:SHIFT].copy()
