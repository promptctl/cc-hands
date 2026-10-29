"""Speech the tests hear: each text spoken once by macOS's Samantha and committed, so every run on every machine hears the
same samples. Said afresh, a test would hear the machine's voice and not the text: each macOS release's Samantha says a
text differently enough to move the wake word, the detector, or Smart Turn's verdict (promptctl-voice-i15t.qg8.71j), and
the speech daemon can stall a test to its timeout (promptctl-testing-px1f.iov).

Run as a script, `uv run python tests/speech.py TEXT...` records each TEXT, naming in the file the macOS it was said on."""

import hashlib
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import soundfile

RECORDINGS = Path(__file__).parent / "fixtures" / "speech"
# As the desk's microphone hands audio over: 16 kHz, 16-bit mono.
RATE = 16_000


def quiet(seconds: float) -> bytes:
    """Silence as the desk's microphone hands it over."""
    return bytes(round(RATE * seconds) * 2)


def recording(text: str) -> Path:
    """Where the recording of `text` is: a slug to find it by, and its digest, since `say` speaks two texts that differ
    only in punctuation differently."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return RECORDINGS / f"{slug}-{hashlib.sha256(text.encode()).hexdigest()[:8]}.flac"


def audio(path: Path) -> tuple[NDArray[np.int16], int]:
    """The audio in `path` as 16-bit samples, and the rate it was sampled at."""
    return cast(tuple[NDArray[np.int16], int], soundfile.read(path, dtype="int16"))  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)


def spoken(text: str) -> bytes:
    """`text` as Samantha said it, as 16 kHz mono 16-bit audio, the way the desk's microphone is heard."""
    path = recording(text)
    if not path.is_file():
        raise FileNotFoundError(f"{text!r} has no recording at {path}: record it with `uv run python tests/speech.py {shlex.quote(text)}`")
    samples, rate = audio(path)
    assert (rate, samples.ndim) == (RATE, 1), (path, rate, samples.shape)
    return samples.tobytes()


def record(text: str) -> Path:
    """Records `text` in Samantha's voice, and never over a recording of it already made: said on another macOS, it is
    another voice. A `say` that fails or hangs raises, naming `say` and the text."""
    path = recording(text)
    RECORDINGS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        said = Path(scratch) / "said.wav"
        subprocess.run(["say", "-v", "Samantha", "-o", str(said), f"--data-format=LEI16@{RATE}", text], check=True, timeout=60)
        samples, rate = audio(said)
    with soundfile.SoundFile(path, "x", samplerate=rate, channels=1, subtype="PCM_16", format="FLAC") as flac:
        flac.comment = f"say -v Samantha, macOS {platform.mac_ver()[0]}"
        flac.write(samples)  # pyright: ignore[reportUnknownMemberType]  (untyped in soundfile)
    return path


if __name__ == "__main__":
    for text in sys.argv[1:]:
        print(record(text))
