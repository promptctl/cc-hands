"""Everything the run fetches, fetched before any test in a process of its own (conftest): the test process neither
connects past loopback nor resolves a name, before it goes offline or after. macOS's resolver holds its connection to
mDNSResponder for the life of the process that first resolves a name, and the sandbox refuses only new connections, so
a run that resolved a name to fetch would go on resolving names once it is offline.

Warm, each fetch is a look at what is already there. The process reads its Fetches on stdin, writes a line for each
fetch, and fails naming what it could not fetch and why."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from http.client import HTTPException
from pathlib import Path

import httpx
import huggingface_hub.constants
import pytest
from huggingface_hub.errors import RevisionResolutionError
from huggingface_hub.utils.tqdm import disable_progress_bars
from mlx_audio.utils import get_model_path

from hands.brain import pinned
from hands.clock import SYSTEM
from hands.voice import dtln, engaged, streaming, transcription, voiceprints
from hands.voice import ortweb
from hands.voice.pinnedfile import Mismatch, Release, Unserved
from hands.voice.wakemodels import WakeModels
from hands.voice.wakeword import Pretrained
from npm_registry import REGISTRY
from sentences import INSTALL_FAILURES, PUNKT_TAB, install

# The canonical mock of Anthropic's API the brain's end-to-end tests run against, at the release they were written
# against. Moving it is one commit, and CI's cache of it is keyed on it.
MOCK_MODULE = "github.com/promptctl/mock-anthropic"
MOCK_RELEASE = "v0.1.0"


@dataclass(frozen=True)
class Weights:
    """A model's weights on the Hugging Face hub, at the revision hands pins it to: `what` of them is fetched, and
    `fetch` is the call its load makes to fetch them, so the run fetches what a load would."""

    what: str
    fetch: Callable[[], object]


def loaded_by_mlx_audio(repo: str, revision: str) -> Weights:
    return Weights(f"{repo} at {revision}", partial(get_model_path, repo, revision=revision))


# [LAW:one-source-of-truth] every speech weight a test loads, the one list the run fetches and CI keys its cache of the
# hub on (hub_key): each weight's `what` names the pins its fetch reads.
WEIGHTS = (
    loaded_by_mlx_audio(streaming.MODEL, streaming.REVISION),
    loaded_by_mlx_audio(streaming.DIARIZATION, streaming.DIARIZATION_REVISION),
    loaded_by_mlx_audio(engaged.SMART_TURN, engaged.SMART_TURN_REVISION),
    Weights(f"{transcription.MODEL} at {transcription.REVISION}", transcription.weights),
    # The voiceprint model is one file of a repository of many, so only it is fetched.
    Weights(f"{voiceprints.FILE} from {voiceprints.MODEL} at {voiceprints.REVISION}", voiceprints.weights),
)


def hub_key() -> str:
    """What CI keys its cache of the hub on: a digest of what of each weight WEIGHTS fetches, so a weight added, or moved
    to another repository, revision or file, is a new key."""
    return hashlib.sha256("\n".join(weights.what for weights in WEIGHTS).encode()).hexdigest()[:16]


# The wake words whose models a switch to the wake word loads.
WAKE_WORDS = (Pretrained("Hey Jarvis"), Pretrained("Hey Mycroft"))


@dataclass(frozen=True)
class Fetches:
    """What the run's fetches need: where each goes in pytest's cache, but the hub's weights, which go to the Hugging Face
    cache the process's environment names; and the machine's environment, which the mock is built with."""

    nltk_data: Path
    wake_word: Path
    claude_code: Path
    mock: Path
    dtln: Path
    runtime: Path
    machine: dict[str, str]
    # The run is the unit layer's, held to nothing outside the process (tests/offline.py): of all this, its tests read only
    # the sentence tokenizer and the echo canceller's model, so that is all it fetches. The rest is a weight it cannot read or a process it cannot start.
    unit: bool = False

    def mock_binary(self) -> Path:
        """The mock at MOCK_RELEASE, in its release's own directory."""
        return self.mock / MOCK_RELEASE / "mock-anthropic"

    def written(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def read(text: str) -> "Fetches":
        read = json.loads(text)
        return Fetches(Path(read["nltk_data"]), Path(read["wake_word"]), Path(read["claude_code"]), Path(read["mock"]), Path(read["dtln"]), Path(read["runtime"]), read["machine"], read["unit"])


def fetching[T](write: Callable[[str], object], what: str, failures: tuple[type[Exception], ...], fetch: Callable[[], T]) -> T:
    """What `fetch` gave, `what` it is written as it starts and how long it took once it is done, so a fetch that hangs
    is the one named last. [LAW:no-silent-failure] one that fails as a fetch does, in one of its `failures`, stops the
    run before any test, naming what it could not fetch and why, never a test timed out on it; anything else is a
    defect, raised as itself with its stack."""
    write(f"fetching {what}: ")
    started = time.monotonic()
    try:
        fetched = fetch()
    except failures as error:
        write("failed\n")
        raise pytest.UsageError(f"the run cannot fetch {what}: {type(error).__name__}: {error}") from error
    write(f"{time.monotonic() - started:.2f} s\n")
    return fetched


def installed_mock(binary: Path, machine: Mapping[str, str]) -> Path:
    """The mock at MOCK_RELEASE, built by `go install` into its release's own directory unless it is there already, so
    the build of one release is never taken for another. Built with the machine's own environment, its toolchain and
    caches, the one fetch that is."""
    if not binary.is_file():
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["go", "install", f"{MOCK_MODULE}@{MOCK_RELEASE}"], check=True, env={**machine, "GOBIN": str(binary.parent)})
    return binary


def fetch_all(fetches: Fetches, write: Callable[[str], object]) -> None:
    """Every fetch the run makes, as `fetches` says, each named by `write` as it goes."""
    # The line each fetch writes says what the run is fetching; the hub's bars would break it.
    disable_progress_bars()
    fetching(write, f"NLTK's punkt_tab sentence tokenizer at {PUNKT_TAB.sha256[:12]}", INSTALL_FAILURES, partial(asyncio.run, install(PUNKT_TAB, fetches.nltk_data, pinned.download)))
    # The echo canceller's model, which the unit layer's tests of the canceller run for real, so it is read there too.
    fetching(write, f"DTLN-aec's models at {dtln.REVISION[:12]}", (OSError, HTTPException, Mismatch), partial(dtln.fetched, fetches.dtln))
    if fetches.unit:
        return
    for weights in WEIGHTS:
        # The hub's failures are its client's, its own OSErrors, and a revision it could not resolve with nothing in the cache.
        fetching(write, f"{weights.what} from the Hugging Face hub", (httpx.HTTPError, OSError, RevisionResolutionError), weights.fetch)
    for word in WAKE_WORDS:
        fetching(write, f"the {word.phrase} wake word's models", (Unserved,), partial(asyncio.run, WakeModels(fetches.wake_word, SYSTEM).ensured(word)))
    # The runtime the phone's page runs the wake word on, which its page test has the page fetch from hands.
    fetching(write, f"ONNX Runtime Web {ortweb.VERSION}", (Unserved,), partial(asyncio.run, Release(fetches.runtime, SYSTEM).ensured(file.pin for file in ortweb.FILES)))
    fetching(write, f"Claude Code {pinned.PIN.version} from npm", (pinned.Uninstallable,), partial(asyncio.run, pinned.install(fetches.claude_code, pinned.PIN, REGISTRY)))
    fetching(write, f"{MOCK_MODULE}@{MOCK_RELEASE}", (subprocess.CalledProcessError, OSError), partial(installed_mock, fetches.mock_binary(), fetches.machine))


# The exit of a fetch process that could not fetch something, having said what and why: sysexits' "service unavailable",
# which Python never exits with of its own. Any other failure is a defect, whose stack the process wrote as it died.
CANNOT_FETCH = os.EX_UNAVAILABLE


def fetched_apart(fetches: Fetches) -> None:
    """[LAW:single-enforcer] every fetch the run makes, in a process of this module's own, all it writes on the terminal
    as it writes it, a fetch's reason or a defect's stack among it. One that fails stops the run. The process takes this
    one's environment, the hub's online, the run's own environment holding the hub offline for the tests, and the
    developer's Xet cache, which a fetch from the hub dedupes its chunks against."""
    done = subprocess.run(
        [sys.executable, __file__],
        input=fetches.written(),
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "0", "HF_XET_CACHE": huggingface_hub.constants.HF_XET_CACHE},
    )
    if done.returncode:
        raise pytest.UsageError(failed(done.returncode))


def failed(returncode: int) -> str:
    """What a fetch process that exited `returncode`, not 0, did: could not fetch, was ended by a signal, or died of a
    defect, whose stack it wrote."""
    if returncode == CANNOT_FETCH:
        return "the run cannot fetch what it needs: why is above"
    if returncode < 0:
        return f"the run's fetches were ended by {signal.Signals(-returncode).name}"
    return f"the run's fetches failed with a defect (exit {returncode}): its stack is above"


if __name__ == "__main__":
    try:
        fetch_all(Fetches.read(sys.stdin.read()), partial(print, end="", flush=True))
    except pytest.UsageError as error:
        print(error, file=sys.stderr)
        sys.exit(CANNOT_FETCH)
