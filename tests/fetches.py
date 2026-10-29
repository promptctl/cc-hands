"""Everything the run fetches, fetched before any test in a process of its own (conftest): the test process neither
connects past loopback nor resolves a name, before it goes offline or after. macOS's resolver holds its connection to
mDNSResponder for the life of the process that first resolves a name, and the sandbox refuses only new connections, so
a run that resolved a name to fetch would go on resolving names once it is offline.

Warm, each fetch is a look at what is already there. The process reads its Fetches on stdin, writes a line for each
fetch, and fails naming what it could not fetch and why."""

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import RevisionResolutionError
from huggingface_hub.utils.tqdm import disable_progress_bars
from mlx_audio.utils import get_model_path

from hands.brain import pinned
from hands.voice import engaged, streaming
from hands.voice.wake import fetched
from hands.voice.wakeword import Pretrained
from npm_registry import REGISTRY
from sentences import INSTALL_FAILURES, PUNKT_TAB, install

# The canonical mock of Anthropic's API the brain's end-to-end tests run against, at the release they were written
# against. Moving it is one commit, and CI's cache of it is keyed on it.
MOCK_MODULE = "github.com/promptctl/mock-anthropic"
MOCK_RELEASE = "v0.1.0"

# Every speech weight a test loads, each at the revision hands pins it to.
WEIGHTS = ((streaming.MODEL, streaming.REVISION), (streaming.DIARIZATION, streaming.DIARIZATION_REVISION), (engaged.SMART_TURN, engaged.SMART_TURN_REVISION))

# The wake words whose models a switch to the wake word loads.
WAKE_WORDS = (Pretrained("Hey Jarvis"), Pretrained("Hey Mycroft"))


@dataclass(frozen=True)
class Fetches:
    """What the run's fetches need: where each goes in pytest's cache, but the weights, which go to the Hugging Face
    cache the process's environment names; and the machine's environment, which the mock is built with."""

    nltk_data: Path
    wake_word: Path
    claude_code: Path
    mock: Path
    machine: dict[str, str]

    def mock_binary(self) -> Path:
        """The mock at MOCK_RELEASE, in its release's own directory."""
        return self.mock / MOCK_RELEASE / "mock-anthropic"

    def written(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def read(text: str) -> "Fetches":
        read = json.loads(text)
        return Fetches(Path(read["nltk_data"]), Path(read["wake_word"]), Path(read["claude_code"]), Path(read["mock"]), read["machine"])


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
    for repo, revision in WEIGHTS:
        # What mlx_audio's loads call with these arguments, so the run fetches what a load would. The hub's failures are
        # its client's, its own OSErrors, and a revision it could not resolve with nothing in the cache.
        fetching(write, f"{repo} at {revision} from the Hugging Face hub", (httpx.HTTPError, OSError, RevisionResolutionError), partial(get_model_path, repo, revision=revision))
    fetching(write, f"NLTK's punkt_tab sentence tokenizer at {PUNKT_TAB.sha256[:12]}", INSTALL_FAILURES, partial(asyncio.run, install(PUNKT_TAB, fetches.nltk_data, pinned.download)))
    for word in WAKE_WORDS:
        fetching(write, f"the {word.phrase} wake word's models", (*pinned.DOWNLOAD_FAILURES, OSError), partial(asyncio.run, fetched(fetches.wake_word, word)))
    fetching(write, f"Claude Code {pinned.VERSION} from npm", (pinned.Uninstallable,), partial(asyncio.run, pinned.install(fetches.claude_code, REGISTRY)))
    fetching(write, f"{MOCK_MODULE}@{MOCK_RELEASE}", (subprocess.CalledProcessError, OSError), partial(installed_mock, fetches.mock_binary(), fetches.machine))


def fetched_apart(fetches: Fetches) -> None:
    """[LAW:single-enforcer] every fetch the run makes, in a process of this module's own, its lines on the terminal as it
    writes them; one that fails stops the run, naming what it could not fetch and why. The process takes this one's
    environment, the hub's online: the run's own environment holds the hub offline for the tests."""
    done = subprocess.run(
        [sys.executable, __file__],
        input=fetches.written(),
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "0"},
    )
    # What it said besides its lines, the hub's warnings and go's among them, and, failed, why: last.
    sys.stderr.write(done.stderr)
    if done.returncode:
        said = done.stderr.strip().splitlines()
        raise pytest.UsageError(said[-1] if said else f"the run's fetches exited {done.returncode}, saying nothing")


if __name__ == "__main__":
    try:
        fetch_all(Fetches.read(sys.stdin.read()), partial(print, end="", flush=True))
    except pytest.UsageError as error:
        sys.exit(str(error))
