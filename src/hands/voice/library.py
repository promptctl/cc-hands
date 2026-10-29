"""Kyutai's voice library: the public repository of voice prompts hands can speak in, listed for search, and the Hugging
Face token the cloning weights are fetched with.

[LAW:one-source-of-truth] the repository and the address its voices are fetched from are spelled here and nowhere else in
hands.
"""

import asyncio
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from huggingface_hub import list_repo_files

LIBRARY_REPO = "kyutai/tts-voices"
LIBRARY_URL_PREFIX = f"hf://{LIBRARY_REPO}/"

# The most names one search returns, to the voice tool and the gallery alike: a model's context is the tighter of the two.
MOST_FOUND = 50

# How long a listing may take before it is an OSError: a stalled connection must not hold the search's lock forever.
LISTING_TIMEOUT_SECONDS = 60.0


def address(path: str) -> str:
    """Where a library voice's audio prompt is fetched from, by its path there without ".wav"."""
    return f"{LIBRARY_URL_PREFIX}{path}.wav"

# How long a listing is kept before the library is listed again: a voice added to the library since is found once it is.
LISTING_SECONDS = 60 * 60.0

# The keychain item the Hugging Face token is kept under.
KEYCHAIN_ITEM = "HF_TOKEN"


@dataclass(frozen=True)
class Found:
    """What a search matched, and when the listing it searched was made; `from_cache` says it was not made for this search."""

    names: tuple[str, ...]
    listed_at: float
    from_cache: bool


def list_library() -> list[str]:
    """Every file in the library, by its path there."""
    return list_repo_files(LIBRARY_REPO)


class Library:
    """The one owner of the library's listing: made once and kept for LISTING_SECONDS, under a lock, so concurrent searches
    list the library once. A failed listing raises OSError and is not kept, so the next search tries again."""

    def __init__(self, list_files: Callable[[], list[str]] = list_library, clock: Callable[[], float] = time.time) -> None:
        self._list_files = list_files
        self._clock = clock
        self._lock = asyncio.Lock()
        self._names: tuple[str, ...] = ()
        self._listed_at: float | None = None

    async def search(self, query: str, limit: int) -> Found:
        """The names containing `query`, at most MOST_FOUND of them. Only names with a folder are offered: a bare name is not a
        voice a choice can make (see voices.LIBRARY_PATH), so offering it would be a lie."""
        async with self._lock:
            now = self._clock()
            if self._listed_at is not None and now - self._listed_at < LISTING_SECONDS:
                listed_at, from_cache = self._listed_at, True
            else:
                files = await asyncio.wait_for(asyncio.to_thread(self._list_files), LISTING_TIMEOUT_SECONDS)
                self._names = tuple(name.removesuffix(".wav") for name in files if name.endswith(".wav") and "/" in name)
                self._listed_at = listed_at = now
                from_cache = False
            needle = query.lower()
            matches = tuple(name for name in self._names if needle in name.lower())[: max(1, min(limit, MOST_FOUND))]
            return Found(matches, listed_at, from_cache)


def hub_token(environment: Mapping[str, str]) -> str | None:
    """The Hugging Face token the cloning weights are fetched with: the environment's where it has one, else the keychain's,
    else None. A voice whose weights need a token and get none fails at its choice, naming the cloning model.

    [LAW:single-enforcer] the one reader of the token. pocket_tts fetches the cloning weights with no token argument, so the
    environment is the only channel it reads; cli.main writes the token there, once, where a process starts. The token is
    never logged or written to a file.
    """
    return environment.get("HF_TOKEN") or _keychain_token()


def _keychain_token() -> str | None:
    try:
        # `security` prints the password on stdout alone; neither stream is logged.
        found = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_ITEM, "-w"], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return found.stdout.strip() or None
