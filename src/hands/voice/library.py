"""Kyutai's voice library: the public repository of voice prompts hands can speak in, listed for search, and the Hugging
Face token the cloning weights are fetched with.

[LAW:one-source-of-truth] the repository and the address its voices are fetched from are spelled here and nowhere else in
hands.
"""

import asyncio
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from huggingface_hub import list_repo_files
from loguru import logger

LIBRARY_REPO = "kyutai/tts-voices"
LIBRARY_URL_PREFIX = f"hf://{LIBRARY_REPO}/"

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
        async with self._lock:
            now = self._clock()
            if self._listed_at is not None and now - self._listed_at < LISTING_SECONDS:
                listed_at, from_cache = self._listed_at, True
            else:
                files = await asyncio.to_thread(self._list_files)
                self._names = tuple(name.removesuffix(".wav") for name in files if name.endswith(".wav"))
                self._listed_at = listed_at = now
                from_cache = False
            needle = query.lower()
            matches = tuple(name for name in self._names if needle in name.lower())[:limit]
            return Found(matches, listed_at, from_cache)


def keep_hub_token(environment: Mapping[str, str]) -> None:
    """Makes the Hugging Face token the process holds: the environment's where it has one, else the keychain's.

    [LAW:single-enforcer] the one owner of the process's token. pocket_tts fetches the cloning weights with no token argument,
    so the environment is the only channel it reads; this is the one write to it, made once, before the configuration that
    may fetch a voice is read. The token is never logged or written to a file.
    """
    if environment.get("HF_TOKEN"):
        return
    token = _keychain_token()
    if token is None:
        logger.warning("no Hugging Face token in the environment or the keychain: the cloning weights may not be fetched, and then no voice in Kyutai's library can be spoken in")
        return
    os.environ["HF_TOKEN"] = token


def _keychain_token() -> str | None:
    try:
        # `security` prints the password on stdout alone; neither stream is logged.
        found = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_ITEM, "-w"], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return found.stdout.strip() or None
