"""Kyutai's voice library: the public repository of voice prompts hands can speak in, listed for search, and the Hugging
Face token the cloning weights are fetched with.

[LAW:one-source-of-truth] the repository and the address its voices are fetched from are spelled here and nowhere else in
hands.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from huggingface_hub import list_repo_files
from loguru import logger

from hands.clock import Clock
from hands.sessions.child import Exited, NeverStarted, TimedOut, run_blocking

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
class _Listing:
    """A listing made: when on the monotonic clock, for its freshness, and when on the wall clock, to be said."""

    fresh_since: float
    listed_at: float
    names: tuple[str, ...]


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

    def __init__(self, clock: Clock, list_files: Callable[[], list[str]] = list_library) -> None:
        self._list_files = list_files
        self._clock = clock
        self._lock = asyncio.Lock()
        self._listing: _Listing | None = None
        self._pending: asyncio.Future[list[str]] | None = None

    async def search(self, query: str, limit: int) -> Found:
        """The names containing `query`, at most MOST_FOUND of them. Only names with a folder are offered: a bare name is not a
        voice a choice can make (see voices.LIBRARY_PATH), so offering it would be a lie."""
        async with self._lock:
            # Freshness is on the monotonic clock, which a wall-clock step cannot move; `listed_at` is the wall clock's, as it
            # is said to the model as a time.
            now = self._clock.now()
            listing = self._listing
            if listing is not None and now - listing.fresh_since < LISTING_SECONDS:
                from_cache = True
            else:
                # A listing that timed out keeps running in its thread, so the next search waits for that one rather than
                # starting another; the pending listing is forgotten once it has finished, failed or not.
                if self._pending is None:
                    self._pending = asyncio.ensure_future(asyncio.to_thread(self._list_files))
                try:
                    async with self._clock.timeout(LISTING_TIMEOUT_SECONDS):
                        files = await asyncio.shield(self._pending)
                finally:
                    if self._pending.done():
                        self._pending = None
                names = tuple(name.removesuffix(".wav") for name in files if name.endswith(".wav") and "/" in name)
                self._listing = listing = _Listing(now, self._clock.wall().timestamp(), names)
                from_cache = False
            needle = query.lower()
            matches = tuple(name for name in listing.names if needle in name.lower())[: max(1, min(limit, MOST_FOUND))]
            return Found(matches, listing.listed_at, from_cache)


def hub_token(environment: Mapping[str, str], clock: Clock) -> str | None:
    """The Hugging Face token the cloning weights are fetched with: the environment's where it has one, else the keychain's,
    else None. A voice whose weights need a token and get none fails at its choice, naming the cloning model.

    [LAW:single-enforcer] the one reader of the token. pocket_tts fetches the cloning weights with no token argument, so the
    environment is the only channel it reads; cli.main writes the token there, once, where a process starts. The token is
    never logged or written to a file.
    """
    return environment.get("HF_TOKEN") or _keychain_token(clock)


# `security` answers at once; one still waiting this long waits on a locked keychain's prompt nobody answers.
KEYCHAIN_SECONDS = 10.0
# What `security` exits with when the keychain holds no such item: errSecItemNotFound.
NO_ITEM = 44


def _keychain_token(clock: Clock) -> str | None:
    # `security` prints the password on stdout alone; neither stream is logged.
    match run_blocking("security", "find-generic-password", "-s", KEYCHAIN_ITEM, "-w", timeout=KEYCHAIN_SECONDS, clock=clock):
        case Exited(returncode=0, out=out):
            return out.strip() or None
        case Exited(returncode=code) if code == NO_ITEM:
            return None
        # [LAW:no-silent-failure] a keychain that could not be read is said, so a voice that later needs the token is
        # not taken to have none.
        case Exited(returncode=code):
            logger.warning(f"the keychain's {KEYCHAIN_ITEM} could not be read: `security` exited {code}")
        case TimedOut():
            logger.warning(f"the keychain's {KEYCHAIN_ITEM} could not be read: `security` waited {KEYCHAIN_SECONDS:.0f}s, on a locked keychain")
        case NeverStarted(error=error):
            logger.warning(f"the keychain's {KEYCHAIN_ITEM} could not be read: `security` never started: {error}")
    return None
