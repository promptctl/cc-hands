"""The voice hands speaks in: one of Pocket TTS's own, or any voice in Kyutai's voice library, chosen by the user by ear, and
kept in the home so it outlives the daemon and needs no restart.

[LAW:one-source-of-truth] the file is the choice. The daemon builds its TTS service in the voice the file names, and
each change goes to the file and to the service together; nothing else holds a copy.
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import NewType

from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError
from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings, language_to_pocket_tts_language
from pipecat.transcriptions.language import Language
from pocket_tts.models.tts_model import CONFIGS_DIR, DEFAULT_LANGUAGE
from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES  # pyright: ignore[reportPrivateUsage]
from pocket_tts.utils.utils import download_if_necessary, get_predefined_voice

from hands.sessions.files import replace_whole
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.library import Found, Library, address

# One of pocket_tts's own names: its embedding is primed by the weights it loads, and pipecat resolves the name itself.
Voice = NewType("Voice", str)


@dataclass(frozen=True)
class LibraryVoice:
    """A voice in Kyutai's voice library, by its path there without ".wav": its audio prompt is encoded by the cloning weights."""

    path: str

    def __str__(self) -> str:
        return self.path


# Any voice hands can speak in: a built-in name, or a library voice. The kind is in the type, never in a name's spelling.
Chosen = Voice | LibraryVoice

# [LAW:one-source-of-truth] the voices on offer are the ones the installed pocket_tts resolves by name itself, each to
# a state primed by the very weights it loads; a state primed elsewhere is a cache from another network, and the loader
# cannot tell.
VOICES: tuple[Voice, ...] = tuple(Voice(name) for name in _ORIGINS_OF_PREDEFINED_VOICES)

# The voice hands speaks in until the user chooses another (the user's choice of default, 2026-10-02).
DEFAULT = Voice("charles")

# The language the TTS service speaks, which picks the set of embeddings a voice's name resolves to.
LANGUAGE = Language.EN

# A library path: repo folders joined by single slashes, each a letter or digit followed by letters, digits, `_`, `.`, or `-`.
LIBRARY_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9][A-Za-z0-9_.\-]*)*")

# Whatever a voice needs before it can be spoken in, made ready, or the reason it cannot be.
Fetch = Callable[[Chosen], None]


def parse_voice(name: str) -> Chosen:
    """The voice a name means, read as a person says it: "Bill Boerst" is bill_boerst; a library voice is its path there."""
    # [LAW:parse-dont-validate] the one crossing from a spoken or written name; past it, a Chosen is one hands can speak in
    # or one a fetch will refuse. Whether a library path exists is the fetch's to say, so a choice needs no listing.
    said = Voice("_".join(name.lower().split()))
    if said in VOICES:
        return said
    path = name.strip()
    if LIBRARY_PATH.fullmatch(path) and ".." not in path:
        return LibraryVoice(path)
    raise Rejected(f"hands has no voice called {name!r}; it has {', '.join(VOICES)}, or any voice in Kyutai's voice library by its path")


def spoken(voice: Chosen) -> str:
    """The voice's name as it is said aloud."""
    match voice:
        case LibraryVoice(path=path):
            return path.rsplit("/", 1)[-1].replace("_", " ").replace("-", " ").title()
        case _:
            return voice.replace("_", " ").title()


def chosen(home: Home) -> Chosen:
    """The voice the user chose; the default where they never chose one."""
    try:
        # Bytes, not text: a file edited by hand can hold anything, and whatever names no voice is refused alike.
        written = home.voice.read_bytes().strip()
    except FileNotFoundError:
        return DEFAULT
    try:
        return parse_voice(written.decode())
    except (UnicodeDecodeError, Rejected) as error:
        raise Rejected(f"{home.voice} says {written.decode(errors='replace')!r}, which names no voice hands has; choose another, or remove the file to speak in {spoken(DEFAULT)}") from error


def keep(home: Home, voice: Chosen) -> None:
    # [LAW:no-ambient-temporal-coupling] replaced whole, so a reader never sees half a name.
    replace_whole(home.voice, f"{voice}\n", 0o644)


def source(voice: Chosen) -> str:
    """What pipecat is handed: a built-in name it resolves itself, or a library voice's audio prompt as an hf:// path."""
    match voice:
        case LibraryVoice(path=path):
            return address(path)
        case _:
            return voice


def cloning_available() -> None:
    """Raises Rejected unless the cloning weights hands' TTS model loads can be had: the default language's, as pocket_tts's
    config names them, fetched the way pocket_tts fetches them (cached, so this is offline-safe)."""
    weights = load_config(CONFIGS_DIR / f"{DEFAULT_LANGUAGE}.yaml").weights_path
    if weights is None:
        raise Rejected(f"pocket_tts's config for {DEFAULT_LANGUAGE} names no cloning model, so hands cannot speak in a voice from Kyutai's library")
    try:
        download_if_necessary(weights)
    except OSError as error:
        raise Rejected(f"hands cannot fetch the cloning model {weights}, so it cannot speak in a voice from Kyutai's library: {error}") from error


def fetched(voice: Chosen) -> None:
    """The voice's embedding in the local cache, downloaded the first time, as the TTS service resolves the name; a library
    voice's audio prompt is fetched after the cloning model is checked, and a path the library lacks is refused by name."""
    match voice:
        case LibraryVoice(path=path):
            cloning_available()
            try:
                download_if_necessary(address(path))
            except LocalEntryNotFoundError:
                # Offline, not absent: the library may well have the voice, so it is an OSError the caller reports as unreachable.
                raise
            except (EntryNotFoundError, RepositoryNotFoundError) as error:
                raise Rejected(f"Kyutai's voice library has no voice called {path!r}; hands also speaks in {', '.join(VOICES)}") from error
        case _:
            download_if_necessary(get_predefined_voice(language=language_to_pocket_tts_language(LANGUAGE), name=voice))


def sample(voice: Chosen) -> str:
    """What a voice says when the user hears it: its name, and a line of hands' own that reports nothing, so said again
    after a barge-in it cannot pass for a session's news."""
    return f"This is {spoken(voice)}. This is how I would tell you a session has finished its turn and is waiting for you."


class Voices:
    """The voices hands can speak in, heard and chosen through `lines`, the processor standing ahead of the TTS service, and
    found in `library`.

    A change of voice is a frame in order with the speech around it, so what was handed to the speaker before it is
    said in the old voice and what comes after in the new. Pipecat keeps a settings frame through a barge-in, which
    drops only the speech, so a hearing cut short still ends back in the chosen voice.
    """

    def __init__(self, home: Home, lines: FrameProcessor, fetch: Fetch, library: Library) -> None:
        self._home = home
        self._lines = lines
        self._fetch = fetch
        self._library = library
        # [LAW:no-ambient-temporal-coupling] the one owner of a change of voice: Pipecat runs a reply's calls side by
        # side, and a hearing and a choice interleaved would leave a sample in another's voice, or the speaker in a
        # voice the file does not hold. Each runs whole, one after another.
        self._changing = asyncio.Lock()

    async def speaking_in(self) -> Chosen:
        return await asyncio.to_thread(chosen, self._home)

    async def search(self, query: str, limit: int) -> Found:
        """The library voices whose paths contain `query`, and when the listing searched was made."""
        return await self._library.search(query, limit, frozenset(name.lower() for name in VOICES))

    async def hear(self, voices: tuple[Chosen, ...]) -> Chosen:
        """Each voice says its sample, in order, and then hands goes back to the voice it speaks in, which it returns.
        A voice that cannot be fetched is refused before any is heard."""
        async with self._changing:
            now = await self.speaking_in()
            await asyncio.gather(*(asyncio.to_thread(self._fetch, voice) for voice in voices))
            try:
                for voice in voices:
                    await self._speak_in(voice)
                    await self._lines.push_frame(TTSSpeakFrame(sample(voice), append_to_context=False))
            finally:
                # A hearing cancelled or failed partway still hands the speaker back the voice it speaks in.
                await self._speak_in(now)
            return now

    async def use(self, voice: Chosen) -> None:
        """From the next thing hands says on, and across restarts, it speaks in `voice`; a voice whose prompt cannot be
        fetched is refused before it is kept. Whether the prompt decodes is not checked here: a file that downloads but
        does not encode fails the start, not the choice."""
        async with self._changing:
            await asyncio.to_thread(self._fetch, voice)
            await asyncio.to_thread(keep, self._home, voice)
            await self._speak_in(voice)

    async def _speak_in(self, voice: Chosen) -> None:
        await self._lines.push_frame(TTSUpdateSettingsFrame(delta=PocketTTSSettings(voice=source(voice))))
