"""The voice the user chose, by its name: one of Pocket TTS's own, or one in Kyutai's voice library, and the one kept
in the home for the next start. Free of the speech pipeline, torch's, so what reads a kept voice need not load it."""

import ast
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from hands.sessions.files import replace_whole
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

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


def _predefined() -> tuple[Voice, ...]:
    """The names the installed pocket_tts resolves itself, read from its source: importing any of pocket_tts loads torch."""
    # find_spec of the top package imports nothing; of a module inside it, it would import the package.
    found = importlib.util.find_spec("pocket_tts")
    if found is None or found.origin is None:
        raise ImportError("pocket_tts is not installed, so hands has none of its voices")
    source = Path(found.origin).parent / "utils" / "utils.py"
    for statement in ast.parse(source.read_text()).body:
        match statement:
            case ast.Assign(targets=[ast.Name(id="_ORIGINS_OF_PREDEFINED_VOICES")], value=ast.Dict(keys=keys)):
                return tuple(Voice(ast.literal_eval(key)) for key in keys if key is not None)
            case _:
                pass
    # [LAW:no-silent-failure] a pocket_tts that keeps its voices elsewhere is one hands was not written against.
    raise ImportError(f"{source} names no _ORIGINS_OF_PREDEFINED_VOICES, where the pocket_tts hands was written against keeps its voices")


# [LAW:one-source-of-truth] the voices on offer are the ones the installed pocket_tts resolves by name itself, each to
# a state primed by the very weights it loads; a state primed elsewhere is a cache from another network, and the loader
# cannot tell.
VOICES: tuple[Voice, ...] = _predefined()

# The voice hands speaks in until the user chooses another (the user's choice of default, 2026-10-02).
DEFAULT = Voice("charles")

# A library path: at least one repo folder and a file, joined by single slashes, each part a letter or digit followed by
# letters, digits, `_`, `.`, or `-`. A bare name is never a library voice, so a mistyped built-in name is refused here.
LIBRARY_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9][A-Za-z0-9_.\-]*)+")


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

