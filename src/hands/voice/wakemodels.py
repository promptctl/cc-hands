"""The files the wake word is heard with, and the one owner of fetching them: the desk (`hands.voice.wake`) hears with them
and the phone's page (`hands.voice.phonewake`) runs them in a browser, so what the two ask for together is fetched once.
This module loads none of them.

openWakeWord's release is fetched the way every file hands runs a model from is (`hands.voice.pinnedfile`): each file
held to the digest pinned here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from hands.clock import Clock
from hands.voice.pinnedfile import Fetch, How, Pin, Release, Unserved, fetched_off_the_loop
from hands.voice.wakeword import Phrase, Pretrained, Trained, Word

VERSION = "v0.5.1"
RELEASE = f"https://github.com/dscripka/openWakeWord/releases/download/{VERSION}"


def _pin(name: str, digest: str) -> Pin:
    return Pin(name, f"{RELEASE}/{name}", digest, f"openWakeWord {VERSION}")


# [LAW:one-source-of-truth] every file of the release hands runs, each as the pin that holds it to its digest. The two
# models that turn audio into the features every wake word model hears:
MELSPECTROGRAM = _pin("melspectrogram.onnx", "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f")
EMBEDDING = _pin("embedding_model.onnx", "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f")
# and a model of each of openWakeWord's own wake words, by how each is said.
PRETRAINED: Mapping[Phrase, Pin] = {
    "Hey Jarvis": _pin("hey_jarvis_v0.1.onnx", "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"),
    "Hey Mycroft": _pin("hey_mycroft_v0.1.onnx", "c2a311e8fa1338de89c31b3b46dc4dffd4af2f9a8d6ddead48893c2d301b1f18"),
    "Hey Rhasspy": _pin("hey_rhasspy_v0.1.onnx", "5a9b3ed3be2910e35780e097905aa9f35a9c10038df47914cf2b3ec4d670f6ea"),
    "Alexa": _pin("alexa_v0.1.onnx", "6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e"),
}


@dataclass(frozen=True)
class Files:
    """The three models the wake word is heard with, each in its place, each the file it should be. A field is the role
    the model plays in the stream of openWakeWord's steps, which is also the name the phone's page asks it by."""

    melspectrogram: Path
    embedding: Path
    word: Path

    def by_role(self) -> dict[str, Path]:
        return {role: getattr(self, role) for role in ROLES}


ROLES = tuple(field.name for field in fields(Files))


@dataclass(frozen=True)
class Ensured:
    """What an ask for a word's models got: the files, and how this ask came by each of openWakeWord's among them (a trained
    word's is its owner's, so not one of those)."""

    files: Files
    how: Mapping[str, How]

    def facts(self) -> dict[str, str]:
        """What the unit of work that asked says of it: the files heard with, those it fetched, and those it wrote over."""
        return {
            "files": ", ".join(path.name for path in self.files.by_role().values()),
            "fetched": ", ".join(name for name, how in self.how.items() if how == "fetched"),
            "replaced": ", ".join(name for name, how in self.how.items() if how == "replaced"),
        }


class WakeModels:
    """The wake word's models in `directory`, fetched when they are first asked for."""

    def __init__(self, directory: Path, clock: Clock, fetch: Fetch = fetched_off_the_loop) -> None:
        self._release = Release(directory, clock, fetch)

    async def ensured(self, word: Word) -> Ensured:
        """`word`'s three models, fetched if they are not there; raises Unserved naming why one cannot be had.

        A word the user trained is never fetched: its model is theirs, where they keep it.
        """
        melspectrogram = await self._release.path(MELSPECTROGRAM)
        embedding = await self._release.path(EMBEDDING)
        kept = [(MELSPECTROGRAM, melspectrogram), (EMBEDDING, embedding)]
        match word:
            case Pretrained(phrase=phrase):
                pin = PRETRAINED[phrase]
                got = await self._release.path(pin)
                kept.append((pin, got))
                own_path = got.path
            case Trained(model=model):
                own_path = model
        # [LAW:single-enforcer] the one judge of whether a model is there to be heard with, whoever's it is.
        if not own_path.is_file():
            raise Unserved(f"no wake word model at {own_path}")
        return Ensured(Files(melspectrogram.path, embedding.path, own_path), {pin.name: got.how for pin, got in kept})
