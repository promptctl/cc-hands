"""The files the wake word is heard with, and the one owner of fetching them: the desk (`hands.voice.wake`) hears with them
and the phone's page (`hands.voice.phonewake`) runs them in a browser, so what the two ask for together is fetched once.
This module loads none of them.

openWakeWord's release is fetched the way every file hands runs a model from is (`hands.voice.pinnedfile`): each file
held to the digest pinned here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from hands.voice.pinnedfile import Fetch, Pin, Release, Unserved, fetched_off_the_loop
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained, Word

VERSION = "v0.5.1"
RELEASE = f"https://github.com/dscripka/openWakeWord/releases/download/{VERSION}"
# The two models that turn audio into the features every wake word model hears.
MELSPECTROGRAM = "melspectrogram.onnx"
EMBEDDING = "embedding_model.onnx"
# [LAW:one-source-of-truth] every file of the release hands runs, by the digest of its file: the two feature models and a
# model of each of openWakeWord's own wake words (`PRETRAINED`).
DIGESTS: Mapping[str, str] = {
    MELSPECTROGRAM: "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    EMBEDDING: "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    "hey_jarvis_v0.1.onnx": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    "hey_mycroft_v0.1.onnx": "c2a311e8fa1338de89c31b3b46dc4dffd4af2f9a8d6ddead48893c2d301b1f18",
    "hey_rhasspy_v0.1.onnx": "5a9b3ed3be2910e35780e097905aa9f35a9c10038df47914cf2b3ec4d670f6ea",
    "alexa_v0.1.onnx": "6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e",
}
# openWakeWord's own default: its pretrained models are tuned to score a wake word above it and little else.
THRESHOLD = 0.5


class Unheard(Exception):
    """A model the wake word cannot be heard with, and why."""


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


class WakeModels:
    """The wake word's models in `directory`, fetched when they are first asked for."""

    def __init__(self, directory: Path, fetch: Fetch = fetched_off_the_loop) -> None:
        self._release = Release(directory, (Pin(name, f"{RELEASE}/{name}", digest, f"openWakeWord {VERSION}") for name, digest in DIGESTS.items()), fetch)

    async def ensured(self, word: Word) -> Files:
        """`word`'s three models, fetched if they are not there; raises Unheard naming why `word` cannot be heard.

        A word the user trained is never fetched: its model is theirs, where they keep it.
        """
        try:
            melspectrogram = await self._release.path(MELSPECTROGRAM)
            embedding = await self._release.path(EMBEDDING)
            match word:
                case Pretrained(phrase=phrase):
                    own = await self._release.path(PRETRAINED[phrase])
                case Trained(model=model):
                    own = model
        except Unserved as error:
            raise Unheard(str(error)) from error
        # [LAW:single-enforcer] the one judge of whether a model is there to be heard with, whoever's it is.
        if not own.is_file():
            raise Unheard(f"no wake word model at {own}")
        return Files(melspectrogram, embedding, own)
