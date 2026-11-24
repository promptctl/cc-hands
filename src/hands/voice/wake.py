"""The wake word: at the desk's wake word level (`hands.voice.desk`), saying it opens a turn with no hand on the key,
and end-of-turn detection closes it, with the models engaged conversation hears with (`hands.voice.engaged`).

The word is config.toml's (`hands.daemon.config`): one of openWakeWord's own, "Hey Jarvis" unless another is named, or
one the user trained with openWakeWord. Its model listens to the desk's microphone through the echo canceller. It hears
what engaged conversation hears, the room with hands' own voice taken out by the echo canceller once it has read that
hands' voice reaches the microphone (`hands.voice.echo.EchoPath`), a quarter of a second into a reply, and a second or
more where the microphone or speaker opened mid-reply; an echo under the room's own sound is never read, and is heard
as it was, under that sound, while hands speaks as much as while it is quiet: the wake word
said over a reply opens a turn that cuts it off. The desk listens for as long as the level is the wake word's, so the
clip owner keeps the second before each turn opens and the wake word, and anything said while the detector made sure
of it, are in the turn.

On the Mac's own microphone and speakers, hands' voice left over from the canceller can say the wake word as the user
would, and a turn opened over it would cut the reply off on hands' own word and send the brain hands' voice as the
user's. There, a wake with hands' voice in what the turn would keep opens nothing (`hands.voice.desk`): hands said it, or
the user said it over hands, and the talk key still cuts in.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import cast

import numpy as np
from openwakeword.model import Model

from hands.clock import Clock
from hands.sessions import wide
from hands.sessions.wide import WideEvent, annotate, count, unit
from hands.voice import loads
from hands.voice.desk import Woken
from hands.voice.wakemodels import Files
from hands.voice.wakeword import THRESHOLD, Word

# The only rate openWakeWord's models hear.
SAMPLE_RATE = 16_000


class Unheard(Exception):
    """A model the wake word cannot be heard with, and why."""


class WakeWord:
    """A wake word's model, run locally on the CPU with ONNX Runtime, on 16 kHz mono audio, from `files`
    (`hands.voice.wakemodels`). Raises Unheard for a model it cannot hear the word with."""

    def __init__(self, files: Files) -> None:
        heard = files.word
        try:
            self._model = Model(
                wakeword_models=[str(heard)], melspec_model_path=str(files.melspectrogram), embedding_model_path=str(files.embedding), inference_framework="onnx"
            )
        # ONNX Runtime's errors share no base nearer than Exception; each is said as the model's, with its own words.
        except Exception as error:
            raise Unheard(f"the wake word model at {heard} cannot be loaded: {error}") from error
        # The model's score is given under its file's name, for a model that scores one word; one that scores several
        # gives each under the name of its class.
        self._name = heard.stem
        if (scores := cast(dict[str, int], self._model.model_outputs)[self._name]) != 1:  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
            raise Unheard(f"the wake word model at {heard} scores {scores} words, and a wake word is one")

    def score(self, audio: bytes) -> float:
        """How sure the model is that the wake word has just been said, from 0 to 1, with `audio` heard last."""
        # Without `timing`, each model's score by the model's name.
        scores = cast(dict[str, float], self._model.predict(np.frombuffer(audio, dtype=np.int16)))  # pyright: ignore[reportUnknownMemberType]  (untyped in openWakeWord)
        return float(scores[self._name])

    def reset(self) -> None:
        """Forget the audio heard so far, so one saying of the wake word wakes hands once."""
        self._model.reset()


@asynccontextmanager
async def loaded(clock: Clock, sample_rate: int, files: Files, word: Word, emit: Callable[[WideEvent], None]) -> AsyncGenerator[WakeWord]:
    """`word`'s model, loaded off the loop as its own unit of work."""
    with loads.loading(clock, "desk.wake_word_loaded", emit):
        # [LAW:nothing-unseen] which word the desk listens for, and the file it is heard with.
        annotate(phrase=word.phrase, model=str(files.word))
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"the wake word is heard at {SAMPLE_RATE} Hz, and the desk's microphone runs at {sample_rate} Hz")
        heard = await asyncio.to_thread(WakeWord, files)
    yield heard


def listening(
    clock: Clock, word: WakeWord, over_hands: Callable[[], bool], built_in: Callable[[], bool], emit: Callable[[WideEvent], None]
) -> Callable[[bytes, bool], Awaitable[Woken | None]]:
    """The wake word each buffer of the desk's microphone ends with, if it does, heard afresh from a buffer the wake
    word's level `begun` with, with nothing of the room before it, and what `over_hands` and `built_in` say as it is."""

    async def woken(audio: bytes, begun: bool) -> Woken | None:
        if begun:
            await asyncio.to_thread(word.reset)
        score = await asyncio.to_thread(word.score, audio)
        if score < THRESHOLD:
            return None
        # Off the loop as scoring is: forgetting refills the model with seconds of features. CoreAudio is asked off it too.
        await asyncio.to_thread(word.reset)
        # [LAW:one-source-of-truth] the clip owner's own second, read as the turn is about to open on it: what it holds then
        # is what the turn keeps, however far behind the microphone the desk is.
        hands, on_built_in = over_hands(), await asyncio.to_thread(built_in)
        count(wake_words=1)
        # [LAW:nothing-unseen] each hearing is its own event, with how sure the model was and the facts the desk takes it
        # by. Whether a wake opened a turn is the desk's to say, as its count of starts.
        with unit("desk.wake_word_heard", emit, wide.begun(clock)):
            annotate(score=score, threshold=THRESHOLD, over_hands=hands, built_in=on_built_in)
        return Woken(clock.now(), over_hands=hands, built_in=on_built_in)

    return woken
