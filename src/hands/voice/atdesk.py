"""The desk served: its driver (`hands.voice.engaged`) wired to the gate, the level's owner, the speaker's cues, and the
models it hears with, as the daemon runs it once the pipeline has opened its streams."""

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Protocol

from loguru import logger

from hands.sessions.wide import WideEvent, annotate, unit
from hands.voice import wake
from hands.voice.cues import Cue, Played, level_cue
from hands.voice.desk import Leveled, Made
from hands.voice.engaged import Ears, Hears, drive, loaded
from hands.voice.hold import KeyEvent, Move
from hands.voice.ptt import PushToTalk
from hands.voice.trigger import Levels
from hands.voice.wakeword import Word


class Microphone(Protocol):
    """The desk's microphone, as the desk hears it between turns."""

    @property
    def sample_rate(self) -> int: ...

    def overheard(self) -> AbstractAsyncContextManager[AsyncIterator[bytes]]: ...


class Speaker(Protocol):
    """The speaker attached now: it plays a cue at once, and says whether hands is speaking."""

    @property
    def hands_speaking(self) -> bool: ...

    def cue(self, cue: Cue) -> Played: ...


async def serve_desk(
    *,
    key: PushToTalk,
    levels: Levels,
    tapped: Callable[[Callable[[KeyEvent], None]], AbstractAsyncContextManager[None]],
    microphone: Microphone,
    speaker: Callable[[], Speaker],
    models: Path,
    word: Word,
    taken: Callable[[Move], Awaitable[None]],
    beat: Callable[[], None],
    record: Callable[[WideEvent], None],
) -> None:
    """Drive the desk until cancelled: each move it makes goes to the gate, and what the gate takes it as to `taken`; each
    change of level goes to `levels`, plays its cue, and is beaten."""

    async def at_desk(made: Made) -> None:
        match key.move(made.move, made.by):
            case None:
                # A Shift at the desk while the turn is the phone's: nothing of it is cued, said, or beaten.
                return
            case move:
                await taken(move)

    async def at_level(leveled: Leveled) -> None:
        levels.leveled(leveled)
        cue = level_cue(leveled)
        logger.info(cue.line)
        speaker().cue(cue)
        beat()

    @asynccontextmanager
    async def hearing() -> AsyncGenerator[tuple[Ears, Hears]]:
        rate = microphone.sample_rate
        # [LAW:no-silent-failure] the models every level hears with are fetched and loaded once, beside the talk key as
        # the desk starts: one that cannot be is the run's failure, said with what could not be loaded, never a level
        # that fails as it is chosen.
        # [LAW:nothing-unseen] the fetch is its own unit of work, naming the files it fetched, none once they are kept.
        with unit("desk.fetched", record):
            annotate(fetched=", ".join(await wake.fetched(models, word)))
        async with loaded(rate, record) as ears, wake.loaded(rate, models, word, record) as heard:
            yield ears, wake.listening(heard, lambda: speaker().hands_speaking, record)

    await drive(tapped, microphone.overheard, hearing, levels.chosen, at_desk, at_level, record)
